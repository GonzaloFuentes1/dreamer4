# src/collector.py
# Phase 0 — Episode collection from DMControl environments.
#
# Supports two policy modes:
#   "random" — zero-mean Gaussian actions, no model needed.
#   "agent"  — loads Encoder + Dynamics + TaskEmbedder + PolicyHead from a
#              Phase-2 (finetune.ckpt) or Phase-3 (agent.ckpt) Lightning
#              checkpoint and runs a receding-horizon context window.
#
# Output format (matches WMDataset exactly):
#   {out_data_dir}/{task}.pt          → {"episode": (N,), "action": (N,16), "reward": (N,)}
#   {out_frames_dir}/{task}/
#       {task}_shard{n:04d}.pt        → {"frames": (K, 3, H, W) uint8}
#   {tasks_json}                      → {"task": {"action_dim": int, "text_embedding": [512 × 0.0]}}

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Force headless EGL rendering via NVIDIA drivers — works on GPU servers without X11.
# Override with MUJOCO_GL=osmesa if no GPU EGL drivers are available.
os.environ.setdefault("MUJOCO_GL", "egl")
# Do NOT set PYOPENGL_PLATFORM — dm_control manages it internally via MUJOCO_GL.

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# DMControl helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_env(task: str, img_size: int, camera_id: int = 0):
    """Create a DMControl environment for `task` (e.g. 'walker-walk')."""
    try:
        from dm_control import suite
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "dm_control is required for Phase-0 collection. "
            "Install it with: uv add dm-control"
        )
    # Some task names use a shorthand that differs from the dm_control domain name
    _DOMAIN_ALIAS = {
        "cup": "ball_in_cup",
    }
    domain, task_name = task.split("-", 1)
    domain = _DOMAIN_ALIAS.get(domain, domain)
    # Convert hyphenated task names with more than one segment: e.g. walker-run-backward
    # dm_control uses underscores for tasks
    task_name = task_name.replace("-", "_")
    env = suite.load(domain_name=domain, task_name=task_name,
                     task_kwargs={"random": 0},
                     visualize_reward=False)
    return env


def _render_frame(env, img_size: int, camera_id: int) -> torch.Tensor:
    """Render a (3, H, W) uint8 frame from the environment."""
    frame = env.physics.render(height=img_size, width=img_size, camera_id=camera_id)
    # frame: (H, W, 3) uint8 — copy() needed: dm_control may return negative-stride arrays
    return torch.from_numpy(frame.copy()).permute(2, 0, 1).contiguous()  # (3, H, W)


def _action_spec_dim(env) -> int:
    """Return the actual continuous action dimension for this env."""
    spec = env.action_spec()
    return int(np.prod(spec.shape))


def _sample_random_action(env, action_dim: int, max_dim: int) -> np.ndarray:
    """Sample a uniformly random action from the env's action spec."""
    spec = env.action_spec()
    lo   = spec.minimum  # (act_dim,) or scalar
    hi   = spec.maximum
    act  = np.random.uniform(lo, hi, size=spec.shape).astype(np.float32)
    # Zero-pad to max_dim
    padded = np.zeros(max_dim, dtype=np.float32)
    padded[:act.shape[0]] = act
    return padded


def _flatten_obs(ts) -> np.ndarray:
    """Flatten a DMControl TimeStep observation dict to a 1-D vector (unused by WM, kept for compatibility)."""
    obs = ts.observation
    return np.concatenate([v.flatten() for v in obs.values()]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Agent policy (loads Phase-2 or Phase-3 checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

class AgentPolicy:
    """
    Wraps a trained PolicyHead + Dynamics to produce actions given pixel observations.

    Maintains a receding context window of length `context_len`.
    Compatible with both Phase-2 (finetune.ckpt) and Phase-3 (agent.ckpt)
    Lightning checkpoints — both have the same relevant keys in state_dict.
    """

    def __init__(
        self,
        ckpt_path: str,
        task_name: str,
        context_len: int,
        packing_factor: int,
        action_noise: float,
        device: torch.device,
        verbose: bool = True,
    ):
        self.context_len  = context_len
        self.packing_factor = packing_factor
        self.action_noise = action_noise
        self.device       = device
        self.verbose      = verbose

        # Lazy import src modules (caller inserts src/ into sys.path)
        from model import (
            Encoder, Dynamics, TaskEmbedder,
            temporal_patchify, pack_bottleneck_to_spatial,
        )
        from agent import PolicyHead

        ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd     = ckpt.get("state_dict", {})
        hp     = ckpt.get("hyper_parameters", {})
        cfg    = hp.get("cfg", {})

        # Figure out which phase this checkpoint is from
        # Phase 2 stores params under cfg.finetune, Phase 3 under cfg.agent
        ft_cfg = cfg.get("finetune", {}) or {}
        ag_cfg = cfg.get("agent", {}) or {}
        # Prefer finetune (has more architecture info); fall back to agent
        src_cfg = ft_cfg if ft_cfg else ag_cfg

        # ── Tokenizer / encoder geometry ──────────────────────────────────
        patch        = int(src_cfg.get("patch",        4))
        H            = int(src_cfg.get("H",          128))
        W            = int(src_cfg.get("W",          128))
        C            = int(src_cfg.get("C",            3))
        n_latents    = int(src_cfg.get("n_latents",   16))
        d_bottleneck = int(src_cfg.get("d_bottleneck", 32))
        d_enc_model  = int(src_cfg.get("d_enc_model", 256))
        n_enc_heads  = int(src_cfg.get("n_enc_heads",   4))
        enc_depth    = int(src_cfg.get("enc_depth",     8))
        mlp_ratio    = float(src_cfg.get("mlp_ratio", 4.0))
        time_every   = int(src_cfg.get("time_every",   1))
        lat_only     = bool(src_cfg.get("latents_only_time", True))

        self._patch = patch
        self._H, self._W, self._C = H, W, C
        n_spatial  = n_latents // packing_factor
        d_spatial  = d_bottleneck * packing_factor
        self.n_spatial = n_spatial
        self.d_spatial = d_spatial
        n_patches  = (H // patch) ** 2
        d_patch    = patch ** 2 * C

        # ── Dynamics geometry ──────────────────────────────────────────────
        d_model_dyn  = int(src_cfg.get("d_model_dyn",  512))
        dyn_depth    = int(src_cfg.get("dyn_depth",      8))
        n_agent      = int(src_cfg.get("n_agent",         1))
        n_register   = int(src_cfg.get("n_register",      4))
        n_heads_dyn  = int(src_cfg.get("n_heads",         8))
        k_max        = int(src_cfg.get("k_max",          16))
        space_mode   = str(src_cfg.get("space_mode", "wm_agent_isolated"))
        action_dim   = int(src_cfg.get("action_dim",     16))
        hidden_dim   = int(src_cfg.get("hidden_dim",    512))
        mtp_length   = int(src_cfg.get("mtp_length",      8))
        n_tasks      = int(src_cfg.get("n_tasks",        30))
        use_task_ids = bool(src_cfg.get("use_task_ids", False))

        self.k_max     = k_max
        self.n_agent   = n_agent
        self.d_model   = d_model_dyn
        self.action_dim = action_dim
        self.use_task_ids = use_task_ids

        # ── Build & load Encoder ───────────────────────────────────────────
        enc = Encoder(
            patch_dim=d_patch, d_model=d_enc_model, n_latents=n_latents,
            n_patches=n_patches, n_heads=n_enc_heads, depth=enc_depth,
            d_bottleneck=d_bottleneck, dropout=0.0, mlp_ratio=mlp_ratio,
            time_every=time_every, latents_only_time=lat_only,
            mae_p_min=0.0, mae_p_max=0.0,
        )
        enc_sd = {k[len("_encoder."):]: v for k, v in sd.items() if k.startswith("_encoder.")}
        if enc_sd:
            enc.load_state_dict(enc_sd, strict=True)
        elif verbose:
            print("[AgentPolicy] WARNING: no _encoder keys found in checkpoint")

        # ── Build & load Dynamics ──────────────────────────────────────────
        dyn = Dynamics(
            d_model=d_model_dyn, d_bottleneck=d_bottleneck, d_spatial=d_spatial,
            n_spatial=n_spatial, n_register=n_register, n_agent=n_agent,
            n_heads=n_heads_dyn, depth=dyn_depth, k_max=k_max,
            dropout=0.0, mlp_ratio=mlp_ratio, time_every=time_every,
            space_mode=space_mode,
        )
        dyn_sd = {k[4:]: v for k, v in sd.items() if k.startswith("dyn.")}
        if dyn_sd:
            dyn.load_state_dict(dyn_sd, strict=True)
        elif verbose:
            print("[AgentPolicy] WARNING: no dyn. keys found in checkpoint")

        # ── Build & load TaskEmbedder ──────────────────────────────────────
        te = TaskEmbedder(
            d_model=d_model_dyn, n_agent=n_agent,
            use_ids=use_task_ids, n_tasks=n_tasks, d_task=512,
        )
        te_sd = {k[len("task_embedder."):]: v for k, v in sd.items() if k.startswith("task_embedder.")}
        if te_sd:
            te.load_state_dict(te_sd, strict=True)

        # ── Build & load PolicyHead ────────────────────────────────────────
        state_dim = n_agent * d_model_dyn
        ph = PolicyHead(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dim=hidden_dim, mtp_length=mtp_length,
        )
        ph_sd = {k[len("policy_head."):]: v for k, v in sd.items() if k.startswith("policy_head.")}
        if ph_sd:
            ph.load_state_dict(ph_sd, strict=True)
        elif verbose:
            print("[AgentPolicy] WARNING: no policy_head keys — actions will be near-random")

        for m in (enc, dyn, te, ph):
            m.to(device).eval()
            for p in m.parameters():
                p.requires_grad_(False)

        self._encoder      = enc
        self._dyn          = dyn
        self._task_embedder = te
        self._policy_head  = ph

        # Precomputed forward-pass constants
        emax = int(round(math.log2(k_max)))
        self._step_val  = emax
        self._sig_val   = k_max - 1

        # Rolling context buffers (filled lazily on reset)
        self._z_buf:   Optional[torch.Tensor] = None  # (1, T_ctx, n_spatial, d_spatial)
        self._act_buf: Optional[torch.Tensor] = None  # (1, T_ctx, 16)
        self._task_tok: Optional[torch.Tensor] = None # (1, T_ctx, n_agent, d_model)

        # Task index / lang_emb (set per episode via set_task)
        self._task_input: Optional[torch.Tensor] = None

        if verbose:
            print(f"[AgentPolicy] Loaded from {ckpt_path}  "
                  f"state_dim={state_dim}  action_dim={action_dim}")

    # ------------------------------------------------------------------
    def set_task(self, task_idx: int = 0, lang_emb: Optional[torch.Tensor] = None):
        """Call before each episode to set the task embedding."""
        if self.use_task_ids:
            self._task_input = torch.tensor([[task_idx]], device=self.device, dtype=torch.long)  # (1,1)
        else:
            if lang_emb is None:
                lang_emb = torch.zeros(512, device=self.device)
            self._task_input = lang_emb.to(self.device).unsqueeze(0)  # (1,512)

    def reset(self):
        """Clear context buffers at the start of each episode."""
        self._z_buf    = None
        self._act_buf  = None
        self._task_tok = None

    @torch.no_grad()
    def act(self, frame_u8: torch.Tensor, prev_action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        frame_u8  : (3, H, W) uint8 CPU tensor — current observation.
        prev_action: (16,)  float32 — last executed action (zeros on first step).
        Returns     (16,)  float32 action tensor.
        """
        from model import temporal_patchify, pack_bottleneck_to_spatial

        device = self.device
        B      = 1

        # ── Encode current frame ──────────────────────────────────────────
        frame_f = frame_u8.float().div(255.0).to(device)            # (3,H,W)
        frame_f = frame_f.unsqueeze(0).unsqueeze(0)                  # (1,1,3,H,W)
        patches = temporal_patchify(frame_f, self._patch)            # (1,1,Np,Dp)
        z_btLd, _ = self._encoder(patches)                           # (1,1,n_lat,d_bot)
        z_new = pack_bottleneck_to_spatial(
            z_btLd, n_spatial=self.n_spatial, k=self.packing_factor
        )                                                             # (1,1,n_spatial,d_spatial)

        # ── Extend context buffers ────────────────────────────────────────
        if prev_action is None:
            prev_action = torch.zeros(self.action_dim, device=device)
        act_new = prev_action.to(device).view(1, 1, -1)              # (1,1,16)

        task_new = self._task_embedder(
            self._task_input, B=B, T=1
        )                                                             # (1,1,n_agent,d_model)

        if self._z_buf is None:
            self._z_buf    = z_new
            self._act_buf  = act_new
            self._task_tok = task_new
        else:
            self._z_buf    = torch.cat([self._z_buf,    z_new],    dim=1)
            self._act_buf  = torch.cat([self._act_buf,  act_new],  dim=1)
            self._task_tok = torch.cat([self._task_tok, task_new], dim=1)

        # Trim to context window
        if self._z_buf.shape[1] > self.context_len:
            self._z_buf    = self._z_buf[:,    -self.context_len:]
            self._act_buf  = self._act_buf[:,  -self.context_len:]
            self._task_tok = self._task_tok[:, -self.context_len:]

        T_ctx = self._z_buf.shape[1]

        # ── Clean Dynamics forward → h_t ──────────────────────────────────
        step = torch.full((B, T_ctx), self._step_val, device=device, dtype=torch.long)
        sig  = torch.full((B, T_ctx), self._sig_val,  device=device, dtype=torch.long)

        _, h_t = self._dyn(
            self._act_buf, step, sig, self._z_buf,
            act_mask=None,
            agent_tokens=self._task_tok,
        )                                                             # h_t: (1,T_ctx,n_agent,d_model)

        h_flat = h_t[:, -1].flatten(1)                               # (1, state_dim)

        # ── Sample action ─────────────────────────────────────────────────
        act_mask = torch.ones(B, self.action_dim, device=device)
        _, action = self._policy_head.sample(h_flat, act_mask=act_mask)
        action = action.squeeze(0)                                    # (16,)

        # Exploration noise
        if self.action_noise > 0.0:
            noise  = torch.randn_like(action) * self.action_noise
            action = (action + noise).clamp(-1.0, 1.0)

        return action.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Single-task collector
# ─────────────────────────────────────────────────────────────────────────────

def collect_task(
    task: str,
    cfg,
    policy: Any,          # "random" string or AgentPolicy instance
    task_idx: int = 0,
) -> bool:
    """
    Collect `cfg.n_episodes_per_task` episodes for `task` and write shards.
    Returns True on success, False if task was skipped (already exists + overwrite=False).
    """
    out_demo   = Path(cfg.out_data_dir)   / f"{task}.pt"
    out_frames = Path(cfg.out_frames_dir) / task
    tasks_json = Path(cfg.tasks_json)

    if not cfg.overwrite and out_demo.exists():
        print(f"[collect] Skipping {task}: demo already exists at {out_demo}")
        return False

    out_demo.parent.mkdir(parents=True, exist_ok=True)
    out_frames.mkdir(parents=True, exist_ok=True)

    env = _make_env(task, cfg.img_size, cfg.camera_id)
    env_act_dim = _action_spec_dim(env)
    max_dim     = cfg.action_dim

    all_episodes: List[int] = []
    all_actions:  List[np.ndarray] = []
    all_rewards:  List[float] = []
    all_frames:   List[torch.Tensor] = []   # each: (3, H, W) uint8

    ep_id = 0
    for ep_i in range(cfg.n_episodes_per_task):
        ts = env.reset()

        if isinstance(policy, AgentPolicy):
            policy.set_task(task_idx=task_idx)
            policy.reset()

        prev_action = np.zeros(max_dim, dtype=np.float32)
        step        = 0

        while not ts.last() and step < cfg.episode_len:
            frame = _render_frame(env, cfg.img_size, cfg.camera_id)

            if policy == "random":
                action_np = _sample_random_action(env, env_act_dim, max_dim)
            else:
                action_t  = policy.act(
                    frame,
                    prev_action=torch.from_numpy(prev_action)
                )
                action_np = action_t.numpy()
                # Map from [-1,1] back to env action spec
                spec = env.action_spec()
                lo, hi = spec.minimum, spec.maximum
                env_action = ((action_np[:env_act_dim] + 1.0) / 2.0) * (hi - lo) + lo
                env_action = env_action.clip(lo, hi)
                action_np_env = env_action.astype(np.float32)
                # action_np stores the [-1,1] version for the WM
                # but we step with the env-range version
                ts = env.step(action_np_env)
                all_episodes.append(ep_id)
                all_actions.append(action_np.copy())
                all_rewards.append(float(ts.reward or 0.0))
                all_frames.append(frame)
                prev_action = action_np
                step += 1
                continue

            ts = env.step(action_np[:env_act_dim])
            all_episodes.append(ep_id)
            all_actions.append(action_np.copy())
            all_rewards.append(float(ts.reward or 0.0))
            all_frames.append(frame)
            prev_action = action_np
            step += 1

        ep_id += 1

        if cfg.verbose:
            total_rew = sum(all_rewards[-step:])
            print(f"  [collect/{task}] ep {ep_i+1}/{cfg.n_episodes_per_task}  "
                  f"steps={step}  ep_reward={total_rew:.2f}")

    if len(all_frames) == 0:
        print(f"[collect] WARNING: 0 frames collected for {task}, skipping.")
        return False

    # ── Save demo .pt ──────────────────────────────────────────────────────
    demo_data = {
        "episode": torch.tensor(all_episodes, dtype=torch.int64),
        "action":  torch.tensor(np.stack(all_actions), dtype=torch.float32),
        "reward":  torch.tensor(all_rewards, dtype=torch.float32),
    }
    _safe_save(demo_data, out_demo)

    # ── Save frame shards ──────────────────────────────────────────────────
    shard_size = int(cfg.shard_size)
    frames_tensor = torch.stack(all_frames)   # (N_total, 3, H, W) uint8
    n_frames = frames_tensor.shape[0]
    n_shards = math.ceil(n_frames / shard_size)
    for si in range(n_shards):
        shard_frames = frames_tensor[si * shard_size : (si + 1) * shard_size].clone()
        shard_path   = out_frames / f"{task}_shard{si:04d}.pt"
        _safe_save({"frames": shard_frames}, shard_path)

    print(f"[collect/{task}] saved {n_frames} frames in {n_shards} shards, "
          f"{len(set(all_episodes))} episodes")

    # ── Update tasks.json ──────────────────────────────────────────────────
    tasks_json.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if tasks_json.exists():
        try:
            with open(tasks_json) as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    existing[task] = {
        "action_dim":      env_act_dim,
        "text_embedding":  [0.0] * 512,   # placeholder; replace with real CLIP emb if available
    }
    with open(tasks_json, "w") as f:
        json.dump(existing, f, indent=2)

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_save(obj: Any, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[collect] ERROR saving {path}: {e}")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
