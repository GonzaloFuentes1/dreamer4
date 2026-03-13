# src/vec_collector.py
# Vectorized environment collection for Phase 0 — Option C.
#
# N DMControl envs step sequentially (physics is CPU-bound),
# but a single batched GPU call handles inference for all N envs at once.
# This eliminates N-1 idle GPU cycles vs N independent per-process copies.
#
# Only applies to policy=agent mode. Random policy uses the existing
# multi-process path which is already task-parallelised.
#
# Usage (from collect_phase0_data.py):
#   python collect_phase0_data.py \
#       collect.policy=agent collect.agent_ckpt=./logs/agent_ckpts/last.ckpt \
#       collect.vectorized=true collect.n_envs_per_batch=8

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# VectorizedAgentPolicy
# ─────────────────────────────────────────────────────────────────────────────

class VectorizedAgentPolicy:
    """
    Batched version of AgentPolicy: N env slots share one model copy.

    Each slot maintains its own context window inside a pre-allocated
    fixed-size buffer (n_envs, context_len, ...).  act() gathers the
    requested slot indices, runs one Encoder + Dynamics + PolicyHead
    forward pass, and scatters the updated buffers back.
    """

    def __init__(
        self,
        ckpt_path: str,
        n_envs: int,
        context_len: int,
        packing_factor: int,
        action_noise: float,
        device: torch.device,
        verbose: bool = True,
    ):
        self.n_envs       = n_envs
        self.context_len  = context_len
        self.packing_factor = packing_factor
        self.action_noise = action_noise
        self.device       = device
        self.verbose      = verbose

        from model import (
            Encoder, Dynamics, TaskEmbedder,
            temporal_patchify, pack_bottleneck_to_spatial,
        )
        from agent import PolicyHead

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd   = ckpt.get("state_dict", {})
        hp   = ckpt.get("hyper_parameters", {})
        cfg  = hp.get("cfg", {})

        ft_cfg  = cfg.get("finetune", {}) or {}
        ag_cfg  = cfg.get("agent",    {}) or {}
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

        self._patch    = patch
        self._H        = H
        self._W        = W
        self._C        = C
        n_spatial      = n_latents // packing_factor
        d_spatial      = d_bottleneck * packing_factor
        self.n_spatial = n_spatial
        self.d_spatial = d_spatial
        n_patches      = (H // patch) ** 2
        d_patch        = patch ** 2 * C

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

        self.k_max        = k_max
        self.n_agent      = n_agent
        self.d_model      = d_model_dyn
        self.action_dim   = action_dim
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
            print("[VecPolicy] WARNING: no _encoder keys found in checkpoint")

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
            print("[VecPolicy] WARNING: no dyn. keys found in checkpoint")

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
            print("[VecPolicy] WARNING: no policy_head keys — actions will be near-random")

        for m in (enc, dyn, te, ph):
            m.to(device).eval()
            for p in m.parameters():
                p.requires_grad_(False)

        self._encoder       = enc
        self._dyn           = dyn
        self._task_embedder = te
        self._policy_head   = ph

        emax = int(round(math.log2(k_max)))
        self._step_val = emax
        self._sig_val  = k_max - 1

        # Pre-allocated fixed-size context buffers: (n_envs, context_len, ...)
        # Indexing with slot_indices list uses PyTorch advanced indexing.
        self._z_buf    = torch.zeros(n_envs, context_len, n_spatial, d_spatial, device=device)
        self._act_buf  = torch.zeros(n_envs, context_len, action_dim,           device=device)
        self._task_tok = torch.zeros(n_envs, context_len, n_agent,   d_model_dyn, device=device)

        # Per-slot task input tensors (set via set_task before each task)
        self._task_inputs: List[Optional[torch.Tensor]] = [None] * n_envs

        if verbose:
            print(f"[VecPolicy] Loaded {ckpt_path}  n_envs={n_envs}  "
                  f"state_dim={state_dim}  action_dim={action_dim}")

    # ------------------------------------------------------------------
    def set_task(
        self,
        env_idx: int,
        task_idx: int = 0,
        lang_emb: Optional[torch.Tensor] = None,
    ) -> None:
        """Set the task embedding for slot env_idx. Call once per task assignment."""
        if self.use_task_ids:
            self._task_inputs[env_idx] = torch.tensor(
                task_idx, device=self.device, dtype=torch.long
            )  # scalar
        else:
            if lang_emb is None:
                lang_emb = torch.zeros(512, device=self.device)
            self._task_inputs[env_idx] = lang_emb.to(self.device)  # (512,)

    def reset(self, env_idx: int) -> None:
        """Zero out context buffers for slot env_idx. Call at episode start."""
        self._z_buf[env_idx]    = 0.0
        self._act_buf[env_idx]  = 0.0
        self._task_tok[env_idx] = 0.0

    @torch.no_grad()
    def act(
        self,
        frames_u8: List[torch.Tensor],
        prev_actions: torch.Tensor,
        slot_indices: List[int],
    ) -> torch.Tensor:
        """
        frames_u8   : list of B tensors, each (3, H, W) uint8 CPU.
        prev_actions: (B, action_dim) float32 CPU.
        slot_indices: list of B ints — which buffer slots correspond to these frames.
        Returns     : (B, action_dim) float32 CPU.
        """
        from model import temporal_patchify, pack_bottleneck_to_spatial

        B      = len(slot_indices)
        device = self.device
        sidx   = slot_indices

        # ── Encode frames ─────────────────────────────────────────────────
        frames_f  = torch.stack(frames_u8).float().div(255.0).to(device)  # (B,3,H,W)
        frames_f  = frames_f.unsqueeze(1)                                  # (B,1,3,H,W)
        patches   = temporal_patchify(frames_f, self._patch)               # (B,1,Np,Dp)
        z_btLd, _ = self._encoder(patches)                                 # (B,1,n_lat,d_bot)
        z_new     = pack_bottleneck_to_spatial(
            z_btLd, n_spatial=self.n_spatial, k=self.packing_factor
        )                                                                   # (B,1,n_spatial,d_spatial)

        # ── Action and task tokens ────────────────────────────────────────
        act_new = prev_actions.to(device).unsqueeze(1)                     # (B,1,action_dim)

        if self.use_task_ids:
            task_in = torch.stack([self._task_inputs[i] for i in sidx])   # (B,) long
        else:
            task_in = torch.stack([self._task_inputs[i] for i in sidx])   # (B,512)
        task_new = self._task_embedder(task_in, B=B, T=1)                  # (B,1,n_agent,d_model)

        # ── Shift-and-append into the fixed-size rolling buffers ──────────
        # Gather slots → (B, context_len, ...)
        z_ctx    = self._z_buf[sidx]
        act_ctx  = self._act_buf[sidx]
        task_ctx = self._task_tok[sidx]

        z_ctx    = torch.cat([z_ctx[:,    1:], z_new],    dim=1)
        act_ctx  = torch.cat([act_ctx[:,  1:], act_new],  dim=1)
        task_ctx = torch.cat([task_ctx[:, 1:], task_new], dim=1)

        # Scatter back
        self._z_buf[sidx]    = z_ctx
        self._act_buf[sidx]  = act_ctx
        self._task_tok[sidx] = task_ctx

        # ── Dynamics forward ──────────────────────────────────────────────
        T_ctx = self.context_len
        step  = torch.full((B, T_ctx), self._step_val, device=device, dtype=torch.long)
        sig   = torch.full((B, T_ctx), self._sig_val,  device=device, dtype=torch.long)

        _, h_t = self._dyn(
            act_ctx, step, sig, z_ctx,
            act_mask=None,
            agent_tokens=task_ctx,
        )                                                                   # (B,T_ctx,n_agent,d_model)

        h_flat = h_t[:, -1].flatten(1)                                     # (B, state_dim)

        # ── Sample actions ────────────────────────────────────────────────
        act_mask = torch.ones(B, self.action_dim, device=device)
        _, actions = self._policy_head.sample(h_flat, act_mask=act_mask)   # (B, action_dim)

        if self.action_noise > 0.0:
            noise   = torch.randn_like(actions) * self.action_noise
            actions = (actions + noise).clamp(-1.0, 1.0)

        return actions.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# collect_vectorized
# ─────────────────────────────────────────────────────────────────────────────

def _update_tasks_json(task: str, act_dim: int, cc: Any) -> None:
    tasks_json = Path(cc.tasks_json)
    tasks_json.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if tasks_json.exists():
        try:
            with open(tasks_json) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing[task] = {
        "action_dim":     act_dim,
        "text_embedding": [0.0] * 512,
    }
    with open(tasks_json, "w") as f:
        json.dump(existing, f, indent=2)


def collect_vectorized(
    tasks: List[str],
    task_idxs: List[int],
    cc: Any,
    vpolicy: VectorizedAgentPolicy,
) -> Dict[str, bool]:
    """
    Collect episodes for all tasks using vectorized GPU inference.

    tasks     : list of task names  (e.g. ["walker-walk", "cheetah-run"])
    task_idxs : list of int indices matching tasks
    cc        : OmegaConf collect config (collect namespace)
    vpolicy   : VectorizedAgentPolicy instance

    Returns   : dict {task_name: True/False} (True = data saved OK)
    """
    from collector import _make_env, _render_frame, _action_spec_dim, _safe_save

    n_envs     = vpolicy.n_envs
    out_data   = Path(cc.out_data_dir)
    out_frames = Path(cc.out_frames_dir)
    shard_size = int(cc.shard_size)
    n_episodes = int(cc.n_episodes_per_task)
    episode_len = int(cc.episode_len)

    # Filter already-collected tasks
    pending: List[tuple] = []
    for task, tidx in zip(tasks, task_idxs):
        out_demo = out_data / f"{task}.pt"
        if not cc.overwrite and out_demo.exists():
            print(f"[vec_collect] Skipping {task}: already exists")
        else:
            pending.append((task, tidx))

    results: Dict[str, bool] = {t: False for t in tasks}

    if not pending:
        print("[vec_collect] Nothing to collect.")
        return results

    queue_idx = 0

    # Per-slot state
    envs          : List[Any]            = [None] * n_envs
    slot_task     : List[Optional[str]]  = [None] * n_envs
    slot_task_idx : List[int]            = [0]    * n_envs
    slot_act_dim  : List[int]            = [0]    * n_envs
    slot_done_eps : List[int]            = [0]    * n_envs
    slot_step     : List[int]            = [0]    * n_envs
    slot_ep_id    : List[int]            = [0]    * n_envs

    # Per-episode frame buffer (flushed after each episode to avoid OOM)
    slot_ep_frames: List[List[torch.Tensor]] = [[] for _ in range(n_envs)]

    # Demo metadata per task (cheap: only episode/action/reward scalars)
    slot_meta: List[Dict[str, list]] = [
        {"episode": [], "action": [], "reward": []} for _ in range(n_envs)
    ]

    # Shard-flush state per task
    slot_shard_buf: List[List[torch.Tensor]] = [[] for _ in range(n_envs)]
    slot_shard_idx: List[int]                = [0]  * n_envs

    prev_actions = torch.zeros(n_envs, vpolicy.action_dim)

    # ── Helpers ────────────────────────────────────────────────────────────

    def assign_task(slot: int) -> bool:
        nonlocal queue_idx
        if queue_idx >= len(pending):
            return False
        task, tidx = pending[queue_idx]
        queue_idx += 1

        out_data.mkdir(parents=True, exist_ok=True)
        (out_frames / task).mkdir(parents=True, exist_ok=True)

        envs[slot]          = _make_env(task, cc.img_size, cc.camera_id)
        slot_task[slot]     = task
        slot_task_idx[slot] = tidx
        slot_act_dim[slot]  = _action_spec_dim(envs[slot])
        slot_done_eps[slot] = 0
        slot_step[slot]     = 0
        slot_ep_id[slot]    = 0
        slot_ep_frames[slot] = []
        slot_meta[slot]      = {"episode": [], "action": [], "reward": []}
        slot_shard_buf[slot] = []
        slot_shard_idx[slot] = 0

        vpolicy.set_task(slot, tidx)
        vpolicy.reset(slot)
        envs[slot].reset()
        prev_actions[slot] = 0.0
        return True

    def flush_shards(slot: int) -> None:
        """Flush complete shard_size chunks from the shard buffer."""
        task    = slot_task[slot]
        out_dir = out_frames / task
        while sum(f.shape[0] for f in slot_shard_buf[slot]) >= shard_size:
            concat    = torch.cat(slot_shard_buf[slot], dim=0)
            to_save   = concat[:shard_size]
            remainder = concat[shard_size:] if concat.shape[0] > shard_size else None
            out_path  = out_dir / f"{task}_shard{slot_shard_idx[slot]:04d}.pt"
            _safe_save({"frames": to_save}, out_path)
            slot_shard_idx[slot] += 1
            slot_shard_buf[slot]  = (
                [remainder] if remainder is not None and remainder.shape[0] > 0 else []
            )

    def finish_task(slot: int) -> None:
        """Flush remaining frames and save demo .pt for the completed task."""
        task    = slot_task[slot]
        out_dir = out_frames / task

        # Flush any leftover shard buffer
        if slot_shard_buf[slot]:
            total = sum(f.shape[0] for f in slot_shard_buf[slot])
            if total > 0:
                concat   = torch.cat(slot_shard_buf[slot], dim=0)
                out_path = out_dir / f"{task}_shard{slot_shard_idx[slot]:04d}.pt"
                _safe_save({"frames": concat}, out_path)
                slot_shard_idx[slot] += 1

        # Save demo .pt
        meta = slot_meta[slot]
        if meta["episode"]:
            demo_data = {
                "episode": torch.tensor(meta["episode"],          dtype=torch.int64),
                "action":  torch.tensor(np.stack(meta["action"]), dtype=torch.float32),
                "reward":  torch.tensor(meta["reward"],           dtype=torch.float32),
            }
            out_demo = out_data / f"{task}.pt"
            _safe_save(demo_data, out_demo)
            n_frames = len(meta["episode"])
            n_eps    = len(set(meta["episode"]))
            print(f"[vec_collect/{task}] saved {n_frames} frames, {n_eps} episodes, "
                  f"{slot_shard_idx[slot]} shards → {out_demo}")
            results[task] = True

        _update_tasks_json(task, slot_act_dim[slot], cc)

    # ── Initial task assignment ────────────────────────────────────────────
    active_count = 0
    for i in range(n_envs):
        if assign_task(i):
            active_count += 1
        else:
            envs[i] = None

    if active_count == 0:
        return results

    # ── Main collection loop ───────────────────────────────────────────────
    total_steps = 0
    log_interval = 1000

    while True:
        active = [i for i in range(n_envs) if envs[i] is not None]
        if not active:
            break

        # 1. Render frames for all active slots
        frames_list = [_render_frame(envs[i], cc.img_size, cc.camera_id) for i in active]

        # 2. Batched GPU inference
        pa_batch = prev_actions[active]                                  # (B, action_dim)
        actions  = vpolicy.act(frames_list, pa_batch, slot_indices=active)  # (B, action_dim)

        # 3. Step each environment and record
        done_slots: List[int] = []
        for j, i in enumerate(active):
            frame     = frames_list[j]
            action_np = actions[j].numpy()
            act_dim   = slot_act_dim[i]

            # Map [-1,1] → env action range
            spec = envs[i].action_spec()
            lo, hi     = spec.minimum, spec.maximum
            env_action = ((action_np[:act_dim] + 1.0) / 2.0) * (hi - lo) + lo
            env_action = env_action.clip(lo, hi).astype(np.float32)

            ts = envs[i].step(env_action)

            slot_ep_frames[i].append(frame)
            slot_meta[i]["episode"].append(slot_ep_id[i])
            slot_meta[i]["action"].append(action_np.copy())
            slot_meta[i]["reward"].append(float(ts.reward or 0.0))
            prev_actions[i] = torch.from_numpy(action_np)
            slot_step[i]   += 1
            total_steps    += 1

            if total_steps % log_interval == 0:
                n_done = sum(1 for t in results.values() if t)
                print(f"[vec_collect] steps={total_steps}  tasks_done={n_done}/{len(pending)}")

            if ts.last() or slot_step[i] >= episode_len:
                done_slots.append(i)

        # 4. End-of-episode bookkeeping
        for i in done_slots:
            # Flush episode frames to shard buffer
            if slot_ep_frames[i]:
                ep_tensor = torch.stack(slot_ep_frames[i])  # (T, 3, H, W)
                slot_shard_buf[i].append(ep_tensor)
                flush_shards(i)
                slot_ep_frames[i] = []

            if cc.verbose:
                step_rew = sum(slot_meta[i]["reward"][-slot_step[i]:])
                print(f"  [vec_collect/{slot_task[i]}] "
                      f"ep {slot_done_eps[i]+1}/{n_episodes}  "
                      f"steps={slot_step[i]}  ep_reward={step_rew:.2f}")

            slot_done_eps[i] += 1
            slot_ep_id[i]    += 1
            slot_step[i]      = 0

            if slot_done_eps[i] >= n_episodes:
                finish_task(i)
                if not assign_task(i):
                    envs[i] = None
            else:
                # Next episode: reset env and clear policy context
                vpolicy.reset(i)
                envs[i].reset()
                prev_actions[i] = 0.0

    print(f"[vec_collect] Collection complete. total_steps={total_steps}")
    return results
