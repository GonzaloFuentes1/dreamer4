# dreamer4/lightning/finetune_module.py
# Phase 2 — FinetuneLightningModule: finetune the world model with task-conditioned
# policy and reward heads (eq. 9 of the paper).
#
# Frozen   : Encoder  (Phase-1 tokenizer checkpoint)
# Trainable: Dynamics, TaskEmbedder, PolicyHead, RewardHead
#
# Loss per step:
#   L = L_wm  (shortcut forcing, eq. 7 — continues Phase-1 objective)
#           + β_bc  * L_bc   (behavior cloning with MTP L=8, eq. 9 first sum)
#           + β_rew * L_rew  (reward model with MTP L=8, eq. 9 second sum)
#
# Workflow:
#   Phase 1a: train_phase1a_tokenizer.py  →  tokenizer.ckpt
#   Phase 1b: train_phase1b_dynamics.py   →  dynamics.ckpt
#   Phase 2: THIS                →  finetune.ckpt   (loads dynamics.ckpt)
#   Phase 3: train_phase3_imagination.py  →  loads finetune.ckpt (fully frozen)

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from model import (
    Dynamics,
    TaskEmbedder,
    temporal_patchify,
    pack_bottleneck_to_spatial,
)
from losses import dynamics_pretrain_loss
from agent import PolicyHead, RewardHead


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loader helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_frozen_tokenizer(ckpt_path: str, device: torch.device):
    """Load tokenizer checkpoint → frozen Encoder + Decoder + tok_args dict."""
    from model import Encoder, Decoder

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Support both Lightning checkpoints and legacy raw-dict checkpoints.
    if "state_dict" in ckpt:
        hp = ckpt.get("hyper_parameters", {})
        tc = hp.get("cfg", {}).get("tokenizer", {})
        tok_args = dict(tc) if tc else {}
        full_sd  = ckpt["state_dict"]
        model_sd = {k[len("model."):]: v for k, v in full_sd.items() if k.startswith("model.")}
        # Strip _orig_mod. prefix added by torch.compile()
        model_sd = {k.replace("_orig_mod.", ""): v for k, v in model_sd.items()}
    else:
        tok_args = dict(ckpt.get("args", {}))
        model_sd = ckpt["model"]
        # Strip _orig_mod. prefix added by torch.compile()
        model_sd = {k.replace("_orig_mod.", ""): v for k, v in model_sd.items()}

    H         = int(tok_args.get("H", 128))
    W         = int(tok_args.get("W", 128))
    C         = int(tok_args.get("C", 3))
    patch     = int(tok_args.get("patch", 4))
    n_patches = (H // patch) * (W // patch)
    d_patch   = patch * patch * C

    _enc_kwargs = dict(
        patch_dim=d_patch,
        d_model=int(tok_args.get("d_model", 256)),
        n_latents=int(tok_args.get("n_latents", 16)),
        n_patches=n_patches,
        n_heads=int(tok_args.get("n_heads", 4)),
        depth=int(tok_args.get("depth", 8)),
        dropout=0.0,
        mlp_ratio=float(tok_args.get("mlp_ratio", 4.0)),
        time_every=int(tok_args.get("time_every", 1)),
        latents_only_time=bool(tok_args.get("latents_only_time", True)),
        mae_p_min=0.0,
        mae_p_max=0.0,
    )
    if tok_args.get("discrete", False):
        from model import DiscreteEncoder
        enc = DiscreteEncoder(
            **_enc_kwargs,
            n_categories=int(tok_args.get("d_bottleneck", 32)),
            temperature=float(tok_args.get("temperature", 1.0)),
        )
    else:
        enc = Encoder(
            **_enc_kwargs,
            d_bottleneck=int(tok_args.get("d_bottleneck", 32)),
        )
    dec = Decoder(
        d_bottleneck=int(tok_args.get("d_bottleneck", 32)),
        d_model=int(tok_args.get("d_model", 256)),
        n_heads=int(tok_args.get("n_heads", 4)),
        depth=int(tok_args.get("depth", 8)),
        n_latents=int(tok_args.get("n_latents", 16)),
        n_patches=n_patches,
        d_patch=d_patch,
        dropout=0.0,
        mlp_ratio=float(tok_args.get("mlp_ratio", 4.0)),
        time_every=int(tok_args.get("time_every", 1)),
        latents_only_time=bool(tok_args.get("latents_only_time", True)),
    )
    enc_sd = {k[len("encoder."):]: v for k, v in model_sd.items() if k.startswith("encoder.")}
    dec_sd = {k[len("decoder."):]: v for k, v in model_sd.items() if k.startswith("decoder.")}
    enc.load_state_dict(enc_sd, strict=True)
    dec.load_state_dict(dec_sd, strict=True)
    enc = enc.to(device).eval()
    dec = dec.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    for p in dec.parameters():
        p.requires_grad_(False)
    return enc, dec, tok_args


# ─────────────────────────────────────────────────────────────────────────────
# FinetuneLightningModule
# ─────────────────────────────────────────────────────────────────────────────

class FinetuneLightningModule(pl.LightningModule):
    """
    Phase 2: task-conditioned world model finetuning.

    Agent tokens are inserted into the dynamics transformer via a TaskEmbedder.
    The dynamics then produces h_t (agent-token outputs) from which PolicyHead
    and RewardHead predict future actions and rewards (MTP with L=8).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.dyn:           Optional[Dynamics]     = None
        self.task_embedder: Optional[TaskEmbedder] = None
        self.policy_head:   Optional[PolicyHead]   = None
        self.reward_head:   Optional[RewardHead]   = None

        self._tok_args:      Optional[Dict[str, Any]] = None
        self._patch:         int = 4
        self._H:             int = 128
        self._W:             int = 128
        self._C:             int = 3
        self.n_spatial:      int = 0
        self.d_spatial:      int = 0
        self.packing_factor: int = 2

    # ------------------------------------------------------------------
    # Lightning lifecycle
    # ------------------------------------------------------------------

    def setup(self, stage: str):
        ft = self.cfg.finetune

        # ---- Frozen encoder (stays outside DDP / optimizer) ----------
        encoder, decoder, tok_args = _load_frozen_tokenizer(ft.tokenizer_ckpt, self.device)
        object.__setattr__(self, "_encoder", encoder)
        object.__setattr__(self, "_decoder", decoder)

        self._tok_args       = tok_args
        self._patch          = int(tok_args.get("patch",        4))
        self._H              = int(tok_args.get("H",          128))
        self._W              = int(tok_args.get("W",          128))
        self._C              = int(tok_args.get("C",            3))
        n_latents            = int(tok_args.get("n_latents",   16))
        d_bottleneck         = int(tok_args.get("d_bottleneck", 32))
        pf                   = int(ft.packing_factor)

        assert n_latents % pf == 0
        self.n_spatial      = n_latents // pf
        self.d_spatial      = d_bottleneck * pf
        self.packing_factor = pf

        if self.dyn is not None:
            return  # already initialised (e.g. DDP calls setup on all ranks)

        # ---- Dynamics ------------------------------------------------
        self.dyn = Dynamics(
            d_model=ft.d_model_dyn,
            d_bottleneck=d_bottleneck,
            d_spatial=self.d_spatial,
            n_spatial=self.n_spatial,
            n_register=ft.n_register,
            n_agent=ft.n_agent,
            n_heads=ft.n_heads,
            depth=ft.dyn_depth,
            k_max=ft.k_max,
            dropout=ft.dropout,
            mlp_ratio=ft.mlp_ratio,
            time_every=ft.time_every,
            space_mode=ft.space_mode,
        )

        # Optionally initialise from a Phase-1 dynamics checkpoint
        dyn_ckpt = getattr(ft, "dynamics_ckpt", None)
        if dyn_ckpt and dyn_ckpt != "???":
            ckpt   = torch.load(dyn_ckpt, map_location="cpu", weights_only=False)
            raw_sd = ckpt.get("state_dict", ckpt.get("model", {}))
            dyn_sd = {k[4:]: v for k, v in raw_sd.items() if k.startswith("dyn.")}
            if not dyn_sd:  # checkpoint saved without Lightning wrapper
                dyn_sd = {k: v for k, v in raw_sd.items()
                          if not any(k.startswith(p) for p in ("_encoder.", "_decoder."))}
            missing, _ = self.dyn.load_state_dict(dyn_sd, strict=False)
            if missing:
                print(f"[Finetune] dynamics missing keys: {missing[:3]} ...")

        # ---- TaskEmbedder (new — maps task_id / lang_emb to h_t input) --
        self.task_embedder = TaskEmbedder(
            d_model=ft.d_model_dyn,
            n_agent=ft.n_agent,
            use_ids=bool(ft.use_task_ids),
            n_tasks=int(ft.n_tasks),
            d_task=512,           # language embedding from tasks.json
        )

        # ---- Policy and Reward heads (new) ----------------------------
        state_dim = ft.n_agent * ft.d_model_dyn
        self.policy_head = PolicyHead(
            state_dim=state_dim,
            action_dim=int(ft.action_dim),
            hidden_dim=int(ft.hidden_dim),
            mtp_length=int(ft.mtp_length),
        )
        self.reward_head = RewardHead(
            state_dim=state_dim,
            hidden_dim=int(ft.hidden_dim) // 2,
            mtp_length=int(ft.mtp_length),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_frames(self, obs_u8: torch.Tensor) -> torch.Tensor:
        """(B,T,3,H,W) uint8 → (B,T,n_spatial,d_spatial) packed latents."""
        frames  = obs_u8.float() / 255.0
        patches = temporal_patchify(frames, self._patch)
        z_btLd  = self._encoder(patches)[0]   # [0] works for both Encoder and DiscreteEncoder
        return pack_bottleneck_to_spatial(z_btLd, n_spatial=self.n_spatial, k=self.packing_factor)

    def _task_tokens(self, batch: Dict, B: int, T: int) -> torch.Tensor:
        """Return (B, T, n_agent, d_model) task-conditioned agent tokens."""
        ft = self.cfg.finetune
        if ft.use_task_ids:
            task_input = batch["emb_id"].long()     # (B,)
        else:
            task_input = batch["lang_emb"].float()  # (B, 512)
        return self.task_embedder(task_input, B=B, T=T)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        ft = self.cfg.finetune

        obs_u8   = batch["obs"]       # (B, T+1, 3, H, W)
        act      = batch["act"]       # (B, T, 16)
        act_mask = batch["act_mask"]  # (B, T, 16)
        rew      = batch["rew"]       # (B, T)

        act = (act * act_mask).clamp(-1.0, 1.0)

        z1   = self._encode_frames(obs_u8[:, :-1])   # (B, T, n_s, d_s)
        B, T = z1.shape[:2]

        # Shift actions: action[t] produced obs[t+1]
        act_shifted  = torch.zeros_like(act);  act_shifted[:, 1:]  = act[:, :-1]
        mask_shifted = torch.zeros_like(act_mask); mask_shifted[:, 1:] = act_mask[:, :-1]

        task_tokens = self._task_tokens(batch, B, T)  # (B, T, n_agent, d_model)

        # ── World model loss (eq. 7, same as Phase 1) ─────────────────────
        B_self   = max(0, min(B - 1, int(round(ft.self_fraction * B))))
        wm_loss, wm_aux = dynamics_pretrain_loss(
            self.dyn,
            z1=z1,
            actions=act_shifted,
            act_mask=mask_shifted,
            k_max=ft.k_max,
            B_self=B_self,
            step=self.global_step,
            bootstrap_start=ft.bootstrap_start,
            agent_tokens=task_tokens,
        )

        # ── Clean forward (τ=1) to get h_t for BC and reward losses ───────
        # Paper: "representations are noisy and we continue to apply the video
        # prediction loss" — the BC/reward heads train on the SAME noisy h_t
        # produced during the WM loss forward.  Here we use a separate τ=1 pass
        # to get a clean, stable h_t for the supervised heads.
        emax       = int(round(math.log2(ft.k_max)))
        step_clean = torch.full((B, T), emax, device=self.device, dtype=torch.long)
        sig_clean  = torch.full((B, T), ft.k_max - 1, device=self.device, dtype=torch.long)

        _, h_t = self.dyn(
            act_shifted, step_clean, sig_clean, z1,
            act_mask=mask_shifted,
            agent_tokens=task_tokens,
        )
        # h_t : (B, T, n_agent, d_model_dyn)
        h_flat = h_t.flatten(2)   # (B, T, n_agent * d_model_dyn)

        # ── BC loss (eq. 9 first sum) ──────────────────────────────────────
        bc_loss  = self.policy_head.bc_loss(h_flat, act, act_mask)

        # ── Reward model loss (eq. 9 second sum) ──────────────────────────
        rew_loss = self.reward_head.mtp_loss(h_flat, rew)

        # ── Total ──────────────────────────────────────────────────────────
        loss = wm_loss + ft.bc_coef * bc_loss + ft.reward_coef * rew_loss

        self.log_dict({
            "loss/total":         loss,
            "loss/wm":            wm_loss,
            "loss/bc":            bc_loss,
            "loss/reward":        rew_loss,
            "loss/flow_mse":      wm_aux["flow_mse"],
            "loss/bootstrap_mse": wm_aux["bootstrap_mse"],
            "stats/B_self":       float(B_self),
        }, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)

        self._last_frames = obs_u8[:, :-1].float().div(255.0)
        self._last_actions = act_shifted
        self._last_act_mask = mask_shifted

        return loss

    # ------------------------------------------------------------------
    # Checkpoint hooks — persist frozen encoder for Phase 3
    # ------------------------------------------------------------------

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Strip _encoder.* keys injected by on_save_checkpoint.

        Lightning calls load_state_dict BEFORE setup(), so _encoder is not yet
        registered as a module.  Removing these keys prevents the
        'Unexpected key(s)' RuntimeError; the encoder is re-loaded from disk
        inside setup() anyway.
        """
        sd = checkpoint.get("state_dict", {})
        for key in [k for k in sd if k.startswith("_encoder.")]:
            del sd[key]

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Inject frozen encoder weights into the Lightning checkpoint.

        Phase 3 (_load_finetune_checkpoint) looks for '_encoder.*' keys in the
        state_dict and tok_args at the top level.  Without this hook, the encoder
        is stored via object.__setattr__ and never reaches the Lightning state_dict,
        so Phase 3 would start with a randomly-initialised encoder.
        """
        enc_sd = {f"_encoder.{k}": v for k, v in self._encoder.state_dict().items()}
        checkpoint["state_dict"].update(enc_sd)
        if self._tok_args is not None:
            checkpoint["tok_args"] = dict(self._tok_args)

    # ------------------------------------------------------------------
    # Optimizer — dynamics + task_embedder + heads
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        ft = self.cfg.finetune
        params = (
            list(self.dyn.parameters())
            + list(self.task_embedder.parameters())
            + list(self.policy_head.parameters())
            + list(self.reward_head.parameters())
        )
        return torch.optim.AdamW(
            params,
            lr=float(ft.lr),
            weight_decay=float(ft.weight_decay),
            betas=(0.9, 0.999),
        )
