# dreamer4/lightning/dynamics_module.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from model import (
    Dynamics,
    temporal_patchify,
    pack_bottleneck_to_spatial,
)
from losses import dynamics_pretrain_loss
from checkpoint_utils import load_frozen_tokenizer
from loss_norm import RunningRMS


class DynamicsLightningModule(pl.LightningModule):
    """
    LightningModule for the Dynamics (world model) training stage.

    The tokenizer is loaded as a *frozen* artifact inside setup() so that it
    lands on the correct device.  It is stored as a plain Python attribute
    (via object.__setattr__) so that:
      - DDP does NOT synchronise gradients through it
      - AdamW does NOT receive its parameters
      - Lightning's model summary does not count its parameters

    Only self.dyn is registered as an nn.Module submodule and participates in
    DDP / optimiser.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Dynamics model is built in setup() once tokenizer dims are known.
        # We declare the attribute here so type-checkers are happy.
        self.dyn: Optional[Dynamics] = None

        # RMS loss normalisation (Dreamer 4): normalise total WM loss
        self.rms_wm = RunningRMS()

        # Cached after setup()
        self._tok_args: Optional[Dict[str, Any]] = None
        self._patch: int = 4
        self._H: int = 128
        self._W: int = 128
        self._C: int = 3
        self.n_spatial: int = 0
        self.d_spatial: int = 0
        self.packing_factor: int = 2

    # ------------------------------------------------------------------
    # Lightning lifecycle
    # ------------------------------------------------------------------

    def setup(self, stage: str):
        dc = self.cfg.dynamics

        # Load frozen tokenizer onto the correct device
        encoder, decoder, tok_args = load_frozen_tokenizer(
            dc.tokenizer_ckpt, device=self.device
        )

        # Store as plain Python attributes — outside DDP / optimizer scope
        object.__setattr__(self, "_encoder", encoder)
        object.__setattr__(self, "_decoder", decoder)

        self._tok_args = tok_args
        self._patch    = int(tok_args.get("patch",        4))
        self._H        = int(tok_args.get("H",          128))
        self._W        = int(tok_args.get("W",          128))
        self._C        = int(tok_args.get("C",            3))

        n_latents    = int(tok_args.get("n_latents",   16))
        d_bottleneck = int(tok_args.get("d_bottleneck", 32))
        pf           = int(dc.packing_factor)

        assert n_latents % pf == 0, (
            f"n_latents ({n_latents}) must be divisible by packing_factor ({pf})"
        )
        self.n_spatial       = n_latents // pf
        self.d_spatial       = d_bottleneck * pf
        self.packing_factor  = pf

        # Build dynamics model (only once)
        if self.dyn is None:
            self.dyn = Dynamics(
                d_model=dc.d_model_dyn,
                d_bottleneck=d_bottleneck,
                d_spatial=self.d_spatial,
                n_spatial=self.n_spatial,
                n_register=dc.n_register,
                n_agent=dc.n_agent,
                n_heads=dc.n_heads,
                depth=dc.dyn_depth,
                k_max=dc.k_max,
                dropout=dc.dropout,
                mlp_ratio=dc.mlp_ratio,
                time_every=dc.time_every,
                space_mode=dc.space_mode,
                scale_pos_embeds=bool(dc.get("scale_pos_embeds", False)),
            )
            if self.cfg.get("compile", False):
                self.dyn = torch.compile(self.dyn)

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode (B,T,C,H,W) float [0,1] -> (B,T,n_spatial,d_spatial) packed latents."""
        with torch.no_grad():
            patches = temporal_patchify(frames, self._patch)
            z_btLd  = self._encoder(patches)[0]   # [0] works for both Encoder (2-tuple) and DiscreteEncoder (3-tuple)
            return pack_bottleneck_to_spatial(z_btLd, n_spatial=self.n_spatial, k=self.packing_factor)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        dc = self.cfg.dynamics

        if dc.use_actions:
            obs_u8 = batch["obs"]              # (B, T+1, 3, H, W) uint8
            act    = batch["act"]              # (B, T, 16) float
            mask   = batch["act_mask"]         # (B, T, 16) float

            act = act.clamp(-1.0, 1.0) * mask

            frames = obs_u8[:, :-1].float() / 255.0  # (B, T, 3, H, W)

            # action[t] produced obs[t+1], shift by 1
            actions = torch.zeros_like(act)
            actions[:, 1:] = act[:, :-1]
            act_mask = torch.zeros_like(mask)
            act_mask[:, 1:] = mask[:, :-1]
        else:
            frames   = batch.float()
            if frames.dtype == torch.uint8:
                frames = frames / 255.0
            actions  = None
            act_mask = None

        z1 = self._encode_frames(frames)  # (B, T, Sz, Dz)

        B      = z1.shape[0]
        B_self = max(0, min(B - 1, int(round(dc.self_fraction * B))))

        raw_loss, aux = dynamics_pretrain_loss(
            self.dyn,
            z1=z1,
            actions=actions,
            act_mask=act_mask,
            k_max=dc.k_max,
            B_self=B_self,
            step=self.global_step,
            bootstrap_start=dc.bootstrap_start,
            agent_tokens=None,
        )
        loss = self.rms_wm.normalize(raw_loss)

        self.log_dict(
            {
                "loss/total":          loss,
                "loss/wm_raw":         raw_loss,
                "loss/flow_mse":       aux["flow_mse"],
                "loss/bootstrap_mse":  aux["bootstrap_mse"],
                "loss/loss_emp":       aux["loss_emp"],
                "loss/loss_self":      aux["loss_self"],
                "stats/sigma_mean":    aux["sigma_mean"],
                "stats/B_self":        float(B_self),
                "stats/rms_wm":        self.rms_wm.sq.sqrt(),
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=False,
        )

        # Cache for callbacks (eval + action-shuffle metric)
        self._last_frames   = frames
        self._last_actions  = actions
        self._last_act_mask = act_mask
        self._last_z1       = z1
        self._last_B_self   = B_self
        self._last_raw_loss = raw_loss

        return loss

    # ------------------------------------------------------------------
    # Optimizer — only the dynamics parameters
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        import math
        dc = self.cfg.dynamics
        opt = torch.optim.AdamW(
            self.dyn.parameters(),
            lr=dc.lr,
            weight_decay=dc.weight_decay,
            betas=(0.9, 0.999),
        )
        warmup = int(dc.get("warmup_steps", 0))
        total  = self.trainer.max_steps
        if warmup > 0 and total > 0:
            schedule = str(dc.get("lr_schedule", "constant_with_warmup")).lower()
            def lr_lambda(step: int) -> float:
                if step < warmup:
                    return step / max(1, warmup)
                if schedule == "constant_with_warmup":
                    return 1.0
                progress = (step - warmup) / max(1, total - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        return opt

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        clip = float(self.cfg.dynamics.get("grad_clip", 1.0))
        self.clip_gradients(optimizer, gradient_clip_val=clip, gradient_clip_algorithm="norm")

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Backwards compat: checkpoints saved before RunningRMS was added
        # don't have rms_wm.sq — initialise it to the default value (ones(1)).
        state = checkpoint.get("state_dict", {})
        if "rms_wm.sq" not in state:
            state["rms_wm.sq"] = torch.ones(1)
