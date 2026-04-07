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


def _load_frozen_tokenizer(ckpt_path: str, device: torch.device):
    """
    Load tokenizer checkpoint, return (encoder, decoder, tok_args).
    Identical logic to load_frozen_tokenizer_from_pt_ckpt in the original
    train_phase1b_dynamics.py — kept here to avoid cross-file imports during setup.
    """
    from model import Encoder, Decoder, Tokenizer

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Support both legacy raw-dict checkpoints and Lightning checkpoints.
    if "state_dict" in ckpt:
        # Lightning checkpoint: hyper_parameters holds the config,
        # state_dict has keys prefixed with "model." (Tokenizer) and
        # "lpips_loss." (LPIPS, not needed here).
        hp = ckpt.get("hyper_parameters", {})
        tc = hp.get("cfg", {}).get("tokenizer", {})
        tok_args = dict(tc) if tc else {}

        full_sd = ckpt["state_dict"]
        model_sd = {
            k[len("model."):]: v
            for k, v in full_sd.items()
            if k.startswith("model.")
        }
        # Strip _orig_mod. prefix added by torch.compile()
        model_sd = {
            k.replace("_orig_mod.", ""): v for k, v in model_sd.items()
        }
    else:
        # Legacy format saved directly as a plain dict with "model" and "args".
        tok_args = dict(ckpt.get("args", {}))
        model_sd = ckpt["model"]
        # Strip _orig_mod. prefix added by torch.compile()
        model_sd = {
            k.replace("_orig_mod.", ""): v for k, v in model_sd.items()
        }

    H      = int(tok_args.get("H", 128))
    W      = int(tok_args.get("W", 128))
    C      = int(tok_args.get("C", 3))
    patch  = int(tok_args.get("patch", 4))
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
        scale_pos_embeds=bool(tok_args.get("scale_pos_embeds", True)),
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
        scale_pos_embeds=bool(tok_args.get("scale_pos_embeds", True)),
    )

    tok = Tokenizer(enc, dec)
    tok.load_state_dict(model_sd, strict=True)
    tok = tok.to(device)
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)

    return tok.encoder, tok.decoder, tok_args


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
        encoder, decoder, tok_args = _load_frozen_tokenizer(
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

        loss, aux = dynamics_pretrain_loss(
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

        self.log_dict(
            {
                "loss/total":          loss,
                "loss/flow_mse":       aux["flow_mse"],
                "loss/bootstrap_mse":  aux["bootstrap_mse"],
                "loss/loss_emp":       aux["loss_emp"],
                "loss/loss_self":      aux["loss_self"],
                "stats/sigma_mean":    aux["sigma_mean"],
                "stats/B_self":        float(B_self),
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
            def lr_lambda(step: int) -> float:
                if step < warmup:
                    return step / max(1, warmup)
                progress = (step - warmup) / max(1, total - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        return opt

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        clip = float(self.cfg.dynamics.get("grad_clip", 1.0))
        self.clip_gradients(optimizer, gradient_clip_val=clip, gradient_clip_algorithm="norm")
