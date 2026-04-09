# dreamer4/lightning/tokenizer_module.py
from __future__ import annotations

from typing import Optional

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from model import (
    Encoder,
    Decoder,
    Tokenizer,
    DiscreteEncoder,
    DiscreteTokenizer,
    temporal_patchify,
    recon_loss_from_mae,
    RunningRMSNorm,
)
from losses import LPIPSLoss


class TokenizerLightningModule(pl.LightningModule):
    """
    LightningModule wrapping the Tokenizer (Encoder + Decoder).

    Supports two encoder modes selected by ``cfg.tokenizer.discrete``:
    - ``False`` (default): continuous tanh bottleneck (original architecture)
    - ``True``:  DreamerV3-style straight-through categorical codes

    Hyperparameters live in cfg.tokenizer (architecture + loss) and cfg.trainer.
    The module itself holds no distributed / AMP boilerplate — Lightning handles
    that via Trainer(strategy="ddp", precision="16-mixed").
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        tc = cfg.tokenizer

        # Derived patch dimensions
        assert tc.H % tc.patch == 0 and tc.W % tc.patch == 0
        n_patches = (tc.H // tc.patch) * (tc.W // tc.patch)
        d_patch   = tc.patch * tc.patch * tc.C

        assert tc.d_model % tc.n_heads == 0, "d_model must be divisible by n_heads"

        self._discrete = bool(tc.get("discrete", False))

        dec = Decoder(
            d_bottleneck=tc.d_bottleneck,
            d_model=tc.d_model,
            n_heads=tc.n_heads,
            depth=tc.depth,
            n_latents=tc.n_latents,
            n_patches=n_patches,
            d_patch=d_patch,
            dropout=tc.dropout,
            mlp_ratio=tc.mlp_ratio,
            time_every=tc.time_every,
            latents_only_time=tc.latents_only_time,
            scale_pos_embeds=bool(tc.get("scale_pos_embeds", True)),
        )

        if self._discrete:
            enc = DiscreteEncoder(
                patch_dim=d_patch,
                d_model=tc.d_model,
                n_latents=tc.n_latents,
                n_patches=n_patches,
                n_heads=tc.n_heads,
                depth=tc.depth,
                n_categories=tc.d_bottleneck,   # d_bottleneck = n_categories for discrete
                dropout=tc.dropout,
                mlp_ratio=tc.mlp_ratio,
                time_every=tc.time_every,
                latents_only_time=tc.latents_only_time,
                mae_p_min=tc.mae_p_min,
                mae_p_max=tc.mae_p_max,
                temperature=float(tc.get("temperature", 1.0)),
                scale_pos_embeds=bool(tc.get("scale_pos_embeds", True)),
            )
            self.model = DiscreteTokenizer(enc, dec)
        else:
            enc = Encoder(
                patch_dim=d_patch,
                d_model=tc.d_model,
                n_latents=tc.n_latents,
                n_patches=n_patches,
                n_heads=tc.n_heads,
                depth=tc.depth,
                d_bottleneck=tc.d_bottleneck,
                dropout=tc.dropout,
                mlp_ratio=tc.mlp_ratio,
                time_every=tc.time_every,
                latents_only_time=tc.latents_only_time,
                mae_p_min=tc.mae_p_min,
                mae_p_max=tc.mae_p_max,
                scale_pos_embeds=bool(tc.get("scale_pos_embeds", True)),
            )
            self.model = Tokenizer(enc, dec)

        # LPIPS — lazy init in setup() once the device is known
        if tc.lpips_weight > 0.0:
            self.lpips_loss: Optional[LPIPSLoss] = LPIPSLoss(
                net=tc.lpips_net,
                weight=tc.lpips_weight,
                subsample_frac=tc.lpips_frac,
            )
        else:
            self.lpips_loss = None

        # Cache frequently used scalars for training_step
        self._patch = int(tc.patch)
        self._H     = int(tc.H)
        self._W     = int(tc.W)
        self._C     = int(tc.C)

        self._rms_mse   = RunningRMSNorm()
        self._rms_lpips = RunningRMSNorm()
        self._use_loss_norm = bool(tc.get("use_loss_norm", False))

        if cfg.get("compile", False):
            self.model = torch.compile(self.model, mode="reduce-overhead")

    # ------------------------------------------------------------------
    # Lightning lifecycle
    # ------------------------------------------------------------------

    def setup(self, stage: str):
        if self.lpips_loss is not None:
            self.lpips_loss.setup(self.device)

    def forward(self, patches_btnd: torch.Tensor):
        return self.model(patches_btnd)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        # batch: (B, T, C, H, W) float32 in [0, 1] from FrameDataModule
        x = batch
        B, T, C, H, W = x.shape
        assert H == self._H and W == self._W, (
            f"Frame size {H}x{W} != model size {self._H}x{self._W}. "
            f"Pre-convert your dataset to {self._H}x{self._W} with scripts/convert_frames_to_res.py"
        )
        patches = temporal_patchify(x, self._patch)  # (B,T,Np,Dp)

        if self._discrete:
            pred, mae_mask, keep_prob, entropy = self(patches)
            entropy_mean = entropy.mean()
            # Maximise entropy → encourage all categories to be used
            entropy_loss = -float(self.cfg.tokenizer.get("entropy_scale", 0.01)) * entropy_mean
            z_for_stats = None
        else:
            # Call encoder directly to expose z for diagnostics (no extra compute)
            z, (mae_mask, keep_prob) = self.model.encoder(patches)
            pred = self.model.decoder(z)
            entropy_loss = torch.zeros((), device=self.device)
            entropy_mean = torch.zeros((), device=self.device)
            z_for_stats = z

        mse = recon_loss_from_mae(pred, patches, mae_mask)

        if self.lpips_loss is not None:
            lp   = self.lpips_loss(pred, patches, mae_mask, H=self._H, W=self._W, C=self._C, patch=self._patch)
            plain_total = mse + lp + entropy_loss
            if self._use_loss_norm:
                loss = self._rms_mse(mse) + self._rms_lpips(lp) + entropy_loss
            else:
                loss = plain_total
        else:
            lp   = torch.zeros((), device=self.device)
            plain_total = mse + entropy_loss
            loss = mse + entropy_loss

        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))

        log_dict = {
            "loss/total":        loss,
            "loss/plain_total":  plain_total,
            "loss/mse":          mse,
            "loss/lpips":        lp,
            "stats/psnr":        psnr,
            "stats/keep_prob":   keep_prob.mean(),
            "stats/masked_frac": mae_mask.float().mean(),
        }
        if z_for_stats is not None:
            log_dict["stats/z_std"]      = z_for_stats.std()
            log_dict["stats/z_mean_abs"] = z_for_stats.abs().mean()  # tanh satura en ±1
        if self._discrete:
            n_cat = self.model.encoder.d_bottleneck
            perplexity = torch.exp(entropy_mean)   # effective categories used
            log_dict["discrete/entropy"]    = entropy_mean
            log_dict["discrete/perplexity"] = perplexity
            log_dict["discrete/util_frac"]  = perplexity / n_cat   # 1.0 = all cats used equally

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=False,  # rank-0 only logging matches original behaviour
        )

        # Store batch for viz callback (avoids re-running the forward pass)
        self._last_batch  = x
        self._last_pred   = pred
        self._last_mask   = mae_mask

        return loss

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        tc = self.cfg.tokenizer
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=tc.lr,
            weight_decay=tc.weight_decay,
        )
        warmup = int(tc.get("warmup_steps", 0))
        total  = int(self.trainer.max_steps)
        schedule = str(tc.get("lr_schedule", "cosine")).lower()

        if schedule == "constant":
            return opt

        if warmup > 0 and total > 0:
            def lr_lambda(step: int) -> float:
                if step < warmup:
                    return step / max(1, warmup)
                if schedule == "constant_with_warmup":
                    return 1.0
                progress = (step - warmup) / max(1, total - warmup)
                return 0.5 * (1.0 + __import__("math").cos(3.14159265 * progress))

            if schedule not in {"cosine", "constant_with_warmup"}:
                raise ValueError(f"Unsupported tokenizer.lr_schedule={schedule}")
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

        if schedule not in {"cosine", "constant_with_warmup"}:
            raise ValueError(f"Unsupported tokenizer.lr_schedule={schedule}")

        return opt

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        clip = float(self.cfg.tokenizer.get("grad_clip", 1.0))
        self.clip_gradients(optimizer, gradient_clip_val=clip, gradient_clip_algorithm="norm")
