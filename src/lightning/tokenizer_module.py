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
    temporal_patchify,
    recon_loss_from_mae,
)
from losses import LPIPSLoss


class TokenizerLightningModule(pl.LightningModule):
    """
    LightningModule wrapping the Tokenizer (Encoder + Decoder).

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
        )
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

        if cfg.get("compile", False):
            self.model = torch.compile(self.model)

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
        patches = temporal_patchify(x, self._patch)  # (B,T,Np,Dp)

        pred, mae_mask, keep_prob = self(patches)

        mse = recon_loss_from_mae(pred, patches, mae_mask)

        if self.lpips_loss is not None:
            lp   = self.lpips_loss(pred, patches, mae_mask, H=self._H, W=self._W, C=self._C, patch=self._patch)
            loss = mse + lp
        else:
            lp   = torch.zeros((), device=self.device)
            loss = mse

        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))

        self.log_dict(
            {
                "loss/total":        loss,
                "loss/mse":          mse,
                "loss/lpips":        lp,
                "stats/psnr":        psnr,
                "stats/keep_prob":   keep_prob.mean(),
                "stats/masked_frac": mae_mask.float().mean(),
            },
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
        return torch.optim.AdamW(
            self.parameters(),
            lr=tc.lr,
            weight_decay=tc.weight_decay,
        )
