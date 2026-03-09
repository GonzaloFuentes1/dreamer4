#!/usr/bin/env python
# train_tokenizer.py — PyTorch Lightning + Hydra entry point
#
# Usage (single GPU):
#   python train_tokenizer.py trainer.devices=1
#
# Usage (8 GPUs, single node):
#   python train_tokenizer.py trainer.devices=8
#
# Override any config key at the CLI:
#   python train_tokenizer.py tokenizer=small wandb.name=exp_001
#
# Hyperparameter sweep (Hydra multirun):
#   python train_tokenizer.py -m tokenizer.d_model=128,256 tokenizer.depth=4,8
#
# Resume from a Lightning checkpoint:
#   python train_tokenizer.py resume=./logs/tokenizer_ckpts/last.ckpt
import sys
import os

# Allow importing dreamer4 package modules without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dreamer4"))

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from lightning.tokenizer_module import TokenizerLightningModule
from lightning.frame_datamodule import FrameDataModule
from lightning.callbacks import TokenizerVizCallback

torch_backends_setup = """
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
"""
exec(torch_backends_setup)  # noqa: S102


@hydra.main(config_path="configs", config_name="train_tokenizer", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    # ---- logger ----
    logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        entity=cfg.wandb.entity if cfg.wandb.entity else None,
        log_model=False,
    )

    # ---- callbacks ----
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.checkpoint.dirpath,
            every_n_train_steps=cfg.checkpoint.every_n_train_steps,
            save_top_k=cfg.checkpoint.save_top_k,
            filename=cfg.checkpoint.filename,
            save_last=True,
        ),
        TokenizerVizCallback(
            viz_every=cfg.tokenizer.viz_every,
            max_items=cfg.tokenizer.viz_max_items,
            max_T=cfg.tokenizer.viz_max_T,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # ---- trainer ----
    tc = cfg.trainer
    trainer = pl.Trainer(
        max_steps=tc.max_steps,
        precision=tc.precision,
        accumulate_grad_batches=tc.accumulate_grad_batches,
        gradient_clip_val=tc.gradient_clip_val,
        log_every_n_steps=tc.log_every_n_steps,
        num_nodes=tc.num_nodes,
        devices=tc.devices,
        strategy=tc.strategy,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=tc.enable_progress_bar,
    )

    # ---- fit ----
    model      = TokenizerLightningModule(cfg)
    datamodule = FrameDataModule(cfg)

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.resume)


if __name__ == "__main__":
    main()
