#!/usr/bin/env python
# train_dynamics.py — PyTorch Lightning + Hydra entry point
#
# Usage (single GPU):
#   python train_dynamics.py dynamics.tokenizer_ckpt=/path/to/tokenizer.ckpt trainer.devices=1
#
# Usage (8 GPUs):
#   python train_dynamics.py dynamics.tokenizer_ckpt=/path/to/ckpt trainer.devices=8
#
# Override any config key at the CLI:
#   python train_dynamics.py dynamics=small wandb.name=exp_001
#
# Hyperparameter sweep (Hydra multirun):
#   python train_dynamics.py -m dynamics.d_model_dyn=256,512 dynamics.dyn_depth=4,8
#
# Resume from a Lightning checkpoint:
#   python train_dynamics.py dynamics.tokenizer_ckpt=... resume=./logs/dynamics_ckpts/last.ckpt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dreamer4"))

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from lightning.dynamics_module import DynamicsLightningModule
from lightning.wm_datamodule import WMDataModule
from lightning.callbacks import DynamicsEvalCallback, ActionShuffleMetricCallback

torch_backends_setup = """
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
"""
exec(torch_backends_setup)  # noqa: S102


@hydra.main(config_path="configs", config_name="train_dynamics", version_base=None)
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
        DynamicsEvalCallback(eval_every=cfg.dynamics.eval_every),
        ActionShuffleMetricCallback(log_every=cfg.trainer.log_every_n_steps * 5),
        LearningRateMonitor(logging_interval="step"),
    ]

    # ---- trainer ----
    tc = cfg.trainer
    trainer = pl.Trainer(
        max_steps=tc.max_steps,
        precision=tc.precision,
        accumulate_grad_batches=tc.accumulate_grad_batches,
        log_every_n_steps=tc.log_every_n_steps,
        num_nodes=tc.num_nodes,
        devices=tc.devices,
        strategy=tc.strategy,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=tc.enable_progress_bar,
        gradient_clip_val=cfg.dynamics.grad_clip if cfg.dynamics.grad_clip > 0 else None,
    )

    # ---- fit ----
    model      = DynamicsLightningModule(cfg)
    datamodule = WMDataModule(cfg)

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.resume)


if __name__ == "__main__":
    main()
