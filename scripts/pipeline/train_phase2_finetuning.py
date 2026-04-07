#!/usr/bin/env python
# scripts/pipeline/train_phase2_finetuning.py — Phase 2: Agent Finetuning (BC + reward model).
#
# Loads a frozen Phase-1 tokenizer and (optionally) a Phase-1 dynamics checkpoint,
# then jointly finetunes the dynamics transformer, TaskEmbedder, PolicyHead, and
# RewardHead using behavior cloning and reward prediction with MTP (L=8).
#
# Usage (single GPU):
#   python train_agent_finetune.py \
#     finetune.tokenizer_ckpt=/path/to/tokenizer.ckpt \
#     finetune.dynamics_ckpt=/path/to/dynamics.ckpt   \
#     trainer.devices=1
#
# Usage (8 GPUs):
#   python train_agent_finetune.py \
#     finetune.tokenizer_ckpt=... finetune.dynamics_ckpt=... trainer.devices=8
#
# Small model (debug):
#   python train_agent_finetune.py finetune=small \
#     finetune.tokenizer_ckpt=... finetune.dynamics_ckpt=...
#
# Resume:
#   python train_agent_finetune.py \
#     finetune.tokenizer_ckpt=... finetune.dynamics_ckpt=... \
#     resume=./logs/finetune_ckpts/last.ckpt

import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
CONFIG_PATH = os.path.join(REPO_ROOT, "configs")

import torch
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from lightning.finetune_module import FinetuneLightningModule
from lightning.wm_datamodule import WMDataModule
from lightning.callbacks import DynamicsEvalCallback

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra.main(config_path=CONFIG_PATH, config_name="train_phase2_finetuning", version_base=None)
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
        DynamicsEvalCallback(eval_every=cfg.finetune.eval_every),
        LearningRateMonitor(logging_interval="step"),
    ]

    # ---- model ----
    # Inject dynamics.use_actions=True so WMDataModule loads action-conditioned data.
    # Use to_container first to strip struct constraints before merging.
    from omegaconf import OmegaConf
    cfg_dm = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.update(cfg_dm, "dynamics.use_actions", True, merge=True)

    model      = FinetuneLightningModule(cfg)
    datamodule = WMDataModule(cfg_dm)

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
        gradient_clip_val=cfg.finetune.grad_clip if cfg.finetune.grad_clip > 0 else None,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("resume", None))


if __name__ == "__main__":
    main()
