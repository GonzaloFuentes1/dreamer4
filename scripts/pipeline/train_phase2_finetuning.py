#!/usr/bin/env python
# scripts/pipeline/train_phase2_finetuning.py — Phase 2: Agent finetuning (BC + reward model).
#
# Loads a frozen Phase-1a tokenizer and (optionally) a Phase-1b dynamics checkpoint,
# then jointly finetunes the dynamics transformer, TaskEmbedder, PolicyHead, and
# RewardHead using behavior cloning and reward prediction with MTP (L=8).
#
# Usage (single GPU):
#   python train_phase2_finetuning.py \
#     finetune.tokenizer_ckpt=/path/to/tokenizer.ckpt \
#     finetune.dynamics_ckpt=/path/to/dynamics.ckpt   \
#     trainer.devices=1
#
# Usage (8 GPUs):
#   python train_phase2_finetuning.py \
#     finetune.tokenizer_ckpt=... finetune.dynamics_ckpt=... trainer.devices=8
#
# Resume:
#   python train_phase2_finetuning.py \
#     finetune.tokenizer_ckpt=... finetune.dynamics_ckpt=... \
#     resume=./logs/finetune_ckpts/last.ckpt

from train_utils import REPO_ROOT, CONFIG_PATH, make_checkpoint_callback, make_trainer  # noqa: E402 (sets sys.path)

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor

from lightning.finetune_module import FinetuneLightningModule
from lightning.wm_datamodule import WMDataModule
from lightning.callbacks import DynamicsEvalCallback


@hydra.main(config_path=CONFIG_PATH, config_name="train_phase2_finetuning", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    callbacks = [
        make_checkpoint_callback(cfg),
        DynamicsEvalCallback(eval_every=cfg.finetune.eval_every),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = make_trainer(cfg, callbacks, grad_clip=cfg.finetune.grad_clip)

    model      = FinetuneLightningModule(cfg)
    datamodule = WMDataModule(cfg, use_actions=True)

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("resume", None))


if __name__ == "__main__":
    main()
