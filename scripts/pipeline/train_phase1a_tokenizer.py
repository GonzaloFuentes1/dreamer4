#!/usr/bin/env python
# scripts/pipeline/train_phase1a_tokenizer.py — Phase 1a: Tokenizer training.
#
# Usage (single GPU):
#   python train_phase1a_tokenizer.py trainer.devices=1
#
# Usage (8 GPUs, single node):
#   python train_phase1a_tokenizer.py trainer.devices=8
#
# Override any config key at the CLI:
#   python train_phase1a_tokenizer.py tokenizer=small wandb.name=exp_001
#
# Hyperparameter sweep (Hydra multirun):
#   python train_phase1a_tokenizer.py -m tokenizer.d_model=128,256 tokenizer.depth=4,8
#
# Resume from a Lightning checkpoint:
#   python train_phase1a_tokenizer.py resume=./logs/tokenizer_ckpts/last.ckpt

from train_utils import REPO_ROOT, CONFIG_PATH, make_checkpoint_callback, make_trainer  # noqa: E402 (sets sys.path)

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor

from lightning.tokenizer_module import TokenizerLightningModule
from lightning.frame_datamodule import FrameDataModule
from lightning.callbacks import TokenizerVizCallback


@hydra.main(config_path=CONFIG_PATH, config_name="train_phase1a_tokenizer", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    callbacks = [
        make_checkpoint_callback(cfg),
        TokenizerVizCallback(
            viz_every=cfg.tokenizer.viz_every,
            max_items=cfg.tokenizer.viz_max_items,
            max_T=cfg.tokenizer.viz_max_T,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = make_trainer(
        cfg,
        callbacks,
        grad_clip=cfg.trainer.get("gradient_clip_val"),
    )

    model      = TokenizerLightningModule(cfg)
    datamodule = FrameDataModule(cfg)

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.resume)


if __name__ == "__main__":
    main()
