#!/usr/bin/env python
# train_phase3_imagination.py — Phase 3: Imagination Training (PMPO, PyTorch Lightning + Hydra)
#
# Requires a Phase-2 finetune.ckpt (from train_agent_finetune.py).
#
# Usage (single GPU):
#   python train_agent.py agent.finetune_ckpt=/path/to/finetune.ckpt trainer.devices=1
#
# Usage (8 GPUs):
#   python train_agent.py agent.finetune_ckpt=... trainer.devices=8
#
# Smaller model (debug):
#   python train_agent.py agent=small agent.finetune_ckpt=... trainer.devices=1
#
# Resume from Lightning checkpoint:
#   python train_agent.py agent.finetune_ckpt=... resume=./logs/agent_ckpts/last.ckpt

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from lightning.agent_module import AgentLightningModule
from lightning.callbacks import AgentEvalCallback
from lightning.wm_datamodule import WMDataModule

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra.main(config_path="configs", config_name="train_phase3_imagination", version_base=None)
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
    eval_tasks = list(cfg.data.tasks) if cfg.data.get("tasks") else []
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.checkpoint.dirpath,
            every_n_train_steps=cfg.checkpoint.every_n_train_steps,
            save_top_k=cfg.checkpoint.save_top_k,
            filename=cfg.checkpoint.filename,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        AgentEvalCallback(
            tasks=eval_tasks,
            eval_every=int(cfg.agent.eval_every),
            n_episodes=int(cfg.agent.get("eval_n_episodes", 3)),
            episode_len=int(cfg.agent.get("eval_episode_len", 500)),
            context_len=int(cfg.agent.ctx_length),
            img_size=128,
        ),
    ]

    # ---- model ----
    # WMDataModule reads cfg.dynamics.use_actions — inject it from agent section
    from omegaconf import OmegaConf
    cfg_with_dyn_flag = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.update(cfg_with_dyn_flag, "dynamics.use_actions", True, merge=True)

    model      = AgentLightningModule(cfg)
    datamodule = WMDataModule(cfg_with_dyn_flag)

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
        gradient_clip_val=cfg.agent.grad_clip if cfg.agent.grad_clip > 0 else None,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("resume", None))


if __name__ == "__main__":
    main()
