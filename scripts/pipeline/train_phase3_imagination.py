#!/usr/bin/env python
# scripts/pipeline/train_phase3_imagination.py — Phase 3: Imagination training (PMPO).
#
# Requires a Phase-2 finetune.ckpt (from train_phase2_finetuning.py).
#
# Usage (single GPU):
#   python train_phase3_imagination.py agent.finetune_ckpt=/path/to/finetune.ckpt trainer.devices=1
#
# Usage (8 GPUs):
#   python train_phase3_imagination.py agent.finetune_ckpt=... trainer.devices=8
#
# Smaller model (debug):
#   python train_phase3_imagination.py agent=small agent.finetune_ckpt=... trainer.devices=1
#
# Resume from Lightning checkpoint:
#   python train_phase3_imagination.py agent.finetune_ckpt=... resume=./logs/agent_ckpts/last.ckpt

from train_utils import REPO_ROOT, CONFIG_PATH, make_checkpoint_callback, make_trainer  # noqa: E402 (sets sys.path)

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor

from lightning.agent_module import AgentLightningModule
from lightning.wm_datamodule import WMDataModule
from lightning.callbacks import AgentEvalCallback


@hydra.main(config_path=CONFIG_PATH, config_name="train_phase3_imagination", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    eval_tasks = list(cfg.data.tasks) if cfg.data.get("tasks") else []
    callbacks = [
        make_checkpoint_callback(cfg),
        LearningRateMonitor(logging_interval="step"),
        AgentEvalCallback(
            tasks=eval_tasks,
            eval_every=int(cfg.agent.eval_every),
            n_episodes=int(cfg.agent.get("eval_n_episodes", 1)),
            episode_len=int(cfg.agent.get("eval_episode_len", 500)),
            context_len=int(cfg.agent.ctx_length),
            img_size=int(cfg.data.get("img_size", 64)),
            frame_skip=int(cfg.data.get("frame_skip", 2)),
        ),
    ]

    trainer = make_trainer(cfg, callbacks, grad_clip=cfg.agent.grad_clip)

    model      = AgentLightningModule(cfg)
    datamodule = WMDataModule(cfg, use_actions=True)

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("resume", None))


if __name__ == "__main__":
    main()
