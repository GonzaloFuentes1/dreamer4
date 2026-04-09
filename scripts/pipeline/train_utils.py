# scripts/pipeline/train_utils.py
# Shared helpers for training entry-point scripts (phases 1a–3).
from __future__ import annotations

import os
import sys
from typing import Any

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------------
# Repo paths — imported by every train_phase*.py
# ---------------------------------------------------------------------------
REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(REPO_ROOT, "configs")

# ---------------------------------------------------------------------------
# Load .env from repo root (sets WANDB_API_KEY etc. for subprocess-launched
# workers that don't go through _pipeline_lib.sh)
# ---------------------------------------------------------------------------
_env_file = os.path.join(REPO_ROOT, ".env")
if os.path.isfile(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#"):
                continue
            if _line.startswith("export "):
                _line = _line[len("export "):]
            if "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# Add src/ to the path once, here, so scripts that import train_utils get it
# for free.
_src = os.path.join(REPO_ROOT, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_wandb_logger(cfg: Any):
    """Create a WandbLogger from standard cfg.wandb fields."""
    from pytorch_lightning.loggers import WandbLogger

    return WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        entity=cfg.wandb.entity if cfg.wandb.entity else None,
        log_model=False,
    )


def make_checkpoint_callback(cfg: Any):
    """Create a ModelCheckpoint from standard cfg.checkpoint fields."""
    from pytorch_lightning.callbacks import ModelCheckpoint

    return ModelCheckpoint(
        dirpath=cfg.checkpoint.dirpath,
        every_n_train_steps=cfg.checkpoint.every_n_train_steps,
        save_top_k=cfg.checkpoint.save_top_k,
        filename=cfg.checkpoint.filename,
        save_last=True,
    )


def make_trainer(cfg: Any, callbacks: list, grad_clip: float | None = None):
    """
    Create a pl.Trainer from standard cfg.trainer fields.

    Args:
        cfg        — Hydra config with a `trainer` section
        callbacks  — list of Lightning callbacks
        grad_clip  — gradient clip value; None disables clipping
    """
    import pytorch_lightning as pl

    tc = cfg.trainer
    return pl.Trainer(
        max_steps=tc.max_steps,
        precision=tc.precision,
        accumulate_grad_batches=tc.accumulate_grad_batches,
        log_every_n_steps=tc.log_every_n_steps,
        num_nodes=tc.num_nodes,
        devices=tc.devices,
        strategy=tc.strategy,
        logger=make_wandb_logger(cfg),
        callbacks=callbacks,
        enable_progress_bar=tc.enable_progress_bar,
        gradient_clip_val=grad_clip if grad_clip and grad_clip > 0 else None,
    )
