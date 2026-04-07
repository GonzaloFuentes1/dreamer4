# dreamer4/lightning/wm_datamodule.py
from __future__ import annotations

import random

import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from task_set import TASK_SET
from wm_dataset import WMDataset, collate_batch


def _worker_init_fn(worker_id: int):
    info = torch.utils.data.get_worker_info()
    s = int(info.seed) % (2 ** 32)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


class WMDataModule(pl.LightningDataModule):
    """
    DataModule wrapping WMDataset for dynamics training.

    Supports action-conditioned and action-free (frame-only) modes depending
    on cfg.dynamics.use_actions.  Lightning handles DistributedSampler
    automatically in DDP mode.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self._dataset: WMDataset | None = None

    def setup(self, stage: str):
        dc  = self.cfg.data
        dyn = self.cfg.dynamics
        tasks = list(dc.tasks) if dc.get("tasks") else TASK_SET

        is_rank0 = (not hasattr(self, "trainer")) or (self.trainer is None) or (self.trainer.global_rank == 0)

        if dyn.use_actions:
            self._dataset = WMDataset(
                data_dir=OmegaConf.to_container(dc.data_dirs, resolve=True),
                frames_dir=OmegaConf.to_container(dc.frame_dirs, resolve=True),
                seq_len=int(dc.seq_len_dynamics),
                img_size=int(dc.get("img_size", 128)),
                action_dim=int(dc.get("action_dim", 16)),
                tasks_json=str(dc.tasks_json),
                tasks=tasks,
                verbose=is_rank0,
            )
        else:
            from sharded_frame_dataset import ShardedFrameDataset
            self._dataset = ShardedFrameDataset(
                outdirs=OmegaConf.to_container(dc.frame_dirs, resolve=True),
                tasks=tasks,
                seq_len=int(dc.seq_len_dynamics),
            )

    def train_dataloader(self) -> DataLoader:
        dc  = self.cfg.data
        dyn = self.cfg.dynamics

        kwargs = dict(
            batch_size=int(dc.batch_size_dynamics),
            shuffle=True,
            num_workers=int(dc.num_workers),
            pin_memory=True,
            drop_last=True,
            persistent_workers=(int(dc.num_workers) > 0),
            worker_init_fn=_worker_init_fn,
        )
        if dyn.use_actions:
            kwargs["collate_fn"] = collate_batch

        return DataLoader(self._dataset, **kwargs)
