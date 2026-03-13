# dreamer4/lightning/frame_datamodule.py
from __future__ import annotations

import random

import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from task_set import TASK_SET
from sharded_frame_dataset import ShardedFrameDataset


def _worker_init_fn(worker_id: int):
    info = torch.utils.data.get_worker_info()
    s = int(info.seed) % (2 ** 32)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


class FrameDataModule(pl.LightningDataModule):
    """
    DataModule wrapping ShardedFrameDataset for tokenizer training.

    Lightning automatically wraps the DataLoader with DistributedSampler
    when strategy="ddp", so no manual sampler is needed here.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self._dataset: ShardedFrameDataset | None = None

    def setup(self, stage: str):
        dc = self.cfg.data
        tasks = list(dc.tasks) if dc.get("tasks") else TASK_SET
        self._dataset = ShardedFrameDataset(
            outdirs=list(dc.frame_dirs),
            tasks=tasks,
            seq_len=int(dc.seq_len_tokenizer),
            iid_sampling=True,
        )

    def train_dataloader(self) -> DataLoader:
        dc = self.cfg.data
        return DataLoader(
            self._dataset,
            batch_size=int(dc.batch_size_tokenizer),
            shuffle=True,                              # Lightning turns this into DistributedSampler in DDP
            num_workers=int(dc.num_workers),
            pin_memory=True,
            drop_last=True,
            persistent_workers=(int(dc.num_workers) > 0),
            worker_init_fn=_worker_init_fn,
        )
