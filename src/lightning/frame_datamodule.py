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
from hdf5_episode_dataset import HDF5EpisodeDataset

def _discover_h5_paths(data_dirs: list[str], tasks: list[str]) -> list[str]:
    """Descubre archivos .h5 en los data_dirs dados para las tasks."""
    from pathlib import Path
    paths = []
    for dd in data_dirs:
        for t in tasks:
            p = Path(dd) / f"{t}.h5"
            if p.exists():
                paths.append(str(p))
    return paths


def _discover_data_dirs_from_root(root: str) -> list[str]:
    """
    Dado un directorio raíz, descubre automáticamente todos los subdirectorios
    que contienen datos (archivos .h5 o subdirectorios con .pt shards).
    Permite pasar data_root: data/ en lugar de listar cada ciclo.
    """
    from pathlib import Path
    root_path = Path(root)
    if not root_path.is_dir():
        return [root]
    
    subdirs = sorted([
        str(d) for d in root_path.iterdir()
        if d.is_dir() and (
            any(d.glob("*.h5")) or       # HDF5
            any(d.glob("**/*.pt"))        # .pt shards
        )
    ])
    return subdirs if subdirs else [root]



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
        # Detectar HDF5 en data_dirs (si existeн, los usamos en lugar de .pt shards)
        # Soporte para data_root: descubre ciclos automáticamente si se define
        if dc.get("data_root"):
            data_dirs = _discover_data_dirs_from_root(str(dc.data_root))
        else:
            data_dirs = list(dc.get("data_dirs", dc.get("frame_dirs", [])))
        h5_paths = _discover_h5_paths(data_dirs, tasks)
        use_hdf5 = len(h5_paths) > 0 and dc.get("use_hdf5", True)
        if use_hdf5:
            self._dataset = HDF5EpisodeDataset(
                h5_paths=h5_paths,
                seq_len=int(dc.seq_len_tokenizer),
                mode="frames",
            )
        else:
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
