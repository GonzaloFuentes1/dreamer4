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


class WMDataModule(pl.LightningDataModule):
    """
    DataModule wrapping WMDataset for dynamics training.

    Supports action-conditioned and action-free (frame-only) modes.  Pass
    use_actions=True/False explicitly to override cfg.dynamics.use_actions
    (useful in phases 2 and 3 where the config section is named differently).
    Lightning handles DistributedSampler automatically in DDP mode.
    """

    def __init__(self, cfg: DictConfig, use_actions: bool | None = None):
        super().__init__()
        self.cfg = cfg
        self._use_actions = use_actions
        self._dataset: WMDataset | None = None

    def setup(self, stage: str):
        dc  = self.cfg.data
        tasks = list(dc.tasks) if dc.get("tasks") else TASK_SET

        is_rank0 = (not hasattr(self, "trainer")) or (self.trainer is None) or (self.trainer.global_rank == 0)

        if self._use_actions is not None:
            use_actions = self._use_actions
        else:
            use_actions = self.cfg.dynamics.use_actions
        # Detectar si hay archivos HDF5 disponibles (escritos por Phase-0 con use_hdf5=true)
        # Soporte para data_root: descubre ciclos automáticamente si se define
        if dc.get("data_root"):
            data_dirs = _discover_data_dirs_from_root(str(dc.data_root))
            frame_dirs = data_dirs  # mismo root para frames
        else:
            data_dirs = OmegaConf.to_container(dc.data_dirs, resolve=True)
            frame_dirs = OmegaConf.to_container(dc.frame_dirs, resolve=True)
        h5_paths = _discover_h5_paths(
            data_dirs if isinstance(data_dirs, list) else [data_dirs], tasks
        )
        use_hdf5 = len(h5_paths) > 0 and dc.get("use_hdf5", True)
        if use_hdf5 and is_rank0:
            print(f"[WMDataModule] Usando HDF5EpisodeDataset ({len(h5_paths)} archivos)")

        if use_actions:
            if use_hdf5:
                self._dataset = HDF5EpisodeDataset(
                    h5_paths=h5_paths,
                    seq_len=int(dc.seq_len_dynamics),
                    mode="episodes",
                )
            else:
                self._dataset = WMDataset(
                    data_dir=data_dirs,
                    frames_dir=frame_dirs,
                    seq_len=int(dc.seq_len_dynamics),
                    img_size=int(dc.get("img_size", 128)),
                    action_dim=int(dc.get("action_dim", 16)),
                    tasks_json=str(dc.tasks_json),
                    tasks=tasks,
                    verbose=is_rank0,
                )
        else:
            if use_hdf5:
                self._dataset = HDF5EpisodeDataset(
                    h5_paths=h5_paths,
                    seq_len=int(dc.seq_len_dynamics),
                    mode="frames",
                )
            else:
                from sharded_frame_dataset import ShardedFrameDataset
                self._dataset = ShardedFrameDataset(
                    outdirs=frame_dirs,
                    tasks=tasks,
                    seq_len=int(dc.seq_len_dynamics),
                )

    def train_dataloader(self) -> DataLoader:
        dc  = self.cfg.data

        kwargs = dict(
            batch_size=int(dc.batch_size_dynamics),
            shuffle=True,
            num_workers=int(dc.num_workers),
            pin_memory=True,
            drop_last=True,
            persistent_workers=(int(dc.num_workers) > 0),
            worker_init_fn=_worker_init_fn,
        )
        if self._use_actions is not None:
            use_actions = self._use_actions
        else:
            use_actions = self.cfg.dynamics.use_actions
        if use_actions:
            kwargs["collate_fn"] = collate_batch

        return DataLoader(self._dataset, **kwargs)
