# dreamer4/lightning/frame_datamodule.py
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from task_set import TASK_SET
from hdf5_episode_dataset import HDF5EpisodeDataset


def _resolve_h5_paths(dc, tasks: list[str]) -> list[str]:
    """Descubre archivos .h5 para las tasks dadas.

    Soporta:
      data_root: <dir>   → busca .h5 directamente en ese dir (o ciclos dentro)
      data_dirs: [...]   → busca .h5 en cada directorio listado
    """
    if dc.get("data_root"):
        root = Path(str(dc.data_root))
        dirs = _subdirs_with_h5(root)
    else:
        dirs = [Path(d) for d in dc.get("data_dirs", [])]

    paths = []
    for d in dirs:
        for t in tasks:
            p = d / f"{t}.h5"
            if p.exists():
                paths.append(str(p))
    return paths


def _subdirs_with_h5(root: Path) -> list[Path]:
    """Si root contiene .h5 directamente, lo devuelve. Si no, devuelve subdirs con .h5."""
    if any(root.glob("*.h5")):
        return [root]
    return sorted([d for d in root.iterdir() if d.is_dir() and any(d.glob("*.h5"))])


def _worker_init_fn(worker_id: int):
    info = torch.utils.data.get_worker_info()
    s = int(info.seed) % (2 ** 32)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


class FrameDataModule(pl.LightningDataModule):
    """DataModule de frames para entrenamiento del tokenizador (Phase 1a).

    Usa HDF5EpisodeDataset en mode='frames'. Lightning añade DistributedSampler
    automáticamente en DDP, no hace falta configurarlo aquí.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self._dataset: HDF5EpisodeDataset | None = None

    def setup(self, stage: str):
        dc    = self.cfg.data
        tasks = list(dc.tasks) if dc.get("tasks") else TASK_SET

        h5_paths = _resolve_h5_paths(dc, tasks)
        if not h5_paths:
            raise FileNotFoundError(
                f"No se encontraron archivos .h5 para las tasks={tasks}. "
                "Asegurate de haber corrido train_phase0_collect_episodes.py "
                "o convert_pt_to_hdf5.py primero."
            )

        self._dataset = HDF5EpisodeDataset(
            h5_paths=h5_paths,
            seq_len=int(dc.seq_len_tokenizer),
            mode="frames",
        )

    def train_dataloader(self) -> DataLoader:
        dc = self.cfg.data
        return DataLoader(
            self._dataset,
            batch_size=int(dc.batch_size_tokenizer),
            shuffle=True,
            num_workers=int(dc.num_workers),
            pin_memory=True,
            drop_last=True,
            persistent_workers=(int(dc.num_workers) > 0),
            worker_init_fn=_worker_init_fn,
        )
