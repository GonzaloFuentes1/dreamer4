"""
src/hdf5_episode_dataset.py
===========================
Dataset respaldado por archivos HDF5 (uno por tarea), escritos por Phase-0.

Ventajas vs el formato .pt actual:
  - Escritura SWMR: crash-safe (sin corrupción al interrumpir la colección)
  - Un solo archivo por tarea: sin gestión de shards
  - Acceso indexado O(1): sin necesidad de cargar todos los shards para contar frames
  - Acciones/rewards cacheados en RAM; pixels cargados por slice desde disco

Estructura del archivo HDF5 (por tarea):
  /pixels     (N, 3, H, W)  uint8   — frames crudos
  /action     (N, A)        float32
  /reward     (N,)          float32
  /episode    (N,)          int64   — episode_id por step
  /ep_len     (E,)          int64   — longitud de cada episodio
  /ep_offset  (E,)          int64   — índice global de inicio de cada episodio

Modos:
  "frames"   — devuelve (seq_len, 3, H, W) float32 sin alineación de episodio
               (para entrenar el tokenizador)
  "episodes" — devuelve dict{"frames","action","reward"} alineado por episodio
               (para entrenar dynamics / finetune / agent)
"""
from __future__ import annotations

import bisect
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Writer
# ─────────────────────────────────────────────────────────────────────────────

class HDF5EpisodeWriter:
    """
    Escribe episodios a un archivo HDF5 de forma incremental, con SWMR activado.

    Uso típico dentro de Phase-0:

        writer = HDF5EpisodeWriter(path, img_size=64, action_dim=16, chunk_frames=512)
        # ... colección ...
        writer.append_batch(pixels, actions, rewards, ep_ids)
        # ... más batches ...
        writer.finalize()   # escribe ep_len, ep_offset y cierra el archivo
    """

    def __init__(
        self,
        path: Union[str, Path],
        img_size: int = 64,
        action_dim: int = 16,
        chunk_frames: int = 512,
        compression: Optional[str] = None,   # None | "lzf" | "gzip"
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.img_size = img_size
        self.action_dim = action_dim
        self.chunk_frames = chunk_frames
        self.compression = compression

        self._n = 0          # steps escritos hasta ahora
        self._ep_ids_so_far: List[int] = []

        # Abrir / crear el archivo
        self._f = h5py.File(str(self.path), "w", libver="latest")

        C, H, W = 3, img_size, img_size
        kw = dict(compression=compression) if compression else {}

        self._ds_pixels = self._f.create_dataset(
            "pixels", shape=(0, C, H, W), maxshape=(None, C, H, W),
            dtype="uint8", chunks=(chunk_frames, C, H, W), **kw,
        )
        self._ds_action = self._f.create_dataset(
            "action", shape=(0, action_dim), maxshape=(None, action_dim),
            dtype="float32", chunks=(chunk_frames, action_dim), **kw,
        )
        self._ds_reward = self._f.create_dataset(
            "reward", shape=(0,), maxshape=(None,),
            dtype="float32", chunks=(chunk_frames,),
        )
        self._ds_episode = self._f.create_dataset(
            "episode", shape=(0,), maxshape=(None,),
            dtype="int64", chunks=(chunk_frames,),
        )

        # Activar SWMR para que lectores concurrentes puedan abrir el archivo
        self._f.swmr_mode = True

    # ------------------------------------------------------------------

    def append_batch(
        self,
        pixels:  torch.Tensor,   # (N, 3, H, W) uint8
        actions: torch.Tensor,   # (N, A) float32
        rewards: torch.Tensor,   # (N,) float32
        ep_ids:  torch.Tensor,   # (N,) int64
    ):
        N = int(pixels.shape[0])
        new_n = self._n + N

        self._ds_pixels.resize(new_n, axis=0)
        self._ds_action.resize(new_n, axis=0)
        self._ds_reward.resize(new_n, axis=0)
        self._ds_episode.resize(new_n, axis=0)

        pix_np = pixels.cpu().numpy()
        if pix_np.dtype != np.uint8:
            pix_np = (np.clip(pix_np, 0, 1) * 255).astype(np.uint8)

        self._ds_pixels[self._n:new_n] = pix_np
        self._ds_action[self._n:new_n] = actions.cpu().numpy().astype(np.float32)
        self._ds_reward[self._n:new_n] = rewards.cpu().numpy().astype(np.float32)
        self._ds_episode[self._n:new_n] = ep_ids.cpu().numpy().astype(np.int64)

        self._ep_ids_so_far.extend(ep_ids.cpu().tolist())
        self._n = new_n

        # Flush para que lectores SWMR vean los datos nuevos
        self._f.flush()

    # ------------------------------------------------------------------

    def finalize(self) -> int:
        """Calcula ep_len / ep_offset, los escribe, y cierra el archivo."""
        if self._n == 0:
            self._f.close()
            return 0

        ep_ids_arr = np.array(self._ep_ids_so_far, dtype=np.int64)
        unique_eps, first_idx, counts = np.unique(
            ep_ids_arr, return_index=True, return_counts=True
        )
        # Ordenar por primera aparición
        order = np.argsort(first_idx)
        ep_offsets = first_idx[order].astype(np.int64)
        ep_lens    = counts[order].astype(np.int64)

        self._f.create_dataset("ep_len",    data=ep_lens,    dtype="int64")
        self._f.create_dataset("ep_offset", data=ep_offsets, dtype="int64")
        self._f.attrs["n_steps"]   = self._n
        self._f.attrs["n_episodes"] = len(ep_lens)
        self._f.flush()
        self._f.close()

        print(
            f"[HDF5EpisodeWriter] Finalizado: {self.path.name} — "
            f"{self._n:,} steps, {len(ep_lens):,} episodios"
        )
        return self._n

    # ------------------------------------------------------------------

    def __enter__(self): return self
    def __exit__(self, *_): self.finalize()

    @property
    def n_steps(self) -> int:
        return self._n


# ─────────────────────────────────────────────────────────────────────────────
# Reader / Dataset
# ─────────────────────────────────────────────────────────────────────────────

class HDF5EpisodeDataset(Dataset):
    """
    Dataset PyTorch que lee desde archivos HDF5 escritos por HDF5EpisodeWriter.

    Sirve para dos roles:
      mode="frames"   → tokenizador (ShardedFrameDataset replacement)
      mode="episodes" → dynamics / finetune / agent (WMDataset replacement)

    Parámetros
    ----------
    h5_paths:
        Ruta(s) a archivos .h5. Cada archivo corresponde a UNA tarea, en un
        ciclo de colección. Para multi-tarea / multi-ciclo, pasar varias rutas.
    seq_len:
        Número de pasos a devolver por sample.
    mode:
        "frames"   — sample libre sobre todos los starts válidos
        "episodes" — sample alineado al inicio de episodio, dentro del episodio
    iid_sampling:
        Si True (default), ignora idx y samplea uniformemente al azar.
    cache_action:
        Si True, carga action + reward completos en RAM al inicializar
        (son pequeños: ~8 bytes × N × A). Solo se aplica en mode="episodes".
    rdcc_mb:
        Tamaño del cache de chunks de HDF5 para pixels, en MB.
    """

    def __init__(
        self,
        h5_paths: Union[str, Path, Sequence[Union[str, Path]]],
        seq_len:       int  = 16,
        mode:          str  = "frames",
        iid_sampling:  bool = True,
        cache_action:  bool = True,
        rdcc_mb:       int  = 256,
    ):
        super().__init__()
        self.seq_len      = int(seq_len)
        self.mode         = mode
        self.iid_sampling = iid_sampling
        self.rdcc_bytes   = int(rdcc_mb) * 1024 * 1024

        if isinstance(h5_paths, (str, Path)):
            h5_paths = [h5_paths]
        self.h5_paths = [Path(p) for p in h5_paths]

        # Metadatos cargados al inicio (arrays pequeños)
        self._metas:      List[Dict]  = []
        self._cum_starts: List[int]   = []
        total_starts = 0

        for path in self.h5_paths:
            if not path.exists():
                print(f"[HDF5EpisodeDataset] WARN: {path} no encontrado, saltando")
                continue
            with h5py.File(str(path), "r", swmr=True) as f:
                n_steps   = int(f["pixels"].shape[0])
                ep_len    = f["ep_len"][:]
                ep_offset = f["ep_offset"][:]

                meta: Dict = {
                    "path":      str(path),
                    "n_steps":   n_steps,
                    "ep_len":    ep_len.copy(),
                    "ep_offset": ep_offset.copy(),
                }

                if mode == "frames":
                    n_starts = max(0, n_steps - self.seq_len + 1)
                else:
                    # solo episodios con suficientes frames
                    n_starts = int(np.sum(ep_len >= self.seq_len))

                if mode == "episodes" and cache_action and "action" in f:
                    meta["action_cache"] = torch.from_numpy(f["action"][:])
                    meta["reward_cache"] = torch.from_numpy(f["reward"][:])

                meta["n_starts"] = n_starts
                if n_starts > 0:
                    self._metas.append(meta)
                    total_starts += n_starts
                    self._cum_starts.append(total_starts)

        self.total_starts = total_starts
        self._handles: Dict[str, h5py.File] = {}   # handles lazy por worker

        print(
            f"[HDF5EpisodeDataset] mode={mode}, archivos={len(self._metas)}, "
            f"total_starts={total_starts:,}, seq_len={self.seq_len}"
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.total_starts

    def _open(self, path: str) -> h5py.File:
        if path not in self._handles:
            self._handles[path] = h5py.File(
                path, "r", swmr=True, rdcc_nbytes=self.rdcc_bytes,
            )
        return self._handles[path]

    def _resolve(self, idx: int):
        file_idx = bisect.bisect_right(self._cum_starts, idx)
        prev = 0 if file_idx == 0 else self._cum_starts[file_idx - 1]
        return self._metas[file_idx], idx - prev

    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        if self.total_starts == 0:
            raise IndexError("HDF5EpisodeDataset vacío")

        if self.iid_sampling:
            idx = random.randint(0, self.total_starts - 1)

        meta, local_idx = self._resolve(idx)
        f = self._open(meta["path"])

        if self.mode == "frames":
            start = local_idx
            end   = start + self.seq_len
            pix   = f["pixels"][start:end]              # (T, 3, H, W) uint8
            return torch.from_numpy(pix.astype(np.float32)) / 255.0

        # ── mode = "episodes" ──────────────────────────────────────
        valid_mask    = meta["ep_len"] >= self.seq_len
        valid_indices = np.where(valid_mask)[0]
        ep_i   = valid_indices[local_idx % len(valid_indices)]
        ep_off = int(meta["ep_offset"][ep_i])
        ep_len = int(meta["ep_len"][ep_i])

        max_off = ep_len - self.seq_len
        off     = random.randint(0, max_off) if max_off > 0 else 0
        start   = ep_off + off
        end     = start + self.seq_len

        pix = torch.from_numpy(f["pixels"][start:end].astype(np.float32)) / 255.0

        if "action_cache" in meta:
            action = meta["action_cache"][start:end]
            reward = meta["reward_cache"][start:end]
        else:
            action = torch.from_numpy(f["action"][start:end])
            reward = torch.from_numpy(f["reward"][start:end])

        return {"frames": pix, "action": action, "reward": reward}

    # ------------------------------------------------------------------

    @staticmethod
    def from_demo_dir(
        demo_dir: Union[str, Path],
        tasks: Sequence[str],
        **kwargs,
    ) -> "HDF5EpisodeDataset":
        """
        Construye un dataset buscando <demo_dir>/<task>.h5 para cada tarea.
        """
        demo_dir = Path(demo_dir)
        paths = [demo_dir / f"{t}.h5" for t in tasks if (demo_dir / f"{t}.h5").exists()]
        if not paths:
            raise FileNotFoundError(
                f"No se encontraron archivos .h5 en {demo_dir} para tasks={list(tasks)}"
            )
        return HDF5EpisodeDataset(paths, **kwargs)
