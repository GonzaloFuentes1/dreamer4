"""
goal_dataset.py — GoalDataset wrapper para entrenamiento goal-conditioned.

¿Qué hace?
----------
Envuelve HDF5EpisodeDataset (mode="episodes") y añade un tensor 'goal' a cada
muestra: una imagen que representa el DESTINO visual que el agente tendría que
alcanzar. Esto permite entrenar modelos y políticas goal-conditioned sin
modificar el código de colección de datos.

¿Cómo funciona el muestreo del goal?
-------------------------------------
Cada vez que se pide un sample, el goal se elige al azar entre tres fuentes:

  1. "future":    Un frame aleatorio en el MISMO episodio, DESPUÉS del último
                  frame de la secuencia actual. Captura el objetivo inmediato.
                  → Peso: p_future  (default 0.50)

  2. "end":       El ÚLTIMO frame del episodio actual. El agente ya completó la
                  tarea → frame terminal como meta aspiracional.
                  → Peso: p_end    (default 0.20)

  3. "random":    Un frame aleatorio de CUALQUIER episodio/tarea del dataset.
                  Fuerza al modelo a entender qué estado tiene el goal en lugar
                  de memorizarlo.
                  → Peso: p_random (default 0.30)

El goal tiene shape (3, H, W) float32 en [0, 1] — igual que los frames del
dataset base.

Uso
---
    from hdf5_episode_dataset import HDF5EpisodeDataset
    from goal_dataset import GoalDataset

    base = HDF5EpisodeDataset(
        h5_paths=["data/cycle0/walker-walk.h5"],
        seq_len=16,
        mode="episodes",
    )

    dataset = GoalDataset(base, p_future=0.5, p_random=0.3, p_end=0.2)

    sample = dataset[0]
    # sample["frames"]  → (16, 3, 64, 64)  secuencia de observaciones
    # sample["action"]  → (16, A)           acciones
    # sample["reward"]  → (16,)             recompensas
    # sample["goal"]    → (3, 64, 64)       imagen de destino ← nuevo

    # Usar en DataLoader normalmente:
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    for batch in loader:
        frames = batch["frames"]  # (B, 16, 3, 64, 64)
        goal   = batch["goal"]    # (B, 3, 64, 64)
        ...

Nota sobre seeds / reproducibilidad
--------------------------------------
El muestreo es no-determinista por defecto (útil en entrenamiento). Para
reproducibilidad puedes hacer:
    GoalDataset(base, seed=42)
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from hdf5_episode_dataset import HDF5EpisodeDataset


class GoalDataset(Dataset):
    """
    Wrapper sobre HDF5EpisodeDataset que añade un frame 'goal' a cada muestra.

    Args:
        dataset:   HDF5EpisodeDataset con mode="episodes" ya construido.
        p_future:  Probabilidad de elegir un frame futuro del mismo episodio.
        p_end:     Probabilidad de elegir el último frame del episodio.
        p_random:  Probabilidad de elegir un frame aleatorio del dataset completo.
                   (p_future + p_end + p_random debe sumar 1.0)
        seed:      Semilla para el RNG de muestreo. None = no-determinista.

    Raises:
        ValueError: Si las probabilidades no suman 1 o el dataset no es "episodes".
    """

    def __init__(
        self,
        dataset: "HDF5EpisodeDataset",
        p_future: float = 0.50,
        p_end:    float = 0.20,
        p_random: float = 0.30,
        seed: int | None = None,
    ):
        if abs(p_future + p_end + p_random - 1.0) > 1e-5:
            raise ValueError(
                f"p_future + p_end + p_random debe ser 1.0, "
                f"recibido {p_future + p_end + p_random:.4f}"
            )
        if dataset.mode != "episodes":
            raise ValueError(
                "GoalDataset requiere un HDF5EpisodeDataset con mode='episodes'. "
                f"El dataset recibido tiene mode='{dataset.mode}'."
            )

        self.dataset  = dataset
        self.p_future = p_future
        self.p_end    = p_end
        self.p_random = p_random
        self._rng     = np.random.default_rng(seed)

        # Pre-calculamos el total de steps para muestreo aleatorio global
        # Cada meta tiene ep_len: un array con la longitud de cada episodio
        self._total_steps = sum(
            int(meta["ep_len"].sum()) for meta in dataset._metas
        )
        # Cumulative step counts por archivo → para muestreo global uniforme
        self._cum_steps: list[int] = []
        running = 0
        for meta in dataset._metas:
            running += int(meta["ep_len"].sum())
            self._cum_steps.append(running)

    # ------------------------------------------------------------------
    # Interfaz Dataset
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[idx]           # {"frames", "action", "reward"}
        goal   = self._sample_goal(idx)      # (3, H, W)
        sample["goal"] = goal
        return sample

    # ------------------------------------------------------------------
    # Lógica de muestreo del goal
    # ------------------------------------------------------------------

    def _sample_goal(self, idx: int) -> torch.Tensor:
        """Elige la estrategia de muestreo y devuelve un frame goal (3, H, W)."""
        r = self._rng.random()
        if r < self.p_future:
            return self._goal_future(idx)
        elif r < self.p_future + self.p_end:
            return self._goal_end(idx)
        else:
            return self._goal_random()

    def _goal_future(self, idx: int) -> torch.Tensor:
        """Frame aleatorio FUTURO del mismo episodio."""
        meta, local_idx = self.dataset._resolve(idx)

        # Episodio y offset dentro del episodio
        valid_mask    = meta["ep_len"] >= self.dataset._raw_window
        valid_indices = np.where(valid_mask)[0]
        ep_i   = valid_indices[local_idx % len(valid_indices)]
        ep_off = int(meta["ep_offset"][ep_i])
        ep_len = int(meta["ep_len"][ep_i])

        # El último frame de la secuencia actual es ep_off + off + (raw_window - 1).
        # Como no sabemos el offset exacto de este sample (se elige al azar en
        # __getitem__), tomamos un frame en [ep_off + raw_window, ep_off + ep_len - 1]
        # para garantizar que es futuro respecto al inicio de cualquier secuencia.
        earliest_future = ep_off + self.dataset._raw_window
        last_frame      = ep_off + ep_len - 1

        if earliest_future > last_frame:
            # Episodio demasiado corto para tener futuro → fallback a "end"
            return self._goal_end(idx)

        goal_step = int(self._rng.integers(earliest_future, last_frame + 1))
        return self._load_pixel(meta, goal_step)

    def _goal_end(self, idx: int) -> torch.Tensor:
        """Último frame del episodio actual."""
        meta, local_idx = self.dataset._resolve(idx)
        valid_mask    = meta["ep_len"] >= self.dataset._raw_window
        valid_indices = np.where(valid_mask)[0]
        ep_i   = valid_indices[local_idx % len(valid_indices)]
        ep_off = int(meta["ep_offset"][ep_i])
        ep_len = int(meta["ep_len"][ep_i])
        last_step = ep_off + ep_len - 1
        return self._load_pixel(meta, last_step)

    def _goal_random(self) -> torch.Tensor:
        """Frame aleatorio de cualquier archivo/episodio del dataset."""
        if self._total_steps == 0:
            # Dataset vacío → tensor negro
            return torch.zeros(3, 64, 64, dtype=torch.float32)

        flat_step = int(self._rng.integers(0, self._total_steps))

        # Encontrar en qué archivo cae
        file_idx = int(np.searchsorted(self._cum_steps, flat_step, side="right"))
        file_idx = min(file_idx, len(self.dataset._metas) - 1)
        prev     = self._cum_steps[file_idx - 1] if file_idx > 0 else 0
        local_step = flat_step - prev

        meta = self.dataset._metas[file_idx]
        # local_step es un índice dentro de los steps del archivo
        total_steps_in_file = int(meta["ep_len"].sum())
        local_step = local_step % total_steps_in_file    # seguridad por off-by-one

        # Convertir step local a step absoluto dentro del archivo HDF5
        # ep_offset ya da la posición absoluta en el HDF5
        cum_ep_len = np.cumsum(meta["ep_len"])
        ep_i = int(np.searchsorted(cum_ep_len, local_step, side="right"))
        ep_i = min(ep_i, len(meta["ep_len"]) - 1)
        prev_ep = int(cum_ep_len[ep_i - 1]) if ep_i > 0 else 0
        within_ep = local_step - prev_ep
        abs_step = int(meta["ep_offset"][ep_i]) + within_ep

        return self._load_pixel(meta, abs_step)

    # ------------------------------------------------------------------
    # I/O de un solo frame
    # ------------------------------------------------------------------

    def _load_pixel(self, meta: dict, abs_step: int) -> torch.Tensor:
        """
        Carga UN solo frame del HDF5 en abs_step (índice absoluto en el archivo).
        Devuelve tensor (3, H, W) float32 en [0, 1].
        """
        f   = self.dataset._open(meta["path"])
        pix = f["pixels"][abs_step]           # (3, H, W) uint8
        return torch.from_numpy(pix.astype(np.float32)) / 255.0
