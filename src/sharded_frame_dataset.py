# sharded_frame_dataset.py
import os
import bisect
import random
from pathlib import Path
from typing import Sequence, List, Dict, Union

import torch
from torch.utils.data import Dataset


class ShardedFrameDataset(Dataset):
    """
    Samples contiguous sequences from preprocessed shards across multiple roots:

      root/<task>/*.pt  with {"frames": (N, 3, H, W) uint8}

    Returns: (T, 3, H, W) float32 in [0,1], where T = seq_len.

    If iid_sampling=True, ignores idx and samples a random starting position
    uniformly over all valid sequence starts across all shards.

"""

    def __init__(
        self,
        outdirs: Union[str, Sequence[str]],
        tasks: Sequence[str] = (),
        seq_len: int = 16,
        iid_sampling: bool = True,
    ):
        super().__init__()
        assert outdirs is not None, "outdirs must be specified"

        if isinstance(outdirs, (str, Path)):
            self.outdirs = [str(outdirs)]
        else:
            self.outdirs = [str(p) for p in outdirs]

        self.tasks = list(tasks)
        self.seq_len = int(seq_len)
        self.iid_sampling = bool(iid_sampling)

        self.shards: List[Dict] = []
        self.cum_starts: List[int] = []
        total_starts = 0

        for root in self.outdirs:
            root = Path(root)
            for task in self.tasks:
                task_dir = root / task
                if not task_dir.exists():
                    continue

                for fname in sorted(os.listdir(task_dir)):
                    if not fname.endswith(".pt"):
                        continue
                    path = task_dir / fname

                    try:
                        td = torch.load(path, map_location="cpu", weights_only=False)
                    except Exception as e:
                        print(f"[ShardedFrameDataset] Skipping shard {path} (load error): {e}")
                        continue

                    frames = td.get("frames", None)
                    if not isinstance(frames, torch.Tensor):
                        print(f"[ShardedFrameDataset] Skipping shard {path} (no 'frames' tensor)")
                        continue
                    if frames.ndim != 4 or frames.shape[1] != 3:
                        print(f"[ShardedFrameDataset] Skipping shard {path} (unexpected shape {frames.shape})")
                        continue

                    N = int(frames.shape[0])
                    if N < self.seq_len:
                        print(f"[ShardedFrameDataset] Skipping shard {path} (N={N} < seq_len={self.seq_len})")
                        continue

                    num_starts = N - self.seq_len + 1
                    self.shards.append(
                        {"path": str(path), "num_frames": N, "num_starts": num_starts}
                    )
                    total_starts += num_starts
                    self.cum_starts.append(total_starts)

        self.total_starts = total_starts
        if self.total_starts == 0:
            print("[ShardedFrameDataset] WARNING: no usable sequences found in outdirs")
        else:
            print(
                f"[ShardedFrameDataset] roots={len(self.outdirs)}, "
                f"shards={len(self.shards):,}, seq_starts={self.total_starts:,}"
            )

        # LRU shard cache — avoids repeated torch.load() with multi-task data
        self._cache_max = min(8, max(1, len(self.shards)))
        self._cache: Dict[str, torch.Tensor] = {}
        self._cache_order: List[str] = []

    def __len__(self) -> int:
        return self.total_starts

    def _load_shard(self, path: str) -> torch.Tensor:
        if path in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(path)
            self._cache_order.append(path)
            return self._cache[path]
        td = torch.load(path, map_location="cpu", weights_only=False)
        frames = td["frames"]
        # Evict oldest if full
        if len(self._cache) >= self._cache_max:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        self._cache[path] = frames
        self._cache_order.append(path)
        return frames

    def _map_global_start_to_shard(self, global_start: int) -> tuple[int, int]:
        # global_start in [0, total_starts)
        shard_idx = bisect.bisect_right(self.cum_starts, global_start)
        prev_cum = 0 if shard_idx == 0 else self.cum_starts[shard_idx - 1]
        start_idx_in_shard = global_start - prev_cum
        return shard_idx, start_idx_in_shard

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.total_starts == 0:
            raise IndexError("Empty dataset")

        if self.iid_sampling:
            global_start = random.randint(0, self.total_starts - 1)
        else:
            if idx < 0 or idx >= self.total_starts:
                raise IndexError(idx)
            global_start = int(idx)

        shard_idx, start = self._map_global_start_to_shard(global_start)

        meta = self.shards[shard_idx]
        frames = self._load_shard(meta["path"])  # (N, 3, H, W)

        end = start + self.seq_len
        seq_u8 = frames[start:end]  # (T, 3, H, W), guaranteed valid by construction
        return seq_u8.to(torch.float32) / 255.0
