#!/usr/bin/env python
"""Quick visual inspection: load 1 shard per task, save a grid of sample frames."""
import torch
from pathlib import Path

DATA_ROOT = Path("runs/active_128x128/dataset/cycle0/frames")
OUT_PATH = Path("runs/active_128x128/debug_frame_grid.png")

tasks = sorted([path.name for path in DATA_ROOT.iterdir() if path.is_dir()])
if not tasks:
    raise RuntimeError(f"No task directories found under {DATA_ROOT}")

all_selected = []
row_labels = []
for task in tasks:
    shard_path = DATA_ROOT / task / f"{task}_shard0000.pt"
    if not shard_path.exists():
        print(f"SKIP {task}: {shard_path} not found")
        continue
    data = torch.load(shard_path, map_location="cpu", weights_only=False)
    frames = data["frames"]  # (N, 3, H, W) uint8
    print(f"{task}: shape={frames.shape}, dtype={frames.dtype}, "
          f"min={frames.min().item()}, max={frames.max().item()}, "
          f"mean={frames.float().mean().item():.1f}")

    # Pick 8 evenly spaced frames
    N = frames.shape[0]
    idxs = torch.linspace(0, N - 1, 8).long()
    selected = frames[idxs].float() / 255.0  # (8, 3, H, W)
    all_selected.append(selected)
    row_labels.append(task)

    # Check for all-black or near-black frames
    per_frame_mean = frames.float().mean(dim=(1, 2, 3))
    n_dark = (per_frame_mean < 10).sum().item()
    n_bright = (per_frame_mean > 245).sum().item()
    print(f"  dark frames (mean<10): {n_dark}/{N}, bright frames (mean>245): {n_bright}/{N}")

# Build grid manually and save as PPM (no numpy needed)
if not all_selected:
    raise RuntimeError(f"No readable shards found under {DATA_ROOT}")

OUT_PPM = OUT_PATH.with_suffix(".ppm")
rows_tensor = []
for sel in all_selected:
    row = torch.cat([sel[i] for i in range(sel.shape[0])], dim=2)  # (3, H, 8*W)
    rows_tensor.append(row)
grid = torch.cat(rows_tensor, dim=1)  # (3, n_tasks*H, 8*W)
grid_uint8 = (grid.clamp(0, 1) * 255).to(torch.uint8)
C, H_total, W_total = grid_uint8.shape
hwc = grid_uint8.permute(1, 2, 0).contiguous().flatten()
raw_bytes = bytes(hwc.tolist())
with open(str(OUT_PPM), "wb") as f:
    f.write(f"P6\n{W_total} {H_total}\n255\n".encode())
    f.write(raw_bytes)
print(f"\nSaved grid to {OUT_PPM} ({len(row_labels)} rows x 8 cols, one row per task)")
print("Row order:", ", ".join(row_labels))
