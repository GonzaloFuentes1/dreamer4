#!/usr/bin/env python
"""
Convert frame shards from one resolution to another (offline, before training).

Usage:
    python scripts/convert_frames_to_res.py \
        --input  ./data/cycle0-pretrained/frames \
        --output ./data/pretrained-64x64/frames \
        --size   64 \
        --workers 16

Structure expected (and preserved):
    <input>/<task>/<task>_shard<NNNN>.pt  ->  <output>/<task>/<task>_shard<NNNN>.pt

Each shard .pt file must contain {"frames": Tensor(N, 3, H, W)} uint8.
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import torch
import torch.nn.functional as F


def _convert_shard(args):
    src_path, dst_path, size = args
    try:
        td = torch.load(src_path, map_location="cpu", weights_only=False)
        frames = td["frames"]  # (N, 3, H, W) uint8

        if frames.shape[-2] == size and frames.shape[-1] == size:
            # Already the right size — just copy
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"frames": frames}, dst_path)
            return (str(src_path), "skipped (already target size)", frames.shape[0])

        frames_f = frames.to(torch.float32) / 255.0
        frames_r = F.interpolate(frames_f, size=(size, size), mode="area")
        frames_out = (frames_r.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst_path.with_suffix(".pt.tmp")
        torch.save({"frames": frames_out}, tmp)
        os.replace(tmp, dst_path)

        return (str(src_path), "ok", frames_out.shape[0])
    except Exception as e:
        return (str(src_path), f"ERROR: {e}", 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True, help="Source frames dir (contains <task>/*.pt)")
    parser.add_argument("--output",  required=True, help="Destination frames dir")
    parser.add_argument("--size",    type=int, default=64, help="Target resolution (square)")
    parser.add_argument("--workers", type=int, default=8,  help="Parallel workers")
    args = parser.parse_args()

    src_root = Path(args.input)
    dst_root = Path(args.output)
    size = args.size

    if not src_root.exists():
        print(f"[ERROR] Input dir not found: {src_root}")
        sys.exit(1)

    shards = sorted(src_root.rglob("*shard*.pt"))
    if not shards:
        print(f"[ERROR] No shard .pt files found under {src_root}")
        sys.exit(1)

    print(f"Found {len(shards)} shards in {src_root}")
    print(f"Output -> {dst_root}  |  target size: {size}x{size}  |  workers: {args.workers}")

    tasks_args = []
    for src in shards:
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        tasks_args.append((src, dst, size))

    t0 = time.time()
    total_frames = 0
    errors = 0

    with Pool(processes=args.workers) as pool:
        for i, (src, status, n) in enumerate(pool.imap_unordered(_convert_shard, tasks_args), 1):
            total_frames += n
            if "ERROR" in status:
                errors += 1
                print(f"  [{i}/{len(shards)}] FAIL  {src}  —  {status}")
            else:
                if i % 50 == 0 or i == len(shards):
                    elapsed = time.time() - t0
                    rate = i / elapsed
                    eta = (len(shards) - i) / rate if rate > 0 else 0
                    print(f"  [{i}/{len(shards)}] {rate:.1f} shards/s  ETA {eta:.0f}s  frames={total_frames:,}")

    elapsed = time.time() - t0
    print(f"\nDone: {len(shards) - errors}/{len(shards)} shards OK  |  {total_frames:,} frames  |  {elapsed:.1f}s")
    if errors:
        print(f"[WARN] {errors} shards failed — check output above")
        sys.exit(1)


if __name__ == "__main__":
    main()
