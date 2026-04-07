#!/usr/bin/env python
# download_pretrain_data.py — Download pretrain data from HuggingFace.
#
# Downloads nicklashansen/dreamer4 dataset (mixed-large or other split):
#   - PNG files: each PNG = multiple episodes stacked horizontally at 224×224
#     → split into individual frames, resize to 128×128, save as sharded .pt
#   - .pt file: episode/action/reward demo data (used by dynamics/finetune/agent)
#
# No DMControl rendering needed — pixel frames come directly from HuggingFace.
#
# Usage:
#   python download_pretrain_data.py
#   python download_pretrain_data.py "download.tasks=[walker-stand,cheetah-run]"
#   python download_pretrain_data.py download.hf_split=expert

import sys
import os
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf


# ─────────────────────────────────────────────────────────────────────────────
# Per-task worker (module-level so it's picklable with spawn)
# ─────────────────────────────────────────────────────────────────────────────

def _download_worker(args: tuple) -> tuple:
    task, dc_dict, src_dir = args
    sys.path.insert(0, src_dir)

    import torch
    import torch.nn.functional as F
    from torchvision.io import read_image
    from pathlib import Path
    from omegaconf import OmegaConf
    from huggingface_hub import hf_hub_download, list_repo_files

    dc = OmegaConf.create(dc_dict)
    out_demo   = Path(dc.out_data_dir)   / f"{task}.pt"
    out_frames = Path(dc.out_frames_dir) / task

    if not dc.overwrite and out_demo.exists() and any(out_frames.glob(f"{task}_shard*.pt")):
        print(f"[download/{task}] Already exists — skipping.")
        return task, False

    target_size = int(dc.img_size)    # 128
    shard_size  = int(dc.shard_size)  # 2048
    frame_size  = int(dc.get("hf_frame_size", 224))  # PNG frame size in HF

    # ── 1. Download demo .pt (actions / rewards / episodes) ──────────────────
    hf_demo = f"{dc.hf_split}/{task}.pt"
    print(f"[download/{task}] Downloading demo: {hf_demo}")
    try:
        demo_path = hf_hub_download(repo_id=str(dc.hf_repo), filename=hf_demo, repo_type="dataset")
    except Exception as e:
        print(f"[download/{task}] Demo not found — skipping: {e}")
        return task, False

    hf_data = torch.load(demo_path, map_location="cpu", weights_only=False)
    demo_data = {
        "episode": hf_data["episode"].to(torch.int64),
        "action":  hf_data["action"].to(torch.float32),
        "reward":  hf_data["reward"].to(torch.float32),
    }
    n_demo_steps = len(demo_data["episode"])

    # ── 2. Discover available PNG files ──────────────────────────────────────
    all_hf_files = set(list_repo_files(str(dc.hf_repo), repo_type="dataset"))
    png_files = []
    i = 0
    while f"{dc.hf_split}/{task}-{i}.png" in all_hf_files:
        png_files.append(f"{dc.hf_split}/{task}-{i}.png")
        i += 1

    if not png_files:
        print(f"[download/{task}] No PNG files found in {dc.hf_split}/ — skipping.")
        return task, False

    print(f"[download/{task}] Found {len(png_files)} PNGs — downloading + processing frames ...")

    # ── 3. Process PNGs → frame shards ───────────────────────────────────────
    out_frames.mkdir(parents=True, exist_ok=True)
    out_demo.parent.mkdir(parents=True, exist_ok=True)

    shard_buffer = []   # list of (N_i, 3, 128, 128) uint8 tensors
    shard_idx    = 0
    total_frames = 0

    def _flush_shard(buf, idx):
        concat = torch.cat(buf, dim=0)
        to_save   = concat[:shard_size]
        remainder = concat[shard_size:] if concat.shape[0] > shard_size else None
        out_path  = out_frames / f"{task}_shard{idx:04d}.pt"
        tmp_path  = out_path.with_suffix(".pt.tmp")
        torch.save({"frames": to_save}, tmp_path)
        tmp_path.rename(out_path)
        return remainder, idx + 1

    for png_hf in png_files:
        try:
            png_path = hf_hub_download(repo_id=str(dc.hf_repo), filename=png_hf, repo_type="dataset")
        except Exception as e:
            print(f"[download/{task}] Failed downloading {png_hf}: {e}")
            continue

        try:
            img = read_image(png_path)   # (3, H, H*N_frames) uint8
        except Exception as e:
            print(f"[download/{task}] Failed reading {png_hf}: {e}")
            continue

        C, H, W_total = img.shape
        if H != frame_size or W_total % frame_size != 0:
            print(f"[download/{task}] Unexpected PNG shape {img.shape} in {png_hf} — skipping.")
            continue

        n_frames = W_total // frame_size
        # Split: (3, H, H*N) → (N, 3, H, H)
        frames = img.view(C, frame_size, n_frames, frame_size)  # (3, H, N, H)
        frames = frames.permute(2, 0, 1, 3).contiguous()        # (N, 3, H, H)

        # Resize to target_size if needed
        if frame_size != target_size:
            frames_f = frames.to(torch.float32) / 255.0
            frames_f = F.interpolate(frames_f, size=(target_size, target_size),
                                     mode="bilinear", align_corners=False)
            frames = (frames_f.clamp(0, 1) * 255).to(torch.uint8)

        shard_buffer.append(frames)
        total_frames += n_frames

        # Flush complete shards
        while sum(f.shape[0] for f in shard_buffer) >= shard_size:
            remainder, shard_idx = _flush_shard(shard_buffer, shard_idx)
            shard_buffer = [remainder] if remainder is not None and remainder.shape[0] > 0 else []

    # Flush remaining frames
    if shard_buffer and sum(f.shape[0] for f in shard_buffer) > 0:
        concat   = torch.cat(shard_buffer, dim=0)
        out_path = out_frames / f"{task}_shard{shard_idx:04d}.pt"
        tmp_path = out_path.with_suffix(".pt.tmp")
        torch.save({"frames": concat}, tmp_path)
        tmp_path.rename(out_path)
        shard_idx += 1

    if total_frames == 0:
        print(f"[download/{task}] No frames processed — skipping.")
        return task, False

    if total_frames != n_demo_steps:
        print(f"[download/{task}] WARNING: frame count ({total_frames}) != demo steps ({n_demo_steps}). "
              f"Trimming demo to {total_frames} steps.")
        for k in demo_data:
            demo_data[k] = demo_data[k][:total_frames]

    # ── 4. Save demo .pt ─────────────────────────────────────────────────────
    tmp = out_demo.with_suffix(".pt.tmp")
    torch.save(demo_data, tmp)
    tmp.rename(out_demo)

    print(f"[download/{task}] Done — {total_frames} frames in {shard_idx} shards, "
          f"{demo_data['episode'].unique().numel()} episodes → {out_demo}")
    return task, True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(config_path="configs", config_name="download_pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    import random, numpy as np
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dc = cfg.download

    # ── Resolve task list ────────────────────────────────────────────────────
    if dc.tasks is None:
        from huggingface_hub import list_repo_files
        hf_files = set(list_repo_files(str(dc.hf_repo), repo_type="dataset"))
        prefix = f"{dc.hf_split}/"
        tasks = [f[len(prefix):-3] for f in hf_files if f.startswith(prefix) and f.endswith(".pt")]
        print(f"[download] Auto-detected {len(tasks)} tasks in {dc.hf_repo}/{dc.hf_split}")
    else:
        tasks = list(OmegaConf.to_container(dc.tasks, resolve=True))

    print(f"[download] hf_split={dc.hf_split}  tasks={len(tasks)}")
    print(f"[download] out_data_dir  → {dc.out_data_dir}")
    print(f"[download] out_frames_dir→ {dc.out_frames_dir}")

    n_workers = min(int(dc.get("num_workers", 1)), len(tasks))
    src_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    dc_dict   = dict(OmegaConf.to_container(dc, resolve=True))
    dc_dict["out_data_dir"]   = os.path.abspath(str(dc.out_data_dir))
    dc_dict["out_frames_dir"] = os.path.abspath(str(dc.out_frames_dir))

    n_ok = 0; n_skip = 0

    if n_workers <= 1:
        for task in tasks:
            _, ok = _download_worker((task, dc_dict, src_dir))
            if ok: n_ok += 1
            else:  n_skip += 1
    else:
        print(f"[download] Parallel mode: {n_workers} workers for {len(tasks)} tasks")
        worker_args = [(task, dc_dict, src_dir) for task in tasks]
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futs = {executor.submit(_download_worker, a): a[0] for a in worker_args}
            for fut in as_completed(futs):
                task_name, ok = fut.result()
                if ok: n_ok += 1
                else:  n_skip += 1

    print(f"\n[download] Done. downloaded={n_ok}  skipped={n_skip}")
    print(f"\nNext — train con pretrain data:")
    print(f"  python scripts/pipeline/train_phase1a_tokenizer.py \\")
    print(f'    "data.frame_dirs=[{dc.out_frames_dir}]"')
    print(f"  python scripts/pipeline/train_phase1b_dynamics.py \\")
    print(f'    "data.data_dirs=[{dc.out_data_dir}]" \\')
    print(f'    "data.frame_dirs=[{dc.out_frames_dir}]"')


if __name__ == "__main__":
    main()
