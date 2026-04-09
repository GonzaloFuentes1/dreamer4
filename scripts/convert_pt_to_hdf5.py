"""
convert_pt_to_hdf5.py — Convierte datos legacy .pt → HDF5 (.h5)

Lee los shards de `frames/<task>/*.pt` (pixels uint8) y el archivo
`demos/<task>.pt` (action + reward + episode) y los combina en un
único `<out>/<task>.h5` compatible con HDF5EpisodeDataset.

Uso:
    python scripts/convert_pt_to_hdf5.py --data data/pretrained-64x64
    python scripts/convert_pt_to_hdf5.py --data data/pretrained-64x64 --tasks walker-walk cheetah-run
    python scripts/convert_pt_to_hdf5.py --data data/pretrained-64x64 --out data/pretrained-64x64-h5
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from hdf5_episode_dataset import HDF5EpisodeWriter


def load_frames(frames_dir: Path, task: str) -> np.ndarray:
    """Carga y concatena todos los shards de frames en orden."""
    task_dir = frames_dir / task
    shards = sorted(task_dir.glob("*.pt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not shards:
        raise FileNotFoundError(f"No se encontraron shards en {task_dir}")

    all_frames = []
    for shard in shards:
        data = torch.load(shard, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            frames = data["frames"]
        else:
            frames = data
        all_frames.append(frames.numpy())

    return np.concatenate(all_frames, axis=0)  # (N, 3, H, W) uint8


def load_demos(demos_dir: Path, task: str) -> dict:
    """Carga action, reward y episode ids desde el .pt de demos."""
    path = demos_dir / f"{task}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    return {
        "action":  data["action"],   # (N, A) float32
        "reward":  data["reward"],   # (N,)   float32
        "episode": data["episode"],  # (N,)   int64
    }


def convert_task(task: str, data_root: Path, out_dir: Path, overwrite: bool) -> bool:
    out_path = out_dir / f"{task}.h5"
    if out_path.exists() and not overwrite:
        print(f"  [SKIP] {task}.h5 ya existe (usa --overwrite para reescribir)")
        return False

    frames_dir = data_root / "frames"
    demos_dir  = data_root / "demos"

    print(f"  Cargando frames de {frames_dir / task}...")
    t0 = time.time()
    pixels = load_frames(frames_dir, task)           # (N, 3, H, W) uint8
    demos  = load_demos(demos_dir, task)
    t_load = time.time() - t0

    N_frames = pixels.shape[0]
    N_demos  = demos["action"].shape[0]

    if N_frames != N_demos:
        print(f"  [ERROR] Mismatch: frames={N_frames}, demos={N_demos} — saltando {task}")
        return False

    H, W  = pixels.shape[2], pixels.shape[3]
    A_dim = demos["action"].shape[1]
    print(f"  {N_frames:,} steps · res={H}x{W} · A={A_dim} · cargado en {t_load:.1f}s")

    print(f"  Escribiendo {out_path.name}...")
    t1 = time.time()

    # Convertir episode ids a int64 numpy para iterar episodios
    ep_ids = demos["episode"].numpy().astype(np.int64)
    unique_eps = np.unique(ep_ids)

    with HDF5EpisodeWriter(
        path=out_path,
        img_size=H,
        action_dim=A_dim,
    ) as writer:
        # Escribir de a episodio para que writer compute ep_len/ep_offset
        for ep_id in unique_eps:
            mask = ep_ids == ep_id
            ep_pixels  = torch.from_numpy(pixels[mask])      # (L, 3, H, W) uint8
            ep_action  = demos["action"][mask]
            ep_reward  = demos["reward"][mask]
            ep_ids_t   = demos["episode"][mask]

            writer.append_batch(ep_pixels, ep_action, ep_reward, ep_ids_t)

    t_write = time.time() - t1
    size_mb = out_path.stat().st_size / 1e6
    print(f"  OK → {out_path.name}  ({size_mb:.0f} MB, escrito en {t_write:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convierte datos .pt a HDF5")
    parser.add_argument("--data",      required=True, help="Directorio raíz: debe tener frames/ y demos/")
    parser.add_argument("--out",       default=None,  help="Directorio de salida (default: mismo que --data)")
    parser.add_argument("--tasks",     nargs="*",     help="Tasks a convertir (default: todas)")
    parser.add_argument("--overwrite", action="store_true", help="Sobreescribir .h5 existentes")
    args = parser.parse_args()

    data_root = Path(args.data)
    out_dir   = Path(args.out) if args.out else data_root
    out_dir.mkdir(parents=True, exist_ok=True)

    # Descubrir tasks disponibles
    frames_dir = data_root / "frames"
    demos_dir  = data_root / "demos"
    if not frames_dir.exists() or not demos_dir.exists():
        print(f"ERROR: {data_root} debe tener subcarpetas frames/ y demos/", file=sys.stderr)
        sys.exit(1)

    all_tasks = sorted([d.name for d in frames_dir.iterdir() if d.is_dir()])
    if args.tasks:
        tasks = [t for t in args.tasks if t in all_tasks]
        unknown = [t for t in args.tasks if t not in all_tasks]
        if unknown:
            print(f"WARN: tasks no encontradas en frames/: {unknown}")
    else:
        tasks = all_tasks

    print(f"Convirtiendo {len(tasks)} tasks de {data_root} → {out_dir}")
    print(f"Tasks: {tasks}\n")

    ok, skip, err = 0, 0, 0
    t_total = time.time()
    for task in tasks:
        print(f"[{tasks.index(task)+1}/{len(tasks)}] {task}")
        try:
            result = convert_task(task, data_root, out_dir, args.overwrite)
            if result:
                ok += 1
            else:
                skip += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            err += 1
        print()

    elapsed = time.time() - t_total
    print(f"{'='*50}")
    print(f"Completado en {elapsed:.0f}s — OK={ok}  SKIP={skip}  ERR={err}")
    print(f"Archivos .h5 en: {out_dir}")


if __name__ == "__main__":
    main()
