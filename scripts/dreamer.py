#!/usr/bin/env python3
"""
scripts/dreamer.py — CLI de inspección para dreamer4.

Uso:
  python scripts/dreamer.py datasets [--root ./data]
  python scripts/dreamer.py runs     [--root ./runs]
  python scripts/dreamer.py inspect  <path_al_h5_o_demo_dir> [--tasks walker-walk,...]

Ejemplos:
  python scripts/dreamer.py datasets
  python scripts/dreamer.py runs
  python scripts/dreamer.py inspect data/pretrained-64x64
  python scripts/dreamer.py inspect data/pretrained-64x64/walker-walk.h5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def _fmt_num(n: int) -> str:
    return f"{n:,}"


CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
BOLD  = "\033[1m"
RESET = "\033[0m"


def _h1(s: str): print(f"\n{BOLD}{CYAN}{'═'*50}{RESET}")
def _header(s: str): print(f"{BOLD}{CYAN}  {s}{RESET}")
def _sep(): print(f"{CYAN}{'─'*50}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Comando: datasets
# ─────────────────────────────────────────────────────────────────────────────

def cmd_datasets(args):
    """Lista todos los datasets (.h5 y .pt) bajo --root."""
    root = Path(args.root)
    if not root.exists():
        print(f"{RED}Error: directorio no encontrado: {root}{RESET}")
        sys.exit(1)

    h5_files  = sorted(root.rglob("*.h5"))
    pt_files  = sorted(root.rglob("*.pt"))
    meta_files= sorted(root.rglob("*_metadata.json"))

    _h1("Datasets")
    _header(f"Datasets en {root}")
    _sep()

    if not h5_files and not pt_files:
        print(f"  {YELLOW}No se encontraron datasets.{RESET}")
        return

    # HDF5
    if h5_files:
        print(f"\n{BOLD}  HDF5 ({len(h5_files)} archivos):{RESET}")
        for p in h5_files:
            size = _fmt_size(p.stat().st_size)
            try:
                import h5py
                with h5py.File(str(p), "r", swmr=True) as f:
                    n  = int(f["pixels"].shape[0]) if "pixels" in f else "?"
                    ne = len(f["ep_len"]) if "ep_len" in f else "?"
                    h, w = (f["pixels"].shape[2], f["pixels"].shape[3]) if "pixels" in f else ("?","?")
                print(f"    {GREEN}✓{RESET} {p.relative_to(root)}")
                print(f"        {_fmt_num(n) if isinstance(n,int) else n} steps · "
                      f"{ne} episodios · {h}×{w} · {size}")
            except Exception as e:
                print(f"    {YELLOW}?{RESET} {p.relative_to(root)} ({size}) — {e}")

    # PT shards (agrupar por tarea)
    if pt_files:
        tasks: dict = {}
        for p in pt_files:
            key = p.parent
            tasks.setdefault(str(key), []).append(p)

        print(f"\n{BOLD}  .pt shards ({len(pt_files)} archivos en {len(tasks)} dirs):{RESET}")
        for dir_str, files in sorted(tasks.items()):
            dir_path = Path(dir_str)
            total_size = sum(f.stat().st_size for f in files)
            # Leer metadata si existe
            meta = None
            for mf in meta_files:
                if mf.parent == dir_path or mf.parent == dir_path.parent:
                    try:
                        meta = json.loads(mf.read_text())
                        break
                    except Exception:
                        pass
            print(f"    {GREEN}✓{RESET} {Path(dir_str).relative_to(root)} "
                  f"({len(files)} shards · {_fmt_size(total_size)})")
            if meta:
                ep  = meta.get("episodes_collected", "?")
                fr  = meta.get("total_frames", "?")
                pol = meta.get("policy_used", "?")
                print(f"        {ep} episodios · {_fmt_num(fr) if isinstance(fr,int) else fr} frames · policy={pol}")

    _sep()
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Comando: runs
# ─────────────────────────────────────────────────────────────────────────────

def cmd_runs(args):
    """Lista todos los runs bajo --root (default ./runs)."""
    root = Path(args.root)
    if not root.exists():
        print(f"{YELLOW}No hay directorio runs en {root}{RESET}")
        return

    run_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not run_dirs:
        print(f"{YELLOW}No hay runs en {root}{RESET}")
        return

    _h1("Runs")
    _header(f"Runs en {root}  ({len(run_dirs)} total)")
    _sep()

    for run_dir in run_dirs:
        phases = {}
        for phase in ("tokenizer", "dynamics", "finetune", "agent"):
            ckpt_dir = run_dir / phase
            if ckpt_dir.exists():
                ckpts = sorted(ckpt_dir.glob("*.ckpt"))
                last  = run_dir / phase / "last.ckpt"
                phases[phase] = {
                    "n_ckpts": len(ckpts),
                    "has_last": last.exists(),
                    "size": _fmt_size(sum(c.stat().st_size for c in ckpts)),
                }

        cycles_dir = run_dir / "cycles"
        n_cycles   = len(list(cycles_dir.glob("*.ckpt"))) if cycles_dir.exists() else 0

        print(f"\n  {BOLD}{run_dir.name}{RESET}")
        for phase, info in phases.items():
            tick = GREEN + "✓" + RESET if info["has_last"] else YELLOW + "…" + RESET
            print(f"    {tick} {phase:<12} {info['n_ckpts']} ckpts · {info['size']}")
        if n_cycles:
            print(f"    {GREEN}✓{RESET} cycles       {n_cycles} snapshots")
        if not phases:
            print(f"    {YELLOW}(vacío){RESET}")

    _sep()
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Comando: inspect
# ─────────────────────────────────────────────────────────────────────────────

def cmd_inspect(args):
    """Inspecciona un archivo .h5 o un directorio de demos."""
    path = Path(args.path)

    # Determinar archivos a inspeccionar
    if path.is_file() and path.suffix == ".h5":
        h5_files = [path]
    elif path.is_dir():
        tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
        if tasks:
            h5_files = [path / f"{t}.h5" for t in tasks if (path / f"{t}.h5").exists()]
        else:
            h5_files = sorted(path.rglob("*.h5"))
    else:
        print(f"{RED}Error: {path} no es un .h5 ni un directorio.{RESET}")
        sys.exit(1)

    if not h5_files:
        print(f"{YELLOW}No se encontraron archivos .h5 en {path}{RESET}")
        sys.exit(0)

    try:
        import h5py
        import numpy as np
    except ImportError:
        print(f"{RED}h5py no instalado. Ejecuta: pip install h5py{RESET}")
        sys.exit(1)

    _h1("Inspect")
    _header(f"Inspeccionando {path}")

    for h5_path in h5_files:
        _sep()
        print(f"\n  {BOLD}{h5_path.name}{RESET}  ({_fmt_size(h5_path.stat().st_size)})")

        try:
            with h5py.File(str(h5_path), "r", swmr=True) as f:
                keys = list(f.keys())

                if "pixels" in f:
                    pix = f["pixels"]
                    N, C, H, W = pix.shape
                    print(f"    pixels:   {_fmt_num(N)} steps · {C}×{H}×{W} · dtype={pix.dtype}")

                if "action" in f:
                    act = f["action"]
                    print(f"    action:   shape={act.shape} · dtype={act.dtype}")

                if "reward" in f:
                    rew = np.array(f["reward"])
                    print(f"    reward:   mean={rew.mean():.4f} · std={rew.std():.4f} "
                          f"· min={rew.min():.4f} · max={rew.max():.4f}")

                if "ep_len" in f:
                    ep_len = np.array(f["ep_len"])
                    print(f"    episodios: {len(ep_len):,} · "
                          f"len mean={ep_len.mean():.1f} · "
                          f"min={ep_len.min()} · max={ep_len.max()}")

                # Atributos globales
                if f.attrs:
                    attrs = dict(f.attrs)
                    print(f"    attrs:    {attrs}")

        except Exception as e:
            print(f"    {RED}Error al leer: {e}{RESET}")

    _sep()
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="dreamer",
        description="CLI de inspección para dreamer4",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # datasets
    p_ds = sub.add_parser("datasets", help="Lista datasets disponibles")
    p_ds.add_argument("--root", default="./data", help="Directorio raíz de datos")

    # runs
    p_runs = sub.add_parser("runs", help="Lista runs de entrenamiento")
    p_runs.add_argument("--root", default="./runs", help="Directorio raíz de runs")

    # inspect
    p_ins = sub.add_parser("inspect", help="Inspecciona un dataset .h5 o directorio")
    p_ins.add_argument("path", help="Ruta a un .h5 o a un directorio de demos")
    p_ins.add_argument("--tasks", default=None,
                       help="Tareas a inspeccionar (comma-separated, ej: walker-walk,cheetah-run)")

    args = parser.parse_args()
    {"datasets": cmd_datasets, "runs": cmd_runs, "inspect": cmd_inspect}[args.cmd](args)


if __name__ == "__main__":
    main()
