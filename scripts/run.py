#!/usr/bin/env python
# scripts/run.py — Coordinator del pipeline completo de Dreamer 4.
#
# Reemplaza los scripts run_cycles*.sh con un coordinador Python que:
#   - Lee configs/run.yaml para los parámetros generales del entrenamiento
#   - Invoca cada fase como subprocess (compatible con SLURM/DDP)
#   - Acumula directorios de datos a lo largo de los ciclos
#   - Detecta checkpoints existentes y retoma automáticamente
#   - Calcula steps con esquema cold/warm por ciclo
#
# Uso:
#   python scripts/run.py
#   python scripts/run.py run.tag=experimento1 run.cycles=5
#   python scripts/run.py model.tokenizer=discrete_base_64x64 run.res=64
#   python scripts/run.py "data.data_dirs=[./data/pretrained-64x64]" run.tag=pt_active
#
# En SLURM (single-node, multi-GPU):
#   srun --nodes=1 --ntasks=1 --gpus=2 --cpus-per-task=64 --mem=128G --time=24:00:00 \\
#     python scripts/run.py run.tag=mi_run
#
#   SLURM setea CUDA_VISIBLE_DEVICES automáticamente según --gpus.
#   El coordinador detecta los GPUs disponibles y se los pasa a cada fase.
#   Si querés forzar la cantidad: trainer.devices=2
#
#   Con 4 GPUs:
#   srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=96 --mem=256G --time=72:00:00 \\
#     python scripts/run.py run.tag=pt_active

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ── Repo root y src en path ───────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# ── Cargar .env antes de cualquier import que use env vars ───────────────────
_env_file = REPO_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#"):
            continue
        if _line.startswith("export "):
            _line = _line[len("export "):]
        if "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Entorno headless (MuJoCo / EGL) ─────────────────────────────────────────
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault(
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
)

import torch
from omegaconf import OmegaConf, DictConfig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def detect_devices(cfg_devices) -> int:
    """Devuelve el número de GPUs a usar.

    Prioridad:
      1. trainer.devices del YAML / CLI (override explícito)
      2. CUDA_VISIBLE_DEVICES — seteado por SLURM al asignar GPUs al job
      3. SLURM_GPUS_ON_NODE / SLURM_GPUS_PER_NODE
      4. SLURM_GPUS  (puede venir como "2" o "gpu:2")
      5. torch.cuda.device_count()

    Con `srun --ntasks=1 --gpus=2`, SLURM setea CUDA_VISIBLE_DEVICES=0,1
    y los subprocesos de entrenamiento lo heredan automáticamente.
    """
    if cfg_devices is not None:
        n = int(cfg_devices)
        print(f"[detect_devices] trainer.devices override → {n}", flush=True)
        return n

    # CUDA_VISIBLE_DEVICES es lo más confiable — lo escribe SLURM al pedir --gpus
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_vis and cuda_vis.strip() not in ("", "NoDevFiles"):
        gpus = [d.strip() for d in cuda_vis.split(",") if d.strip()]
        n = len(gpus)
        print(f"[detect_devices] CUDA_VISIBLE_DEVICES={cuda_vis} → {n} GPU(s)", flush=True)
        return n

    # SLURM_GPUS_ON_NODE / SLURM_GPUS_PER_NODE  (ej. "2" o "4")
    for env_var in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE"):
        val = os.environ.get(env_var, "")
        if val:
            try:
                n = int(val.split(":")[-1].split("(")[0])
                print(f"[detect_devices] {env_var}={val} → {n} GPU(s)", flush=True)
                return n
            except (ValueError, IndexError):
                pass

    # SLURM_GPUS puede venir como "2", "gpu:2", "gpu:2(IDX:0,1)"
    slurm_gpus = os.environ.get("SLURM_GPUS", "")
    if slurm_gpus:
        try:
            raw = slurm_gpus.split(":")[-1].split("(")[0]
            n = int(raw)
            print(f"[detect_devices] SLURM_GPUS={slurm_gpus} → {n} GPU(s)", flush=True)
            return n
        except (ValueError, IndexError):
            pass

    n = torch.cuda.device_count()
    n = n if n > 0 else 1
    print(f"[detect_devices] torch.cuda.device_count() → {n} GPU(s)", flush=True)
    return n


def phase_steps(cold: int, warm: int, ckpt: Optional[Path], cycle: int) -> tuple[int, list[str]]:
    """Calcula steps y resume override para una fase."""
    if ckpt is not None and ckpt.exists():
        overrides = [f"resume={ckpt}"]
        cur_steps = cold if cycle == 0 else cold + cycle * warm
    else:
        overrides = []
        cur_steps = cold
    return cur_steps, overrides


def run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> None:
    """Ejecuta un comando en subprocess, falla si exit code != 0."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n[run.py] $ {cmd_str}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(cwd))


def tasks_str(tasks) -> str:
    return ",".join(tasks)


def join_dirs(dirs: list[str]) -> str:
    return "[" + ",".join(dirs) + "]"


def header(title: str, cycle: int, total: int) -> None:
    bar = "═" * 44
    print(f"\n{bar}\n  Ciclo {cycle}/{total - 1}  ·  {title}\n{bar}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Fases
# ─────────────────────────────────────────────────────────────────────────────

def phase_collect(cfg: DictConfig, cycle: int, run_root: Path,
                  agent_ckpt: Path, data_root: Path) -> Path:
    """Ejecuta Phase 0: colecta episodios y escribe <task>.h5 en out_data."""
    out_data   = data_root / f"cycle{cycle}"
    out_videos = data_root / f"cycle{cycle}" / "videos"
    header("Phase 0 · Collect", cycle, cfg.run.cycles)
    run([
        sys.executable, "scripts/pipeline/launch_phase0_dist.py",
        "collect.policy=agent",
        f"collect.num_envs_per_task={cfg.collect.num_envs}",
        f"collect.agent_ckpt={agent_ckpt}",
        f"collect.n_episodes_per_task={cfg.collect.n_episodes_per_task}",
        f"++collect.episode_len={cfg.collect.episode_len}",
        f"collect.frame_skip={cfg.collect.frame_skip}",
        f"collect.img_size={cfg.run.res}",
        f"collect.out_data_dir={out_data}",
        f"collect.out_videos_dir={out_videos}",
        f"++collect.save_preview_video={str(cfg.collect.save_video).lower()}",
        "++collect.preview_video_backend=torchrl",
        f"++collect.preview_video_fps={cfg.collect.video_fps}",
        f"++collect.preview_video_max_frames={cfg.collect.episode_len}",
        f"collect.tasks=[{tasks_str(cfg.data.tasks)}]",
    ])
    return out_data


def phase_tokenizer(cfg: DictConfig, cycle: int, run_root: Path,
                    data_dirs: list[str], devices: int) -> Path:
    ckpt_dir = run_root / "tokenizer"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / "last.ckpt"
    if cycle == 0 and last_ckpt.exists():
        print(f"[Phase 1a] Checkpoint existente — saltando ({last_ckpt})")
        return last_ckpt
    header("Phase 1a · Tokenizer", cycle, cfg.run.cycles)
    cur_steps, resume = phase_steps(
        cfg.steps.tokenizer_cold, cfg.steps.tokenizer_warm,
        last_ckpt if last_ckpt.exists() else None, cycle,
    )
    run([
        sys.executable, "scripts/pipeline/train_phase1a_tokenizer.py",
        f"tokenizer={cfg.model.tokenizer}",
        f"data.data_dirs={join_dirs(data_dirs)}",
        f"data.img_size={cfg.run.res}",
        f"trainer.devices={devices}",
        f"trainer.max_steps={cur_steps}",
        f"data.batch_size_tokenizer={cfg.batch.tokenizer}",
        f"data.num_workers={cfg.data.num_workers}",
        f"checkpoint.dirpath={ckpt_dir}",
        f"checkpoint.every_n_train_steps={cfg.checkpoint.every_tokenizer}",
        f"checkpoint.save_top_k={cfg.checkpoint.save_top_k}",
        f"wandb.project={cfg.wandb.project}",
        f"wandb.name=c{cycle}_tok_{cfg.run.tag}",
        *([f"wandb.entity={cfg.wandb.entity}"] if cfg.wandb.entity else []),
        f"data.tasks=[{tasks_str(cfg.data.tasks)}]",
        *resume,
    ])
    return last_ckpt


def phase_dynamics(cfg: DictConfig, cycle: int, run_root: Path,
                   data_dirs: list[str],
                   tok_ckpt: Path, devices: int) -> Path:
    ckpt_dir = run_root / "dynamics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / "last.ckpt"
    if cycle == 0 and last_ckpt.exists():
        print(f"[Phase 1b] Checkpoint existente — saltando ({last_ckpt})")
        return last_ckpt
    header("Phase 1b · Dynamics", cycle, cfg.run.cycles)
    cur_steps, resume = phase_steps(
        cfg.steps.dynamics_cold, cfg.steps.dynamics_warm,
        last_ckpt if last_ckpt.exists() else None, cycle,
    )
    run([
        sys.executable, "scripts/pipeline/train_phase1b_dynamics.py",
        f"dynamics={cfg.model.dynamics}",
        f"dynamics.tokenizer_ckpt={tok_ckpt}",
        f"data.data_dirs={join_dirs(data_dirs)}",
        f"data.img_size={cfg.run.res}",
        f"trainer.devices={devices}",
        f"trainer.max_steps={cur_steps}",
        f"data.batch_size_dynamics={cfg.batch.dynamics}",
        f"data.num_workers={cfg.data.num_workers}",
        f"checkpoint.dirpath={ckpt_dir}",
        f"checkpoint.every_n_train_steps={cfg.checkpoint.every_dynamics}",
        f"checkpoint.save_top_k={cfg.checkpoint.save_top_k}",
        f"wandb.project={cfg.wandb.project}",
        f"wandb.name=c{cycle}_dyn_{cfg.run.tag}",
        *([f"wandb.entity={cfg.wandb.entity}"] if cfg.wandb.entity else []),
        f"data.tasks=[{tasks_str(cfg.data.tasks)}]",
        *resume,
    ])
    return last_ckpt


def phase_finetune(cfg: DictConfig, cycle: int, run_root: Path,
                   data_dirs: list[str],
                   tok_ckpt: Path, dyn_ckpt: Path, devices: int) -> Path:
    ckpt_dir = run_root / "finetune"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / "last.ckpt"
    if cycle == 0 and last_ckpt.exists():
        print(f"[Phase 2] Checkpoint existente — saltando ({last_ckpt})")
        return last_ckpt
    header("Phase 2 · Finetune", cycle, cfg.run.cycles)
    cur_steps, resume = phase_steps(
        cfg.steps.finetune_cold, cfg.steps.finetune_warm,
        last_ckpt if last_ckpt.exists() else None, cycle,
    )
    run([
        sys.executable, "scripts/pipeline/train_phase2_finetuning.py",
        f"finetune={cfg.model.finetune}",
        f"finetune.tokenizer_ckpt={tok_ckpt}",
        f"finetune.dynamics_ckpt={dyn_ckpt}",
        f"data.data_dirs={join_dirs(data_dirs)}",
        f"data.img_size={cfg.run.res}",
        f"trainer.devices={devices}",
        f"trainer.max_steps={cur_steps}",
        f"data.batch_size_dynamics={cfg.batch.finetune}",
        f"data.num_workers={cfg.data.num_workers}",
        f"checkpoint.dirpath={ckpt_dir}",
        f"checkpoint.every_n_train_steps={cfg.checkpoint.every_finetune}",
        f"checkpoint.save_top_k={cfg.checkpoint.save_top_k}",
        f"wandb.project={cfg.wandb.project}",
        f"wandb.name=c{cycle}_ft_{cfg.run.tag}",
        *([f"wandb.entity={cfg.wandb.entity}"] if cfg.wandb.entity else []),
        f"data.tasks=[{tasks_str(cfg.data.tasks)}]",
        *resume,
    ])
    return last_ckpt


def phase_agent(cfg: DictConfig, cycle: int, run_root: Path,
                data_dirs: list[str],
                ft_ckpt: Path, devices: int) -> Path:
    ckpt_dir = run_root / "agent"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / "last.ckpt"
    if cycle == 0 and last_ckpt.exists():
        print(f"[Phase 3] Checkpoint existente — saltando ({last_ckpt})")
        return last_ckpt
    header("Phase 3 · Agent (imagination)", cycle, cfg.run.cycles)
    cur_steps, resume = phase_steps(
        cfg.steps.agent_cold, cfg.steps.agent_warm,
        last_ckpt if last_ckpt.exists() else None, cycle,
    )
    run([
        sys.executable, "scripts/pipeline/train_phase3_imagination.py",
        f"agent={cfg.model.agent}",
        f"agent.finetune_ckpt={ft_ckpt}",
        f"data.data_dirs={join_dirs(data_dirs)}",
        f"data.img_size={cfg.run.res}",
        f"trainer.devices={devices}",
        f"trainer.max_steps={cur_steps}",
        f"data.batch_size_dynamics={cfg.batch.agent}",
        f"data.num_workers={cfg.data.num_workers}",
        f"agent.imagination_batch_size={cfg.batch.agent}",
        f"checkpoint.dirpath={ckpt_dir}",
        f"checkpoint.every_n_train_steps={cfg.checkpoint.every_agent}",
        f"checkpoint.save_top_k={cfg.checkpoint.save_top_k}",
        f"wandb.project={cfg.wandb.project}",
        f"wandb.name=c{cycle}_agent_{cfg.run.tag}",
        *([f"wandb.entity={cfg.wandb.entity}"] if cfg.wandb.entity else []),
        f"data.tasks=[{tasks_str(cfg.data.tasks)}]",
        *resume,
    ])
    return last_ckpt


def phase_eval(cfg: DictConfig, run_root: Path, agent_ckpt: Path) -> None:
    if not agent_ckpt.exists():
        print("\n⚠ No se encontró checkpoint del agente — evaluación saltada.")
        return
    eval_root = run_root / "dataset" / "eval"
    (eval_root / "videos").mkdir(parents=True, exist_ok=True)
    bar = "═" * 44
    print(f"\n{bar}\n  Evaluación final\n{bar}", flush=True)
    run([
        sys.executable, "scripts/pipeline/launch_phase0_dist.py",
        "collect.policy=agent",
        f"collect.num_envs_per_task={cfg.collect.num_envs}",
        f"collect.agent_ckpt={agent_ckpt}",
        "collect.n_episodes_per_task=5",
        f"++collect.episode_len={cfg.collect.episode_len}",
        f"++collect.min_frames_per_task={5 * cfg.collect.episode_len}",
        f"collect.frame_skip={cfg.collect.frame_skip}",
        f"collect.out_data_dir={eval_root / 'demos'}",
        f"collect.out_frames_dir={eval_root / 'frames'}",
        f"collect.out_videos_dir={eval_root / 'videos'}",
        f"collect.img_size={cfg.run.res}",
        "++collect.save_preview_video=true",
        "++collect.preview_video_backend=torchrl",
        f"++collect.preview_video_fps={cfg.collect.video_fps}",
        f"++collect.preview_video_max_frames={cfg.collect.episode_len}",
        f"++collect.wandb_project={cfg.wandb.project}",
        f"++collect.wandb_run_name=eval_{cfg.run.tag}",
        f"collect.tasks=[{tasks_str(cfg.data.tasks)}]",
    ])
    print(f"\n✓ Evaluación completada — Videos en {eval_root / 'videos'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "configs" / "run.yaml")

    # Aplicar overrides CLI estilo key=value
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, val = arg.split("=", 1)
            # Intentar parsear como tipo nativo; si falla, dejar como string
            try:
                parsed = OmegaConf.create({key.split(".")[-1]: val})
                OmegaConf.update(cfg, key, list(parsed.values())[0], merge=True)
            except Exception:
                OmegaConf.update(cfg, key, val, merge=True)
        else:
            print(f"[run.py] Argumento ignorado: {arg!r}")

    if cfg.run.res not in (64, 128):
        raise ValueError(f"run.res debe ser 64 o 128, recibido: {cfg.run.res}")

    devices  = detect_devices(cfg.trainer.devices)
    run_root = REPO_ROOT / "runs" / cfg.run.tag
    data_root = run_root / "dataset"
    cycles_dir = run_root / "cycles"
    for d in (run_root, data_root, cycles_dir):
        d.mkdir(parents=True, exist_ok=True)

    base_dirs = [Path(d) for d in cfg.data.data_dirs]
    for d in base_dirs:
        if not d.exists():
            raise FileNotFoundError(
                f"data.data_dirs: directorio no encontrado: {d}\n"
                "Descargá los datos con: python scripts/download_pretrain_data.py\n"
                "O convertí datos legacy con: python scripts/convert_pt_to_hdf5.py --data <dir>"
            )

    bar = "═" * 44
    print(f"\n{bar}")
    print(f"  Dreamer 4 — run={cfg.run.tag}  cycles={cfg.run.cycles}")
    print(f"  tokenizer={cfg.model.tokenizer}  dynamics={cfg.model.dynamics}")
    print(f"  devices={devices}  res={cfg.run.res}  seed={cfg.run.seed}")
    print(f"  data={[str(d) for d in base_dirs]}")
    print(f"{bar}\n", flush=True)

    data_dirs: list[str] = [str(d) for d in base_dirs]
    agent_ckpt = run_root / "agent" / "last.ckpt"

    for cycle in range(cfg.run.cycles):
        print(f"\n{bar}\n  Ciclo {cycle} / {cfg.run.cycles - 1}\n{bar}")

        # Phase 0 — datos
        if cycle == 0:
            print(f"[Phase 0] Datos fijos: {data_dirs}")
        else:
            out_data = phase_collect(cfg, cycle, run_root, agent_ckpt, data_root)
            data_dirs.append(str(out_data))

        tok_ckpt = phase_tokenizer(cfg, cycle, run_root, data_dirs, devices)
        dyn_ckpt = phase_dynamics(cfg, cycle, run_root, data_dirs, tok_ckpt, devices)
        ft_ckpt  = phase_finetune(cfg, cycle, run_root, data_dirs, tok_ckpt, dyn_ckpt, devices)
        agent_ckpt = phase_agent(cfg, cycle, run_root, data_dirs, ft_ckpt, devices)

        if agent_ckpt.exists():
            snap = cycles_dir / f"cycle{cycle}.ckpt"
            shutil.copy2(agent_ckpt, snap)
            print(f"\n✓ Ciclo {cycle} completo → {snap}")
        else:
            print(f"\n✓ Ciclo {cycle} completado (sin snapshot de agente)")

    phase_eval(cfg, run_root, agent_ckpt)

    print(f"\n{bar}")
    print(f"  Pipeline completo — {cfg.run.cycles} ciclos OK")
    print(f"  Run guardado en: runs/{cfg.run.tag}/")
    print(f"{bar}\n")


if __name__ == "__main__":
    main()

