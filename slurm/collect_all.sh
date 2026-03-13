#!/bin/bash
# slurm/collect_all.sh — Colección de episodios para ambientes DMControl
#
# Uso:
#   srun --cpus-per-task=16 --mem=32G --time=6:00:00 \
#     bash slurm/collect_all.sh [OUT_DIR] [N_EPISODES] [EPISODE_LEN] [TASKS]
#
# Ejemplos:
#   srun ... bash slurm/collect_all.sh                                          # todos los tasks, defaults
#   srun ... bash slurm/collect_all.sh ./data/cycle0 50 1000                   # todos los tasks, ciclo 0
#   srun ... bash slurm/collect_all.sh ./data/cycle0 50 1000 "walker-walk,cheetah-run"  # solo 2 tasks

set -e

# ── Argumentos ───────────────────────────────────────────────────────────────
OUT_DIR=${1:-"./data/collect_all"}
N_EPISODES=${2:-25}
EPISODE_LEN=${3:-1000}
TASKS=${4:-""}   # vacío = todos los tasks del TASK_SET

# ── Entorno ──────────────────────────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/.env"
source "$ROOT_DIR/.venv/bin/activate"

mkdir -p logs/slurm

# ── Construir override de tasks ───────────────────────────────────────────────
if [[ -n "$TASKS" ]]; then
    # Convertir "walker-walk,cheetah-run" → "[walker-walk,cheetah-run]"
    TASKS_ARG="collect.tasks=[${TASKS}]"
    echo "[collect_all] tasks=$TASKS  n_episodes=$N_EPISODES  episode_len=$EPISODE_LEN"
else
    TASKS_ARG="collect.tasks=null"
    echo "[collect_all] tasks=ALL  n_episodes=$N_EPISODES  episode_len=$EPISODE_LEN"
fi
echo "[collect_all] out → $OUT_DIR"

python collect_phase0_data.py \
    collect.n_episodes_per_task=$N_EPISODES \
    collect.episode_len=$EPISODE_LEN \
    collect.out_data_dir="$OUT_DIR/demos" \
    collect.out_frames_dir="$OUT_DIR/frames" \
    "$TASKS_ARG" \
    collect.num_workers=8
