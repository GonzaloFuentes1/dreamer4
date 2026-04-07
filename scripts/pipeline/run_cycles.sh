#!/bin/bash
# run_cycles.sh — Ciclo recursivo completo dreamer4
#
# Uso:
#   srun --gres=gpu:h100:1 --cpus-per-task=8  --mem=64G  --time=48:00:00 bash run_cycles.sh ...
#   srun --gres=gpu:h100:2 --cpus-per-task=16 --mem=128G --time=48:00:00 bash run_cycles.sh ...
#   srun --gres=gpu:h100:3 --cpus-per-task=24 --mem=192G --time=48:00:00 bash run_cycles.sh ...
#
# Ejemplos:
#   bash run_cycles.sh 10 50 1000 "walker-walk,cheetah-run"   # real ~5h con los defaults
#   bash run_cycles.sh 2 5 200 "walker-walk,cheetah-run" 500  # smoke test
#
# Ciclo 0:  datos random  → Phase 1a → 1b → 2 → 3  → agent_v0
# Ciclo N:  datos agent   → Phase 1a → 1b → 2 → 3  → agent_vN  (1a+1b siempre)

set -e
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
if [[ -f "$ROOT_DIR/.env" ]]; then source "$ROOT_DIR/.env"; fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARGUMENTOS DE LÍNEA DE COMANDOS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
K=${1:-20}             # número de ciclos
N_EPISODES=${2:-25}    # episodios por tarea — ciclo 0 (colección random)
EPISODE_LEN=${3:-1000} # pasos por episodio  — ciclo 0
TASKS=${4:-"walker-stand,walker-walk,walker-run,cheetah-run,hopper-stand,hopper-hop,reacher-easy,reacher-hard,cartpole-swingup,finger-spin"}
MAX_STEPS=${5:-""}     # smoke test: sobreescribe TODOS los STEPS_* si se pasa

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURACIÓN DE HARDWARE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [[ -n "${DEVICES:-}" ]]; then
    DEVICES="$DEVICES"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    DEVICES="$SLURM_GPUS_ON_NODE"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra _cuda_ids <<< "$CUDA_VISIBLE_DEVICES"
    DEVICES="${#_cuda_ids[@]}"
else
    DEVICES=1
fi
echo "[run_cycles] trainer.devices=$DEVICES"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEPS POR FASE  (calibrados para DEVICES=5, 10 tasks, 25 eps, 1000 steps)
#
#  Referencia rápida con DEVICES=5, 10 tasks, batch defaults abajo:
#    Ciclo 0 datos:  10 tasks × 25 eps × 1000 steps = 250k frames / 242k windows
#    Ciclos >0 data: +10 tasks × 20 eps × 500 steps = +100k frames / +97k windows
#
#    1 época tokenizer  = 250k / (24×5) = 2,083 steps
#    1 época dynamics   = 242k / (96×5) =   505 steps
#
#    Cold start (ciclo 0): más steps para arrancar desde cero
#    Warm start (ciclos >0): menos steps, parte del checkpoint anterior
#
#    Phase 1a: cold=3 épocas tok, warm=1 época tok
#    Phase 1b: cold=5 épocas dyn, warm=2 épocas dyn
#    Phase 2:  cold=5 épocas dyn, warm=3 épocas dyn
#    Phase 3:  cold=10 épocas,    warm=10 épocas
#
#    Total 20 ciclos: ~12k steps/ciclo (ciclos 1-19) + cold ≈ 24h con 5×H100
#
#  Para smoke test rápido: bash run_cycles.sh 2 5 200 "" 500
#    → MAX_STEPS=500 sobreescribe todo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cold start — ciclo 0 (desde cero, sin checkpoint previo)
COLD_1A=10_000         # tokenizer   ~3.0 épocas cold  (2,083 steps/época)
COLD_1B=2_500         # dynamics    ~5.0 épocas cold  (505 steps/época)
COLD_2=2_500          # finetuning  ~5.0 épocas cold  (505 steps/época)
COLD_3=5_000          # agente PMPO ~10 épocas cold

# # Warm start — ciclos 1+ (resume desde checkpoint anterior)
STEPS_1A=5_000        # tokenizer   ~1.0 época  (2,083 steps/época)
STEPS_1B=5_000        # dynamics    ~2.0 épocas (505 steps/época)
STEPS_2=5_000         # finetuning  ~3.0 épocas (505 steps/época)
STEPS_3=5_000         # agente PMPO ~10 épocas equiv.

# Si MAX_STEPS está seteado (modo smoke test), sobreescribe todo
if [[ -n "$MAX_STEPS" ]]; then
    COLD_1A=$MAX_STEPS; COLD_1B=$MAX_STEPS; COLD_2=$MAX_STEPS; COLD_3=$MAX_STEPS
    STEPS_1A=$MAX_STEPS; STEPS_1B=$MAX_STEPS; STEPS_2=$MAX_STEPS; STEPS_3=$MAX_STEPS
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BATCH SIZES  (optimizados para H100 80GB — aumentar si hay OOM reducir si crash)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BATCH_TOK=24          # frames por GPU — tokenizer  (S=1040 tokens, atención O(S²), NO subir mucho)
BATCH_DYN=64          # seqs  por GPU  — dynamics/finetune/agente (S~19 tokens, puede ser grande)
NUM_WORKERS=4         # workers DataLoader por GPU (4 GPUs × 4 = 16 total, conserva RAM)
IMAG_BATCH=96         # rollouts de imaginación por step (agent_module)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COLECCIÓN CON POLÍTICA AGENTE (ciclos > 0)
#   Menos episodios que ciclo 0 para que collect no sea el cuello de botella
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGENT_N_EPISODES=20   # vs $N_EPISODES del ciclo 0
AGENT_EPISODE_LEN=1000 # DMControl + mjwarp: mantener horizonte completo

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECKPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOK_CKPT=./logs/tokenizer_ckpts/last.ckpt
DYN_CKPT=./logs/dynamics_ckpts/last.ckpt
FT_CKPT=./logs/finetune_ckpts/last.ckpt
AGENT_CKPT=./logs/agent_ckpts/last.ckpt

# ── Overrides de tasks ───────────────────────────────────────────────────────
TASKS_ARG=()
DATA_TASKS_ARG=()
if [[ -n "$TASKS" ]]; then
    TASKS_ARG=("collect.tasks=[$TASKS]")
    DATA_TASKS_ARG=("data.tasks=[$TASKS]")
fi

# ── Tag W&B ──────────────────────────────────────────────────────────────────
DATETAG=$(date +%m%d_%H%M)
TASKTAG=${TASKS:-"all"}
TASKTAG=$(echo "$TASKTAG" | tr ',' '+' | cut -c1-30)
RUN_TAG="${DATETAG}_${TASKTAG}"

# ── Acumulador de directorios de datos ───────────────────────────────────────
DATA_DIRS=""
FRAME_DIRS=""

append_dirs() {
    local data=$1 frames=$2
    if [[ -z "$DATA_DIRS" ]]; then
        DATA_DIRS="$data"; FRAME_DIRS="$frames"
    else
        DATA_DIRS="${DATA_DIRS},${data}"; FRAME_DIRS="${FRAME_DIRS},${frames}"
    fi
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BUCLE PRINCIPAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
for CYCLE in $(seq 0 $((K - 1))); do
    echo ""
    echo "════════════════════════════════════════"
    echo " Ciclo $CYCLE / $((K-1))"
    echo "════════════════════════════════════════"

    OUT_DATA="./data/cycle${CYCLE}/demos"
    OUT_FRAMES="./data/cycle${CYCLE}/frames"

    # ── Phase 0 — Colección ─────────────────────────────────────────────────
    echo "[Phase 0] Recolectando datos..."
    _all_tasks_present=1
    IFS=',' read -ra _task_list <<< "$TASKS"
    for _t in "${_task_list[@]}"; do
        [[ -f "$OUT_DATA/${_t}.pt" ]] || { _all_tasks_present=0; break; }
    done
    if [[ "$CYCLE" -eq 0 ]] && [[ "$_all_tasks_present" -eq 1 ]]; then
        echo "[Phase 0] Todos los tasks ya existen en $OUT_DATA — saltando recolección."
    elif [[ "$CYCLE" -eq 0 ]]; then
        python scripts/pipeline/launch_phase0_dist.py \
            collect.n_episodes_per_task=$N_EPISODES \
            collect.episode_len=$EPISODE_LEN \
            collect.out_data_dir=$OUT_DATA \
            collect.out_frames_dir=$OUT_FRAMES \
            "${TASKS_ARG[@]}"
    else
        python scripts/pipeline/launch_phase0_dist.py \
            collect.policy=agent \
            collect.agent_ckpt=$AGENT_CKPT \
            collect.n_episodes_per_task=$AGENT_N_EPISODES \
            collect.episode_len=$AGENT_EPISODE_LEN \
            collect.out_data_dir=$OUT_DATA \
            collect.out_frames_dir=$OUT_FRAMES \
            "${TASKS_ARG[@]}"
    fi

    append_dirs "$OUT_DATA" "$OUT_FRAMES"

    # ── Phase 1a — Tokenizer (todos los ciclos, warm desde ciclo 1) ──────────
    echo "[Phase 1a] Entrenando tokenizer..."
    RESUME_TOK_ARG=()
    if [[ "$CYCLE" -eq 0 ]] && [[ -f "$TOK_CKPT" ]]; then
        echo "[Phase 1a] Checkpoint existente en ciclo 0 — saltando (usar ciclos>0 para warm)."
    else
        if [[ "$CYCLE" -gt 0 ]] && [[ -f "$TOK_CKPT" ]]; then
            RESUME_TOK_ARG=("resume=$TOK_CKPT")
            CUR_STEPS_1A=$STEPS_1A
        else
            CUR_STEPS_1A=$COLD_1A
        fi
        python scripts/pipeline/train_phase1a_tokenizer.py \
            "data.frame_dirs=[$FRAME_DIRS]" \
            trainer.devices=$DEVICES \
            trainer.max_steps=$CUR_STEPS_1A \
            data.batch_size_tokenizer=$BATCH_TOK \
            data.num_workers=$NUM_WORKERS \
            wandb.name="c${CYCLE}_tok_${RUN_TAG}" \
            "${RESUME_TOK_ARG[@]}" \
            "${DATA_TASKS_ARG[@]}"
    fi

    # ── Phase 1b — Dynamics (todos los ciclos) ──────────────────────────────
    echo "[Phase 1b] Entrenando dynamics..."
    RESUME_DYN_ARG=()
    if [[ "$CYCLE" -eq 0 ]] && [[ -f "$DYN_CKPT" ]]; then
        echo "[Phase 1b] Checkpoint existente en ciclo 0 — saltando (usar ciclos>0 para warm)."
    else
        if [[ "$CYCLE" -gt 0 ]] && [[ -f "$DYN_CKPT" ]]; then
            RESUME_DYN_ARG=("resume=$DYN_CKPT")
            CUR_STEPS_1B=$STEPS_1B
        else
            CUR_STEPS_1B=$COLD_1B
        fi
        python scripts/pipeline/train_phase1b_dynamics.py \
            dynamics.tokenizer_ckpt=$TOK_CKPT \
            "data.data_dirs=[$DATA_DIRS]" \
            "data.frame_dirs=[$FRAME_DIRS]" \
            trainer.devices=$DEVICES \
            trainer.max_steps=$CUR_STEPS_1B \
            data.batch_size_dynamics=$BATCH_DYN \
            data.num_workers=$NUM_WORKERS \
            wandb.name="c${CYCLE}_dyn_${RUN_TAG}" \
            "${RESUME_DYN_ARG[@]}" \
            "${DATA_TASKS_ARG[@]}"
    fi

    # ── Phase 2 — Finetuning ────────────────────────────────────────────────
    echo "[Phase 2] Finetuning (BC + reward)..."
    CUR_STEPS_2=$([[ "$CYCLE" -eq 0 ]] && echo $COLD_2 || echo $STEPS_2)
    python scripts/pipeline/train_phase2_finetuning.py \
        finetune.tokenizer_ckpt=$TOK_CKPT \
        finetune.dynamics_ckpt=$DYN_CKPT \
        "data.data_dirs=[$DATA_DIRS]" \
        "data.frame_dirs=[$FRAME_DIRS]" \
        trainer.devices=$DEVICES \
        trainer.max_steps=$CUR_STEPS_2 \
        data.batch_size_dynamics=$BATCH_DYN \
        data.num_workers=$NUM_WORKERS \
        wandb.name="c${CYCLE}_ft_${RUN_TAG}" \
        "${DATA_TASKS_ARG[@]}"

    # ── Phase 3 — Imagination Training ─────────────────────────────────────
    echo "[Phase 3] Imagination training (PMPO)..."
    CUR_STEPS_3=$([[ "$CYCLE" -eq 0 ]] && echo $COLD_3 || echo $STEPS_3)
    python scripts/pipeline/train_phase3_imagination.py \
        agent.finetune_ckpt=$FT_CKPT \
        "data.data_dirs=[$DATA_DIRS]" \
        "data.frame_dirs=[$FRAME_DIRS]" \
        trainer.devices=$DEVICES \
        trainer.max_steps=$CUR_STEPS_3 \
        data.batch_size_dynamics=$BATCH_DYN \
        data.num_workers=$NUM_WORKERS \
        agent.imagination_batch_size=$IMAG_BATCH \
        wandb.name="c${CYCLE}_agent_${RUN_TAG}" \
        "${DATA_TASKS_ARG[@]}"

    # ── Guardar checkpoint del ciclo ────────────────────────────────────────
    cp $AGENT_CKPT ./logs/agent_ckpts/cycle${CYCLE}.ckpt
    echo ""
    echo "✓ Ciclo $CYCLE completo → ./logs/agent_ckpts/cycle${CYCLE}.ckpt"
done

echo ""
echo "════════════════════════════════════════"
echo " Pipeline completo — $K ciclos OK"
echo "════════════════════════════════════════"

