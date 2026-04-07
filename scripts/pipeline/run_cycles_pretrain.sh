#!/bin/bash
# run_cycles_pretrain.sh — Ciclo recursivo usando datos pretrain de HuggingFace (mixed-large)
#
# Uso:
#   srun --gpus=4 --cpus-per-task=64 --mem=384G --time=72:00:00 bash scripts/pipeline/run_cycles_pretrain.sh
#
# Diferencias vs run_cycles.sh:
#   - Ciclo 0 usa ./data/pretrain/cycle0/ (mixed-large HF) en lugar de colección random
#   - Ciclos 1+ siguen colectando con la política entrenada
#   - K ciclos (default K=10), multi-GPU, logs separados (logs_pretrain/)
#   - Optimizado para 4x NVIDIA H100 (80GB)

set -e
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# Habilitar aceleración GPU por hardware para MuJoCo
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

# Eliminado export EGL_DEVICE_ID=0 para permitir asignación dinámica por subproceso

# Forzar explícitamente el uso de los drivers nativos de NVIDIA para EGL
# export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json

cd "$ROOT_DIR"
if [[ -f "$ROOT_DIR/.env" ]]; then source "$ROOT_DIR/.env"; fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURACIÓN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
K=${K:-10}              # ciclos totales
if [[ -n "${DEVICES:-}" ]]; then
    DEVICES="$DEVICES"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    DEVICES="$SLURM_GPUS_ON_NODE"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra _cuda_ids <<< "$CUDA_VISIBLE_DEVICES"
    DEVICES="${#_cuda_ids[@]}"
else
    DEVICES=4
fi
echo "[run_cycles_pretrain] trainer.devices=$DEVICES"

# Tasks disponibles en el pretrain reducidas a 4 clave para experimentación controlada
TASKS="walker-walk,walker-run,cheetah-run,walker-stand"

# Colección ciclos > 0 (con política aprendida)
AGENT_N_EPISODES=40     # 40 eps * 4 tasks * 1000 steps = 160,000 frames recolectados por ciclo
AGENT_EPISODE_LEN=1000
COLLECT_NUM_ENVS=4       # EnvPool workers por task por GPU (bajado a 4)
COLLECT_SAVE_PREVIEW_VIDEO=${COLLECT_SAVE_PREVIEW_VIDEO:-true}
COLLECT_PREVIEW_VIDEO_BACKEND=${COLLECT_PREVIEW_VIDEO_BACKEND:-torchrl}
COLLECT_PREVIEW_VIDEO_FPS=${COLLECT_PREVIEW_VIDEO_FPS:-30}
COLLECT_PREVIEW_VIDEO_MAX_FRAMES=${COLLECT_PREVIEW_VIDEO_MAX_FRAMES:-$AGENT_EPISODE_LEN}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BATCH SIZES Y WORKERS
# Para 4 GPUs evitando OOM (la memoria local debe no exceder el máximo posible)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BATCH_TOK=${BATCH_TOK:-24}     # local=24 * 4 GPUs = Global 96 frames (No supera el límite de 30 local)
BATCH_DYN=${BATCH_DYN:-64}     # local=64 * 4 GPUs = Global 256 seqs
BATCH_FT=${BATCH_FT:-24}        # local=24 * 4 GPUs = Global 96 seqs (Finetuning más pequeño para mayor updates y menos caos multivariante)
BATCH_AGT=${BATCH_AGT:-24}     # local=24 * 4 GPUs = Global 96 seqs (data loader de Phase 3, independiente de FT)
IMAG_BATCH=${IMAG_BATCH:-24}   # local=24 * 4 GPUs = Global 96 (batch de rollouts imaginados para PMPO)
NUM_WORKERS=${NUM_WORKERS:-4}  # Bajado de 16 a 4 para prevenir OOM en RAM del sistema

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━══
# STEPS POR FASE
#
# Objetivo por defecto:
#   - COLD: ~5 epochs
#   - WARM: ~2 epochs
#
# Aproximación usada (con settings actuales):
#   - Tokenizer:  batch global = 24*4=96
#   - Dynamics:   batch global = 64*4=256
#   - FT / Agent: batch global = 24*4=96
#   - COLD se estima sobre ciclo0 (~400k frames)
#   - WARM se estima sobre ciclo1 (~560k frames = 400k + 160k recolectados)
# ━═════════════════════════════════════════════════════════════════════════
COLD_1A=${COLD_1A:-20800}        # tokenizer ~5 epochs (ciclo0)
COLD_1B=${COLD_1B:-7800}         # dynamics ~5 epochs (ciclo0)
COLD_2=${COLD_2:-10000}          # finetuning ~5 epochs (ciclo0)
COLD_3=${COLD_3:-10000}          # PMPO agent ~5 epochs (ciclo0)

STEPS_1A=${STEPS_1A:-2000}      # tokenizer warm ~2 epochs (ciclo1 aprox)
STEPS_1B=${STEPS_1B:-2000}      # dynamics warm ~2 epochs (ciclo1 aprox)
STEPS_2=${STEPS_2:-10000}        # finetuning warm ~2 epochs (ciclo1 aprox) - duplicado
STEPS_3=${STEPS_3:-10000}        # PMPO agent warm ~2 epochs (ciclo1 aprox) - duplicado


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUNS Y CHECKPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASKS_ARG=("collect.tasks=[$TASKS]")
DATA_TASKS_ARG=("data.tasks=[$TASKS]")

RUN_TAG="${RUN_TAG:-pt_active}"
RUN_ROOT="./runs/${RUN_TAG}"

# Dataset base (fijo) vive en data/. Todo lo generado por este run vive en runs/.
# Permite comparar contra variantes como ./data/cycle0-random sin mezclar historicos.
BASE_DATA_ROOT="${BASE_DATA_ROOT:-./data/cycle0-pretrained}"
RUN_DATA_ROOT="${RUN_ROOT}/dataset"

TOK_CKPT_DIR="${RUN_ROOT}/tokenizer"
DYN_CKPT_DIR="${RUN_ROOT}/dynamics"
FT_CKPT_DIR="${RUN_ROOT}/finetune"
AGENT_CKPT_DIR="${RUN_ROOT}/agent"
AGENT_CYCLE_CKPT_DIR="${RUN_ROOT}/cycles"

TOK_CKPT="${TOK_CKPT_DIR}/last.ckpt"
DYN_CKPT="${DYN_CKPT_DIR}/last.ckpt"
FT_CKPT="${FT_CKPT_DIR}/last.ckpt"
AGENT_CKPT="${AGENT_CKPT_DIR}/last.ckpt"

mkdir -p \
    "$TOK_CKPT_DIR" \
    "$DYN_CKPT_DIR" \
    "$FT_CKPT_DIR" \
    "$AGENT_CKPT_DIR" \
    "$AGENT_CYCLE_CKPT_DIR" \
    "$RUN_DATA_ROOT"

echo "[run_cycles_pretrain] run_tag=$RUN_TAG"
echo "[run_cycles_pretrain] run_root=$RUN_ROOT"
echo "[run_cycles_pretrain] base_data_root=$BASE_DATA_ROOT"
echo "[run_cycles_pretrain] run_data_root=$RUN_DATA_ROOT"

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
    echo " Ciclo $CYCLE / $((K-1))  [pretrain run]"
    echo "════════════════════════════════════════"

    if [[ "$CYCLE" -eq 0 ]]; then
        # Ciclo 0 — usar dataset base fijo (semigood / random / etc.) en data/
        OUT_DATA="${BASE_DATA_ROOT}/demos"
        OUT_FRAMES="${BASE_DATA_ROOT}/frames"
        echo "[Phase 0] Usando dataset base en $BASE_DATA_ROOT — sin colección."
    else
        # Ciclos 1+ — colectar con la política aprendida (todo dentro de runs/)
        OUT_DATA="${RUN_DATA_ROOT}/cycle${CYCLE}/demos"
        OUT_FRAMES="${RUN_DATA_ROOT}/cycle${CYCLE}/frames"
        OUT_VIDEOS="${RUN_DATA_ROOT}/cycle${CYCLE}/videos"
        echo "[Phase 0] Recolectando con política agente (ciclo $CYCLE)..."
        python scripts/pipeline/launch_phase0_dist.py \
            collect.policy=agent \
            collect.num_envs_per_task=$COLLECT_NUM_ENVS \
            collect.agent_ckpt=$AGENT_CKPT \
            collect.n_episodes_per_task=$AGENT_N_EPISODES \
            ++collect.episode_len=$AGENT_EPISODE_LEN \
            collect.out_data_dir=$OUT_DATA \
            collect.out_frames_dir=$OUT_FRAMES \
            collect.out_videos_dir=$OUT_VIDEOS \
            ++collect.save_preview_video=$COLLECT_SAVE_PREVIEW_VIDEO \
            ++collect.preview_video_backend=$COLLECT_PREVIEW_VIDEO_BACKEND \
            ++collect.preview_video_fps=$COLLECT_PREVIEW_VIDEO_FPS \
            ++collect.preview_video_max_frames=$COLLECT_PREVIEW_VIDEO_MAX_FRAMES \
            "${TASKS_ARG[@]}"
    fi

    append_dirs "$OUT_DATA" "$OUT_FRAMES"

    # ── Phase 1a — Tokenizer ──────────────────────────────────────────────
    echo "[Phase 1a] Entrenando tokenizer..."
    RESUME_TOK_ARG=()
    if [[ "$CYCLE" -eq 0 ]] && [[ -f "$TOK_CKPT" ]]; then
        echo "[Phase 1a] Checkpoint existente en ciclo 0 — saltando (usar ciclos>0 para warm)."
    else
        if [[ "$CYCLE" -gt 0 ]] && [[ -f "$TOK_CKPT" ]]; then
            RESUME_TOK_ARG=("resume=$TOK_CKPT")
            CUR_STEPS_1A=$(( COLD_1A + CYCLE * STEPS_1A ))
        else
            CUR_STEPS_1A=$COLD_1A
        fi

        if [[ "$CUR_STEPS_1A" -gt 0 ]]; then
            python scripts/pipeline/train_phase1a_tokenizer.py \
                tokenizer=base_128x128 \
                "data.frame_dirs=[$FRAME_DIRS]" \
                trainer.devices=$DEVICES \
                trainer.max_steps=$CUR_STEPS_1A \
                data.batch_size_tokenizer=$BATCH_TOK \
                data.num_workers=$NUM_WORKERS \
                checkpoint.dirpath=$TOK_CKPT_DIR \
                wandb.name="c${CYCLE}_tok_${RUN_TAG}" \
                "${RESUME_TOK_ARG[@]}" \
                "${DATA_TASKS_ARG[@]}"
        else
            echo "Saltando entrenamiento del tokenizer (CUR_STEPS_1A = $CUR_STEPS_1A)..."
        fi
    fi

    # ── Phase 1b — Dynamics ───────────────────────────────────────────────
    echo "[Phase 1b] Entrenando dynamics..."
    RESUME_DYN_ARG=()
    if [[ "$CYCLE" -eq 0 ]] && [[ -f "$DYN_CKPT" ]]; then
        echo "[Phase 1b] Checkpoint existente en ciclo 0 — saltando (usar ciclos>0 para warm)."
    else
        if [[ "$CYCLE" -gt 0 ]] && [[ -f "$DYN_CKPT" ]]; then
            RESUME_DYN_ARG=("resume=$DYN_CKPT")
            CUR_STEPS_1B=$(( COLD_1B + CYCLE * STEPS_1B ))
        else
            CUR_STEPS_1B=$COLD_1B
        fi
        python scripts/pipeline/train_phase1b_dynamics.py \
            dynamics=base_128x128 \
            dynamics.tokenizer_ckpt=$TOK_CKPT \
            "data.data_dirs=[$DATA_DIRS]" \
            "data.frame_dirs=[$FRAME_DIRS]" \
            trainer.devices=$DEVICES \
            trainer.max_steps=$CUR_STEPS_1B \
            data.batch_size_dynamics=$BATCH_DYN \
            data.num_workers=$NUM_WORKERS \
            checkpoint.dirpath=$DYN_CKPT_DIR \
            wandb.name="c${CYCLE}_dyn_${RUN_TAG}" \
            "${RESUME_DYN_ARG[@]}" \
            "${DATA_TASKS_ARG[@]}"
    fi

    # ── Phase 2 — Finetuning ──────────────────────────────────────────────
    echo "[Phase 2] Finetuning (BC + reward)..."
    RESUME_FT_ARG=()
    if [[ "$CYCLE" -eq 0 ]] && [[ -f "$FT_CKPT" ]]; then
        echo "[Phase 2] Checkpoint existente en ciclo 0 — saltando (usar ciclos>0 para warm)."
    else
        if [[ "$CYCLE" -gt 0 ]] && [[ -f "$FT_CKPT" ]]; then
            RESUME_FT_ARG=("resume=$FT_CKPT")
            CUR_STEPS_2=$(( COLD_2 + CYCLE * STEPS_2 ))
        else
            CUR_STEPS_2=$COLD_2
        fi
        python scripts/pipeline/train_phase2_finetuning.py \
            finetune=base_128x128 \
            finetune.tokenizer_ckpt=$TOK_CKPT \
            finetune.dynamics_ckpt=$DYN_CKPT \
            "data.data_dirs=[$DATA_DIRS]" \
            "data.frame_dirs=[$FRAME_DIRS]" \
            trainer.devices=$DEVICES \
            trainer.max_steps=$CUR_STEPS_2 \
            data.batch_size_dynamics=$BATCH_FT \
            data.num_workers=$NUM_WORKERS \
            checkpoint.dirpath=$FT_CKPT_DIR \
            wandb.name="c${CYCLE}_ft_${RUN_TAG}" \
            "${RESUME_FT_ARG[@]}" \
            "${DATA_TASKS_ARG[@]}"
    fi

    # ── Phase 3 — Imagination Training ───────────────────────────────────
    echo "[Phase 3] Imagination training (PMPO)..."
    RESUME_AGENT_ARG=()
    if [[ "$CYCLE" -eq 0 ]] && [[ -f "$AGENT_CKPT" ]]; then
        echo "[Phase 3] Checkpoint existente en ciclo 0 — saltando (usar ciclos>0 para warm)."
    else
        if [[ "$CYCLE" -gt 0 ]] && [[ -f "$AGENT_CKPT" ]]; then
            RESUME_AGENT_ARG=("resume=$AGENT_CKPT")
            CUR_STEPS_3=$(( COLD_3 + CYCLE * STEPS_3 ))
        else
            CUR_STEPS_3=$COLD_3
        fi
        python scripts/pipeline/train_phase3_imagination.py \
            agent=base_128x128 \
            agent.finetune_ckpt=$FT_CKPT \
            "data.data_dirs=[$DATA_DIRS]" \
            "data.frame_dirs=[$FRAME_DIRS]" \
            trainer.devices=$DEVICES \
            trainer.max_steps=$CUR_STEPS_3 \
            data.batch_size_dynamics=$BATCH_AGT \
            data.num_workers=$NUM_WORKERS \
            agent.imagination_batch_size=$IMAG_BATCH \
            checkpoint.dirpath=$AGENT_CKPT_DIR \
            wandb.name="c${CYCLE}_agent_${RUN_TAG}" \
            "${RESUME_AGENT_ARG[@]}" \
            "${DATA_TASKS_ARG[@]}"
    fi

    [[ -f $AGENT_CKPT ]] && cp $AGENT_CKPT "$AGENT_CYCLE_CKPT_DIR/cycle${CYCLE}.ckpt"
    echo ""
    if [[ -f "$AGENT_CYCLE_CKPT_DIR/cycle${CYCLE}.ckpt" ]]; then
        echo "✓ Ciclo $CYCLE completo → $AGENT_CYCLE_CKPT_DIR/cycle${CYCLE}.ckpt"
    else
        echo "✓ Ciclo $CYCLE completado (sin snapshot de agent)"
    fi
done

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUACIÓN Y GENERACIÓN DE VIDEOS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [[ -f "$AGENT_CKPT" ]]; then
    echo ""
    echo "════════════════════════════════════════"
    echo " Evaluación — Generando videos con politica entrenada"
    echo "════════════════════════════════════════"
    
    EVAL_DATA="${RUN_DATA_ROOT}/eval/demos"
    EVAL_FRAMES="${RUN_DATA_ROOT}/eval/frames"
    EVAL_VIDEOS="${RUN_DATA_ROOT}/eval/videos"
    
    mkdir -p "$EVAL_VIDEOS"
    
    python scripts/pipeline/launch_phase0_dist.py \
        collect.policy=agent \
        collect.num_envs_per_task=$COLLECT_NUM_ENVS \
        collect.agent_ckpt=$AGENT_CKPT \
        collect.n_episodes_per_task=5 \
        ++collect.episode_len=1000 \
        collect.out_data_dir=$EVAL_DATA \
        collect.out_frames_dir=$EVAL_FRAMES \
        collect.out_videos_dir=$EVAL_VIDEOS \
        ++collect.save_preview_video=true \
        ++collect.preview_video_backend=torchrl \
        ++collect.preview_video_fps=30 \
        ++collect.preview_video_max_frames=1000 \
        collect.wandb_project=dreamer4 \
        collect.wandb_run_name="eval_${RUN_TAG}" \
        trainer.devices=$DEVICES \
        "${TASKS_ARG[@]}"
    
    echo ""
    echo "✓ Evaluación completada — Videos guardados en $EVAL_VIDEOS"
else
    echo ""
    echo "⚠ No se encontró checkpoint del agente — evaluación saltada."
fi

echo ""
echo "════════════════════════════════════════"
echo " Pipeline pretrain completo — $K ciclos OK"
echo "════════════════════════════════════════"
