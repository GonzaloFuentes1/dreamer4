#!/bin/bash
# run_cycles_64x64.sh — Pipeline activo sin pretraining, resolución 64×64
#
# Uso:
#   srun --gpus=3 --cpus-per-task=48 --mem=256G --time=72:00:00 bash scripts/pipeline/run_cycles_64x64.sh
#
# Configuración:
#   - Sin datos de pretraining: ciclo 0 colecta con política aleatoria
#   - Resolución 64×64 (patch=4)
#   - Misma arquitectura base del pipeline 128×128 en el tokenizer continuo:
#       d_model=256, n_heads=4, depth=8, n_latents=16, d_bottleneck=32
#   - K=30 ciclos, 5 tasks (walker-run, cheetah-run, hopper-hop, finger-spin, reacher-hard)
#   - 10 episodios/task → 50k frames/ciclo

set -e
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json

cd "$ROOT_DIR"
if [[ -f "$ROOT_DIR/.env" ]]; then source "$ROOT_DIR/.env"; fi

VENV_PATH=${VENV_PATH:-$ROOT_DIR/.venv}
if [[ -d "$VENV_PATH" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
    echo "[run_cycles_64x64] usando venv=$VENV_PATH"
else
    echo "[run_cycles_64x64] aviso: no se encontro venv en $VENV_PATH"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURACIÓN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
K=${K:-30}
START_FROM_DYNAMICS_CYCLE0=${START_FROM_DYNAMICS_CYCLE0:-true}
SOURCE_RUN_TAG=${SOURCE_RUN_TAG:-active_64x64_v2}

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
echo "[run_cycles_64x64] trainer.devices=$DEVICES"

TASKS="${TASKS:-walker-run,cheetah-run,hopper-hop,finger-spin,reacher-hard}"

# Colección — todos los ciclos (ciclo 0: aleatoria, ciclos 1+: con agente)
AGENT_N_EPISODES=10
AGENT_EPISODE_LEN=1000
COLLECT_MIN_FRAMES_PER_TASK=${COLLECT_MIN_FRAMES_PER_TASK:-$((AGENT_N_EPISODES * AGENT_EPISODE_LEN))}
COLLECT_NUM_ENVS=16
COLLECT_SAVE_PREVIEW_VIDEO=${COLLECT_SAVE_PREVIEW_VIDEO:-true}
COLLECT_PREVIEW_VIDEO_BACKEND=${COLLECT_PREVIEW_VIDEO_BACKEND:-torchrl}
COLLECT_PREVIEW_VIDEO_FPS=${COLLECT_PREVIEW_VIDEO_FPS:-30}
COLLECT_PREVIEW_VIDEO_MAX_FRAMES=${COLLECT_PREVIEW_VIDEO_MAX_FRAMES:-$AGENT_EPISODE_LEN}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BATCH SIZES Y WORKERS
# 64×64 reduce 4× los patches respecto a 128×128, así que el tokenizer puede
# ir con batch mayor manteniendo la misma arquitectura base.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BATCH_TOK=${BATCH_TOK:-48}
BATCH_DYN=${BATCH_DYN:-16}
BATCH_FT=${BATCH_FT:-16}
IMAG_BATCH=${IMAG_BATCH:-24}
NUM_WORKERS=${NUM_WORKERS:-4}
AGENT_DDP_STRATEGY=${AGENT_DDP_STRATEGY:-ddp_find_unused_parameters_true}
TOKENIZER_MAE_P_MAX=${TOKENIZER_MAE_P_MAX:-0.75}
TOKENIZER_D_MODEL=${TOKENIZER_D_MODEL:-256}
TOKENIZER_N_HEADS=${TOKENIZER_N_HEADS:-4}
TOKENIZER_DEPTH=${TOKENIZER_DEPTH:-8}
TOKENIZER_N_LATENTS=${TOKENIZER_N_LATENTS:-16}
TOKENIZER_D_BOTTLENECK=${TOKENIZER_D_BOTTLENECK:-32}
TOKENIZER_TIME_EVERY=${TOKENIZER_TIME_EVERY:-1}
TOKENIZER_LATENTS_ONLY_TIME=${TOKENIZER_LATENTS_ONLY_TIME:-true}
TOKENIZER_SCALE_POS_EMBEDS=${TOKENIZER_SCALE_POS_EMBEDS:-false}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEPS POR FASE
# Mismo esquema corto y multi-ciclo que 128×128, con tokenizer un poco más barato.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLD_1A=${COLD_1A:-20000}
COLD_1B=${COLD_1B:-20000}
COLD_2=${COLD_2:-3000}
COLD_3=${COLD_3:-10000}

STEPS_1A=${STEPS_1A:-2000}
STEPS_1B=${STEPS_1B:-2000}
STEPS_2=${STEPS_2:-2000}
STEPS_3=${STEPS_3:-4000}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUNS Y CHECKPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASKS_ARG=("collect.tasks=[$TASKS]")
DATA_TASKS_ARG=("data.tasks=[$TASKS]")

RUN_TAG="${RUN_TAG:-active_64x64_dynfix_v1}"
RUN_ROOT="./runs/${RUN_TAG}"
RUN_DATA_ROOT="${RUN_ROOT}/dataset"
SOURCE_RUN_ROOT="./runs/${SOURCE_RUN_TAG}"

TOK_CKPT_DIR="${RUN_ROOT}/tokenizer"
DYN_CKPT_DIR="${RUN_ROOT}/dynamics"
FT_CKPT_DIR="${RUN_ROOT}/finetune"
AGENT_CKPT_DIR="${RUN_ROOT}/agent"
AGENT_CYCLE_CKPT_DIR="${RUN_ROOT}/cycles"

TOK_CKPT="${TOK_CKPT_DIR}/last.ckpt"
DYN_CKPT="${DYN_CKPT_DIR}/last.ckpt"
FT_CKPT="${FT_CKPT_DIR}/last.ckpt"
AGENT_CKPT="${AGENT_CKPT_DIR}/last.ckpt"

if [[ "$START_FROM_DYNAMICS_CYCLE0" == "true" ]]; then
    CYCLE0_PRETRAINED_DATA=${CYCLE0_PRETRAINED_DATA:-${SOURCE_RUN_ROOT}/dataset/cycle0}
    SEED_TOKENIZER_CKPT=${SEED_TOKENIZER_CKPT:-${SOURCE_RUN_ROOT}/tokenizer/last.ckpt}
fi

mkdir -p \
    "$TOK_CKPT_DIR" \
    "$DYN_CKPT_DIR" \
    "$FT_CKPT_DIR" \
    "$AGENT_CKPT_DIR" \
    "$AGENT_CYCLE_CKPT_DIR" \
    "$RUN_DATA_ROOT"

echo "[run_cycles_64x64] run_tag=$RUN_TAG"
echo "[run_cycles_64x64] run_root=$RUN_ROOT"
echo "[run_cycles_64x64] run_data_root=$RUN_DATA_ROOT"
if [[ "$START_FROM_DYNAMICS_CYCLE0" == "true" ]]; then
    echo "[run_cycles_64x64] start_from_dynamics_cycle0=true"
    echo "[run_cycles_64x64] source_run_root=$SOURCE_RUN_ROOT"
    echo "[run_cycles_64x64] cycle0_pretrained_data=$CYCLE0_PRETRAINED_DATA"
    echo "[run_cycles_64x64] seed_tokenizer_ckpt=$SEED_TOKENIZER_CKPT"
fi

if [[ "$START_FROM_DYNAMICS_CYCLE0" == "true" ]]; then
    if [[ ! -d "$CYCLE0_PRETRAINED_DATA/demos" ]] || [[ ! -d "$CYCLE0_PRETRAINED_DATA/frames" ]]; then
        echo "[run_cycles_64x64] error: CYCLE0_PRETRAINED_DATA invalido: $CYCLE0_PRETRAINED_DATA"
        exit 1
    fi
    if [[ ! -f "$SEED_TOKENIZER_CKPT" ]]; then
        echo "[run_cycles_64x64] error: no se encontro tokenizer seed: $SEED_TOKENIZER_CKPT"
        exit 1
    fi
    if [[ ! -f "$TOK_CKPT" ]]; then
        cp "$SEED_TOKENIZER_CKPT" "$TOK_CKPT"
        echo "[run_cycles_64x64] tokenizer seed copiado a $TOK_CKPT"
    fi
fi

DATA_DIRS=""
FRAME_DIRS=""

append_dirs() {
    local data=$1 frames=$2
    if [[ -z "$DATA_DIRS" ]]; then
        DATA_DIRS="$data"
        FRAME_DIRS="$frames"
    else
        DATA_DIRS="${DATA_DIRS},${data}"
        FRAME_DIRS="${FRAME_DIRS},${frames}"
    fi
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BUCLE PRINCIPAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
for CYCLE in $(seq 0 $((K - 1))); do
    echo ""
    echo "════════════════════════════════════════"
    echo " Ciclo $CYCLE / $((K-1))  [active_64x64]"
    echo "════════════════════════════════════════"

    OUT_DATA="${RUN_DATA_ROOT}/cycle${CYCLE}/demos"
    OUT_FRAMES="${RUN_DATA_ROOT}/cycle${CYCLE}/frames"
    OUT_VIDEOS="${RUN_DATA_ROOT}/cycle${CYCLE}/videos"

    if [[ "$CYCLE" -eq 0 ]] && [[ -n "${CYCLE0_PRETRAINED_DATA:-}" ]]; then
        echo "[Phase 0] Usando datos precolectados: $CYCLE0_PRETRAINED_DATA"
        OUT_DATA="${CYCLE0_PRETRAINED_DATA}/demos"
        OUT_FRAMES="${CYCLE0_PRETRAINED_DATA}/frames"
    elif [[ "$CYCLE" -eq 0 ]]; then
        echo "[Phase 0] Recolectando con política aleatoria (ciclo 0)..."
        python scripts/pipeline/launch_phase0_dist.py \
            collect.policy=random \
            collect.num_envs_per_task=$COLLECT_NUM_ENVS \
            collect.n_episodes_per_task=$AGENT_N_EPISODES \
            ++collect.episode_len=$AGENT_EPISODE_LEN \
            ++collect.min_frames_per_task=$COLLECT_MIN_FRAMES_PER_TASK \
            collect.out_data_dir=$OUT_DATA \
            collect.out_frames_dir=$OUT_FRAMES \
            collect.out_videos_dir=$OUT_VIDEOS \
            collect.img_size=64 \
            ++collect.save_preview_video=$COLLECT_SAVE_PREVIEW_VIDEO \
            ++collect.preview_video_backend=$COLLECT_PREVIEW_VIDEO_BACKEND \
            ++collect.preview_video_fps=$COLLECT_PREVIEW_VIDEO_FPS \
            ++collect.preview_video_max_frames=$COLLECT_PREVIEW_VIDEO_MAX_FRAMES \
            "${TASKS_ARG[@]}"
    else
        echo "[Phase 0] Recolectando con política agente (ciclo $CYCLE)..."
        python scripts/pipeline/launch_phase0_dist.py \
            collect.policy=agent \
            collect.num_envs_per_task=$COLLECT_NUM_ENVS \
            collect.agent_ckpt=$AGENT_CKPT \
            collect.n_episodes_per_task=$AGENT_N_EPISODES \
            ++collect.episode_len=$AGENT_EPISODE_LEN \
            ++collect.min_frames_per_task=$COLLECT_MIN_FRAMES_PER_TASK \
            collect.out_data_dir=$OUT_DATA \
            collect.out_frames_dir=$OUT_FRAMES \
            collect.out_videos_dir=$OUT_VIDEOS \
            collect.img_size=64 \
            ++collect.save_preview_video=$COLLECT_SAVE_PREVIEW_VIDEO \
            ++collect.preview_video_backend=$COLLECT_PREVIEW_VIDEO_BACKEND \
            ++collect.preview_video_fps=$COLLECT_PREVIEW_VIDEO_FPS \
            ++collect.preview_video_max_frames=$COLLECT_PREVIEW_VIDEO_MAX_FRAMES \
            "${TASKS_ARG[@]}"
    fi

    append_dirs "$OUT_DATA" "$OUT_FRAMES"

    if [[ "$CYCLE" -eq 0 ]] && [[ "$START_FROM_DYNAMICS_CYCLE0" == "true" ]]; then
        echo "[Phase 1a] Saltado en ciclo 0 — usando tokenizer seed: $TOK_CKPT"
    else
        echo "[Phase 1a] Entrenando tokenizer..."
        RESUME_TOK_ARG=()
        if [[ -f "$TOK_CKPT" ]]; then
            RESUME_TOK_ARG=("resume=$TOK_CKPT")
            if [[ "$CYCLE" -gt 0 ]]; then
                CUR_STEPS_1A=$(( COLD_1A + CYCLE * STEPS_1A ))
            else
                CUR_STEPS_1A=$COLD_1A
            fi
        else
            CUR_STEPS_1A=$COLD_1A
        fi

        python scripts/pipeline/train_phase1a_tokenizer.py \
            tokenizer=base_64x64 \
            tokenizer.d_model=$TOKENIZER_D_MODEL \
            tokenizer.n_heads=$TOKENIZER_N_HEADS \
            tokenizer.depth=$TOKENIZER_DEPTH \
            tokenizer.n_latents=$TOKENIZER_N_LATENTS \
            tokenizer.d_bottleneck=$TOKENIZER_D_BOTTLENECK \
            tokenizer.time_every=$TOKENIZER_TIME_EVERY \
            tokenizer.latents_only_time=$TOKENIZER_LATENTS_ONLY_TIME \
            tokenizer.scale_pos_embeds=$TOKENIZER_SCALE_POS_EMBEDS \
            tokenizer.mae_p_max=$TOKENIZER_MAE_P_MAX \
            "data.frame_dirs=[$FRAME_DIRS]" \
            trainer.devices=$DEVICES \
            trainer.max_steps=$CUR_STEPS_1A \
            data.batch_size_tokenizer=$BATCH_TOK \
            data.num_workers=$NUM_WORKERS \
            checkpoint.dirpath=$TOK_CKPT_DIR \
            wandb.name="c${CYCLE}_tok_${RUN_TAG}" \
            "${RESUME_TOK_ARG[@]}" \
            "${DATA_TASKS_ARG[@]}"
    fi

    echo "[Phase 1b] Entrenando dynamics..."
    RESUME_DYN_ARG=()
    if [[ -f "$DYN_CKPT" ]]; then
        RESUME_DYN_ARG=("resume=$DYN_CKPT")
        if [[ "$CYCLE" -gt 0 ]]; then
            CUR_STEPS_1B=$(( COLD_1B + CYCLE * STEPS_1B ))
        else
            CUR_STEPS_1B=$COLD_1B
        fi
    else
        CUR_STEPS_1B=$COLD_1B
    fi

    python scripts/pipeline/train_phase1b_dynamics.py \
        dynamics=base_64x64 \
        dynamics.tokenizer_ckpt=$TOK_CKPT \
        "data.data_dirs=[$DATA_DIRS]" \
        "data.frame_dirs=[$FRAME_DIRS]" \
        data.img_size=64 \
        trainer.devices=$DEVICES \
        trainer.max_steps=$CUR_STEPS_1B \
        data.batch_size_dynamics=$BATCH_DYN \
        data.num_workers=$NUM_WORKERS \
        checkpoint.dirpath=$DYN_CKPT_DIR \
        wandb.name="c${CYCLE}_dyn_${RUN_TAG}" \
        "${RESUME_DYN_ARG[@]}" \
        "${DATA_TASKS_ARG[@]}"

    echo "[Phase 2] Finetuning (BC + reward)..."
    RESUME_FT_ARG=()
    if [[ -f "$FT_CKPT" ]]; then
        RESUME_FT_ARG=("resume=$FT_CKPT")
        if [[ "$CYCLE" -gt 0 ]]; then
            CUR_STEPS_2=$(( COLD_2 + CYCLE * STEPS_2 ))
        else
            CUR_STEPS_2=$COLD_2
        fi
    else
        CUR_STEPS_2=$COLD_2
    fi

    python scripts/pipeline/train_phase2_finetuning.py \
        finetune=base_64x64 \
        finetune.tokenizer_ckpt=$TOK_CKPT \
        finetune.dynamics_ckpt=$DYN_CKPT \
        "data.data_dirs=[$DATA_DIRS]" \
        "data.frame_dirs=[$FRAME_DIRS]" \
        data.img_size=64 \
        trainer.devices=$DEVICES \
        trainer.max_steps=$CUR_STEPS_2 \
        data.batch_size_dynamics=$BATCH_FT \
        data.num_workers=$NUM_WORKERS \
        checkpoint.dirpath=$FT_CKPT_DIR \
        wandb.name="c${CYCLE}_ft_${RUN_TAG}" \
        "${RESUME_FT_ARG[@]}" \
        "${DATA_TASKS_ARG[@]}"

    echo "[Phase 3] Imagination training (PMPO)..."
    RESUME_AGENT_ARG=()
    if [[ -f "$AGENT_CKPT" ]]; then
        RESUME_AGENT_ARG=("resume=$AGENT_CKPT")
        if [[ "$CYCLE" -gt 0 ]]; then
            CUR_STEPS_3=$(( COLD_3 + CYCLE * STEPS_3 ))
        else
            CUR_STEPS_3=$COLD_3
        fi
    else
        CUR_STEPS_3=$COLD_3
    fi

    python scripts/pipeline/train_phase3_imagination.py \
        agent=base_64x64 \
        agent.finetune_ckpt=$FT_CKPT \
        "data.data_dirs=[$DATA_DIRS]" \
        "data.frame_dirs=[$FRAME_DIRS]" \
        data.img_size=64 \
        trainer.devices=$DEVICES \
        trainer.strategy=$AGENT_DDP_STRATEGY \
        trainer.max_steps=$CUR_STEPS_3 \
        data.batch_size_dynamics=$IMAG_BATCH \
        data.num_workers=$NUM_WORKERS \
        agent.imagination_batch_size=$IMAG_BATCH \
        checkpoint.dirpath=$AGENT_CKPT_DIR \
        wandb.name="c${CYCLE}_agent_${RUN_TAG}" \
        "${RESUME_AGENT_ARG[@]}" \
        "${DATA_TASKS_ARG[@]}"

    [[ -f $AGENT_CKPT ]] && cp $AGENT_CKPT "$AGENT_CYCLE_CKPT_DIR/cycle${CYCLE}.ckpt"
    echo ""
    if [[ -f "$AGENT_CYCLE_CKPT_DIR/cycle${CYCLE}.ckpt" ]]; then
        echo "✓ Ciclo $CYCLE completo → $AGENT_CYCLE_CKPT_DIR/cycle${CYCLE}.ckpt"
    else
        echo "✓ Ciclo $CYCLE completado (sin snapshot de agent)"
    fi
done

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUACIÓN FINAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [[ -f "$AGENT_CKPT" ]]; then
    echo ""
    echo "════════════════════════════════════════"
    echo " Evaluación — Generando videos con política entrenada"
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
        ++collect.min_frames_per_task=5000 \
        collect.out_data_dir=$EVAL_DATA \
        collect.out_frames_dir=$EVAL_FRAMES \
        collect.out_videos_dir=$EVAL_VIDEOS \
        collect.img_size=64 \
        ++collect.save_preview_video=true \
        ++collect.preview_video_backend=torchrl \
        ++collect.preview_video_fps=30 \
        ++collect.preview_video_max_frames=1000 \
        ++collect.wandb_project=dreamer4 \
        ++collect.wandb_run_name="eval_${RUN_TAG}" \
        "${TASKS_ARG[@]}"

    echo ""
    echo "✓ Evaluación completada — Videos guardados en $EVAL_VIDEOS"
else
    echo ""
    echo "⚠ No se encontró checkpoint del agente — evaluación saltada."
fi

echo ""
echo "════════════════════════════════════════"
echo " Pipeline 64×64 completo — $K ciclos OK"
echo "════════════════════════════════════════"
