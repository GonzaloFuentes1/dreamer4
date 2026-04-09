#!/bin/bash
# run_cycles_pretrain.sh — Pipeline con datos de pretraining HuggingFace (mixed-large)
#
# Uso:
#   srun --gpus=4 --cpus-per-task=64 --mem=384G --time=72:00:00 bash scripts/pipeline/run_cycles_pretrain.sh
#
# Ciclo 0 usa BASE_DATA_ROOT (datos descargados de HF) — sin colección.
# Ciclos 1+ colectan con la política aprendida.
#
# Variables de entorno principales:
#   RES=64|128, K=10, RUN_TAG, TASKS, BASE_DATA_ROOT
#   COLD_*, STEPS_*, BATCH_TOK, BATCH_DYN, BATCH_FT, IMAG_BATCH, NUM_WORKERS, DEVICES

set -e
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
source "$(dirname "$0")/_pipeline_lib.sh"
pipeline_setup "$ROOT_DIR"
cd "$ROOT_DIR"

# ── Resolución ────────────────────────────────────────────────────────────────
RES=${RES:-64}
case "$RES" in
  64|128) IMG_SIZE=$RES ;;
  *) echo "Error: RES debe ser 64 o 128"; exit 1 ;;
esac
# CFG_SUFFIX puede sobreescribirse con env var (ej. CFG_SUFFIX=large_64x64)
CFG_SUFFIX="${CFG_SUFFIX:-base_${RES}x${RES}}"

# ── Configuración ─────────────────────────────────────────────────────────────
K=${K:-10}
detect_devices
TASKS="${TASKS:-walker-walk,walker-run,cheetah-run,walker-stand,hopper-hop}"

AGENT_N_EPISODES=${AGENT_N_EPISODES:-30}
AGENT_EPISODE_LEN=${AGENT_EPISODE_LEN:-500}   # HF data: frame_skip=2 → 500 agent steps = 1000 physics steps
COLLECT_FRAME_SKIP=${COLLECT_FRAME_SKIP:-2}    # debe coincidir con los datos HF (frame_skip=2)
COLLECT_NUM_ENVS=${COLLECT_NUM_ENVS:-10}
COLLECT_SAVE_PREVIEW_VIDEO=${COLLECT_SAVE_PREVIEW_VIDEO:-true}
COLLECT_PREVIEW_VIDEO_BACKEND=${COLLECT_PREVIEW_VIDEO_BACKEND:-torchrl}
COLLECT_PREVIEW_VIDEO_FPS=${COLLECT_PREVIEW_VIDEO_FPS:-30}
COLLECT_PREVIEW_VIDEO_MAX_FRAMES=${COLLECT_PREVIEW_VIDEO_MAX_FRAMES:-$AGENT_EPISODE_LEN}

BATCH_TOK=${BATCH_TOK:-28};  BATCH_DYN=${BATCH_DYN:-64}
BATCH_FT=${BATCH_FT:-64};    IMAG_BATCH=${IMAG_BATCH:-64}
NUM_WORKERS=${NUM_WORKERS:-4}

COLD_1A=${COLD_1A:-90000};  STEPS_1A=${STEPS_1A:-3000}
COLD_1B=${COLD_1B:-90000};   STEPS_1B=${STEPS_1B:-3000}
COLD_2=${COLD_2:-25000};    STEPS_2=${STEPS_2:-10000}
COLD_3=${COLD_3:-30000};    STEPS_3=${STEPS_3:-10000}

# ── Rutas ─────────────────────────────────────────────────────────────────────
RUN_TAG="${RUN_TAG:-pt_active}"
BASE_DATA_ROOT="${BASE_DATA_ROOT:-./data/cycle0-pretrained}"
RUN_ROOT="./runs/${RUN_TAG}"
RUN_DATA_ROOT="${RUN_ROOT}/dataset"
TOK_CKPT_DIR="${RUN_ROOT}/tokenizer";    TOK_CKPT="${TOK_CKPT_DIR}/last.ckpt"
DYN_CKPT_DIR="${RUN_ROOT}/dynamics";     DYN_CKPT="${DYN_CKPT_DIR}/last.ckpt"
FT_CKPT_DIR="${RUN_ROOT}/finetune";      FT_CKPT="${FT_CKPT_DIR}/last.ckpt"
AGENT_CKPT_DIR="${RUN_ROOT}/agent";      AGENT_CKPT="${AGENT_CKPT_DIR}/last.ckpt"
AGENT_CYCLE_CKPT_DIR="${RUN_ROOT}/cycles"
mkdir -p "$TOK_CKPT_DIR" "$DYN_CKPT_DIR" "$FT_CKPT_DIR" \
         "$AGENT_CKPT_DIR" "$AGENT_CYCLE_CKPT_DIR" "$RUN_DATA_ROOT"

echo "[run_cycles_pretrain] devices=$DEVICES  run_tag=$RUN_TAG  base_data=$BASE_DATA_ROOT"

# ── Bucle principal ───────────────────────────────────────────────────────────
for CYCLE in $(seq 0 $((K - 1))); do
    echo ""; echo "════════════════════════════════════════"
    echo " Ciclo $CYCLE / $((K-1))  [pretrain]"
    echo "════════════════════════════════════════"

    # Phase 0 — ciclo 0 usa datos fijos; ciclos 1+ colectan con el agente
    if [[ "$CYCLE" -eq 0 ]]; then
        OUT_DATA="${BASE_DATA_ROOT}/demos"
        OUT_FRAMES="${BASE_DATA_ROOT}/frames"
        echo "[Phase 0] Usando dataset base en $BASE_DATA_ROOT"
    else
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
            collect.frame_skip=$COLLECT_FRAME_SKIP \
            collect.img_size=$IMG_SIZE \
            collect.out_data_dir=$OUT_DATA \
            collect.out_frames_dir=$OUT_FRAMES \
            collect.out_videos_dir=$OUT_VIDEOS \
            ++collect.save_preview_video=$COLLECT_SAVE_PREVIEW_VIDEO \
            ++collect.preview_video_backend=$COLLECT_PREVIEW_VIDEO_BACKEND \
            ++collect.preview_video_fps=$COLLECT_PREVIEW_VIDEO_FPS \
            ++collect.preview_video_max_frames=$COLLECT_PREVIEW_VIDEO_MAX_FRAMES \
            "collect.tasks=[$TASKS]"
    fi
    append_dirs "$OUT_DATA" "$OUT_FRAMES"

    # Phase 1a — Tokenizer (skip en ciclo 0 si ya existe checkpoint)
    if [[ "$CYCLE" -eq 0 && -f "$TOK_CKPT" ]]; then
        echo "[Phase 1a] Checkpoint existente en ciclo 0 — saltando"
    else
        echo "[Phase 1a] Entrenando tokenizer..."
        phase_steps $COLD_1A $STEPS_1A "$TOK_CKPT" $CYCLE
        python scripts/pipeline/train_phase1a_tokenizer.py \
            tokenizer=$CFG_SUFFIX \
            "data.frame_dirs=[$FRAME_DIRS]" \
            data.img_size=$IMG_SIZE \
            trainer.devices=$DEVICES \
            trainer.max_steps=$CUR_STEPS \
            data.batch_size_tokenizer=$BATCH_TOK \
            data.num_workers=$NUM_WORKERS \
            checkpoint.dirpath=$TOK_CKPT_DIR \
            wandb.name="c${CYCLE}_tok_${RUN_TAG}" \
            "data.tasks=[$TASKS]" \
            "${RESUME_ARG[@]}"
    fi

    # Phase 1b — Dynamics (skip en ciclo 0 si ya existe checkpoint)
    if [[ "$CYCLE" -eq 0 && -f "$DYN_CKPT" ]]; then
        echo "[Phase 1b] Checkpoint existente en ciclo 0 — saltando"
    else
        echo "[Phase 1b] Entrenando dynamics..."
        phase_steps $COLD_1B $STEPS_1B "$DYN_CKPT" $CYCLE
        python scripts/pipeline/train_phase1b_dynamics.py \
            dynamics=$CFG_SUFFIX \
            dynamics.tokenizer_ckpt=$TOK_CKPT \
            "data.data_dirs=[$DATA_DIRS]" \
            "data.frame_dirs=[$FRAME_DIRS]" \
            data.img_size=$IMG_SIZE \
            trainer.devices=$DEVICES \
            trainer.max_steps=$CUR_STEPS \
            data.batch_size_dynamics=$BATCH_DYN \
            data.num_workers=$NUM_WORKERS \
            checkpoint.dirpath=$DYN_CKPT_DIR \
            wandb.name="c${CYCLE}_dyn_${RUN_TAG}" \
            "data.tasks=[$TASKS]" \
            "${RESUME_ARG[@]}"
    fi

    # Phase 2 — Finetuning (skip en ciclo 0 si ya existe checkpoint)
    if [[ "$CYCLE" -eq 0 && -f "$FT_CKPT" ]]; then
        echo "[Phase 2] Checkpoint existente en ciclo 0 — saltando"
    else
        echo "[Phase 2] Finetuning (BC + reward)..."
        phase_steps $COLD_2 $STEPS_2 "$FT_CKPT" $CYCLE
        python scripts/pipeline/train_phase2_finetuning.py \
            finetune=$CFG_SUFFIX \
            finetune.tokenizer_ckpt=$TOK_CKPT \
            finetune.dynamics_ckpt=$DYN_CKPT \
            "data.data_dirs=[$DATA_DIRS]" \
            "data.frame_dirs=[$FRAME_DIRS]" \
            data.img_size=$IMG_SIZE \
            trainer.devices=$DEVICES \
            trainer.max_steps=$CUR_STEPS \
            data.batch_size_dynamics=$BATCH_FT \
            data.num_workers=$NUM_WORKERS \
            checkpoint.dirpath=$FT_CKPT_DIR \
            wandb.name="c${CYCLE}_ft_${RUN_TAG}" \
            "data.tasks=[$TASKS]" \
            "${RESUME_ARG[@]}"
    fi

    # Phase 3 — Imagination Training (skip en ciclo 0 si ya existe checkpoint)
    if [[ "$CYCLE" -eq 0 && -f "$AGENT_CKPT" ]]; then
        echo "[Phase 3] Checkpoint existente en ciclo 0 — saltando"
    else
        echo "[Phase 3] Imagination training (PMPO)..."
        phase_steps $COLD_3 $STEPS_3 "$AGENT_CKPT" $CYCLE
        python scripts/pipeline/train_phase3_imagination.py \
            agent=$CFG_SUFFIX \
            agent.finetune_ckpt=$FT_CKPT \
            "data.data_dirs=[$DATA_DIRS]" \
            "data.frame_dirs=[$FRAME_DIRS]" \
            data.img_size=$IMG_SIZE \
            trainer.devices=$DEVICES \
            trainer.max_steps=$CUR_STEPS \
            data.batch_size_dynamics=$IMAG_BATCH \
            data.num_workers=$NUM_WORKERS \
            agent.imagination_batch_size=$IMAG_BATCH \
            checkpoint.dirpath=$AGENT_CKPT_DIR \
            wandb.name="c${CYCLE}_agent_${RUN_TAG}" \
            "data.tasks=[$TASKS]" \
            "${RESUME_ARG[@]}"
    fi

    [[ -f "$AGENT_CKPT" ]] && cp "$AGENT_CKPT" "$AGENT_CYCLE_CKPT_DIR/cycle${CYCLE}.ckpt"
    echo ""
    if [[ -f "$AGENT_CYCLE_CKPT_DIR/cycle${CYCLE}.ckpt" ]]; then
        echo "✓ Ciclo $CYCLE completo → $AGENT_CYCLE_CKPT_DIR/cycle${CYCLE}.ckpt"
    else
        echo "✓ Ciclo $CYCLE completado (sin snapshot de agent)"
    fi
done

run_eval "$AGENT_CKPT" "$RUN_DATA_ROOT" "$IMG_SIZE" "$RUN_TAG" "$COLLECT_NUM_ENVS" "$TASKS" "$COLLECT_FRAME_SKIP" "$AGENT_EPISODE_LEN"

echo ""; echo "════════════════════════════════════════"
echo " Pipeline pretrain completo — $K ciclos OK"
echo "════════════════════════════════════════"
