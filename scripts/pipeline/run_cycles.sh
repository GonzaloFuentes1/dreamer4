#!/bin/bash
# run_cycles.sh — Pipeline activo dreamer4 (resolución 64×64 o 128×128)
#
# Uso:
#   RES=64  srun --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=64 --mem=128G --time=24:00:00 bash scripts/pipeline/run_cycles.sh
#   RES=128 srun --nodes=1 --ntasks=1 --gpus=3 --cpus-per-task=48 --mem=256G --time=72:00:00 bash scripts/pipeline/run_cycles.sh
#
# Variables de entorno principales (todas opcionales):
#   RES=64|128                     resolución (default: 64)
#   K=30                           número de ciclos
#   RUN_TAG=mi_experimento         nombre del run
#   TASKS=walker-run,cheetah-run   tareas separadas por coma
#   START_FROM_DYNAMICS_CYCLE0=true  usar tokenizer y datos de otro run en ciclo 0
#   SOURCE_RUN_TAG=active_64x64_v2   run fuente (requiere START_FROM_DYNAMICS_CYCLE0=true)
#   CYCLE0_PRETRAINED_DATA=./runs/…/dataset/cycle0
#   SEED_TOKENIZER_CKPT=./runs/…/tokenizer/last.ckpt
#   COLD_1A, STEPS_1A, COLD_1B, STEPS_1B, COLD_2, STEPS_2, COLD_3, STEPS_3
#   BATCH_TOK, BATCH_DYN, BATCH_FT, IMAG_BATCH, NUM_WORKERS, DEVICES

set -e
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
source "$(dirname "$0")/_pipeline_lib.sh"
pipeline_setup "$ROOT_DIR"
cd "$ROOT_DIR"

# ── Resolución ────────────────────────────────────────────────────────────────
RES=${RES:-64}
case "$RES" in
  64)
    CFG_SUFFIX=base_64x64
    IMG_SIZE=64
    BATCH_TOK=${BATCH_TOK:-48};  BATCH_DYN=${BATCH_DYN:-16}
    BATCH_FT=${BATCH_FT:-16};    IMAG_BATCH=${IMAG_BATCH:-24}
    COLD_1A=${COLD_1A:-20000};   STEPS_1A=${STEPS_1A:-2000}
    COLD_1B=${COLD_1B:-20000};   STEPS_1B=${STEPS_1B:-2000}
    COLD_2=${COLD_2:-3000};      STEPS_2=${STEPS_2:-2000}
    COLD_3=${COLD_3:-10000};     STEPS_3=${STEPS_3:-4000}
    RUN_TAG="${RUN_TAG:-active_64x64}"
    ;;
  128)
    CFG_SUFFIX=base_128x128
    IMG_SIZE=128
    BATCH_TOK=${BATCH_TOK:-16};  BATCH_DYN=${BATCH_DYN:-16}
    BATCH_FT=${BATCH_FT:-16};    IMAG_BATCH=${IMAG_BATCH:-24}
    COLD_1A=${COLD_1A:-20000};   STEPS_1A=${STEPS_1A:-1500}
    COLD_1B=${COLD_1B:-20000};   STEPS_1B=${STEPS_1B:-1000}
    COLD_2=${COLD_2:-1000};      STEPS_2=${STEPS_2:-1000}
    COLD_3=${COLD_3:-2000};      STEPS_3=${STEPS_3:-2000}
    RUN_TAG="${RUN_TAG:-active_128x128}"
    ;;
  *)
    echo "Error: RES debe ser 64 o 128"; exit 1
    ;;
esac

# ── Configuración ─────────────────────────────────────────────────────────────
K=${K:-30}
detect_devices
TASKS="${TASKS:-walker-run,cheetah-run,hopper-hop,finger-spin,reacher-hard}"

AGENT_N_EPISODES=${AGENT_N_EPISODES:-32}   # múltiplo de COLLECT_NUM_ENVS para evitar overshoot
AGENT_EPISODE_LEN=${AGENT_EPISODE_LEN:-1000}
COLLECT_MIN_FRAMES_PER_TASK=${COLLECT_MIN_FRAMES_PER_TASK:-$((AGENT_N_EPISODES * AGENT_EPISODE_LEN))}
COLLECT_NUM_ENVS=${COLLECT_NUM_ENVS:-16}
COLLECT_SAVE_PREVIEW_VIDEO=${COLLECT_SAVE_PREVIEW_VIDEO:-true}
COLLECT_PREVIEW_VIDEO_BACKEND=${COLLECT_PREVIEW_VIDEO_BACKEND:-torchrl}
COLLECT_PREVIEW_VIDEO_FPS=${COLLECT_PREVIEW_VIDEO_FPS:-30}
COLLECT_PREVIEW_VIDEO_MAX_FRAMES=${COLLECT_PREVIEW_VIDEO_MAX_FRAMES:-$AGENT_EPISODE_LEN}

NUM_WORKERS=${NUM_WORKERS:-4}
AGENT_DDP_STRATEGY=${AGENT_DDP_STRATEGY:-ddp_find_unused_parameters_true}

# Hiperparámetros del tokenizer (sobreescribibles por env var)
TOKENIZER_MAE_P_MAX=${TOKENIZER_MAE_P_MAX:-0.75}
TOKENIZER_D_MODEL=${TOKENIZER_D_MODEL:-256}
TOKENIZER_N_HEADS=${TOKENIZER_N_HEADS:-4}
TOKENIZER_DEPTH=${TOKENIZER_DEPTH:-8}
TOKENIZER_N_LATENTS=${TOKENIZER_N_LATENTS:-16}
TOKENIZER_D_BOTTLENECK=${TOKENIZER_D_BOTTLENECK:-32}
TOKENIZER_TIME_EVERY=${TOKENIZER_TIME_EVERY:-1}
TOKENIZER_LATENTS_ONLY_TIME=${TOKENIZER_LATENTS_ONLY_TIME:-true}
TOKENIZER_SCALE_POS_EMBEDS=${TOKENIZER_SCALE_POS_EMBEDS:-false}

# ── Rutas ─────────────────────────────────────────────────────────────────────
RUN_ROOT="./runs/${RUN_TAG}"
RUN_DATA_ROOT="${RUN_ROOT}/dataset"
TOK_CKPT_DIR="${RUN_ROOT}/tokenizer";    TOK_CKPT="${TOK_CKPT_DIR}/last.ckpt"
DYN_CKPT_DIR="${RUN_ROOT}/dynamics";     DYN_CKPT="${DYN_CKPT_DIR}/last.ckpt"
FT_CKPT_DIR="${RUN_ROOT}/finetune";      FT_CKPT="${FT_CKPT_DIR}/last.ckpt"
AGENT_CKPT_DIR="${RUN_ROOT}/agent";      AGENT_CKPT="${AGENT_CKPT_DIR}/last.ckpt"
AGENT_CYCLE_CKPT_DIR="${RUN_ROOT}/cycles"
mkdir -p "$TOK_CKPT_DIR" "$DYN_CKPT_DIR" "$FT_CKPT_DIR" \
         "$AGENT_CKPT_DIR" "$AGENT_CYCLE_CKPT_DIR" "$RUN_DATA_ROOT"

echo "[run_cycles] RES=${RES}x${RES}  devices=$DEVICES  run_tag=$RUN_TAG"

# ── Ciclo 0: opción de arrancar desde datos y tokenizer preexistentes ─────────
START_FROM_DYNAMICS_CYCLE0=${START_FROM_DYNAMICS_CYCLE0:-false}
SOURCE_RUN_TAG=${SOURCE_RUN_TAG:-active_${RES}x${RES}_v2}
SOURCE_RUN_ROOT="./runs/${SOURCE_RUN_TAG}"

if [[ "$START_FROM_DYNAMICS_CYCLE0" == "true" ]]; then
    CYCLE0_PRETRAINED_DATA=${CYCLE0_PRETRAINED_DATA:-${SOURCE_RUN_ROOT}/dataset/cycle0}
    SEED_TOKENIZER_CKPT=${SEED_TOKENIZER_CKPT:-${SOURCE_RUN_ROOT}/tokenizer/last.ckpt}
    echo "[run_cycles] start_from_dynamics_cycle0=true  source=$SOURCE_RUN_ROOT"

    if [[ ! -d "$CYCLE0_PRETRAINED_DATA/demos" ]] || [[ ! -d "$CYCLE0_PRETRAINED_DATA/frames" ]]; then
        echo "[run_cycles] error: CYCLE0_PRETRAINED_DATA inválido: $CYCLE0_PRETRAINED_DATA"; exit 1
    fi
    if [[ ! -f "$SEED_TOKENIZER_CKPT" ]]; then
        echo "[run_cycles] error: tokenizer seed no encontrado: $SEED_TOKENIZER_CKPT"; exit 1
    fi
    if [[ ! -f "$TOK_CKPT" ]]; then
        cp "$SEED_TOKENIZER_CKPT" "$TOK_CKPT"
        echo "[run_cycles] tokenizer seed copiado → $TOK_CKPT"
    fi
fi

# ── Bucle principal ───────────────────────────────────────────────────────────
for CYCLE in $(seq 0 $((K - 1))); do
    echo ""; echo "════════════════════════════════════════"
    echo " Ciclo $CYCLE / $((K-1))  [${RES}x${RES}]"
    echo "════════════════════════════════════════"

    OUT_DATA="${RUN_DATA_ROOT}/cycle${CYCLE}/demos"
    OUT_FRAMES="${RUN_DATA_ROOT}/cycle${CYCLE}/frames"
    OUT_VIDEOS="${RUN_DATA_ROOT}/cycle${CYCLE}/videos"

    # Phase 0 — colección
    if [[ "$CYCLE" -eq 0 && "$START_FROM_DYNAMICS_CYCLE0" == "true" ]]; then
        echo "[Phase 0] Usando datos precolectados: $CYCLE0_PRETRAINED_DATA"
        OUT_DATA="${CYCLE0_PRETRAINED_DATA}/demos"
        OUT_FRAMES="${CYCLE0_PRETRAINED_DATA}/frames"
    else
        COLLECT_POLICY=$([[ "$CYCLE" -eq 0 ]] && echo "random" || echo "agent")
        echo "[Phase 0] Recolectando — política=$COLLECT_POLICY ciclo=$CYCLE"
        COLLECT_ARGS=(
            collect.policy=$COLLECT_POLICY
            collect.num_envs_per_task=$COLLECT_NUM_ENVS
            collect.n_episodes_per_task=$AGENT_N_EPISODES
            ++collect.episode_len=$AGENT_EPISODE_LEN
            ++collect.min_frames_per_task=$COLLECT_MIN_FRAMES_PER_TASK
            collect.out_data_dir=$OUT_DATA
            collect.out_frames_dir=$OUT_FRAMES
            collect.out_videos_dir=$OUT_VIDEOS
            collect.img_size=$IMG_SIZE
            ++collect.save_preview_video=$COLLECT_SAVE_PREVIEW_VIDEO
            ++collect.preview_video_backend=$COLLECT_PREVIEW_VIDEO_BACKEND
            ++collect.preview_video_fps=$COLLECT_PREVIEW_VIDEO_FPS
            ++collect.preview_video_max_frames=$COLLECT_PREVIEW_VIDEO_MAX_FRAMES
            "collect.tasks=[$TASKS]"
        )
        [[ "$CYCLE" -gt 0 ]] && COLLECT_ARGS+=(collect.agent_ckpt=$AGENT_CKPT)
        python scripts/pipeline/launch_phase0_dist.py "${COLLECT_ARGS[@]}"
    fi
    append_dirs "$OUT_DATA"

    # Phase 1a — Tokenizer
    if [[ "$CYCLE" -eq 0 && "$START_FROM_DYNAMICS_CYCLE0" == "true" ]]; then
        echo "[Phase 1a] Saltado — usando tokenizer seed"
    else
        echo "[Phase 1a] Entrenando tokenizer..."
        phase_steps $COLD_1A $STEPS_1A "$TOK_CKPT" $CYCLE
        python scripts/pipeline/train_phase1a_tokenizer.py \
            tokenizer=$CFG_SUFFIX \
            tokenizer.d_model=$TOKENIZER_D_MODEL \
            tokenizer.n_heads=$TOKENIZER_N_HEADS \
            tokenizer.depth=$TOKENIZER_DEPTH \
            tokenizer.n_latents=$TOKENIZER_N_LATENTS \
            tokenizer.d_bottleneck=$TOKENIZER_D_BOTTLENECK \
            tokenizer.time_every=$TOKENIZER_TIME_EVERY \
            tokenizer.latents_only_time=$TOKENIZER_LATENTS_ONLY_TIME \
            tokenizer.scale_pos_embeds=$TOKENIZER_SCALE_POS_EMBEDS \
            tokenizer.mae_p_max=$TOKENIZER_MAE_P_MAX \
            "data.data_dirs=[$DATA_DIRS]" \
            trainer.devices=$DEVICES \
            trainer.max_steps=$CUR_STEPS \
            data.batch_size_tokenizer=$BATCH_TOK \
            data.num_workers=$NUM_WORKERS \
            checkpoint.dirpath=$TOK_CKPT_DIR \
            wandb.name="c${CYCLE}_tok_${RUN_TAG}" \
            "data.tasks=[$TASKS]" \
            "${RESUME_ARG[@]}"
    fi

    # Phase 1b — Dynamics
    echo "[Phase 1b] Entrenando dynamics..."
    phase_steps $COLD_1B $STEPS_1B "$DYN_CKPT" $CYCLE
    python scripts/pipeline/train_phase1b_dynamics.py \
        dynamics=$CFG_SUFFIX \
        dynamics.tokenizer_ckpt=$TOK_CKPT \
        "data.data_dirs=[$DATA_DIRS]" \
        data.img_size=$IMG_SIZE \
        trainer.devices=$DEVICES \
        trainer.max_steps=$CUR_STEPS \
        data.batch_size_dynamics=$BATCH_DYN \
        data.num_workers=$NUM_WORKERS \
        checkpoint.dirpath=$DYN_CKPT_DIR \
        wandb.name="c${CYCLE}_dyn_${RUN_TAG}" \
        "data.tasks=[$TASKS]" \
        "${RESUME_ARG[@]}"

    # Phase 2 — Finetuning
    echo "[Phase 2] Finetuning (BC + reward)..."
    phase_steps $COLD_2 $STEPS_2 "$FT_CKPT" $CYCLE
    python scripts/pipeline/train_phase2_finetuning.py \
        finetune=$CFG_SUFFIX \
        finetune.tokenizer_ckpt=$TOK_CKPT \
        finetune.dynamics_ckpt=$DYN_CKPT \
        "data.data_dirs=[$DATA_DIRS]" \
        data.img_size=$IMG_SIZE \
        trainer.devices=$DEVICES \
        trainer.max_steps=$CUR_STEPS \
        data.batch_size_dynamics=$BATCH_FT \
        data.num_workers=$NUM_WORKERS \
        checkpoint.dirpath=$FT_CKPT_DIR \
        wandb.name="c${CYCLE}_ft_${RUN_TAG}" \
        "data.tasks=[$TASKS]" \
        "${RESUME_ARG[@]}"

    # Phase 3 — Imagination (PMPO)
    echo "[Phase 3] Imagination training (PMPO)..."
    phase_steps $COLD_3 $STEPS_3 "$AGENT_CKPT" $CYCLE
    python scripts/pipeline/train_phase3_imagination.py \
        agent=$CFG_SUFFIX \
        agent.finetune_ckpt=$FT_CKPT \
        "data.data_dirs=[$DATA_DIRS]" \
        data.img_size=$IMG_SIZE \
        trainer.devices=$DEVICES \
        trainer.strategy=$AGENT_DDP_STRATEGY \
        trainer.max_steps=$CUR_STEPS \
        data.batch_size_dynamics=$IMAG_BATCH \
        data.num_workers=$NUM_WORKERS \
        agent.imagination_batch_size=$IMAG_BATCH \
        checkpoint.dirpath=$AGENT_CKPT_DIR \
        wandb.name="c${CYCLE}_agent_${RUN_TAG}" \
        "data.tasks=[$TASKS]" \
        "${RESUME_ARG[@]}"

    [[ -f "$AGENT_CKPT" ]] && cp "$AGENT_CKPT" "$AGENT_CYCLE_CKPT_DIR/cycle${CYCLE}.ckpt"
    echo ""
    if [[ -f "$AGENT_CYCLE_CKPT_DIR/cycle${CYCLE}.ckpt" ]]; then
        echo "✓ Ciclo $CYCLE completo → $AGENT_CYCLE_CKPT_DIR/cycle${CYCLE}.ckpt"
    else
        echo "✓ Ciclo $CYCLE completado (sin snapshot de agent)"
    fi
done

run_eval "$AGENT_CKPT" "$RUN_DATA_ROOT" "$IMG_SIZE" "$RUN_TAG" "$COLLECT_NUM_ENVS" "$TASKS"

echo ""; echo "════════════════════════════════════════"
echo " Pipeline ${RES}x${RES} completo — $K ciclos OK"
echo "════════════════════════════════════════"
