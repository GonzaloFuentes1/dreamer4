#!/bin/bash
# run_cycles_128x128.sh — Pipeline activo sin pretraining, resolución 128×128
#
# Uso:
#   srun --gpus=3 --cpus-per-task=48 --mem=256G --time=72:00:00 bash scripts/pipeline/run_cycles_128x128.sh
#
# Configuración:
#   - Sin datos de pretraining: ciclo 0 colecta con política aleatoria
#   - Resolución 128×128 (patch=4, n_latents=16, d_bottleneck=32, packing=2 → n_spatial=8, d_spatial=64, dyn_depth=8)
#   - K=20 ciclos, 5 tasks (walker-run, cheetah-run, hopper-hop, finger-spin, reacher-hard)
#   - 10 episodios/task → 50k frames/ciclo
#   - Optimizado para 2-3× H100 80GB (DEVICES se autodetecta)
#
# Latente: ratio patches:latents = 1024:16 = 64:1 (aligned with nicklashansen/dreamer4)
# Tokenizer: 16+1024=1040 tokens en encoder (patch=4), d_bottleneck=32
# Dinámica: n_spatial=8 (16/2 packing) → 16 tokens por timestep (1 act + 1 sig + 1 step + 8 spatial + 4 reg + 1 agent)
#
# Steps/época con 50k frames/ciclo @ 2 GPUs (ciclo 0):
#   Tokenizer: 50k / (24×2=48) global_batch             ≈ 1042 steps/época
#   Dynamics:  ~50k ventanas / (16×2=32) global          ≈ 1562 steps/época
#   Finetune:  ~50k ventanas / (16×2=32) global          ≈ 1562 steps/época
#   Agent:     usa imaginación pura (no iteraciones de dataset)
#
# Steps COLD (ciclo 0):
#   Tokenizer:  3000 ≈ 2.9 épocas
#   Dynamics:   1000 ≈ 1.0 época corta
#   Finetune:   1000 ≈ 1.0 época corta
#   Agent:      2000 pasos PMPO
#
# Steps WARM (por ciclo):
#   Tokenizer:  1500 ≈ ~1.4 épocas de datos nuevos
#   Dynamics:   1000 ≈ ~1.0 época de datos nuevos
#   Finetune:   1000 ≈ ~1.0 época de datos nuevos
#   Agent:      2000 pasos PMPO/ciclo

set -e
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# Habilitar aceleración GPU por hardware para MuJoCo
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json

cd "$ROOT_DIR"
if [[ -f "$ROOT_DIR/.env" ]]; then source "$ROOT_DIR/.env"; fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURACIÓN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
K=${K:-30}              # ciclos totales (más ciclos, menos steps — datos mejoran con cada ciclo)
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
echo "[run_cycles_128x128] trainer.devices=$DEVICES"

TASKS="${TASKS:-walker-run,cheetah-run,hopper-hop,finger-spin,reacher-hard}"

# Colección — todos los ciclos (ciclo 0: aleatoria, ciclos 1+: con agente)
AGENT_N_EPISODES=10     # 10 eps × 5 tasks × 1000 steps = 50 000 frames/ciclo
AGENT_EPISODE_LEN=1000
COLLECT_MIN_FRAMES_PER_TASK=${COLLECT_MIN_FRAMES_PER_TASK:-$((AGENT_N_EPISODES * AGENT_EPISODE_LEN))}
COLLECT_NUM_ENVS=16
COLLECT_SAVE_PREVIEW_VIDEO=${COLLECT_SAVE_PREVIEW_VIDEO:-true}
COLLECT_PREVIEW_VIDEO_BACKEND=${COLLECT_PREVIEW_VIDEO_BACKEND:-torchrl}
COLLECT_PREVIEW_VIDEO_FPS=${COLLECT_PREVIEW_VIDEO_FPS:-30}
COLLECT_PREVIEW_VIDEO_MAX_FRAMES=${COLLECT_PREVIEW_VIDEO_MAX_FRAMES:-$AGENT_EPISODE_LEN}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BATCH SIZES Y WORKERS — H100 80GB
# Tokenizer  patch=4 @ 128×128 → 1024 patches + 16 latents = 1040 tokens, d=256, depth=8
# Dynamics   n_spatial=8 (16 latents / packing=2), 16 tok/step, T=32, d=512, depth=8
# Finetune   misma backbone de dynamics + policy heads → misma memoria por sample
# Agent/Imag horizon=15 rollouts × ODE steps → el coste escala con horizon, no con tokens
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BATCH_TOK=${BATCH_TOK:-16}      # patch=4 → 1040 tokens; 24 por GPU suele caber en H100 80GB
BATCH_DYN=${BATCH_DYN:-16}      # más conservador para evitar OOM/worker kills en DDP
BATCH_FT=${BATCH_FT:-16}        # misma carga base que dynamics
IMAG_BATCH=${IMAG_BATCH:-24}    # horizonte fijo=15, ODE barato por step; 24 cabe sin problema
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
# Ver comentarios del encabezado para el desglose de épocas.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLD_1A=${COLD_1A:-20000}        # tokenizer cold  (recorte fuerte, pero sin irse al extremo de 1000)
COLD_1B=${COLD_1B:-20000}        # dynamics cold   (evita sobre-entrenar sobre datos random)
COLD_2=${COLD_2:-1000}          # finetuning cold (BC/reward con budget corto al inicio)
COLD_3=${COLD_3:-2000}          # PMPO agent cold (seed inicial, luego se refina por ciclos)

STEPS_1A=${STEPS_1A:-1500}       # tokenizer warm por ciclo  (mantiene adaptación sin comerse el presupuesto)
STEPS_1B=${STEPS_1B:-1000}       # dynamics warm por ciclo   (dataset mejora en cada ciclo)
STEPS_2=${STEPS_2:-1000}         # finetuning warm por ciclo (mismo criterio)
STEPS_3=${STEPS_3:-2000}         # PMPO agent warm por ciclo

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUNS Y CHECKPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASKS_ARG=("collect.tasks=[$TASKS]")
DATA_TASKS_ARG=("data.tasks=[$TASKS]")

RUN_TAG="${RUN_TAG:-active_128x128_v2}"
RUN_ROOT="./runs/${RUN_TAG}"
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

echo "[run_cycles_128x128] run_tag=$RUN_TAG"
echo "[run_cycles_128x128] run_root=$RUN_ROOT"
echo "[run_cycles_128x128] run_data_root=$RUN_DATA_ROOT"

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
    echo " Ciclo $CYCLE / $((K-1))  [active_128x128]"
    echo "════════════════════════════════════════"

    OUT_DATA="${RUN_DATA_ROOT}/cycle${CYCLE}/demos"
    OUT_FRAMES="${RUN_DATA_ROOT}/cycle${CYCLE}/frames"
    OUT_VIDEOS="${RUN_DATA_ROOT}/cycle${CYCLE}/videos"

    if [[ "$CYCLE" -eq 0 ]] && [[ -n "${CYCLE0_PRETRAINED_DATA:-}" ]]; then
        # Ciclo 0 — datos precolectados (skip colección)
        echo "[Phase 0] Usando datos precolectados: $CYCLE0_PRETRAINED_DATA"
        OUT_DATA="${CYCLE0_PRETRAINED_DATA}/demos"
        OUT_FRAMES="${CYCLE0_PRETRAINED_DATA}/frames"
    elif [[ "$CYCLE" -eq 0 ]]; then
        # Ciclo 0 — política aleatoria (sin pretraining)
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
            collect.img_size=128 \
            ++collect.save_preview_video=$COLLECT_SAVE_PREVIEW_VIDEO \
            ++collect.preview_video_backend=$COLLECT_PREVIEW_VIDEO_BACKEND \
            ++collect.preview_video_fps=$COLLECT_PREVIEW_VIDEO_FPS \
            ++collect.preview_video_max_frames=$COLLECT_PREVIEW_VIDEO_MAX_FRAMES \
            "${TASKS_ARG[@]}"
    else
        # Ciclos 1+ — política aprendida
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
            collect.img_size=128 \
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
        tokenizer=base_128x128 \
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

    # ── Phase 1b — Dynamics ───────────────────────────────────────────────
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
        dynamics=base_128x128 \
        dynamics.tokenizer_ckpt=$TOK_CKPT \
        "data.data_dirs=[$DATA_DIRS]" \
        "data.frame_dirs=[$FRAME_DIRS]" \
        data.img_size=128 \
        trainer.devices=$DEVICES \
        trainer.max_steps=$CUR_STEPS_1B \
        data.batch_size_dynamics=$BATCH_DYN \
        data.num_workers=$NUM_WORKERS \
        checkpoint.dirpath=$DYN_CKPT_DIR \
        wandb.name="c${CYCLE}_dyn_${RUN_TAG}" \
        "${RESUME_DYN_ARG[@]}" \
        "${DATA_TASKS_ARG[@]}"

    # ── Phase 2 — Finetuning ──────────────────────────────────────────────
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
        finetune=base_128x128 \
        finetune.tokenizer_ckpt=$TOK_CKPT \
        finetune.dynamics_ckpt=$DYN_CKPT \
        "data.data_dirs=[$DATA_DIRS]" \
        "data.frame_dirs=[$FRAME_DIRS]" \
        data.img_size=128 \
        trainer.devices=$DEVICES \
        trainer.max_steps=$CUR_STEPS_2 \
        data.batch_size_dynamics=$BATCH_FT \
        data.num_workers=$NUM_WORKERS \
        checkpoint.dirpath=$FT_CKPT_DIR \
        wandb.name="c${CYCLE}_ft_${RUN_TAG}" \
        "${RESUME_FT_ARG[@]}" \
        "${DATA_TASKS_ARG[@]}"

    # ── Phase 3 — Imagination Training ───────────────────────────────────
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
        agent=base_128x128 \
        agent.finetune_ckpt=$FT_CKPT \
        "data.data_dirs=[$DATA_DIRS]" \
        "data.frame_dirs=[$FRAME_DIRS]" \
        data.img_size=128 \
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
        collect.img_size=128 \
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
echo " Pipeline 128×128 completo — $K ciclos OK"
echo "════════════════════════════════════════"
