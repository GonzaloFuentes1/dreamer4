#!/bin/bash
# run_cycles_pretrain.sh — Ciclo recursivo usando datos pretrain de HuggingFace (mixed-large)
#
# Uso:
#   srun --gpus=2 --cpus-per-task=16 --mem=128G --time=72:00:00 bash run_cycles_pretrain.sh
#
# Diferencias vs run_cycles.sh:
#   - Ciclo 0 usa ./data/pretrain/cycle0/ (mixed-large HF) en lugar de colección random
#   - Ciclos 1+ siguen colectando con la política entrenada
#   - 3 ciclos, 2 GPUs, logs separados (logs_pretrain/)
#   - Más steps en 1a/1b ciclo 0 por la mayor cantidad de datos (~1.8M frames, 18 tasks)

set -e
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
if [[ -f "$ROOT_DIR/.env" ]]; then source "$ROOT_DIR/.env"; fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURACIÓN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
K=3                    # ciclos totales
DEVICES=4              # GPUs

# Tasks disponibles en el pretrain (las 18 que dm_control soporta)
TASKS="walker-stand,walker-walk,walker-run,cheetah-run,hopper-stand,hopper-hop,reacher-easy,reacher-hard,cartpole-swingup,cartpole-balance,cartpole-swingup-sparse,cartpole-balance-sparse,cup-catch,finger-spin,finger-turn-easy,finger-turn-hard,acrobot-swingup,pendulum-swingup"

# Colección ciclos > 0 (con política aprendida)
AGENT_N_EPISODES=20
AGENT_EPISODE_LEN=500

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEPS POR FASE
#
#  Datos pretrain ciclo 0: 18 tasks × 200 eps × ~500 steps ≈ 1.8M frames
#    1 época tokenizer  = 1.8M / (24×2) ≈ 37,500 steps
#    1 época dynamics   = 1.8M / (64×2) ≈ 14,000 steps
#
#  Cold (ciclo 0): más steps por mayor volumen de datos
#  Warm (ciclos 1-2): menos steps, resume desde checkpoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLD_1A=30_000        # tokenizer  ~0.8 épocas cold  (datos pretrain son densos)
COLD_1B=10_000        # dynamics   ~0.7 épocas cold
COLD_2=5_000          # finetuning ~0.4 épocas cold
COLD_3=5_000          # agente PMPO

STEPS_1A=10_000       # tokenizer  warm
STEPS_1B=5_000        # dynamics   warm
STEPS_2=5_000         # finetuning warm
STEPS_3=5_000         # agente PMPO warm

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BATCH SIZES  (2×H100)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BATCH_TOK=24          # frames por GPU — tokenizer
BATCH_DYN=64          # seqs  por GPU  — dynamics/finetune/agente
NUM_WORKERS=6         # workers DataLoader por GPU (2×6=12 total)
IMAG_BATCH=96         # rollouts imaginación

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECKPOINTS  (carpetas separadas de run_cycles.sh)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOK_CKPT=./logs_pretrain/tokenizer_ckpts/last.ckpt
DYN_CKPT=./logs_pretrain/dynamics_ckpts/last.ckpt
FT_CKPT=./logs_pretrain/finetune_ckpts/last.ckpt
AGENT_CKPT=./logs_pretrain/agent_ckpts/last.ckpt

TASKS_ARG=("collect.tasks=[$TASKS]")
DATA_TASKS_ARG=("data.tasks=[$TASKS]")

DATETAG=$(date +%m%d_%H%M)
RUN_TAG="pt_${DATETAG}"

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
        # Ciclo 0 — usar datos pretrain descargados de HuggingFace
        OUT_DATA="./data/pretrain/cycle0/demos"
        OUT_FRAMES="./data/pretrain/cycle0/frames"
        echo "[Phase 0] Usando datos pretrain HF en $OUT_DATA — sin colección."
    else
        # Ciclos 1+ — colectar con la política aprendida
        OUT_DATA="./data/pretrain/cycle${CYCLE}/demos"
        OUT_FRAMES="./data/pretrain/cycle${CYCLE}/frames"
        echo "[Phase 0] Recolectando con política agente (ciclo $CYCLE)..."
        python collect_phase0_data.py \
            collect.policy=agent \
            collect.agent_ckpt=$AGENT_CKPT \
            collect.n_episodes_per_task=$AGENT_N_EPISODES \
            collect.episode_len=$AGENT_EPISODE_LEN \
            collect.out_data_dir=$OUT_DATA \
            collect.out_frames_dir=$OUT_FRAMES \
            "${TASKS_ARG[@]}"
    fi

    append_dirs "$OUT_DATA" "$OUT_FRAMES"

    # ── Phase 1a — Tokenizer ──────────────────────────────────────────────
    echo "[Phase 1a] Entrenando tokenizer..."
    RESUME_TOK_ARG=()
    if [[ "$CYCLE" -gt 0 ]] && [[ -f "$TOK_CKPT" ]]; then
        RESUME_TOK_ARG=("resume=$TOK_CKPT")
        CUR_STEPS_1A=$STEPS_1A
    else
        CUR_STEPS_1A=$COLD_1A
    fi
    python train_phase1a_tokenizer.py \
        "data.frame_dirs=[$FRAME_DIRS]" \
        trainer.devices=$DEVICES \
        trainer.max_steps=$CUR_STEPS_1A \
        data.batch_size_tokenizer=$BATCH_TOK \
        data.num_workers=$NUM_WORKERS \
        checkpoint.dirpath=./logs_pretrain/tokenizer_ckpts \
        wandb.name="c${CYCLE}_tok_${RUN_TAG}" \
        "${RESUME_TOK_ARG[@]}" \
        "${DATA_TASKS_ARG[@]}"

    # ── Phase 1b — Dynamics ───────────────────────────────────────────────
    echo "[Phase 1b] Entrenando dynamics..."
    RESUME_DYN_ARG=()
    if [[ "$CYCLE" -gt 0 ]] && [[ -f "$DYN_CKPT" ]]; then
        RESUME_DYN_ARG=("resume=$DYN_CKPT")
        CUR_STEPS_1B=$STEPS_1B
    else
        CUR_STEPS_1B=$COLD_1B
    fi
    python train_phase1b_dynamics.py \
        dynamics.tokenizer_ckpt=$TOK_CKPT \
        "data.data_dirs=[$DATA_DIRS]" \
        "data.frame_dirs=[$FRAME_DIRS]" \
        trainer.devices=$DEVICES \
        trainer.max_steps=$CUR_STEPS_1B \
        data.batch_size_dynamics=$BATCH_DYN \
        data.num_workers=$NUM_WORKERS \
        checkpoint.dirpath=./logs_pretrain/dynamics_ckpts \
        wandb.name="c${CYCLE}_dyn_${RUN_TAG}" \
        "${RESUME_DYN_ARG[@]}" \
        "${DATA_TASKS_ARG[@]}"

    # ── Phase 2 — Finetuning ──────────────────────────────────────────────
    echo "[Phase 2] Finetuning (BC + reward)..."
    CUR_STEPS_2=$([[ "$CYCLE" -eq 0 ]] && echo $COLD_2 || echo $STEPS_2)
    python train_phase2_finetuning.py \
        finetune.tokenizer_ckpt=$TOK_CKPT \
        finetune.dynamics_ckpt=$DYN_CKPT \
        "data.data_dirs=[$DATA_DIRS]" \
        "data.frame_dirs=[$FRAME_DIRS]" \
        trainer.devices=$DEVICES \
        trainer.max_steps=$CUR_STEPS_2 \
        data.batch_size_dynamics=$BATCH_DYN \
        data.num_workers=$NUM_WORKERS \
        checkpoint.dirpath=./logs_pretrain/finetune_ckpts \
        wandb.name="c${CYCLE}_ft_${RUN_TAG}" \
        "${DATA_TASKS_ARG[@]}"

    # ── Phase 3 — Imagination Training ───────────────────────────────────
    echo "[Phase 3] Imagination training (PMPO)..."
    CUR_STEPS_3=$([[ "$CYCLE" -eq 0 ]] && echo $COLD_3 || echo $STEPS_3)
    python train_phase3_imagination.py \
        agent.finetune_ckpt=$FT_CKPT \
        "data.data_dirs=[$DATA_DIRS]" \
        "data.frame_dirs=[$FRAME_DIRS]" \
        trainer.devices=$DEVICES \
        trainer.max_steps=$CUR_STEPS_3 \
        data.batch_size_dynamics=$BATCH_DYN \
        data.num_workers=$NUM_WORKERS \
        agent.imagination_batch_size=$IMAG_BATCH \
        checkpoint.dirpath=./logs_pretrain/agent_ckpts \
        wandb.name="c${CYCLE}_agent_${RUN_TAG}" \
        "${DATA_TASKS_ARG[@]}"

    cp $AGENT_CKPT ./logs_pretrain/agent_ckpts/cycle${CYCLE}.ckpt
    echo ""
    echo "✓ Ciclo $CYCLE completo → ./logs_pretrain/agent_ckpts/cycle${CYCLE}.ckpt"
done

echo ""
echo "════════════════════════════════════════"
echo " Pipeline pretrain completo — $K ciclos OK"
echo "════════════════════════════════════════"
