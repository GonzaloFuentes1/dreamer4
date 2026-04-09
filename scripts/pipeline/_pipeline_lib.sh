#!/bin/bash
# _pipeline_lib.sh — Funciones compartidas para los scripts run_cycles*.sh
# Fuente: source "$(dirname "$0")/_pipeline_lib.sh"

# ── Entorno + venv ─────────────────────────────────────────────────────────────
pipeline_setup() {
    local root="$1"
    export PYTHONPATH="$root:${PYTHONPATH:-}"
    export MUJOCO_GL="egl"
    export PYOPENGL_PLATFORM="egl"
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
    [[ -f "$root/.env" ]] && source "$root/.env"

    local venv="${VENV_PATH:-$root/.venv}"
    if [[ -d "$venv" ]]; then
        # shellcheck disable=SC1091
        source "$venv/bin/activate"
        echo "[pipeline] venv=$venv"
    fi
}

# ── Autodetectar número de GPUs → DEVICES ─────────────────────────────────────
detect_devices() {
    if [[ -n "${DEVICES:-}" ]]; then
        :
    elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
        DEVICES="$SLURM_GPUS_ON_NODE"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -ra _ids <<< "$CUDA_VISIBLE_DEVICES"
        DEVICES="${#_ids[@]}"
    else
        DEVICES=4
    fi
}

# ── Acumular directorios de datos a lo largo de los ciclos ────────────────────
# Uso: append_dirs "$OUT_DATA"
# Modifica global: DATA_DIRS
DATA_DIRS=""
append_dirs() {
    if [[ -z "$DATA_DIRS" ]]; then
        DATA_DIRS="$1"
    else
        DATA_DIRS="${DATA_DIRS},$1"
    fi
}

# ── Calcular steps y resume arg con esquema cold/warm ─────────────────────────
# Uso: phase_steps <cold> <warm> <ckpt_path> <cycle>
# Escribe globales: CUR_STEPS (int), RESUME_ARG (array)
phase_steps() {
    local cold=$1 warm=$2 ckpt=$3 cycle=$4
    RESUME_ARG=()
    CUR_STEPS=$cold
    if [[ -f "$ckpt" ]]; then
        RESUME_ARG=("resume=$ckpt")
        [[ "$cycle" -gt 0 ]] && CUR_STEPS=$(( cold + cycle * warm ))
    fi
}

# ── Evaluación final ───────────────────────────────────────────────────────────
# Uso: run_eval <agent_ckpt> <run_data_root> <img_size> <run_tag> <num_envs> <tasks>
run_eval() {
    local agent_ckpt="$1" run_data_root="$2" img_size="$3" run_tag="$4"
    local num_envs="${5:-16}" tasks="$6"

    if [[ ! -f "$agent_ckpt" ]]; then
        echo ""; echo "⚠ No se encontró checkpoint del agente — evaluación saltada."
        return
    fi
    echo ""; echo "════════════════════════════════════════"
    echo " Evaluación final"; echo "════════════════════════════════════════"

    local eval_root="${run_data_root}/eval"
    mkdir -p "$eval_root/videos"

    local frame_skip="${7:-1}"
    local episode_len="${8:-1000}"

    python scripts/pipeline/launch_phase0_dist.py \
        collect.policy=agent \
        collect.num_envs_per_task="$num_envs" \
        collect.agent_ckpt="$agent_ckpt" \
        collect.n_episodes_per_task=5 \
        ++collect.episode_len="$episode_len" \
        ++collect.min_frames_per_task=$(( 5 * episode_len )) \
        collect.frame_skip="$frame_skip" \
        collect.out_data_dir="${eval_root}/demos" \
        collect.out_frames_dir="${eval_root}/frames" \
        collect.out_videos_dir="${eval_root}/videos" \
        collect.img_size="$img_size" \
        ++collect.save_preview_video=true \
        ++collect.preview_video_backend=torchrl \
        ++collect.preview_video_fps=30 \
        ++collect.preview_video_max_frames="$episode_len" \
        ++collect.wandb_project=dreamer4 \
        ++collect.wandb_run_name="eval_${run_tag}" \
        "collect.tasks=[$tasks]"

    echo ""; echo "✓ Evaluación completada — Videos en ${eval_root}/videos"
}
