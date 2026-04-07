#!/bin/bash
#SBATCH --job-name=bench_2gpu_ckpt
#SBATCH --output=/workspace1/gofuentes/dreamer4/docs/bench_2gpu_ckpt_%j.log
#SBATCH --error=/workspace1/gofuentes/dreamer4/docs/bench_2gpu_ckpt_%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --ntasks=1

cd /workspace1/gofuentes/dreamer4
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

echo "Corriendo Benchmark escalado con 2 procesos independientes (1 GPU cada uno)..."

CKPT="/workspace1/gofuentes/dreamer4/logs/agent_ckpts/last.ckpt"

# Lanzamos el Proceso 0 anclado a la GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/benchmarks/generate_scale_bench.py \
    --gpus 1 --ckpt "$CKPT" \
    --out "docs/benchmark_scale_indep_gpu0.md" \
    > docs/bench_indep_gpu0.log 2>&1 &
PID1=$!

# Lanzamos el Proceso 1 anclado a la GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/benchmarks/generate_scale_bench.py \
    --gpus 1 --ckpt "$CKPT" \
    --out "docs/benchmark_scale_indep_gpu1.md" \
    > docs/bench_indep_gpu1.log 2>&1 &
PID2=$!

echo "Lanzados procesos independientes PID $PID1 (GPU0) y $PID2 (GPU1)."
echo "Esperando a que terminen ambos..."
wait $PID1
wait $PID2
echo "Completado."
