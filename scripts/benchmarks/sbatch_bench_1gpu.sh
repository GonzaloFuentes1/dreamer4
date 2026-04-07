#!/bin/bash
#SBATCH --job-name=bench_1gpu_ckpt
#SBATCH --output=/workspace1/gofuentes/dreamer4/docs/bench_1gpu_ckpt_%j.log
#SBATCH --error=/workspace1/gofuentes/dreamer4/docs/bench_1gpu_ckpt_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --ntasks=1

cd /workspace1/gofuentes/dreamer4
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

echo "Corriendo Benchmark escalado (1 GPU + Ckpt)..."
python scripts/benchmarks/generate_scale_bench.py --gpus 1 --ckpt "/workspace1/gofuentes/dreamer4/logs/agent_ckpts/last.ckpt"
