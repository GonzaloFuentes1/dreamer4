#!/bin/bash
tasks=("dmc_cheetah_run" "atari_pong" "dmc_walker_walk")
envs=(16 64 128)
total_steps=50000

echo "| Entorno | Instancias Paralelas (num_envs) | Pasos Totales | Throughput (SPS) |" > /workspace1/gofuentes/dreamer4/benchmark_results.md
echo "|---|---|---|---|" >> /workspace1/gofuentes/dreamer4/benchmark_results.md

for t in "${tasks[@]}"; do
    for e in "${envs[@]}"; do
        echo "Lanzando test: $t con $e workers..."
        
        output=$(srun --gpus=2 --mem=64G bash -c "source /workspace1/gofuentes/dreamer4/.venv/bin/activate && python /workspace1/gofuentes/dreamer4/scripts/pipeline/benchmark_sps.py --task $t --num_envs $e --total_steps $total_steps" 2>&1)
        
        sps=$(echo "$output" | grep "Throughput" | awk -F': ' '{print $2}' | awk '{print $1}')
        
        if [ -z "$sps" ]; then
            sps="Error/TimeOut"
            echo "--- ERROR LOG ---"
            echo "$output"
            echo "-----------------"
        fi
        
        echo "| $t | $e | $total_steps | $sps |" >> /workspace1/gofuentes/dreamer4/benchmark_results.md
    done
done

echo ""
cat /workspace1/gofuentes/dreamer4/benchmark_results.md
