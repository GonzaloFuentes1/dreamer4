import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="gym")
#!/usr/bin/env python
import time
import os
import torch
import argparse
from torchrl.collectors import SyncDataCollector
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from envs import make_torchrl_envpool
from policies import make_random_policy, make_agent_policy

def main():
    parser = argparse.ArgumentParser(description="Envpool + TorchRL Multi-Task SPS")
    parser.add_argument("--tasks", type=str, default="dmc_cheetah_run", help="Comma-separated list of tasks")
    parser.add_argument("--num_envs", type=int, default=16, help="Worker count per task")
    parser.add_argument("--steps_per_task", type=int, default=2000, help="Steps to collect per task")
    parser.add_argument("--policy", type=str, default="agent", choices=["random", "agent"])
    args = parser.parse_args()
    
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    device_str = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    all_tasks = args.tasks.split(",")
    # Repartir las tareas uniformemente entre las GPUs (Ranks)
    my_tasks = all_tasks[local_rank :: world_size]

    print(f"\n{'='*40}")
    print(f"=== SPS Benchmark (Rank {local_rank}) ===")
    print(f"Target Device : {device_str}")
    print(f"Envs / Task   : {args.num_envs}")
    print(f"Steps / Task  : {args.steps_per_task}")
    print(f"Assigned Tasks: {len(my_tasks)}")
    print(f"{'='*40}\n")

    total_steps_collected = 0
    total_sim_time = 0.0

    for task in my_tasks:
        try:
            start_env_time = time.time()
            env = make_torchrl_envpool(task, num_envs=args.num_envs)
            env_time = time.time() - start_env_time
            
            if args.policy == "random":
                policy = make_random_policy(env.action_spec)
            else:
                policy = make_agent_policy(ckpt_path=None, action_spec=env.action_spec, num_envs=args.num_envs, device=device_str)
                
            frames_per_batch = min(args.num_envs * 100, args.steps_per_task)
            # Asegurar que sea divisible
            if args.steps_per_task % frames_per_batch != 0:
                frames_per_batch = args.steps_per_task
                
            collector = SyncDataCollector(
                env,
                policy,
                frames_per_batch=frames_per_batch,
                total_frames=args.steps_per_task,
                device="cpu",
                compile_policy=False
            )

            start_sim = time.time()
            for _ in collector:
                pass 
            
            sim_time = time.time() - start_sim
            sps = args.steps_per_task / sim_time
            
            print(f"[Rank {local_rank}] Task '{task}' \t-> {sps:8.2f} SPS  (Sim Time: {sim_time:.2f}s | Env Load: {env_time:.2f}s)")
            
            total_steps_collected += args.steps_per_task
            total_sim_time += sim_time
            env.close()
            
            # Limpiar memoria de la GPU entre ciclos de tareas pesadas
            del env, policy, collector
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"[Rank {local_rank}] [!] Error procesando '{task}': {e}")

    # Resumen Final del Rank
    if total_sim_time > 0:
        avg_sps = total_steps_collected / total_sim_time
        print("\n" + "-"*40)
        print(f"[Rank {local_rank}] SUMMARY:")
        print(f" Total Steps     : {total_steps_collected}")
        print(f" Total Sim Time  : {total_sim_time:.2f}s")
        print(f" Mean Throughput : {avg_sps:.2f} SPS")
        print("-"*40 + "\n")

if __name__ == "__main__":
    main()
