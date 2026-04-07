#!/usr/bin/env python
import time
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from policies import make_random_policy

from torchrl.envs import ParallelEnv, EnvCreator
from torchrl.envs.libs.gym import GymEnv
from torchrl.collectors import SyncDataCollector

def make_env(task):
    return EnvCreator(lambda: GymEnv(task))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="ALE/Pong-v5")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--total_steps", type=int, default=50000)
    args = parser.parse_args()

    print(f"=== SPS Benchmark (Nativo TorchRL) ===")
    print(f"Task: {args.task}")
    print(f"Envs Paralelos: {args.num_envs}")
    
    start_env = time.time()
    env = ParallelEnv(args.num_envs, make_env(args.task))
    print(f"[1] Entornos levantados en {time.time()-start_env:.2f}s")
    
    policy = make_random_policy(env.action_spec)
    
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=min(args.num_envs * 100, args.total_steps),
        total_frames=args.total_steps,
        device="cpu"
    )

    start_sim = time.time()
    for i, td in enumerate(collector):
        pass
    
    total_time = time.time() - start_sim
    sps = args.total_steps / total_time
    
    print("\n" + "="*30)
    print(f"Throughput (SPS nativo): {sps:.2f} Steps per Second")
    print("="*30 + "\n")
    
    env.close()

if __name__ == "__main__":
    main()
