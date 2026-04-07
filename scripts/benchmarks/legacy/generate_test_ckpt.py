import time
import os
import sys
import torch
from torchrl.collectors import SyncDataCollector

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from envs import make_torchrl_envpool
from policies import make_agent_policy

task = "dmc_cheetah_run"
n_envs = 32
steps = 64

ckpt_path = "/workspace1/gofuentes/dreamer4/logs/agent_ckpts/last.ckpt"

import contextlib
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    env = make_torchrl_envpool(task, num_envs=n_envs)
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"\nIniciando test con modelo cargado desde log: {ckpt_path}\n")
    
    policy = make_agent_policy(ckpt_path=ckpt_path, action_spec=env.action_spec, num_envs=n_envs, device=device_str)
    
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=steps,
        total_frames=steps,
        device="cpu",
        compile_policy=False
    )
    
    for td in collector: 
        print(f"Recolectado tensordict! Acciones shape: {td['action'].shape}")
        break
        
    env.close()
    
print("\n[OK] Run exitoso con modelo pesando completado.")
