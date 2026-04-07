import time
import os
import sys
import torch
import concurrent.futures
from torchrl.collectors import SyncDataCollector

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from envs import make_torchrl_envpool
from policies import make_random_policy, make_agent_policy

tasks = [
    "dmc_cheetah_run", "dmc_walker_walk", "dmc_hopper_hop", "dmc_cartpole_swingup",
    "dmc_pendulum_swingup", "dmc_reacher_easy", "dmc_acrobot_swingup", "dmc_finger_spin",
    "atari_pong", "atari_breakout", "atari_space_invaders", "atari_seaquest",
    "atari_alien", "atari_qbert", "atari_asteroids", "atari_ms_pacman",
    "atari_enduro", "atari_hero", "atari_freeway", "atari_riverraid"
]

def run_collector(task, n_envs, policy_name, steps, device_str, ckpt):
    import contextlib
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env = make_torchrl_envpool(task, num_envs=n_envs)
        
        if policy_name == "random":
            policy = make_random_policy(env.action_spec)
        else:
            policy = make_agent_policy(ckpt_path=ckpt, action_spec=env.action_spec, num_envs=n_envs, device=device_str)
            
        frames_per_batch = min(n_envs * 100, steps)
        if steps % frames_per_batch != 0:
            frames_per_batch = steps
            
        collector = SyncDataCollector(
            env,
            policy,
            frames_per_batch=frames_per_batch,
            total_frames=steps,
            device="cpu",
            compile_policy=False
        )
        
        start = time.time()
        for _ in collector: pass
        sim_time = time.time() - start
        
        env.close()
        del env, policy, collector
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
    return sim_time

def run_bench(task, num_envs, policy_name, steps, num_gpus=1, ckpt=None):
    if num_gpus == 1:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        sim_time = run_collector(task, num_envs, policy_name, steps, device, ckpt)
        return sim_time
    else:
        n_envs_per_gpu = num_envs // num_gpus
        steps_per_gpu = steps // num_gpus
        
        start = time.time()
        futures = []
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
            for i in range(num_gpus):
                device = f"cuda:{i}" if torch.cuda.is_available() else "cpu"
                futures.append(executor.submit(run_collector, task, n_envs_per_gpu, policy_name, steps_per_gpu, device, ckpt))
            
            for future in futures:
                future.result() # Wait for all and raise exceptions if any
                
        sim_time = time.time() - start
        return sim_time

def generate_table(env_counts, num_gpus, out_file, ckpt=None):
    results = []
    print(f"\nIniciando recoleccion masiva (Envs: {env_counts}, GPUs: {num_gpus}, Ckpt: {ckpt})...", flush=True)

    for task in tasks:
        for n_env in env_counts:
            steps = max(2000, n_env * 20)
            
            try:
                print(f"-> Procesando {task} con {n_env} envs (GPUs: {num_gpus})...", flush=True)
                
                # 1) Random
                random_time = run_bench(task, n_env, "random", steps, num_gpus, None)
                batch_steps = steps / n_env
                render_ms = (random_time / batch_steps) * 1000.0
                
                # 2) Agent
                agent_time = run_bench(task, n_env, "agent", steps, num_gpus, ckpt)
                agent_batch_ms = (agent_time / batch_steps) * 1000.0
                
                inf_ms = agent_batch_ms - render_ms
                if inf_ms < 0: inf_ms = 0.0
                
                sps = steps / agent_time
                fps_env = sps / n_env
                
                results.append(
                    f"| `{task}` | {n_env} | **{sps:.2f}** | {fps_env:.2f} | {agent_batch_ms:.2f} ms | {render_ms:.2f} ms | {inf_ms:.2f} ms |"
                )
            except Exception as e:
                print(f"[!] Archivo omitido por fallo {task} ({n_env}): {e}", flush=True)

    md_content = f"""# Benchmark de Recolección Escalado ({num_gpus} GPU) - Preentrenado

Evaluación de latencia y rendimiento variando grandes volúmenes de workers y GPUs, **cargando pesos de un modelo real**.

Mide la simulación empleando infraestructura Multi-worker ({env_counts}) distribuida en {num_gpus} GPUs.
                                                                
| Ambiente (Task) | Envs Simultáneos | SPS Total | FPS por Env | MS Totales por Batch | Render (MS C++) | Inferencia (MS GPU) |
|:---|:---:|---:|---:|---:|---:|---:|
""" + "\n".join(results) + "\n"

    os.makedirs("docs", exist_ok=True)
    with open(out_file, "w") as f:
        f.write(md_content)
    print(f"\n[OK] Resultados guardados en {out_file}!", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    
    envs_to_test = [64, 128, 256]
    
    if args.out:
        out_name = args.out
    else:
        if args.gpus == 1:
            out_name = "docs/benchmark_scale_1gpu_ckpt.md" if args.ckpt else "docs/benchmark_scale_1gpu.md"
        else:
            out_name = "docs/benchmark_scale_2gpu_ckpt.md" if args.ckpt else "docs/benchmark_scale_2gpu.md"
            
    generate_table(envs_to_test, args.gpus, out_name, args.ckpt)