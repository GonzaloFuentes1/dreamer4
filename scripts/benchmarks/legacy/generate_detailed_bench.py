import time
import os
import sys
import torch
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

results = []

def run_bench(task, num_envs, policy_name, steps):
    import contextlib
    import warnings
    # Suppress output to keep console clean
    with warnings.catch_warnings(), open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        
        env = make_torchrl_envpool(task, num_envs=num_envs)
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if policy_name == "random":
            policy = make_random_policy(env.action_spec)
        else:
            policy = make_agent_policy(ckpt_path=None, action_spec=env.action_spec, num_envs=num_envs, device=device_str)
            
        frames_per_batch = min(num_envs * 100, steps)
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

print("Iniciando recoleccion masiva para 20 entornos (DMC y Atari)...")

for task in tasks:
    for n_env in [1, 32]:
        steps = 200 if n_env == 1 else 2000
        
        try:
            print(f"-> Procesando {task} con {n_env} envs...")
            # 1) Random
            random_time = run_bench(task, n_env, "random", steps)
            batch_steps = steps / n_env
            render_ms = (random_time / batch_steps) * 1000.0
            
            # 2) Agent
            agent_time = run_bench(task, n_env, "agent", steps)
            agent_batch_ms = (agent_time / batch_steps) * 1000.0
            
            inf_ms = agent_batch_ms - render_ms
            if inf_ms < 0: inf_ms = 0.0
            
            sps = steps / agent_time
            fps_env = sps / n_env
            
            results.append(
                f"| `{task}` | {n_env} | **{sps:.2f}** | {fps_env:.2f} | {agent_batch_ms:.2f} ms | {render_ms:.2f} ms | {inf_ms:.2f} ms |"
            )
        except Exception as e:
            print(f"[!] Archivo omitido por fallo {task} ({n_env}): {e}")

md_content = """# Benchmark Detallado de Recolección (Phase 0)

Este reporte evalúa la latencia y rendimiento de TorchRL + EnvPool + Transformers a lo largo de **20 entornos distribuidos** (mezclando librerías `dmc` y `atari`).

Se mide la simulación sobre 1 worker y sobre 32 workers (procesamiento batch simulado en 1 GPU).

| Ambiente (Task) | Envs Simultáneos | SPS Total | FPS por Env | MS Totales por Batch | Render (MS C++) | Inferencia (MS GPU) |
|:---|:---:|---:|---:|---:|---:|---:|
""" + "\n".join(results) + "\n"

os.makedirs("docs", exist_ok=True)
with open("docs/benchmark_20_envs_detailed.md", "w") as f:
    f.write(md_content)

print("\n[OK] ¡Resultados guardados exitosamente en docs/benchmark_20_envs_detailed.md!")
