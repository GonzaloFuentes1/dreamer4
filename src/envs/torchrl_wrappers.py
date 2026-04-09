import os
from typing import Any

# ==============================================================================
# MAGIC EGL FIX: Pre-inicializar el render de MuJoCo *ANTES* de importar torchrl
# porque torch/torchrl corrompe los bindings del driver gráfico EGL al cargar.
# ==============================================================================
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
if 'DISPLAY' in os.environ:
    del os.environ['DISPLAY']
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ["RL_WARNINGS"] = "False"
os.environ.setdefault('EGL_LOG_LEVEL', 'fatal')   # suprime "failed to create dri2 screen"


# El launcher `launch_phase0_dist.py` ahora setea EGL_DEVICE_ID correctamente
# para coincidir con la GPU de SLURM. No debemos borrarlo! Ya no interfiere.

try:
    from dm_control import mujoco
    mujoco.Physics.from_xml_string('<mujoco></mujoco>')
except Exception:
    pass

import torch
from torchrl.envs import ParallelEnv, EnvCreator
from torchrl.envs.transforms import TransformedEnv, Compose, Resize, ToTensorImage

def parse_dmc_task(task: str):
    """
    dmc_cheetah_run -> cheetah, run
    cheetah-run -> cheetah, run
    """
    if task.startswith("dmc_"):
        parts = task.replace("dmc_", "").split("_")
    else:
        parts = task.split("-")
        
    domain = parts[0]
    task_name = "_".join(parts[1:])
    return domain, task_name

def parse_atari_task(task: str):
    """
    atari_pong -> PongNoFrameskip-v4
    """
    parts = task.replace("atari_", "").split("_")
    name = "".join([p.capitalize() for p in parts])
    return f"{name}NoFrameskip-v4"

def _make_single_env(task: str, img_size: int = 64, frame_skip: int = 1):
    """
    Contexto constructor anónimo para cada hilo de Multiprocessing local
    (Propuesta 1)
    """
    import os
    # --- LA MAGIA NUEVA TAMBIÉN EN LOS WORKERS ---
    # Respetamos el EGL_DEVICE_ID heredado!
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    if 'DISPLAY' in os.environ:
        del os.environ['DISPLAY']
    os.environ.setdefault('EGL_LOG_LEVEL', 'fatal')

    # Environment config is already safely global now.

    # Tasks usually come as dmc_cheetah_run or just cheetah-run
    # If it is not atari_ it is assumed to be dmc
    is_dmc = task.startswith("dmc_") or ("-" in task and not task.startswith("atari_"))

    if is_dmc:
        from torchrl.envs.libs.dm_control import DMControlEnv
        domain, task_name = parse_dmc_task(task)

        env = DMControlEnv(
            env_name=domain,
            task_name=task_name,
            from_pixels=True,
            pixels_only=False,
            frame_skip=frame_skip,
        )
    else:
        # Fallback Atari
        from torchrl.envs.libs.gym import GymEnv
        atari_name = parse_atari_task(task)
        env = GymEnv(
            env_name=atari_name,
            frame_skip=frame_skip,
            from_pixels=True,
            pixels_only=False
        )

    return TransformedEnv(
        env,
        Compose(
            ToTensorImage(in_keys=["pixels"]),
            Resize(w=img_size, h=img_size, in_keys=["pixels"])
        )
    )

def make_torchrl_env(task: str, num_envs: int = 16, seed: int = 42, img_size: int = 64, frame_skip: int = 1) -> ParallelEnv:
    """
    Propuesta 1: TorchRL ParallelEnv con constructores directos, sin pasar por Envpool.
    """
    # EnvCreator empaqueta la inicialización de manera segura entre procesos
    create_env_fn = EnvCreator(lambda: _make_single_env(task, img_size=img_size, frame_skip=frame_skip))
    
    parallel_env = ParallelEnv(
        num_workers=num_envs,
        create_env_fn=create_env_fn,
        mp_start_method="spawn"
    )
    
    parallel_env.set_seed(seed)
    
    print(f"[TorchRL Wrapper] Construido {num_envs} envs nativos en paralelo para {task}.")
    
    return parallel_env
