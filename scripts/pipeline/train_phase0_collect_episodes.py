#!/usr/bin/env python
import os
import sys

# Compatibilidad: ignorar override obsoleto para evitar fallo de Hydra.
_deprecated_overrides = []
_sanitized_argv = [sys.argv[0]]
for _arg in sys.argv[1:]:
    if _arg.startswith("collect.env_backend=") or _arg.startswith("+collect.env_backend="):
        _deprecated_overrides.append(_arg)
        continue
    _sanitized_argv.append(_arg)
sys.argv = _sanitized_argv
if _deprecated_overrides:
    print(f"[Phase 0] Ignorando override(s) obsoleto(s): {_deprecated_overrides}", flush=True)

# 1. PURGA ABSOLUTA Y CONFIGURACIÓN EGL
# (Debe ser la primera acción real del script)
# --- LA MAGIA NUEVA ---
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl" 
# Intentamos obtener un DEVICE_ID específico (de launch_phase0_dist) o el primero disponible en el scope
gpu_id = os.environ.get("EGL_DEVICE_ID", os.environ.get("CUDA_VISIBLE_DEVICES", "0")).split(",")[0]
os.environ["EGL_DEVICE_ID"] = str(gpu_id)
os.environ["DISPLAY"] = ""

os.environ['LAZY_LEGACY_OP'] = '0'
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "1800"
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"

import warnings
warnings.filterwarnings('ignore')

# 2. Inicialización forzada de MuJoCo EGL
try:
    from dm_control import mujoco
    # Esto inicializa el backend interno de C++ correctamente con EGL
    mujoco.Physics.from_xml_string('<mujoco></mujoco>')
except Exception:
    pass

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
from torchrl.collectors import SyncDataCollector
import imageio
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Aseguramos que los módulos nativos puedan ser encontrados
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from envs import make_torchrl_env
from policies import get_collect_policy

try:
    from hdf5_episode_dataset import HDF5EpisodeWriter
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    HDF5EpisodeWriter = None

def _extract_frames(td, target_size=128):
    """
    Extrae la observación (imágenes) del TensorDict y las estandariza a (N, 3, H, W) uint8.
    """
    # Envpool / Torchrl pueden guardar las observaciones bajo distintas llaves dependiendo del env.
    if "pixels" in td.keys():
        obs = td["pixels"]
    elif "observation" in td.keys():
        obs = td["observation"]
    else:
        # Fallback ciego al primer tensor que parezca un espacio de obs
        obs = next(iter([v for k, v in td.items() if isinstance(v, torch.Tensor) and v.ndim >= 3]))
        
    # Asume batch y timesteps [B, T, ...]
    if obs.ndim >= 4:
        flat_batch = int(torch.prod(torch.tensor(obs.shape[:-3])))
        obs = obs.reshape(flat_batch, *obs.shape[-3:])
    elif obs.ndim == 3: # En caso de que haya aplastado antes
        B = obs.shape[0]
        T = 1
        
    # Formato Torch: (N, C, H, W). Envpool a veces envía (N, H, W, C).
    if obs.shape[-1] in [1, 3, 4] and obs.shape[-3] not in [1, 3, 4]:
        obs = obs.permute(0, 3, 1, 2)
    
    # Estandarizamos tamaño a target_size x target_size
    if obs.shape[-1] != target_size or obs.shape[-2] != target_size:
        obs_f = obs.to(torch.float32)
        if obs.dtype == torch.uint8:
            obs_f = obs_f / 255.0
        obs_f = F.interpolate(obs_f, size=(target_size, target_size), mode="bilinear", align_corners=False)
        obs = (obs_f.clamp(0, 1) * 255).to(torch.uint8)
        
    if obs.dtype != torch.uint8:
        obs = (obs.clamp(0, 1) * 255).to(torch.uint8)
        
    return obs

@hydra.main(config_path="../../configs", config_name="collect_phase0", version_base=None)
def main(cfg: DictConfig):
    # Por si usamos el base.yaml dictado en configs/collect/
    dc = cfg.collect if "collect" in cfg else cfg
    
    # Inicializar W&B si está disponible y habilitado
    wandb_run = None
    if WANDB_AVAILABLE and dc.get("wandb_project", None):
        wandb_project = dc.get("wandb_project", "dreamer4")
        wandb_entity = dc.get("wandb_entity", None)
        wandb_run_name = dc.get("wandb_run_name", None)
        wandb_mode = dc.get("wandb_mode", "online")
        
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=wandb_mode
        )
    
    out_demo = Path(dc.get("out_data_dir", "./data/collected/demos"))
    out_frames = Path(dc.get("out_frames_dir", "./data/collected/frames"))
    
    out_demo.mkdir(parents=True, exist_ok=True)
    out_frames.mkdir(parents=True, exist_ok=True)

    use_hdf5 = bool(dc.get("use_hdf5", HDF5_AVAILABLE))
    if use_hdf5 and not HDF5_AVAILABLE:
        print("[Phase 0] WARN: use_hdf5=true pero h5py no instalado. Usando .pt.")
        use_hdf5 = False
    
    print(f"\n[Phase 0] Iniciando Recolección Múltiple. Destinos:")
    print(f"  - Demos:  {out_demo}")
    print(f"  - Frames: {out_frames}\n")
    
    tasks = dc.get("tasks", ["atari_pong"])
    num_envs = dc.get("num_envs_per_task", 16)
    episodes_target = dc.get("n_episodes_per_task", 100)
    episode_len = int(dc.get("episode_len", 1000))
    frame_skip = int(dc.get("frame_skip", 1))
    shard_size = dc.get("shard_size", 2048)
    target_img_size = dc.get("img_size", 128)
    target_action_dim = int(dc.get("action_dim", 16))
    min_frames_per_task = int(dc.get("min_frames_per_task", episodes_target * episode_len))
    if min_frames_per_task < 0:
        min_frames_per_task = 0
    max_collect_frames_per_task = int(
        dc.get(
            "max_collect_frames_per_task",
            max(episodes_target * episode_len * num_envs, max(min_frames_per_task * 4, shard_size)),
        )
    )
    max_collect_frames_per_task = max(max_collect_frames_per_task, shard_size)
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"▶ Procesando Task: {task}")
        print(f"{'='*60}")
        print(
            f"[Config] episodes_target={episodes_target}, episode_len={episode_len}, "
            f"min_frames_per_task={min_frames_per_task}, max_collect_frames_per_task={max_collect_frames_per_task}"
        )
        
        # 1. Pipeline de Simulación
        print("[1/4] Levantando entorno EnvPool + TorchRL Wrappers...")
        env = make_torchrl_env(task, num_envs=num_envs, seed=cfg.get("seed", 42), img_size=target_img_size, frame_skip=frame_skip)
        
        # 2. Pipeline de la Política
        print("[2/4] Instanciando Política...")
        policy = get_collect_policy(dc, env.action_spec, env.observation_spec, device="cuda" if torch.cuda.is_available() else "cpu", task_name=task)
        
        # 3. Collector
        # Usamos SyncDataCollector para extraer lotes asincronos del ParallelEnv subyacente.
        # Solo para acotar, decimos que trabaje 200 iteraciones máximas como safety.
        print("[3/4] Inicializando Motor de Recolección...")
        collector = SyncDataCollector(
            env, 
            policy, 
            frames_per_batch=num_envs * 64, # Un batch moderado para no reventar la RAM
            total_frames=max_collect_frames_per_task,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Estructuras de almacenamiento
        buffer_frames = []

        # HDF5 writer (si está habilitado)
        hdf5_writer = None
        if use_hdf5:
            hdf5_path = out_demo / f"{task}.h5"
            hdf5_writer = HDF5EpisodeWriter(
                hdf5_path,
                img_size=target_img_size,
                action_dim=target_action_dim,
                chunk_frames=512,
            )
            print(f"[Phase 0] HDF5 writer: {hdf5_path}")
        all_actions = []
        all_rewards = []
        all_episodes = []
        
        shard_idx = 0
        episodes_collected = 0
        steps_collected = 0
        current_episode_ids = torch.arange(num_envs) # trackers de id
        global_episode_counter = num_envs

        # Buffer para previsualización de video (extraemos solo un env y un max_frames)
        save_video = dc.get("save_preview_video", False)
        max_video_frames = dc.get("preview_video_max_frames", 300)
        video_fps = dc.get("preview_video_fps", 20)
        video_env_idx = int(dc.get("preview_video_env_idx", 0))
        out_videos = Path(dc.get("out_videos_dir", "./data/collected/videos"))
        if save_video:
            out_videos.mkdir(parents=True, exist_ok=True)
            video_frames_list = []
            video_episode_done = False

        
        print(f"[4/4] Recolectando {episodes_target} episodios...")
        start_time = time.time()
        
        for i, tensordict in enumerate(collector):
            
            # tensordict shape: [batch_size, time_steps] -> [num_envs, frames_per_batch // num_envs]
            B = tensordict.shape[0]
            T = tensordict.shape[1] if tensordict.ndim > 1 else 1
            steps_collected += B * T

            # Extraer píxeles
            frames = _extract_frames(tensordict, target_size=target_img_size) # (B*T, 3, 128, 128)
            buffer_frames.append(frames)
            
            # --- VIDEO CAPTURE ---
            # Capturamos un episodio completo del env seleccionado (frame a frame),
            # no solo 1 frame por batch, para evitar videos en "camara rapida".
            if save_video and not video_episode_done:
                dones_shaped = tensordict["next", "done"].reshape(B, T)

                if B > 0 and frames.shape[0] == (B * T):
                    env_idx = min(max(video_env_idx, 0), B - 1)
                    frames_btchw = frames.reshape(B, T, *frames.shape[-3:])
                    env_frames = frames_btchw[env_idx]  # (T, C, H, W)

                    for t_vid in range(T):
                        if len(video_frames_list) >= max_video_frames:
                            video_episode_done = True
                            break

                        np_frame = env_frames[t_vid].permute(1, 2, 0).cpu().numpy()
                        video_frames_list.append(np_frame)

                        if bool(dones_shaped[env_idx, t_vid].item()):
                            video_episode_done = True
                            break

            
            # Aplanar Tensores
            if "action" in tensordict.keys():
                acts = tensordict["action"].reshape(-1, *tensordict["action"].shape[2:])
            else:
                acts = torch.zeros((B*T, 1)) # Fallback safe

            # Normalizamos acciones a un ancho fijo para mezclar tareas/sources sin romper el dataset.
            if acts.ndim == 1:
                acts = acts.unsqueeze(-1)
            elif acts.ndim > 2:
                acts = acts.reshape(acts.shape[0], -1)
            acts = acts.to(torch.float32)
            if acts.shape[-1] != target_action_dim:
                acts_norm = torch.zeros((acts.shape[0], target_action_dim), dtype=torch.float32)
                copy_dim = min(int(acts.shape[-1]), target_action_dim)
                if copy_dim > 0:
                    acts_norm[:, :copy_dim] = acts[:, :copy_dim]
                acts = acts_norm
            
            if ("next", "reward") in tensordict.keys(include_nested=True):
                rews = tensordict["next", "reward"].reshape(-1)
            else:
                rews = torch.zeros(B*T)
                
            dones = tensordict["next", "done"].reshape(-1)
            
            # Asignar a cada step su respectivo episode_id
            # Esto es clave para que los dataloaders de la Fase 1-2-3 sepan donde cortan las secuencias.
            ep_id_t = current_episode_ids.view(B, 1).expand(B, T).reshape(-1).clone()
            
            # Track de episodios completados 
            dones_shaped = dones.view(B, T)
            for b in range(B):
                for t in range(T):
                    if dones_shaped[b, t]:
                        episodes_collected += 1
                        current_episode_ids[b] = global_episode_counter
                        global_episode_counter += 1
            
            all_actions.append(acts)
            all_rewards.append(rews)
            all_episodes.append(ep_id_t)

            # Flush HDF5 incremental (SWMR crash-safe)
            if hdf5_writer is not None:
                hdf5_writer.append_batch(frames, acts, rews, ep_id_t)
            
            # Flush a Disco (Sharding compatible con sharded_frame_dataset.py)
            current_buffer_size = sum([f.shape[0] for f in buffer_frames])
            if current_buffer_size >= shard_size:
                concat_f = torch.cat(buffer_frames, dim=0)
                
                # Escribir fragmentos enteros de tamaño shard_size
                while concat_f.shape[0] >= shard_size:
                    to_save = concat_f[:shard_size]
                    concat_f = concat_f[shard_size:]
                    
                    task_frame_dir = out_frames / task
                    task_frame_dir.mkdir(parents=True, exist_ok=True)
                    
                    out_path = task_frame_dir / f"{task}_shard{shard_idx:04d}.pt"
                    torch.save({"frames": to_save}, out_path)
                    shard_idx += 1
                
                buffer_frames = [concat_f] if concat_f.shape[0] > 0 else []
                
            # Condición de salida si acumulamos los episodios target
            if episodes_collected >= episodes_target and steps_collected >= min_frames_per_task:
                print(
                    f"\n[✓] Alcanzado target: episodes={episodes_collected}/{episodes_target}, "
                    f"frames={steps_collected}/{min_frames_per_task}."
                )
                break
            if (
                episodes_collected >= episodes_target
                and steps_collected < min_frames_per_task
                and (i % 10 == 0)
            ):
                print(
                    f"[!] Episodios objetivo alcanzados, pero faltan frames: "
                    f"{steps_collected}/{min_frames_per_task}. Continuando recoleccion..."
                )
                
        # Limpieza residual
        if len(buffer_frames) > 0 and buffer_frames[0].shape[0] > 0:
            concat_f = torch.cat(buffer_frames, dim=0)
            task_frame_dir = out_frames / task
            task_frame_dir.mkdir(parents=True, exist_ok=True)
            out_path = task_frame_dir / f"{task}_shard{shard_idx:04d}.pt"
            torch.save({"frames": concat_f}, out_path)
            shard_idx += 1
        # --- GUARDADO DE VIDEO ---
        if save_video and len(video_frames_list) > 0:
            vid_path = out_videos / f"{task}_preview.mp4"
            print(f"Creando video previsualizacion: {vid_path} ({len(video_frames_list)} frames)")
            try:
                imageio.mimwrite(str(vid_path), video_frames_list, fps=video_fps, codec='libx264')
            except Exception as e:
                print(f"[!] Error guardando video: {e}")

            
        # Guardar data relacional estructurada
        demo_data = {
            "episode": torch.cat(all_episodes, dim=0).to(torch.int64),
            "action":  torch.cat(all_actions, dim=0).to(torch.float32),
            "reward":  torch.cat(all_rewards, dim=0).to(torch.float32),
        }
        
        # Recorte al tamaño real de frames extraídos para que calce 1:1 localmente con el sharded loader.
        if hdf5_writer is not None:
            total_steps_saved = hdf5_writer.n_steps
            hdf5_writer.finalize()
            hdf5_writer = None
        else:
            total_steps_saved = sum([
                torch.load(p, weights_only=False)["frames"].shape[0]
                for p in sorted((out_frames / task).glob("*.pt"))
            ])
        demo_data = {k: v[:total_steps_saved] for k, v in demo_data.items()}
        avg_ep_len = float(total_steps_saved) / float(max(episodes_collected, 1))

        if episodes_collected < episodes_target:
            raise RuntimeError(
                f"[Phase 0] Recoleccion insuficiente para '{task}': "
                f"episodes={episodes_collected} < episodes_target={episodes_target}. "
                f"frames={total_steps_saved}, avg_ep_len={avg_ep_len:.1f}. "
                f"Aumenta max_collect_frames_per_task o reduce n_episodes_per_task."
            )

        if total_steps_saved < min_frames_per_task:
            raise RuntimeError(
                f"[Phase 0] Recoleccion insuficiente para '{task}': "
                f"frames={total_steps_saved} < min_frames_per_task={min_frames_per_task}. "
                f"episodes={episodes_collected}, avg_ep_len={avg_ep_len:.1f}. "
                f"Aumenta max_collect_frames_per_task o reduce min_frames_per_task."
            )
            
        task_demo_path = out_demo / f"{task}.pt"
        torch.save(demo_data, task_demo_path)
        
        elapsed = time.time() - start_time
        print(f"[Done] '{task}': Guardados {total_steps_saved} frames/steps. "
              f"Data en {task_demo_path} ({elapsed:.1f}s generacion)")
        print(f"       Episodios: {episodes_collected} | Largo promedio: {avg_ep_len:.1f} steps")
        
        # Opcional metadatos (Gobernanza Etapa 1)
        meta_path = out_demo / f"{task}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump({
                "task": task,
                "envs": num_envs,
                "episodes_collected": episodes_collected,
                "total_frames": total_steps_saved,
                "avg_episode_len": avg_ep_len,
                "min_frames_per_task": min_frames_per_task,
                "policy_used": dc.get("policy", "random"),
                "build_backend": "envpool->torchrl"
            }, f, indent=4)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # W&B LOGGING — Métricas por tarea
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if wandb_run is not None and all_rewards:
            all_rews_np = torch.cat(all_rewards).cpu().numpy()
            reward_mean = float(np.mean(all_rews_np))
            reward_std = float(np.std(all_rews_np))
            reward_min = float(np.min(all_rews_np))
            reward_max = float(np.max(all_rews_np))
            
            metrics = {
                f"eval/{task}/reward_mean": reward_mean,
                f"eval/{task}/reward_std": reward_std,
                f"eval/{task}/reward_min": reward_min,
                f"eval/{task}/reward_max": reward_max,
                f"eval/{task}/episodes_collected": episodes_collected,
                f"eval/{task}/total_frames": total_steps_saved,
            }
            wandb_run.log(metrics)
            print(f"[W&B] Logged metrics for {task}:")
            print(f"      reward_mean={reward_mean:.4f}, reward_std={reward_std:.4f}")
            
        env.close()

    # Cerrar W&B al final
    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    main()
