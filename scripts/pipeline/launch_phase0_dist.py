import sys
import os
import subprocess
import torch

def main():
    args = sys.argv[1:]

    # Compatibilidad retroactiva: ignorar override obsoleto para evitar fallos de Hydra.
    filtered_args = []
    dropped_deprecated = []
    for arg in args:
        if arg.startswith("collect.env_backend=") or arg.startswith("+collect.env_backend="):
            dropped_deprecated.append(arg)
            continue
        filtered_args.append(arg)
    args = filtered_args

    if dropped_deprecated:
        print(
            f"[Launch Phase0] Ignorando override(s) obsoleto(s): {dropped_deprecated}",
            flush=True,
        )
    
    # Extraer y parsear collect.tasks=[...]
    task_arg_idx = -1
    tasks_str = None
    for i, arg in enumerate(args):
        if arg.startswith("collect.tasks=["):
            task_arg_idx = i
            # extraemos el contenido y hacemos strip del ] final
            tasks_str = arg.replace("collect.tasks=[", "").replace("]", "")
            break
            
    if tasks_str is None:
        print("[Launch Phase0] No se encontro collect.tasks=[...]. Ejecutando secuencial...", flush=True)
        # Fallback a single run
        cmd = [sys.executable, "scripts/pipeline/train_phase0_collect_episodes.py"] + args
        subprocess.run(cmd, check=True)
        return

    tasks = [t.strip() for t in tasks_str.split(",") if t.strip()]
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1 or len(tasks) == 1:
        print(f"[Launch Phase0] {num_gpus} GPUs disponibles. Ejecutando de forma estandar...", flush=True)
        cmd = [sys.executable, "scripts/pipeline/train_phase0_collect_episodes.py"] + args
        subprocess.run(cmd, check=True)
        return
        
    # Split de los tasks entre GPUs
    if num_gpus > len(tasks):
        num_gpus = len(tasks)
        
    chunks = [[] for _ in range(num_gpus)]
    for idx, t in enumerate(tasks):
        chunks[idx % num_gpus].append(t)
        
    print(f"\n[Launch Phase0 Dist] Escalamiento Perfecto! {len(tasks)} Tareas, {num_gpus} GPUs.", flush=True)
    print("Distribuyendo recoleccion como Procesos Independientes:", flush=True)
    
    processes = []
    base_cmd = [sys.executable, "scripts/pipeline/train_phase0_collect_episodes.py"]
    base_args = [a for i, a in enumerate(args) if i != task_arg_idx]
    
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_devices:
        # Slurm usually sets something like "4,5,6"
        avail_devices = [d.strip() for d in cuda_devices.split(",") if d.strip()]
    else:
        avail_devices = [str(x) for x in range(num_gpus)]

    for i, chunk in enumerate(chunks):
        if not chunk: continue
        env = os.environ.copy()
        
        # Mapeo vital: Asignamos el ID físico real de CUDA si hay un allocation en cluster
        target_gpu_id = avail_devices[i] if i < len(avail_devices) else str(i)
        
        # Al setear CUDA_VISIBLE_DEVICES al device específico, ese device se vuelve el index 0
        # para todo lo demás dentro de ese subprocess, incluyendo EGL.
        env["CUDA_VISIBLE_DEVICES"] = target_gpu_id
        
        # --- LA MAGIA NUEVA ---
        # 1. Obligamos a MuJoCo a usar EGL
        env["MUJOCO_GL"] = "egl"
        
        # 2. Obligamos a PyOpenGL (el motor que usa dm_control) a usar EGL
        env["PYOPENGL_PLATFORM"] = "egl" 
        
        # 3. Asignamos la GPU
        env["EGL_DEVICE_ID"] = str(target_gpu_id)
        
        # Evita que intente abrir una ventana X11 real
        env["DISPLAY"] = ""
        
        # reconstruimos el argumento exacto pero con las tareas filtradas 
        chunk_arg = "collect.tasks=[" + ",".join(chunk) + "]"
        cmd = base_cmd + base_args + [chunk_arg]
        
        print(f"  -> GPU {i} se lleva: {chunk}")
        p = subprocess.Popen(cmd, env=env)
        processes.append((i, p))
        
    # Esperamos a que todos terminen (Data Parallel de facto)
    has_error = False
    for i, p in processes:
        p.wait()
        if p.returncode != 0:
            print(f"[!] Error: Proceso de la GPU {i} fallo con codigo {p.returncode}")
            has_error = True
            
    if has_error:
        sys.exit(1)
    else:
        print("[Launch Phase0 Dist] Todas las GPUs finalizaron su recoleccion con exito.\n")

if __name__ == "__main__":
    main()
