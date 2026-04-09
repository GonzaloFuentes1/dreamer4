# dreamer4 — PyTorch Lightning

> ⚠️ **Implementación no oficial.** Este repositorio es una re-implementación independiente basada en el paper
> [Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527) (Hafner, Yan, Lillicrap, 2025).
> El repositorio oficial de los autores es [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4).

Implementación del pipeline completo de Dreamer V4 sobre DMControl, usando PyTorch Lightning + Hydra.
Extiende la referencia oficial con las Fases 2 y 3 (agent finetuning + imagination training),
el ciclo de recolección recursivo (self-play) y soporte para tokenizador discreto.

---

## Instalación

```bash
# Requiere uv (https://github.com/astral-sh/uv)
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

Crea un `.env` en la raíz del repo con tus credenciales:

```bash
export WANDB_API_KEY=tu_clave_aqui
```

### Entorno headless (cluster / SLURM)

El rendering headless usa EGL. Las variables necesarias las configura automáticamente `_pipeline_lib.sh`:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
```

---

## Pipeline completo

Cinco fases que se ejecutan en orden. Las Fases 1a y 1b tienen checkpoints preentrenados disponibles en [HuggingFace](https://huggingface.co/nicklashansen/dreamer4).

```
Phase 0  →  scripts/pipeline/launch_phase0_dist.py      # recolección de episodios
Phase 1a →  scripts/pipeline/train_phase1a_tokenizer.py # tokenizador de frames (VQVAE-like)
Phase 1b →  scripts/pipeline/train_phase1b_dynamics.py  # world model (RSSM)
Phase 2  →  scripts/pipeline/train_phase2_finetuning.py # BC + reward head
Phase 3  →  scripts/pipeline/train_phase3_imagination.py # RL en imaginación (PMPO)
```

El entrenamiento acumula datos de todos los ciclos anteriores: cada ciclo `k` entrena sobre `cycle0 + cycle1 + … + cycleK`.

---

## Modos de ejecución

### Modo 1 — Desde cero (sin datos previos)

El pipeline completo parte de política random en el ciclo 0 y va mejorando el agente ciclo a ciclo.

```bash
# 64×64, 1 GPU, 30 ciclos
RES=64 srun --nodes=1 --ntasks=1 --gpus=1 \
    --cpus-per-task=64 --mem=128G --time=24:00:00 \
    bash scripts/pipeline/run_cycles.sh

# 128×128, 3 GPUs, 30 ciclos
RES=128 srun --nodes=1 --ntasks=1 --gpus=3 \
    --cpus-per-task=48 --mem=256G --time=72:00:00 \
    bash scripts/pipeline/run_cycles.sh
```

Variables de entorno opcionales para `run_cycles.sh`:

| Variable | Default | Descripción |
|---|---|---|
| `RES` | `64` | Resolución: `64` o `128` |
| `K` | `30` | Número de ciclos |
| `RUN_TAG` | `active_64x64` | Nombre del run (carpeta en `./runs/`) |
| `TASKS` | 5 tareas DMC | Lista separada por comas |
| `START_FROM_DYNAMICS_CYCLE0` | `false` | Arrancar desde tokenizer+datos externos |
| `SOURCE_RUN_TAG` | — | Run fuente (requiere la opción anterior) |
| `DEVICES` | auto-detect | Número de GPUs |

### Modo 2 — Desde datos preentrenados (HuggingFace)

Ciclo 0 usa datos descargados de HF sin colección. Ciclos posteriores colectan con el agente aprendido.

```bash
# Convertir frames a la resolución deseada (solo una vez)
python scripts/convert_frames_to_res.py \
    --input  ./data/cycle0-pretrained/frames \
    --output ./data/pretrained-64x64/frames \
    --size 64 --workers 32

# Crear symlink de demos (los demos ya están en la resolución correcta)
ln -s ./data/cycle0-pretrained/demos ./data/pretrained-64x64/demos

# Lanzar pipeline
RES=64 BASE_DATA_ROOT=./data/pretrained-64x64 \
    srun --gpus=4 --cpus-per-task=64 --mem=384G --time=72:00:00 \
    bash scripts/pipeline/run_cycles_pretrain.sh
```

Variables adicionales para `run_cycles_pretrain.sh`:

| Variable | Default | Descripción |
|---|---|---|
| `BASE_DATA_ROOT` | `./data/cycle0-pretrained` | Directorio con datos del ciclo 0 (HF) |
| `RUN_TAG` | `pt_active` | Nombre del run |
| `K` | `10` | Número de ciclos |

---

## Preprocesamiento de frames

Los frames deben estar en la resolución del modelo **antes** del entrenamiento. El entrenamiento no hace resize en tiempo de ejecución.

```bash
python scripts/convert_frames_to_res.py \
    --input  <ruta/frames/originales> \
    --output <ruta/frames/nuevos> \
    --size   64          # o 128
    --workers 32         # procesos en paralelo
```

El script preserva la estructura de subdirectorios y hace saves atómicos (`.tmp` → rename).

---

## Estructura del proyecto

```
dreamer4/
├── scripts/pipeline/
│   ├── run_cycles.sh                   # Pipeline activo (desde cero)
│   ├── run_cycles_pretrain.sh          # Pipeline desde datos HF
│   ├── _pipeline_lib.sh                # Funciones compartidas (setup, steps, eval)
│   ├── launch_phase0_dist.py           # Phase 0 — colección distribuida
│   ├── train_phase1a_tokenizer.py      # Phase 1a — tokenizer
│   ├── train_phase1b_dynamics.py       # Phase 1b — world model
│   ├── train_phase2_finetuning.py      # Phase 2 — BC + reward
│   ├── train_phase3_imagination.py     # Phase 3 — RL en imaginación
│   └── train_utils.py                  # Helpers compartidos (trainer, logger, ckpt)
│
├── scripts/
│   └── convert_frames_to_res.py        # Conversión offline de resolución
│
├── configs/
│   ├── collect/base.yaml               # Config Phase 0
│   ├── tokenizer/base_64x64.yaml       # Config Phase 1a (64×64)
│   ├── tokenizer/base_128x128.yaml     # Config Phase 1a (128×128)
│   ├── dynamics/base_64x64.yaml        # Config Phase 1b
│   ├── finetune/base_64x64.yaml        # Config Phase 2
│   ├── agent/base_64x64.yaml           # Config Phase 3
│   └── data/
│       ├── local.yaml                  # Config de datos local
│       └── cluster.yaml                # Config de datos en cluster
│
├── src/
│   ├── model.py                        # Tokenizer, Dynamics, TaskEmbedder
│   ├── agent.py                        # PolicyHead, RewardHead, ValueHead
│   ├── distributions.py                # SymExp TwoHot (eq.10), PMPO (eq.11)
│   ├── losses.py                       # dynamics_pretrain_loss (eq.7)
│   ├── wm_dataset.py                   # WMDataset (acciones + frames)
│   ├── sharded_frame_dataset.py        # ShardedFrameDataset (frames por shard)
│   ├── task_set.py                     # TASK_SET — 18 tareas DMControl
│   ├── envs/
│   │   └── torchrl_wrappers.py         # ParallelEnv + DMControlEnv (EGL)
│   └── lightning/
│       ├── tokenizer_module.py         # Phase 1a
│       ├── frame_datamodule.py         # DataModule para tokenizer
│       ├── dynamics_module.py          # Phase 1b
│       ├── wm_datamodule.py            # DataModule para dynamics/finetune/agent
│       ├── finetune_module.py          # Phase 2
│       └── agent_module.py             # Phase 3
│
├── tasks.json                          # action_dim + CLIP embeddings (234 tareas)
└── .env                                # WANDB_API_KEY (no commitear)
```

---

## Formato de datos

```
<data_root>/
├── demos/
│   └── <task>.pt       # dict con keys: obs, action, reward, done  [T, ...]
└── frames/
    └── <task>/
        └── <task>_shard<N>.pt   # tensor uint8 [S, T, C, H, W]
```

- Los demos contienen acciones y recompensas; los frames son las observaciones visuales en uint8.
- Los shards de frames permiten carga eficiente y sampling distribuido.

---

## Tareas soportadas

18 tareas de control continuo de DMControl en `TASK_SET` (ver `src/task_set.py`):

```
walker-{stand,walk,run}    hopper-{stand,hop}         reacher-{easy,hard}
cheetah-run                cartpole-{swingup,balance,swingup-sparse,balance-sparse}
cup-catch                  finger-{spin,turn-easy,turn-hard}
acrobot-swingup            pendulum-swingup
```

`tasks.json` contiene 234 tareas con embeddings CLIP y dimensiones de acción.

---

## Citas

```bibtex
@misc{Hafner2025TrainingAgents,
    title={Training Agents Inside of Scalable World Models},
    author={Danijar Hafner and Wilson Yan and Timothy Lillicrap},
    year={2025},
    eprint={2509.24527},
    archivePrefix={arXiv},
}

@misc{Hansen2026Dreamer4PyTorch,
    title={Dreamer 4 in PyTorch},
    author={Nicklas Hansen},
    year={2026},
    publisher={GitHub},
    howpublished={\url{https://github.com/nicklashansen/dreamer4}},
}
```
