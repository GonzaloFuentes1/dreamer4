# dreamer4 — PyTorch Lightning

[English](README.md) | Español

> **Re-implementación independiente y no oficial.** Este repositorio no está afiliado a los autores del paper
> [Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527) (Hafner, Yan, Lillicrap, 2025;
> arXiv:2509.24527). No afirma reproducir los resultados reportados en el paper, y no se reproducen aquí números
> del paper. Consulta el paper para el método y sus resultados.

Implementación del pipeline completo de Dreamer 4 sobre DMControl, usando PyTorch Lightning + Hydra.
Se apoya en la estructura de [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4), una implementación
en PyTorch con licencia MIT, también no oficial, que provee entrenamiento de tokenizador y de dinámica (ver
[Agradecimientos](#agradecimientos)), y añade las Fases 2 y 3 (agent finetuning + imagination training),
el ciclo de recolección recursivo (self-play) y soporte para tokenizador discreto.

El paper no se redistribuye en este repositorio; está disponible en <https://arxiv.org/abs/2509.24527>.

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

Cinco fases que se ejecutan en orden.

```
Phase 0  →  scripts/pipeline/launch_phase0_dist.py      # recolección de episodios
Phase 1a →  scripts/pipeline/train_phase1a_tokenizer.py # tokenizador de frames (transformer con masked autoencoding y cuello de botella tanh; variante discreta disponible)
Phase 1b →  scripts/pipeline/train_phase1b_dynamics.py  # world model (transformer block-causal, shortcut forcing)
Phase 2  →  scripts/pipeline/train_phase2_finetuning.py # BC + reward head
Phase 3  →  scripts/pipeline/train_phase3_imagination.py # RL en imaginación (PMPO)
```

El entrenamiento acumula datos de todos los ciclos anteriores: cada ciclo `k` entrena sobre `cycle0 + cycle1 + … + cycleK`.

Además de los pipelines en shell descritos más abajo, `scripts/run.py` es un coordinador en Python que lee
`configs/run.yaml` y ejecuta las mismas fases (por ejemplo `python scripts/run.py profile=smoke_test`), y
`scripts/dreamer.py` es una CLI pequeña para inspeccionar datasets y runs.

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
El dataset es el que publica [nicklashansen/dreamer4](https://huggingface.co/datasets/nicklashansen/dreamer4)
y se descarga con `scripts/download_pretrain_data.py`.

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

El entrenamiento lee un archivo `.h5` por tarea (ver [Formato de datos](#formato-de-datos)). Los datos en el formato
`.pt` heredado (el que escribe `scripts/download_pretrain_data.py` y procesa `scripts/convert_frames_to_res.py`)
deben convertirse con `scripts/convert_pt_to_hdf5.py` (`--data`, `--out`) antes de entrenar.

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
│   ├── run_cycles.sh                    # Pipeline activo (desde cero)
│   ├── run_cycles_pretrain.sh           # Pipeline desde datos HF
│   ├── run_cycles_discrete_base.sh      # Pipeline con tokenizador discreto (64×64)
│   ├── _pipeline_lib.sh                 # Funciones compartidas (setup, steps, eval)
│   ├── launch_phase0_dist.py            # Phase 0 — colección distribuida
│   ├── train_phase0_collect_episodes.py # Phase 0 — worker de colección por tarea
│   ├── train_phase1a_tokenizer.py       # Phase 1a — tokenizer
│   ├── train_phase1b_dynamics.py        # Phase 1b — world model
│   ├── train_phase2_finetuning.py       # Phase 2 — BC + reward
│   ├── train_phase3_imagination.py      # Phase 3 — RL en imaginación
│   └── train_utils.py                   # Helpers compartidos (trainer, logger, ckpt)
│
├── scripts/
│   ├── run.py                           # Coordinador en Python del pipeline completo
│   ├── dreamer.py                       # CLI de inspección de datasets y runs
│   ├── convert_frames_to_res.py         # Conversión offline de resolución
│   ├── convert_pt_to_hdf5.py            # Conversión .pt heredado → HDF5
│   ├── download_pretrain_data.py        # Descarga del dataset de HF
│   └── generate_task_embeddings.py      # Embeddings CLIP de texto para tasks.json
│
├── configs/
│   ├── collect/base.yaml                # Config Phase 0
│   ├── tokenizer/base_64x64.yaml        # Config Phase 1a (64×64)
│   ├── tokenizer/base_128x128.yaml      # Config Phase 1a (128×128)
│   ├── dynamics/base_64x64.yaml         # Config Phase 1b
│   ├── finetune/base_64x64.yaml         # Config Phase 2
│   ├── agent/base_64x64.yaml            # Config Phase 3
│   ├── profiles/smoke_test.yaml         # Perfil para scripts/run.py
│   └── data/
│       ├── local.yaml                   # Config de datos local
│       └── cluster.yaml                 # Config de datos en cluster
│
├── src/
│   ├── model.py                         # Tokenizer, Dynamics, TaskEmbedder
│   ├── agent.py                         # PolicyHead, RewardHead, ValueHead
│   ├── distributions.py                 # symlog/symexp, SymExp TwoHot (eq.10), PMPO (eq.11)
│   ├── losses.py                        # dynamics_pretrain_loss (eq.7), schedules de tau
│   ├── loss_norm.py                     # Normalización de losses con RunningRMS
│   ├── hdf5_episode_dataset.py          # HDF5EpisodeWriter / HDF5EpisodeDataset
│   ├── goal_dataset.py                  # GoalDataset (muestreo goal-conditioned)
│   ├── checkpoint_utils.py              # Carga del tokenizer congelado
│   ├── viz.py                           # Helpers de muestreo y visualización
│   ├── task_set.py                      # TASK_SET — 18 tareas DMControl
│   ├── envs/
│   │   └── torchrl_wrappers.py          # ParallelEnv + DMControlEnv (EGL)
│   ├── policies/
│   │   └── exploration.py               # Políticas de colección (random y agente aprendido)
│   └── lightning/
│       ├── tokenizer_module.py          # Phase 1a
│       ├── frame_datamodule.py          # DataModule para tokenizer
│       ├── dynamics_module.py           # Phase 1b
│       ├── wm_datamodule.py             # DataModule para dynamics/finetune/agent
│       ├── finetune_module.py           # Phase 2
│       ├── agent_module.py              # Phase 3
│       └── callbacks.py                 # Callbacks de visualización y evaluación
│
├── tasks.json                           # action_dim + embeddings CLIP (234 tareas)
└── .env                                 # WANDB_API_KEY (no commitear)
```

---

## Formato de datos

Cada tarea se guarda en un único archivo HDF5, escrito de forma incremental por Phase 0 (SWMR, a prueba de
caídas) y leído por `HDF5EpisodeDataset` (`src/hdf5_episode_dataset.py`):

```
<data_dir>/
└── <task>.h5
    ├── pixels     (N, 3, H, W)  uint8     # frames crudos
    ├── action     (N, A)        float32
    ├── reward     (N,)          float32
    ├── episode    (N,)          int64     # id de episodio por step
    ├── ep_len     (E,)          int64     # longitud de cada episodio
    └── ep_offset  (E,)          int64     # índice global de inicio de cada episodio
```

- El entrenamiento apunta a los directorios con los `.h5` mediante `data.data_dirs` (lista explícita) o
  `data.data_root` (un directorio con `.h5`, o con subdirectorios de ciclos que los contengan).
- El dataset tiene dos modos: `frames` (secuencias de frames sin alineación por episodio, para el tokenizer) y
  `episodes` (frames, acciones y recompensas alineados por episodio, para dynamics / finetune / agent).
- El formato heredado (`demos/<task>.pt` con `episode, action, reward`, y `frames/<task>/<task>_shard<N>.pt` con
  tensores `uint8`) es el que escribe `scripts/download_pretrain_data.py` y el que espera
  `scripts/convert_frames_to_res.py`; se convierte con `scripts/convert_pt_to_hdf5.py`.

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

## Estado y alcance

- Es código de investigación en desarrollo. Este repositorio no reporta resultados, y no se afirma reproducir
  ninguno de los resultados del paper.
- El paper aplica Dreamer 4 a Minecraft; este repositorio apunta a DMControl (acciones continuas), por lo que no se
  espera que coincida con el setup ni con los números del paper.
- Para hiperparámetros y detalles de arquitectura, guíate por el código y las configs de este repositorio, no por
  supuestos tomados del paper.

---

## Agradecimientos

- **Paper.** El método es de Hafner, Yan y Lillicrap, *Training Agents Inside of Scalable World Models*
  (arXiv:2509.24527). El crédito del algoritmo corresponde a los autores del paper.
- **[nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4)** (licencia MIT), una implementación no
  oficial de Dreamer 4 en PyTorch de Nicklas Hansen. Este repositorio sigue su estructura general (entrenamiento de
  tokenizer y dinámica sobre DMControl multi-tarea; por ejemplo `model.py`, `task_set.py` y `tasks.json`). Los
  comentarios de `configs/tokenizer/base_128x128.yaml` y `configs/dynamics/base_128x128.yaml` indican que esas
  configs están alineadas con su configuración de DMControl, y `scripts/download_pretrain_data.py` descarga el
  dataset que publica en HuggingFace, sujeto a los términos indicados allí.

---

## Citas

Si usas el método, cita el paper:

```bibtex
@misc{Hafner2025TrainingAgents,
    title={Training Agents Inside of Scalable World Models},
    author={Danijar Hafner and Wilson Yan and Timothy Lillicrap},
    year={2025},
    eprint={2509.24527},
    archivePrefix={arXiv},
    url={https://arxiv.org/abs/2509.24527},
}
```

Si usas la implementación en PyTorch en la que se apoya este repositorio:

```bibtex
@misc{Hansen2026Dreamer4PyTorch,
    title={Dreamer 4 in PyTorch},
    author={Nicklas Hansen},
    year={2026},
    publisher={GitHub},
    howpublished={\url{https://github.com/nicklashansen/dreamer4}},
}
```
