# dreamer4 — PyTorch Lightning

English | [Español](README.es.md)

> **Unofficial, independent re-implementation.** This repository is not affiliated with the authors of the paper
> [Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527) (Hafner, Yan, Lillicrap, 2025;
> arXiv:2509.24527). It does not claim to reproduce the results reported in the paper, and no numbers from the
> paper are reproduced here. Refer to the paper for the method and its reported results.

Implementation of the full Dreamer 4 pipeline on DMControl, using PyTorch Lightning + Hydra.
It builds on the structure of [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4), an MIT-licensed,
also unofficial, PyTorch implementation that provides tokenizer and dynamics training (see
[Acknowledgements](#acknowledgements)), and adds Phases 2 and 3 (agent finetuning + imagination training),
the recursive collection cycle (self-play) and support for a discrete tokenizer.

The paper is not redistributed in this repository; it is available at <https://arxiv.org/abs/2509.24527>.

---

## Installation

```bash
# Requires uv (https://github.com/astral-sh/uv)
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

Create a `.env` file at the repo root with your credentials:

```bash
export WANDB_API_KEY=your_key_here
```

### Headless environment (cluster / SLURM)

Headless rendering uses EGL. The required variables are set automatically by `_pipeline_lib.sh`:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
```

---

## Full pipeline

Five phases that run in order.

```
Phase 0  →  scripts/pipeline/launch_phase0_dist.py      # episode collection
Phase 1a →  scripts/pipeline/train_phase1a_tokenizer.py # frame tokenizer (masked-autoencoding transformer with tanh bottleneck; discrete variant available)
Phase 1b →  scripts/pipeline/train_phase1b_dynamics.py  # world model (block-causal transformer, shortcut forcing)
Phase 2  →  scripts/pipeline/train_phase2_finetuning.py # BC + reward head
Phase 3  →  scripts/pipeline/train_phase3_imagination.py # RL in imagination (PMPO)
```

Training accumulates data from all previous cycles: cycle `k` trains on `cycle0 + cycle1 + … + cycleK`.

Besides the shell pipelines described below, `scripts/run.py` is a Python coordinator that reads
`configs/run.yaml` and runs the same phases (for example `python scripts/run.py profile=smoke_test`), and
`scripts/dreamer.py` is a small CLI to inspect datasets and runs.

---

## Execution modes

### Mode 1 — From scratch (no prior data)

The full pipeline starts from a random policy in cycle 0 and improves the agent cycle after cycle.

```bash
# 64×64, 1 GPU, 30 cycles
RES=64 srun --nodes=1 --ntasks=1 --gpus=1 \
    --cpus-per-task=64 --mem=128G --time=24:00:00 \
    bash scripts/pipeline/run_cycles.sh

# 128×128, 3 GPUs, 30 cycles
RES=128 srun --nodes=1 --ntasks=1 --gpus=3 \
    --cpus-per-task=48 --mem=256G --time=72:00:00 \
    bash scripts/pipeline/run_cycles.sh
```

Optional environment variables for `run_cycles.sh`:

| Variable | Default | Description |
|---|---|---|
| `RES` | `64` | Resolution: `64` or `128` |
| `K` | `30` | Number of cycles |
| `RUN_TAG` | `active_64x64` | Run name (folder in `./runs/`) |
| `TASKS` | 5 DMC tasks | Comma-separated list |
| `START_FROM_DYNAMICS_CYCLE0` | `false` | Start from an external tokenizer + data |
| `SOURCE_RUN_TAG` | — | Source run (requires the previous option) |
| `DEVICES` | auto-detect | Number of GPUs |

### Mode 2 — From pretrained data (HuggingFace)

Cycle 0 uses data downloaded from HuggingFace, without collection. Later cycles collect with the learned agent.
The dataset is the one published by [nicklashansen/dreamer4](https://huggingface.co/datasets/nicklashansen/dreamer4)
and is fetched with `scripts/download_pretrain_data.py`.

```bash
# Convert frames to the desired resolution (only once)
python scripts/convert_frames_to_res.py \
    --input  ./data/cycle0-pretrained/frames \
    --output ./data/pretrained-64x64/frames \
    --size 64 --workers 32

# Create a symlink for the demos (the demos are already at the right resolution)
ln -s ./data/cycle0-pretrained/demos ./data/pretrained-64x64/demos

# Launch the pipeline
RES=64 BASE_DATA_ROOT=./data/pretrained-64x64 \
    srun --gpus=4 --cpus-per-task=64 --mem=384G --time=72:00:00 \
    bash scripts/pipeline/run_cycles_pretrain.sh
```

Training reads one `.h5` file per task (see [Data format](#data-format)). Data in the legacy `.pt` layout (as written
by `scripts/download_pretrain_data.py` and handled by `scripts/convert_frames_to_res.py`) must be converted with
`scripts/convert_pt_to_hdf5.py` (`--data`, `--out`) before training.

Additional variables for `run_cycles_pretrain.sh`:

| Variable | Default | Description |
|---|---|---|
| `BASE_DATA_ROOT` | `./data/cycle0-pretrained` | Directory with the cycle 0 data (HF) |
| `RUN_TAG` | `pt_active` | Run name |
| `K` | `10` | Number of cycles |

---

## Frame preprocessing

Frames must be at the model resolution **before** training. Training does not resize at run time.

```bash
python scripts/convert_frames_to_res.py \
    --input  <path/to/original/frames> \
    --output <path/to/new/frames> \
    --size   64          # or 128
    --workers 32         # parallel processes
```

The script preserves the subdirectory structure and saves atomically (`.tmp` → rename).

---

## Project structure

```
dreamer4/
├── scripts/pipeline/
│   ├── run_cycles.sh                    # Active pipeline (from scratch)
│   ├── run_cycles_pretrain.sh           # Pipeline from HF data
│   ├── run_cycles_discrete_base.sh      # Pipeline with the discrete tokenizer (64×64)
│   ├── _pipeline_lib.sh                 # Shared functions (setup, steps, eval)
│   ├── launch_phase0_dist.py            # Phase 0 — distributed collection
│   ├── train_phase0_collect_episodes.py # Phase 0 — per-task collection worker
│   ├── train_phase1a_tokenizer.py       # Phase 1a — tokenizer
│   ├── train_phase1b_dynamics.py        # Phase 1b — world model
│   ├── train_phase2_finetuning.py       # Phase 2 — BC + reward
│   ├── train_phase3_imagination.py      # Phase 3 — RL in imagination
│   └── train_utils.py                   # Shared helpers (trainer, logger, ckpt)
│
├── scripts/
│   ├── run.py                           # Python coordinator for the full pipeline
│   ├── dreamer.py                       # CLI to inspect datasets and runs
│   ├── convert_frames_to_res.py         # Offline resolution conversion
│   ├── convert_pt_to_hdf5.py            # Legacy .pt → HDF5 conversion
│   ├── download_pretrain_data.py        # Download the HF dataset
│   └── generate_task_embeddings.py      # CLIP text embeddings for tasks.json
│
├── configs/
│   ├── collect/base.yaml                # Phase 0 config
│   ├── tokenizer/base_64x64.yaml        # Phase 1a config (64×64)
│   ├── tokenizer/base_128x128.yaml      # Phase 1a config (128×128)
│   ├── dynamics/base_64x64.yaml         # Phase 1b config
│   ├── finetune/base_64x64.yaml         # Phase 2 config
│   ├── agent/base_64x64.yaml            # Phase 3 config
│   ├── profiles/smoke_test.yaml         # Profile for scripts/run.py
│   └── data/
│       ├── local.yaml                   # Local data config
│       └── cluster.yaml                 # Cluster data config
│
├── src/
│   ├── model.py                         # Tokenizer, Dynamics, TaskEmbedder
│   ├── agent.py                         # PolicyHead, RewardHead, ValueHead
│   ├── distributions.py                 # symlog/symexp, SymExp TwoHot (eq.10), PMPO (eq.11)
│   ├── losses.py                        # dynamics_pretrain_loss (eq.7), tau schedules
│   ├── loss_norm.py                     # RunningRMS loss normalisation
│   ├── hdf5_episode_dataset.py          # HDF5EpisodeWriter / HDF5EpisodeDataset
│   ├── goal_dataset.py                  # GoalDataset (goal-conditioned sampling)
│   ├── checkpoint_utils.py              # Frozen tokenizer loading
│   ├── viz.py                           # Sampling and visualisation helpers
│   ├── task_set.py                      # TASK_SET — 18 DMControl tasks
│   ├── envs/
│   │   └── torchrl_wrappers.py          # ParallelEnv + DMControlEnv (EGL)
│   ├── policies/
│   │   └── exploration.py               # Random and learned-agent collection policies
│   └── lightning/
│       ├── tokenizer_module.py          # Phase 1a
│       ├── frame_datamodule.py          # DataModule for the tokenizer
│       ├── dynamics_module.py           # Phase 1b
│       ├── wm_datamodule.py             # DataModule for dynamics/finetune/agent
│       ├── finetune_module.py           # Phase 2
│       ├── agent_module.py              # Phase 3
│       └── callbacks.py                 # Visualisation and evaluation callbacks
│
├── tasks.json                           # action_dim + CLIP embeddings (234 tasks)
└── .env                                 # WANDB_API_KEY (do not commit)
```

---

## Data format

Each task is stored in a single HDF5 file, written incrementally by Phase 0 (SWMR, crash-safe) and read by
`HDF5EpisodeDataset` (`src/hdf5_episode_dataset.py`):

```
<data_dir>/
└── <task>.h5
    ├── pixels     (N, 3, H, W)  uint8     # raw frames
    ├── action     (N, A)        float32
    ├── reward     (N,)          float32
    ├── episode    (N,)          int64     # episode id per step
    ├── ep_len     (E,)          int64     # length of each episode
    └── ep_offset  (E,)          int64     # global start index of each episode
```

- Training is pointed at the directories holding the `.h5` files with `data.data_dirs` (explicit list) or
  `data.data_root` (a directory with `.h5` files, or with cycle subdirectories that contain them).
- The dataset has two modes: `frames` (frame sequences without episode alignment, used for the tokenizer) and
  `episodes` (frames, actions and rewards aligned by episode, used for dynamics / finetune / agent).
- The legacy layout (`demos/<task>.pt` with `episode, action, reward`, and `frames/<task>/<task>_shard<N>.pt` with
  `uint8` tensors) is what `scripts/download_pretrain_data.py` writes and what `scripts/convert_frames_to_res.py`
  expects; convert it with `scripts/convert_pt_to_hdf5.py`.

---

## Supported tasks

18 DMControl continuous-control tasks in `TASK_SET` (see `src/task_set.py`):

```
walker-{stand,walk,run}    hopper-{stand,hop}         reacher-{easy,hard}
cheetah-run                cartpole-{swingup,balance,swingup-sparse,balance-sparse}
cup-catch                  finger-{spin,turn-easy,turn-hard}
acrobot-swingup            pendulum-swingup
```

`tasks.json` contains 234 tasks with CLIP embeddings and action dimensions.

---

## Status and scope

- This is research code under development. No results are reported in this repository, and none of the paper's
  results are claimed to be reproduced.
- The paper applies Dreamer 4 to Minecraft; this repository targets DMControl (continuous actions), so it is not
  expected to match the paper's setup or numbers.
- For hyperparameters and architecture details, rely on the code and configs in this repository rather than on
  assumptions taken from the paper.

---

## Acknowledgements

- **Paper.** The method is from Hafner, Yan and Lillicrap, *Training Agents Inside of Scalable World Models*
  (arXiv:2509.24527). Credit for the algorithm belongs to the paper's authors.
- **[nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4)** (MIT license), an unofficial PyTorch
  implementation of Dreamer 4 by Nicklas Hansen. This repository follows its overall layout (tokenizer and
  dynamics training on multi-task DMControl; for example `model.py`, `task_set.py` and `tasks.json`). The comments in
  `configs/tokenizer/base_128x128.yaml` and `configs/dynamics/base_128x128.yaml` state that those configs are
  aligned with its DMControl configuration, and `scripts/download_pretrain_data.py` downloads the dataset it
  publishes on HuggingFace, which is subject to the terms stated there.

---

## Citation

If you use the method, cite the paper:

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

If you use the upstream PyTorch implementation this repository builds on:

```bibtex
@misc{Hansen2026Dreamer4PyTorch,
    title={Dreamer 4 in PyTorch},
    author={Nicklas Hansen},
    year={2026},
    publisher={GitHub},
    howpublished={\url{https://github.com/nicklashansen/dreamer4}},
}
```
