# dreamer4 — PyTorch Lightning

Implementación del pipeline completo de [Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527) (Hafner, Yan, Lillicrap, 2025) sobre DMControl, usando PyTorch Lightning + Hydra.

Basado en la implementación de referencia [nicklashansen/dreamer4](https://github.com/nicklashansen/dreamer4), con las Fases 2 y 3 (agent finetuning + imagination training) y el ciclo de recolección recursivo añadidos.

---

## Instalación

```bash
conda env create -f environment.yaml
conda activate dreamer4
```

---

## Pipeline completo

El entrenamiento tiene 5 etapas que se ejecutan en orden. Las Fases 1a y 1b tienen checkpoints preentrenados disponibles en [HuggingFace](https://huggingface.co/nicklashansen/dreamer4) — puedes saltártelas y entrar directo en la Fase 2.

```
Phase 0  →  collect_phase0_data.py     # recolección de episodios
Phase 1a →  train_phase1a_tokenizer.py # tokenizador de frames
Phase 1b →  train_phase1b_dynamics.py  # world model (dynamics)
Phase 2  →  train_phase2_finetuning.py # BC + reward head
Phase 3  →  train_phase3_imagination.py# RL en imaginación (PMPO)
```

---

## Uso rápido con checkpoints preentrenados

Si ya tienes `tokenizer.ckpt` y `dynamics.ckpt` (de HuggingFace o entrenados tú mismo), puedes empezar desde Phase 2:

```bash
# Phase 2 — Agent Finetuning
python train_phase2_finetuning.py \
  finetune.tokenizer_ckpt=./checkpoints/tokenizer.ckpt \
  finetune.dynamics_ckpt=./checkpoints/dynamics.ckpt \
  data.data_dirs=[./data/demos] \
  data.frame_dirs=[./data/frames] \
  trainer.devices=2

# Phase 3 — Imagination Training
python train_phase3_imagination.py \
  agent.finetune_ckpt=./logs/finetune_ckpts/last.ckpt \
  data.data_dirs=[./data/demos] \
  data.frame_dirs=[./data/frames] \
  trainer.devices=2
```

---

## Ciclo recursivo offline

La idea central es que el agente puede **mejorar iterativamente** sin acceso online al entorno: cada ciclo genera datos de mejor calidad que el ciclo anterior, que a su vez producen un mejor agente.

```
Ciclo 0:  datos random  →  entrenar  →  agent_v0.ckpt
Ciclo 1:  datos con agent_v0  →  re-entrenar  →  agent_v1.ckpt
Ciclo N:  datos con agent_vN-1  →  re-entrenar  →  agent_vN.ckpt
```

### Ciclo 0 — bootstrap desde cero (sin datos previos)

```bash
# 1. Recolectar episodios con política random
python collect_phase0_data.py \
  collect.out_data_dir=./data/cycle0/demos \
  collect.out_frames_dir=./data/cycle0/frames \
  collect.n_episodes_per_task=50

# 2. Phase 1a — Tokenizer
srun --nodes=1 --ntasks-per-node=2 --gres=gpu:2 --mem=64G --time=48:00:00 \
  python train_phase1a_tokenizer.py \
  data.frame_dirs=[./data/cycle0/frames] \
  trainer.devices=2

# 3. Phase 1b — Dynamics
srun --nodes=1 --ntasks-per-node=2 --gres=gpu:2 --mem=64G --time=48:00:00 \
  python train_phase1b_dynamics.py \
  dynamics.tokenizer_ckpt=./logs/tokenizer_ckpts/last.ckpt \
  data.data_dirs=[./data/cycle0/demos] \
  data.frame_dirs=[./data/cycle0/frames] \
  trainer.devices=2

# 4. Phase 2 — Finetuning
srun --nodes=1 --ntasks-per-node=2 --gres=gpu:2 --mem=64G --time=24:00:00 \
  python train_phase2_finetuning.py \
  finetune.tokenizer_ckpt=./logs/tokenizer_ckpts/last.ckpt \
  finetune.dynamics_ckpt=./logs/dynamics_ckpts/last.ckpt \
  data.data_dirs=[./data/cycle0/demos] \
  data.frame_dirs=[./data/cycle0/frames] \
  trainer.devices=2

# 5. Phase 3 — Imagination Training
srun --nodes=1 --ntasks-per-node=2 --gres=gpu:2 --mem=64G --time=24:00:00 \
  python train_phase3_imagination.py \
  agent.finetune_ckpt=./logs/finetune_ckpts/last.ckpt \
  data.data_dirs=[./data/cycle0/demos] \
  data.frame_dirs=[./data/cycle0/frames] \
  trainer.devices=2
# → produce: ./logs/agent_ckpts/last.ckpt  (agent_v0)
```

### Ciclo N — usar el agente entrenado para generar mejores datos

```bash
# Recolectar con la política entrenada del ciclo anterior
python collect_phase0_data.py \
  collect.policy=agent \
  collect.agent_ckpt=./logs/agent_ckpts/last.ckpt \
  collect.out_data_dir=./data/cycle1/demos \
  collect.out_frames_dir=./data/cycle1/frames \
  collect.n_episodes_per_task=50

# Re-entrenar Phase 2 y 3 acumulando TODOS los ciclos de datos
# (WMDataModule acepta listas de directorios — une todos los datos)
srun --nodes=1 --ntasks-per-node=2 --gres=gpu:2 --mem=64G --time=24:00:00 \
  python train_phase2_finetuning.py \
  finetune.tokenizer_ckpt=./logs/tokenizer_ckpts/last.ckpt \
  finetune.dynamics_ckpt=./logs/dynamics_ckpts/last.ckpt \
  "data.data_dirs=[./data/cycle0/demos,./data/cycle1/demos]" \
  "data.frame_dirs=[./data/cycle0/frames,./data/cycle1/frames]" \
  trainer.devices=2

srun --nodes=1 --ntasks-per-node=2 --gres=gpu:2 --mem=64G --time=24:00:00 \
  python train_phase3_imagination.py \
  agent.finetune_ckpt=./logs/finetune_ckpts/last.ckpt \
  "data.data_dirs=[./data/cycle0/demos,./data/cycle1/demos]" \
  "data.frame_dirs=[./data/cycle0/frames,./data/cycle1/frames]" \
  trainer.devices=2
# → produce: agent_v1.ckpt
```

> **Nota:** En ciclos posteriores no es necesario re-entrenar el Tokenizer ni el Dynamics desde cero. Basta con re-entrenar las Fases 2 y 3 (o hacer fine-tuning de las 4 a la vez si quieres máxima calidad).

### Script de ciclo automático (bash)

Para automatizar N ciclos completos:

```bash
#!/bin/bash
# run_cycles.sh — ejecuta K ciclos del pipeline recursivo
# Uso: bash run_cycles.sh 3

K=${1:-3}
DEVICES=2
MEM=64G
TIME_LONG=48:00:00   # Phase 1a, 1b
TIME_SHORT=24:00:00  # Phase 2, 3
SRUN="srun --nodes=1 --ntasks-per-node=${DEVICES} --gres=gpu:${DEVICES} --mem=${MEM}"
TOK_CKPT=./logs/tokenizer_ckpts/last.ckpt
DYN_CKPT=./logs/dynamics_ckpts/last.ckpt

DATA_DIRS="./data/cycle0/demos"
FRAME_DIRS="./data/cycle0/frames"

for CYCLE in $(seq 0 $((K - 1))); do
  echo "════════ Ciclo $CYCLE ════════"

  OUT_DATA="./data/cycle${CYCLE}/demos"
  OUT_FRAMES="./data/cycle${CYCLE}/frames"

  # Phase 0 — Colección
  if [ "$CYCLE" -eq 0 ]; then
    python collect_phase0_data.py \
      collect.out_data_dir=$OUT_DATA \
      collect.out_frames_dir=$OUT_FRAMES
  else
    python collect_phase0_data.py \
      collect.policy=agent \
      collect.agent_ckpt=./logs/agent_ckpts/last.ckpt \
      collect.out_data_dir=$OUT_DATA \
      collect.out_frames_dir=$OUT_FRAMES
  fi

  # Acumular datos de todos los ciclos anteriores
  DATA_DIRS="${DATA_DIRS},${OUT_DATA}"
  FRAME_DIRS="${FRAME_DIRS},${OUT_FRAMES}"
  # (ciclo 0: eliminar la primera entrada vacía)
  DATA_DIRS=$(echo $DATA_DIRS | sed 's/^,//')
  FRAME_DIRS=$(echo $FRAME_DIRS | sed 's/^,//')

  # Phase 1a+1b solo en ciclo 0
  if [ "$CYCLE" -eq 0 ]; then
    $SRUN --time=$TIME_LONG \
      python train_phase1a_tokenizer.py \
      "data.frame_dirs=[$FRAME_DIRS]" trainer.devices=$DEVICES

    $SRUN --time=$TIME_LONG \
      python train_phase1b_dynamics.py \
      dynamics.tokenizer_ckpt=$TOK_CKPT \
      "data.data_dirs=[$DATA_DIRS]" \
      "data.frame_dirs=[$FRAME_DIRS]" \
      trainer.devices=$DEVICES
  fi

  # Phase 2
  $SRUN --time=$TIME_SHORT \
    python train_phase2_finetuning.py \
    finetune.tokenizer_ckpt=$TOK_CKPT \
    finetune.dynamics_ckpt=$DYN_CKPT \
    "data.data_dirs=[$DATA_DIRS]" \
    "data.frame_dirs=[$FRAME_DIRS]" \
    trainer.devices=$DEVICES

  # Phase 3
  $SRUN --time=$TIME_SHORT \
    python train_phase3_imagination.py \
    agent.finetune_ckpt=./logs/finetune_ckpts/last.ckpt \
    "data.data_dirs=[$DATA_DIRS]" \
    "data.frame_dirs=[$FRAME_DIRS]" \
    trainer.devices=$DEVICES

  # Guardar checkpoint del ciclo
  cp ./logs/agent_ckpts/last.ckpt ./logs/agent_ckpts/cycle${CYCLE}.ckpt
  echo "Ciclo $CYCLE completo → ./logs/agent_ckpts/cycle${CYCLE}.ckpt"
done
```

---

## Estructura de archivos

```
dreamer4/
├── collect_phase0_data.py        # Phase 0 — recolección
├── train_phase1a_tokenizer.py    # Phase 1a — tokenizer
├── train_phase1b_dynamics.py     # Phase 1b — world model
├── train_phase2_finetuning.py    # Phase 2 — BC + reward
├── train_phase3_imagination.py   # Phase 3 — RL en imaginación
│
├── configs/
│   ├── collect/base.yaml         # config Phase 0
│   ├── tokenizer/                # config Phase 1a
│   ├── dynamics/                 # config Phase 1b
│   ├── finetune/                 # config Phase 2
│   └── agent/                   # config Phase 3
│
├── src/
│   ├── model.py                  # Tokenizer, Dynamics, TaskEmbedder
│   ├── agent.py                  # PolicyHead, RewardHead, ValueHead
│   ├── distributions.py          # SymExp TwoHot (eq.10), PMPO (eq.11)
│   ├── losses.py                 # dynamics_pretrain_loss (eq.7)
│   ├── collector.py              # AgentPolicy + collect_task()
│   ├── wm_dataset.py             # WMDataset
│   └── lightning/
│       ├── tokenizer_module.py   # Phase 1a
│       ├── dynamics_module.py    # Phase 1b
│       ├── finetune_module.py    # Phase 2
│       └── agent_module.py       # Phase 3
│
├── tasks.json                    # action_dim + CLIP embeddings por tarea
└── docs/architecture.md          # diagramas Mermaid del pipeline
```

---

## Tareas soportadas

30 tareas de control continuo de DMControl y MMBench. Ver `src/task_set.py` para la lista completa.

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
