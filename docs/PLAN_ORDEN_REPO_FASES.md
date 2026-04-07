# Plan de Orden del Repo (Fases)

Fecha: 2026-03-16
Estado actual: Fase 2 (benchmarks) aplicada + orden seguro inicial de raiz.

## Objetivo

Reducir ruido en la raiz del repo y mantener flujo operativo estable para training/collect/benchmarks.

## Estado de Fases

- Fase 0 (hecha): consolidacion de benchmarks en un resumen unico y archivado historico.
- Fase 1 (en progreso): orden de raiz sin romper jobs activos.
- Cambio aplicado: `collect_phase0_data.py` renombrado a `scripts/pipeline/train_phase0_collect_episodes.py`.
  - Se removio el wrapper legacy para reducir ruido en raiz.
- Fase 2 (en progreso): mover entrypoints de pipeline fuera de la raiz con compatibilidad.
  - `run_cycles.sh` y `run_cycles_pretrain.sh` se movieron a `scripts/pipeline/`.
  - Se mantuvieron wrappers en raiz para no romper comandos existentes.
- Fase 3 (pendiente): estandarizar runtime artifacts (`logs`, `outputs`, `wandb`, `data`).

## Fase 1 — Cambios seguros (sin disrupcion)

Aplicado:
- eliminado `__pycache__/` de la raiz.
- restaurada ruta de log activo en `benchmarks/benchmark_mjwarp_3gpu_3383.log` mediante symlink al archivo archivado.

Pendiente dentro de Fase 1:
- limpiar archivos temporales sueltos no versionables en raiz (si aparecen nuevos).
- revisar README para que apunte al resumen unico de benchmarks y al archivo historico.

## Fase 2 — Reorganizacion de entrypoints (cuando no haya jobs criticos corriendo)

Propuesta:
- crear `scripts/pipeline/`.
- mover alli:
  - `scripts/pipeline/train_phase0_collect_episodes.py`
  - `scripts/pipeline/train_phase1a_tokenizer.py`
  - `scripts/pipeline/train_phase1b_dynamics.py`
  - `scripts/pipeline/train_phase2_finetuning.py`
  - `scripts/pipeline/train_phase3_imagination.py`
  - `run_cycles.sh`
  - `run_cycles_pretrain.sh`

Compatibilidad:
- dejar wrappers livianos en raiz (1-2 lineas) por una ventana de transicion.
- actualizar README con las rutas nuevas.

## Fase 3 — Politica de artefactos

- mantener artefactos de ejecucion en:
  - `logs/`
  - `data/`
  - `wandb/`
  - `docs/archive/`
- `outputs/` actualmente es un symlink a `logs/runs` (compatibilidad visual).
- configuraciones Hydra de train/collect/download ahora escriben explicitamente en:
  - `logs/runs/...` (run)
  - `logs/multirun/...` (sweeps)
- evitar nuevos `.log` y reportes sueltos en raiz/benchmarks fuera de resumen unico.

## Criterio de cierre

El orden se considera completo cuando:
- la raiz contiene solo codigo/config/documentacion principal.
- los entrypoints estan agrupados en `scripts/pipeline/`.
- los artefactos de ejecucion quedan fuera de la vista principal del repo.
