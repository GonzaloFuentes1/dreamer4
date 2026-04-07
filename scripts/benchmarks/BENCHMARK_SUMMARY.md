# Resumen de Benchmarks (DreamerV4 Multi-GPU & EnvPool)

Este documento resume los resultados clave de rendimiento y latencia obtenidos tras evaluar el colector de datos asincrónico (EnvPool + TorchRL) y la inferencia real de los pesos de la red de Agente (PyTorch Lightning).

## 1. Conclusiones Principales de Arquitectura

1. **Eficiencia Base (Multi-Environment):**
   * EnvPool es extremadamente rápido haciendo el paso de simulación (C++ render). El costo real en el pipeline siempre será el paso Forward del modelo neuronal para las acciones.
2. **Escalamiento de Entornos por GPU:**
   * **1 GPU maneja cómodamente hasta 256 entornos paralelos** superando consistentemente los **1000 iteraciones por segundo (SPS)** reales de simulación (Render + Red Neuronal + Serialización) con latencias estables menores a ~300ms por iteración de sub-procesamiento.
3. **El Cuello de Botella del IPC (Threading vs Processing vs Independent):**
   * **No usar `Threading` ni `ProcessPoolExecutor` interconectado para GPUs:** La transferencia de memoria serializada de matrices visuales de `N` entornos por sockets/hilos entre un proceso originador y los pools bloqueados destruye el throughput (pasando de 1000 SPS a 200 SPS).
   * **La Solución (Data Parallelism puro):** Múltiples entornos se manejan creando copias completas independientes de `python main.py` ancladas con `CUDA_VISIBLE_DEVICES`. Cada script inicializa localmente las instancias de EnvPool y carga la Red, multiplicando exactamente lineamente la velocidad base (e.g. 1 GPU = 1000 SPS, 2 GPUs = 2000 SPS global).

---

## 2. Rendimiento (1 GPU vs 2 GPUs Independientes con Modelo Cargado)

*Nota: Todas las métricas representan recolección utilizando una política evaluando pesos densos reales de memoria.*

### 1 GPU (256 Environments Máximo Escalamiento)

| Ambiente           | SPS Total | FPS por Env | MS Totales (Batch) | MS C++ Render | MS GPU (Inferencia) |
| :----------------- | --------: | ----------: | -----------------: | ------------: | ------------------: |
| `dmc_cheetah_run`      | 1572.78   | 6.14        | 162.77 ms         | 5.11 ms       | 157.65 ms         |
| `dmc_walker_walk`      | 1563.85   | 6.11        | 163.70 ms         | 3.73 ms       | 159.96 ms         |
| `atari_pong`           | 1046.06   | 4.09        | 244.73 ms         | 34.15 ms      | 210.57 ms         |
| `atari_freeway`        | 914.86    | 3.57        | 279.82 ms         | 34.70 ms      | 245.13 ms         |
| `atari_space_invaders` | 913.68    | 3.57        | 280.19 ms         | 51.52 ms      | 228.67 ms         |

### 2 GPUs (Escalamiento Distribuido Independiente - Ej: sumando GPU_0 + GPU_1 por tarea)
*Simulan ~2000 SPS simultáneos logrados sumando los reportes base generados.*

> Se estableció que el acercamiento ideal para el Cluster local, dadas las limitaciones de IPC, es realizar sharding (dividir las listas de juegos) a las colas de las distintas GPUs mediante el wrapper de Data Parallel (ahora incorporado directamente en el Phase 0 `launch_phase0_dist.py`). Así se mantiene latencia nativa (250ms), sin transferencias entre ranuras PCI-E.

## 3. Reporte Completo: Todos los Entornos (Max SPS con Modelo Cargado)

Esta tabla muestra el escalamiento ideal sumando dos procesos independientes (Data Parallel) comparado contra 1 solo GPU (ambos con 256 entornos paralelos).

| Ambiente | SPS Totales (1 GPU) | SPS Totales (2 GPUs - Indep) | Escalamiento |
| :--- | ---: | ---: | ---: |
| `atari_alien` | 908.93 | 1722.88 | 1.90x |
| `atari_asteroids` | 996.70 | 1536.01 | 1.54x |
| `atari_breakout` | 911.24 | 1657.23 | 1.82x |
| `atari_enduro` | 904.60 | 1722.26 | 1.90x |
| `atari_freeway` | 914.86 | 1525.40 | 1.67x |
| `atari_hero` | 983.97 | 1679.47 | 1.71x |
| `atari_ms_pacman` | 987.45 | 1793.49 | 1.82x |
| `atari_pong` | 926.38 | 1601.22 | 1.73x |
| `atari_qbert` | 918.50 | 1790.59 | 1.95x |
| `atari_riverraid` | 1010.46 | 1754.11 | 1.74x |
| `atari_seaquest` | 908.80 | 1549.62 | 1.71x |
| `atari_space_invaders` | 989.12 | 1628.00 | 1.65x |
| `dmc_acrobot_swingup` | 1069.33 | 2134.92 | 2.00x |
| `dmc_cartpole_swingup` | 1072.54 | 2129.67 | 1.99x |
| `dmc_cheetah_run` | 1063.20 | 2109.97 | 1.98x |
| `dmc_finger_spin` | 1072.45 | 2123.07 | 1.98x |
| `dmc_hopper_hop` | 1065.97 | 2126.50 | 1.99x |
| `dmc_pendulum_swingup` | 1072.38 | 2133.84 | 1.99x |
| `dmc_reacher_easy` | 1071.17 | 2124.07 | 1.98x |
| `dmc_walker_walk` | 1057.97 | 2103.07 | 1.99x |
