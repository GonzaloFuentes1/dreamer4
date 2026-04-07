# Benchmark Detallado de Recolección (Phase 0)

Este reporte evalúa la latencia y rendimiento de TorchRL + EnvPool + Transformers a lo largo de **20 entornos distribuidos** (mezclando librerías `dmc` y `atari`).

Se mide la simulación sobre 1 worker y sobre 32 workers (procesamiento batch simulado en 1 GPU).

| Ambiente (Task) | Envs Simultáneos | SPS Total | FPS por Env | MS Totales por Batch | Render (MS C++) | Inferencia (MS GPU) |
|:---|:---:|---:|---:|---:|---:|---:|
| `dmc_cheetah_run` | 1 | **118.19** | 118.19 | 8.46 ms | 1.97 ms | 6.49 ms |
| `dmc_cheetah_run` | 32 | **588.06** | 18.38 | 54.42 ms | 0.66 ms | 53.76 ms |
| `dmc_walker_walk` | 1 | **115.64** | 115.64 | 8.65 ms | 0.69 ms | 7.95 ms |
| `dmc_walker_walk` | 32 | **583.50** | 18.23 | 54.84 ms | 0.90 ms | 53.94 ms |
| `dmc_hopper_hop` | 1 | **112.26** | 112.26 | 8.91 ms | 0.61 ms | 8.30 ms |
| `dmc_hopper_hop` | 32 | **591.49** | 18.48 | 54.10 ms | 0.72 ms | 53.38 ms |
| `dmc_cartpole_swingup` | 1 | **121.63** | 121.63 | 8.22 ms | 0.53 ms | 7.69 ms |
| `dmc_cartpole_swingup` | 32 | **590.07** | 18.44 | 54.23 ms | 0.95 ms | 53.28 ms |
| `dmc_pendulum_swingup` | 1 | **133.42** | 133.42 | 7.50 ms | 0.52 ms | 6.97 ms |
| `dmc_pendulum_swingup` | 32 | **588.98** | 18.41 | 54.33 ms | 0.61 ms | 53.73 ms |
| `dmc_reacher_easy` | 1 | **135.64** | 135.64 | 7.37 ms | 0.55 ms | 6.82 ms |
| `dmc_reacher_easy` | 32 | **878.44** | 27.45 | 36.43 ms | 0.78 ms | 35.64 ms |
| `dmc_acrobot_swingup` | 1 | **139.40** | 139.40 | 7.17 ms | 0.55 ms | 6.62 ms |
| `dmc_acrobot_swingup` | 32 | **833.59** | 26.05 | 38.39 ms | 1.00 ms | 37.38 ms |
| `dmc_finger_spin` | 1 | **137.52** | 137.52 | 7.27 ms | 0.61 ms | 6.66 ms |
| `dmc_finger_spin` | 32 | **877.58** | 27.42 | 36.46 ms | 0.70 ms | 35.77 ms |
| `atari_pong` | 1 | **126.09** | 126.09 | 7.93 ms | 0.87 ms | 7.06 ms |
| `atari_pong` | 32 | **839.59** | 26.24 | 38.11 ms | 1.81 ms | 36.31 ms |
| `atari_breakout` | 1 | **131.08** | 131.08 | 7.63 ms | 2.07 ms | 5.55 ms |
| `atari_breakout` | 32 | **830.55** | 25.95 | 38.53 ms | 3.31 ms | 35.22 ms |
| `atari_space_invaders` | 1 | **131.45** | 131.45 | 7.61 ms | 0.85 ms | 6.76 ms |
| `atari_space_invaders` | 32 | **822.04** | 25.69 | 38.93 ms | 1.87 ms | 37.05 ms |
| `atari_seaquest` | 1 | **131.87** | 131.87 | 7.58 ms | 0.85 ms | 6.74 ms |
| `atari_seaquest` | 32 | **837.09** | 26.16 | 38.23 ms | 3.26 ms | 34.97 ms |
| `atari_alien` | 1 | **114.65** | 114.65 | 8.72 ms | 1.02 ms | 7.70 ms |
| `atari_alien` | 32 | **825.19** | 25.79 | 38.78 ms | 3.30 ms | 35.48 ms |
| `atari_qbert` | 1 | **121.57** | 121.57 | 8.23 ms | 0.84 ms | 7.38 ms |
| `atari_qbert` | 32 | **836.79** | 26.15 | 38.24 ms | 2.05 ms | 36.19 ms |
| `atari_asteroids` | 1 | **124.33** | 124.33 | 8.04 ms | 0.85 ms | 7.19 ms |
| `atari_asteroids` | 32 | **843.91** | 26.37 | 37.92 ms | 2.83 ms | 35.09 ms |
| `atari_ms_pacman` | 1 | **120.99** | 120.99 | 8.27 ms | 0.87 ms | 7.39 ms |
| `atari_ms_pacman` | 32 | **838.39** | 26.20 | 38.17 ms | 2.11 ms | 36.06 ms |
| `atari_enduro` | 1 | **117.68** | 117.68 | 8.50 ms | 0.95 ms | 7.54 ms |
| `atari_enduro` | 32 | **827.38** | 25.86 | 38.68 ms | 2.82 ms | 35.86 ms |
| `atari_hero` | 1 | **114.01** | 114.01 | 8.77 ms | 1.00 ms | 7.77 ms |
| `atari_hero` | 32 | **837.46** | 26.17 | 38.21 ms | 2.28 ms | 35.93 ms |
| `atari_freeway` | 1 | **120.54** | 120.54 | 8.30 ms | 1.07 ms | 7.23 ms |
| `atari_freeway` | 32 | **808.16** | 25.25 | 39.60 ms | 3.26 ms | 36.34 ms |
| `atari_riverraid` | 1 | **125.60** | 125.60 | 7.96 ms | 0.89 ms | 7.07 ms |
| `atari_riverraid` | 32 | **821.36** | 25.67 | 38.96 ms | 2.12 ms | 36.84 ms |
