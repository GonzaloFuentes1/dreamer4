# Benchmark de Recolección Escalado (1 GPU) - Preentrenado

Evaluación de latencia y rendimiento variando grandes volúmenes de workers y GPUs, **cargando pesos de un modelo real**.

Mide la simulación empleando infraestructura Multi-worker ([64, 128, 256]) distribuida en 1 GPUs.
                                                                
| Ambiente (Task) | Envs Simultáneos | SPS Total | FPS por Env | MS Totales por Batch | Render (MS C++) | Inferencia (MS GPU) |
|:---|:---:|---:|---:|---:|---:|---:|
| `dmc_cheetah_run` | 64 | **939.39** | 14.68 | 68.13 ms | 8.69 ms | 59.44 ms |
| `dmc_cheetah_run` | 128 | **1049.39** | 8.20 | 121.98 ms | 1.51 ms | 120.47 ms |
| `dmc_cheetah_run` | 256 | **1061.32** | 4.15 | 241.21 ms | 4.77 ms | 236.44 ms |
| `dmc_walker_walk` | 64 | **939.79** | 14.68 | 68.10 ms | 2.12 ms | 65.98 ms |
| `dmc_walker_walk` | 128 | **1030.36** | 8.05 | 124.23 ms | 4.58 ms | 119.65 ms |
| `dmc_walker_walk` | 256 | **1060.67** | 4.14 | 241.36 ms | 6.03 ms | 235.33 ms |
| `dmc_hopper_hop` | 64 | **941.49** | 14.71 | 67.98 ms | 1.33 ms | 66.64 ms |
| `dmc_hopper_hop` | 128 | **1048.94** | 8.19 | 122.03 ms | 1.97 ms | 120.06 ms |
| `dmc_hopper_hop` | 256 | **1069.43** | 4.18 | 239.38 ms | 4.36 ms | 235.02 ms |
| `dmc_cartpole_swingup` | 64 | **947.27** | 14.80 | 67.56 ms | 1.04 ms | 66.52 ms |
| `dmc_cartpole_swingup` | 128 | **1052.46** | 8.22 | 121.62 ms | 1.61 ms | 120.01 ms |
| `dmc_cartpole_swingup` | 256 | **1071.89** | 4.19 | 238.83 ms | 2.47 ms | 236.36 ms |
| `dmc_pendulum_swingup` | 64 | **945.68** | 14.78 | 67.68 ms | 1.57 ms | 66.11 ms |
| `dmc_pendulum_swingup` | 128 | **1051.31** | 8.21 | 121.75 ms | 1.32 ms | 120.43 ms |
| `dmc_pendulum_swingup` | 256 | **1075.94** | 4.20 | 237.93 ms | 2.15 ms | 235.78 ms |
| `dmc_reacher_easy` | 64 | **934.51** | 14.60 | 68.49 ms | 0.91 ms | 67.58 ms |
| `dmc_reacher_easy` | 128 | **1049.29** | 8.20 | 121.99 ms | 1.70 ms | 120.28 ms |
| `dmc_reacher_easy` | 256 | **1070.89** | 4.18 | 239.05 ms | 3.14 ms | 235.91 ms |
| `dmc_acrobot_swingup` | 64 | **946.71** | 14.79 | 67.60 ms | 0.96 ms | 66.65 ms |
| `dmc_acrobot_swingup` | 128 | **1049.27** | 8.20 | 121.99 ms | 1.78 ms | 120.21 ms |
| `dmc_acrobot_swingup` | 256 | **1075.01** | 4.20 | 238.14 ms | 2.93 ms | 235.20 ms |
| `dmc_finger_spin` | 64 | **939.45** | 14.68 | 68.13 ms | 1.27 ms | 66.86 ms |
| `dmc_finger_spin` | 128 | **1048.59** | 8.19 | 122.07 ms | 2.08 ms | 119.99 ms |
| `dmc_finger_spin` | 256 | **1075.73** | 4.20 | 237.98 ms | 2.23 ms | 235.75 ms |
| `atari_pong` | 64 | **669.02** | 10.45 | 95.66 ms | 30.01 ms | 65.65 ms |
| `atari_pong` | 128 | **761.21** | 5.95 | 168.15 ms | 100.18 ms | 67.97 ms |
| `atari_pong` | 256 | **808.60** | 3.16 | 316.60 ms | 40.11 ms | 276.49 ms |
| `atari_breakout` | 64 | **627.22** | 9.80 | 102.04 ms | 74.34 ms | 27.69 ms |
| `atari_breakout` | 128 | **750.54** | 5.86 | 170.54 ms | 78.75 ms | 91.80 ms |
| `atari_breakout` | 256 | **830.75** | 3.25 | 308.16 ms | 62.29 ms | 245.86 ms |
| `atari_space_invaders` | 64 | **675.60** | 10.56 | 94.73 ms | 34.00 ms | 60.74 ms |
| `atari_space_invaders` | 128 | **684.79** | 5.35 | 186.92 ms | 57.82 ms | 129.10 ms |
| `atari_space_invaders` | 256 | **808.78** | 3.16 | 316.53 ms | 48.87 ms | 267.66 ms |
| `atari_seaquest` | 64 | **665.88** | 10.40 | 96.11 ms | 33.05 ms | 63.06 ms |
| `atari_seaquest` | 128 | **839.77** | 6.56 | 152.42 ms | 56.22 ms | 96.21 ms |
| `atari_seaquest` | 256 | **749.73** | 2.93 | 341.46 ms | 75.00 ms | 266.46 ms |
| `atari_alien` | 64 | **653.49** | 10.21 | 97.94 ms | 31.05 ms | 66.88 ms |
| `atari_alien` | 128 | **819.57** | 6.40 | 156.18 ms | 41.74 ms | 114.44 ms |
| `atari_alien` | 256 | **860.01** | 3.36 | 297.67 ms | 57.30 ms | 240.37 ms |
| `atari_qbert` | 64 | **653.32** | 10.21 | 97.96 ms | 5.71 ms | 92.25 ms |
| `atari_qbert` | 128 | **702.19** | 5.49 | 182.29 ms | 12.31 ms | 169.97 ms |
| `atari_qbert` | 256 | **880.39** | 3.44 | 290.78 ms | 15.82 ms | 274.96 ms |
| `atari_asteroids` | 64 | **830.38** | 12.97 | 77.07 ms | 68.99 ms | 8.08 ms |
| `atari_asteroids` | 128 | **705.94** | 5.52 | 181.32 ms | 32.64 ms | 148.67 ms |
| `atari_asteroids` | 256 | **736.72** | 2.88 | 347.49 ms | 33.10 ms | 314.39 ms |
| `atari_ms_pacman` | 64 | **679.08** | 10.61 | 94.24 ms | 82.61 ms | 11.63 ms |
| `atari_ms_pacman` | 128 | **817.28** | 6.39 | 156.62 ms | 18.04 ms | 138.58 ms |
| `atari_ms_pacman` | 256 | **883.60** | 3.45 | 289.72 ms | 95.20 ms | 194.52 ms |
| `atari_enduro` | 64 | **653.49** | 10.21 | 97.94 ms | 71.40 ms | 26.54 ms |
| `atari_enduro` | 128 | **826.82** | 6.46 | 154.81 ms | 70.68 ms | 84.13 ms |
| `atari_enduro` | 256 | **861.14** | 3.36 | 297.28 ms | 74.09 ms | 223.19 ms |
| `atari_hero` | 64 | **645.16** | 10.08 | 99.20 ms | 40.39 ms | 58.81 ms |
| `atari_hero` | 128 | **789.15** | 6.17 | 162.20 ms | 20.50 ms | 141.70 ms |
| `atari_hero` | 256 | **879.85** | 3.44 | 290.96 ms | 81.15 ms | 209.81 ms |
| `atari_freeway` | 64 | **652.94** | 10.20 | 98.02 ms | 40.35 ms | 57.67 ms |
| `atari_freeway` | 128 | **826.06** | 6.45 | 154.95 ms | 52.98 ms | 101.97 ms |
| `atari_freeway` | 256 | **728.43** | 2.85 | 351.44 ms | 80.85 ms | 270.59 ms |
| `atari_riverraid` | 64 | **656.64** | 10.26 | 97.47 ms | 28.95 ms | 68.52 ms |
| `atari_riverraid` | 128 | **811.56** | 6.34 | 157.72 ms | 46.80 ms | 110.92 ms |
| `atari_riverraid` | 256 | **827.00** | 3.23 | 309.55 ms | 67.88 ms | 241.67 ms |
