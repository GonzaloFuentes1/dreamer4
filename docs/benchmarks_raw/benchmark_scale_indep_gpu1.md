# Benchmark de Recolección Escalado (1 GPU) - Preentrenado

Evaluación de latencia y rendimiento variando grandes volúmenes de workers y GPUs, **cargando pesos de un modelo real**.

Mide la simulación empleando infraestructura Multi-worker ([64, 128, 256]) distribuida en 1 GPUs.
                                                                
| Ambiente (Task) | Envs Simultáneos | SPS Total | FPS por Env | MS Totales por Batch | Render (MS C++) | Inferencia (MS GPU) |
|:---|:---:|---:|---:|---:|---:|---:|
| `dmc_cheetah_run` | 64 | **935.38** | 14.62 | 68.42 ms | 8.66 ms | 59.76 ms |
| `dmc_cheetah_run` | 128 | **1033.83** | 8.08 | 123.81 ms | 1.76 ms | 122.05 ms |
| `dmc_cheetah_run` | 256 | **1048.65** | 4.10 | 244.12 ms | 6.05 ms | 238.08 ms |
| `dmc_walker_walk` | 64 | **933.84** | 14.59 | 68.53 ms | 1.81 ms | 66.73 ms |
| `dmc_walker_walk` | 128 | **1023.30** | 7.99 | 125.09 ms | 4.83 ms | 120.25 ms |
| `dmc_walker_walk` | 256 | **1042.40** | 4.07 | 245.59 ms | 14.86 ms | 230.73 ms |
| `dmc_hopper_hop` | 64 | **932.31** | 14.57 | 68.65 ms | 1.55 ms | 67.10 ms |
| `dmc_hopper_hop` | 128 | **1035.77** | 8.09 | 123.58 ms | 1.98 ms | 121.60 ms |
| `dmc_hopper_hop` | 256 | **1057.07** | 4.13 | 242.18 ms | 3.54 ms | 238.64 ms |
| `dmc_cartpole_swingup` | 64 | **931.51** | 14.55 | 68.71 ms | 0.95 ms | 67.76 ms |
| `dmc_cartpole_swingup` | 128 | **1036.91** | 8.10 | 123.44 ms | 1.63 ms | 121.81 ms |
| `dmc_cartpole_swingup` | 256 | **1057.78** | 4.13 | 242.02 ms | 2.17 ms | 239.85 ms |
| `dmc_pendulum_swingup` | 64 | **939.14** | 14.67 | 68.15 ms | 1.00 ms | 67.14 ms |
| `dmc_pendulum_swingup` | 128 | **1037.28** | 8.10 | 123.40 ms | 1.28 ms | 122.12 ms |
| `dmc_pendulum_swingup` | 256 | **1057.90** | 4.13 | 241.99 ms | 2.84 ms | 239.15 ms |
| `dmc_reacher_easy` | 64 | **936.51** | 14.63 | 68.34 ms | 0.97 ms | 67.37 ms |
| `dmc_reacher_easy` | 128 | **1034.79** | 8.08 | 123.70 ms | 1.30 ms | 122.40 ms |
| `dmc_reacher_easy` | 256 | **1053.18** | 4.11 | 243.07 ms | 1.84 ms | 241.23 ms |
| `dmc_acrobot_swingup` | 64 | **933.61** | 14.59 | 68.55 ms | 1.31 ms | 67.24 ms |
| `dmc_acrobot_swingup` | 128 | **1036.12** | 8.09 | 123.54 ms | 2.05 ms | 121.49 ms |
| `dmc_acrobot_swingup` | 256 | **1059.91** | 4.14 | 241.53 ms | 2.11 ms | 239.42 ms |
| `dmc_finger_spin` | 64 | **937.93** | 14.66 | 68.24 ms | 1.57 ms | 66.67 ms |
| `dmc_finger_spin` | 128 | **1036.40** | 8.10 | 123.50 ms | 2.74 ms | 120.77 ms |
| `dmc_finger_spin` | 256 | **1047.34** | 4.09 | 244.43 ms | 2.50 ms | 241.93 ms |
| `atari_pong` | 64 | **616.10** | 9.63 | 103.88 ms | 66.46 ms | 37.42 ms |
| `atari_pong` | 128 | **834.78** | 6.52 | 153.33 ms | 37.19 ms | 116.15 ms |
| `atari_pong` | 256 | **792.62** | 3.10 | 322.98 ms | 82.24 ms | 240.74 ms |
| `atari_breakout` | 64 | **436.10** | 6.81 | 146.75 ms | 55.81 ms | 90.94 ms |
| `atari_breakout` | 128 | **890.80** | 6.96 | 143.69 ms | 87.14 ms | 56.55 ms |
| `atari_breakout` | 256 | **826.48** | 3.23 | 309.75 ms | 41.44 ms | 268.31 ms |
| `atari_space_invaders` | 64 | **458.25** | 7.16 | 139.66 ms | 50.36 ms | 89.30 ms |
| `atari_space_invaders` | 128 | **827.13** | 6.46 | 154.75 ms | 36.36 ms | 118.39 ms |
| `atari_space_invaders` | 256 | **819.22** | 3.20 | 312.49 ms | 40.29 ms | 272.20 ms |
| `atari_seaquest` | 64 | **459.72** | 7.18 | 139.22 ms | 55.01 ms | 84.20 ms |
| `atari_seaquest` | 128 | **847.44** | 6.62 | 151.04 ms | 7.04 ms | 144.01 ms |
| `atari_seaquest` | 256 | **799.89** | 3.12 | 320.04 ms | 33.53 ms | 286.51 ms |
| `atari_alien` | 64 | **444.20** | 6.94 | 144.08 ms | 51.95 ms | 92.12 ms |
| `atari_alien` | 128 | **831.60** | 6.50 | 153.92 ms | 58.43 ms | 95.49 ms |
| `atari_alien` | 256 | **862.87** | 3.37 | 296.68 ms | 66.35 ms | 230.33 ms |
| `atari_qbert` | 64 | **560.63** | 8.76 | 114.16 ms | 33.83 ms | 80.33 ms |
| `atari_qbert` | 128 | **807.24** | 6.31 | 158.57 ms | 29.85 ms | 128.72 ms |
| `atari_qbert` | 256 | **910.20** | 3.56 | 281.26 ms | 62.52 ms | 218.74 ms |
| `atari_asteroids` | 64 | **420.96** | 6.58 | 152.03 ms | 29.38 ms | 122.66 ms |
| `atari_asteroids` | 128 | **845.90** | 6.61 | 151.32 ms | 30.91 ms | 120.41 ms |
| `atari_asteroids` | 256 | **799.29** | 3.12 | 320.28 ms | 55.31 ms | 264.98 ms |
| `atari_ms_pacman` | 64 | **586.66** | 9.17 | 109.09 ms | 26.44 ms | 82.65 ms |
| `atari_ms_pacman` | 128 | **838.27** | 6.55 | 152.70 ms | 88.39 ms | 64.30 ms |
| `atari_ms_pacman` | 256 | **909.89** | 3.55 | 281.35 ms | 65.14 ms | 216.22 ms |
| `atari_enduro` | 64 | **443.22** | 6.93 | 144.40 ms | 49.32 ms | 95.08 ms |
| `atari_enduro` | 128 | **839.27** | 6.56 | 152.51 ms | 102.42 ms | 50.10 ms |
| `atari_enduro` | 256 | **861.12** | 3.36 | 297.29 ms | 39.91 ms | 257.38 ms |
| `atari_hero` | 64 | **475.42** | 7.43 | 134.62 ms | 35.04 ms | 99.58 ms |
| `atari_hero` | 128 | **823.81** | 6.44 | 155.38 ms | 16.40 ms | 138.97 ms |
| `atari_hero` | 256 | **799.62** | 3.12 | 320.15 ms | 35.24 ms | 284.91 ms |
| `atari_freeway` | 64 | **827.11** | 12.92 | 77.38 ms | 52.02 ms | 25.36 ms |
| `atari_freeway` | 128 | **799.31** | 6.24 | 160.14 ms | 22.40 ms | 137.74 ms |
| `atari_freeway` | 256 | **796.97** | 3.11 | 321.22 ms | 38.15 ms | 283.07 ms |
| `atari_riverraid` | 64 | **473.08** | 7.39 | 135.28 ms | 56.06 ms | 79.22 ms |
| `atari_riverraid` | 128 | **837.62** | 6.54 | 152.81 ms | 45.01 ms | 107.80 ms |
| `atari_riverraid` | 256 | **927.11** | 3.62 | 276.13 ms | 38.54 ms | 237.58 ms |
