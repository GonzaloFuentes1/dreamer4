# Benchmark de Recolección Escalado (1 GPU) - Preentrenado

Evaluación de latencia y rendimiento variando grandes volúmenes de workers y GPUs, **cargando pesos de un modelo real**.

Mide la simulación empleando infraestructura Multi-worker ([64, 128, 256]) distribuida en 1 GPUs.
                                                                
| Ambiente (Task) | Envs Simultáneos | SPS Total | FPS por Env | MS Totales por Batch | Render (MS C++) | Inferencia (MS GPU) |
|:---|:---:|---:|---:|---:|---:|---:|
| `dmc_cheetah_run` | 64 | **941.39** | 14.71 | 67.98 ms | 10.54 ms | 57.45 ms |
| `dmc_cheetah_run` | 128 | **1048.71** | 8.19 | 122.05 ms | 1.42 ms | 120.63 ms |
| `dmc_cheetah_run` | 256 | **1063.20** | 4.15 | 240.78 ms | 5.14 ms | 235.64 ms |
| `dmc_walker_walk` | 64 | **931.93** | 14.56 | 68.68 ms | 1.89 ms | 66.79 ms |
| `dmc_walker_walk` | 128 | **1029.80** | 8.05 | 124.30 ms | 4.87 ms | 119.43 ms |
| `dmc_walker_walk` | 256 | **1057.97** | 4.13 | 241.97 ms | 5.81 ms | 236.16 ms |
| `dmc_hopper_hop` | 64 | **930.70** | 14.54 | 68.77 ms | 1.80 ms | 66.96 ms |
| `dmc_hopper_hop` | 128 | **1044.68** | 8.16 | 122.53 ms | 1.83 ms | 120.70 ms |
| `dmc_hopper_hop` | 256 | **1065.97** | 4.16 | 240.16 ms | 4.93 ms | 235.23 ms |
| `dmc_cartpole_swingup` | 64 | **940.81** | 14.70 | 68.03 ms | 1.14 ms | 66.88 ms |
| `dmc_cartpole_swingup` | 128 | **1045.84** | 8.17 | 122.39 ms | 1.37 ms | 121.02 ms |
| `dmc_cartpole_swingup` | 256 | **1072.54** | 4.19 | 238.69 ms | 2.20 ms | 236.49 ms |
| `dmc_pendulum_swingup` | 64 | **944.53** | 14.76 | 67.76 ms | 0.94 ms | 66.82 ms |
| `dmc_pendulum_swingup` | 128 | **1051.47** | 8.21 | 121.73 ms | 1.14 ms | 120.59 ms |
| `dmc_pendulum_swingup` | 256 | **1072.38** | 4.19 | 238.72 ms | 1.70 ms | 237.02 ms |
| `dmc_reacher_easy` | 64 | **937.69** | 14.65 | 68.25 ms | 1.11 ms | 67.14 ms |
| `dmc_reacher_easy` | 128 | **1040.05** | 8.13 | 123.07 ms | 1.45 ms | 121.62 ms |
| `dmc_reacher_easy` | 256 | **1071.17** | 4.18 | 238.99 ms | 1.83 ms | 237.16 ms |
| `dmc_acrobot_swingup` | 64 | **945.03** | 14.77 | 67.72 ms | 0.97 ms | 66.76 ms |
| `dmc_acrobot_swingup` | 128 | **1045.14** | 8.17 | 122.47 ms | 1.28 ms | 121.19 ms |
| `dmc_acrobot_swingup` | 256 | **1069.33** | 4.18 | 239.40 ms | 2.06 ms | 237.34 ms |
| `dmc_finger_spin` | 64 | **939.60** | 14.68 | 68.11 ms | 1.20 ms | 66.91 ms |
| `dmc_finger_spin` | 128 | **1044.70** | 8.16 | 122.52 ms | 1.56 ms | 120.97 ms |
| `dmc_finger_spin` | 256 | **1072.45** | 4.19 | 238.71 ms | 2.63 ms | 236.08 ms |
| `atari_pong` | 64 | **839.29** | 13.11 | 76.25 ms | 5.91 ms | 70.34 ms |
| `atari_pong` | 128 | **984.34** | 7.69 | 130.04 ms | 21.84 ms | 108.20 ms |
| `atari_pong` | 256 | **926.38** | 3.62 | 276.34 ms | 30.21 ms | 246.13 ms |
| `atari_breakout` | 64 | **643.48** | 10.05 | 99.46 ms | 29.52 ms | 69.94 ms |
| `atari_breakout` | 128 | **794.43** | 6.21 | 161.12 ms | 33.35 ms | 127.77 ms |
| `atari_breakout` | 256 | **911.24** | 3.56 | 280.94 ms | 44.13 ms | 236.81 ms |
| `atari_space_invaders` | 64 | **756.39** | 11.82 | 84.61 ms | 31.60 ms | 53.01 ms |
| `atari_space_invaders` | 128 | **821.85** | 6.42 | 155.75 ms | 33.69 ms | 122.06 ms |
| `atari_space_invaders` | 256 | **989.12** | 3.86 | 258.81 ms | 18.87 ms | 239.95 ms |
| `atari_seaquest` | 64 | **651.21** | 10.18 | 98.28 ms | 12.41 ms | 85.87 ms |
| `atari_seaquest` | 128 | **966.71** | 7.55 | 132.41 ms | 8.64 ms | 123.77 ms |
| `atari_seaquest` | 256 | **908.80** | 3.55 | 281.69 ms | 36.49 ms | 245.20 ms |
| `atari_alien` | 64 | **820.30** | 12.82 | 78.02 ms | 7.85 ms | 70.17 ms |
| `atari_alien` | 128 | **934.91** | 7.30 | 136.91 ms | 22.77 ms | 114.14 ms |
| `atari_alien` | 256 | **908.93** | 3.55 | 281.65 ms | 30.64 ms | 251.02 ms |
| `atari_qbert` | 64 | **635.52** | 9.93 | 100.70 ms | 29.95 ms | 70.75 ms |
| `atari_qbert` | 128 | **816.69** | 6.38 | 156.73 ms | 12.67 ms | 144.06 ms |
| `atari_qbert` | 256 | **918.50** | 3.59 | 278.72 ms | 44.84 ms | 233.88 ms |
| `atari_asteroids` | 64 | **669.85** | 10.47 | 95.54 ms | 29.51 ms | 66.04 ms |
| `atari_asteroids` | 128 | **897.64** | 7.01 | 142.60 ms | 32.57 ms | 110.03 ms |
| `atari_asteroids` | 256 | **996.70** | 3.89 | 256.85 ms | 17.68 ms | 239.17 ms |
| `atari_ms_pacman` | 64 | **762.29** | 11.91 | 83.96 ms | 15.62 ms | 68.34 ms |
| `atari_ms_pacman` | 128 | **898.82** | 7.02 | 142.41 ms | 29.40 ms | 113.01 ms |
| `atari_ms_pacman` | 256 | **987.45** | 3.86 | 259.25 ms | 15.73 ms | 243.53 ms |
| `atari_enduro` | 64 | **853.56** | 13.34 | 74.98 ms | 8.56 ms | 66.42 ms |
| `atari_enduro` | 128 | **972.37** | 7.60 | 131.64 ms | 30.80 ms | 100.84 ms |
| `atari_enduro` | 256 | **904.60** | 3.53 | 283.00 ms | 46.56 ms | 236.44 ms |
| `atari_hero` | 64 | **786.04** | 12.28 | 81.42 ms | 24.10 ms | 57.32 ms |
| `atari_hero` | 128 | **955.91** | 7.47 | 133.90 ms | 19.85 ms | 114.06 ms |
| `atari_hero` | 256 | **983.97** | 3.84 | 260.17 ms | 51.69 ms | 208.48 ms |
| `atari_freeway` | 64 | **667.05** | 10.42 | 95.95 ms | 7.13 ms | 88.82 ms |
| `atari_freeway` | 128 | **790.88** | 6.18 | 161.85 ms | 12.49 ms | 149.36 ms |
| `atari_freeway` | 256 | **914.86** | 3.57 | 279.82 ms | 34.70 ms | 245.13 ms |
| `atari_riverraid` | 64 | **638.97** | 9.98 | 100.16 ms | 5.58 ms | 94.58 ms |
| `atari_riverraid` | 128 | **909.43** | 7.10 | 140.75 ms | 33.47 ms | 107.27 ms |
| `atari_riverraid` | 256 | **1010.46** | 3.95 | 253.35 ms | 24.58 ms | 228.77 ms |
