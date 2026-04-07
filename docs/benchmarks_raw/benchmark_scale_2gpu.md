# Benchmark de Recolección Escalado (2 GPU)

Evaluación de latencia y rendimiento variando grandes volúmenes de workers y GPUs.                                                                             
Mide la simulación empleando infraestructura Multi-worker ([64, 128, 256]) distribuida en 2 GPUs.
                                                                
| Ambiente (Task) | Envs Simultáneos | SPS Total | FPS por Env | MS Totales por Batch | Render (MS C++) | Inferencia (MS GPU) |
|:---|:---:|---:|---:|---:|---:|---:|
| `dmc_cheetah_run` | 64 | **285.89** | 4.47 | 223.86 ms | 198.36 ms | 25.50 ms |
| `dmc_cheetah_run` | 128 | **217.31** | 1.70 | 589.03 ms | 480.40 ms | 108.63 ms |
| `dmc_cheetah_run` | 256 | **215.67** | 0.84 | 1186.99 ms | 985.85 ms | 201.14 ms |
| `dmc_walker_walk` | 64 | **290.09** | 4.53 | 220.62 ms | 163.12 ms | 57.50 ms |
| `dmc_walker_walk` | 128 | **214.11** | 1.67 | 597.82 ms | 498.73 ms | 99.09 ms |
| `dmc_walker_walk` | 256 | **217.75** | 0.85 | 1175.65 ms | 988.63 ms | 187.02 ms |
| `dmc_hopper_hop` | 64 | **294.34** | 4.60 | 217.44 ms | 157.67 ms | 59.77 ms |
| `dmc_hopper_hop` | 128 | **210.47** | 1.64 | 608.17 ms | 472.61 ms | 135.56 ms |
| `dmc_hopper_hop` | 256 | **219.29** | 0.86 | 1167.42 ms | 1008.58 ms | 158.84 ms |
| `dmc_cartpole_swingup` | 64 | **300.66** | 4.70 | 212.86 ms | 157.85 ms | 55.02 ms |
| `dmc_cartpole_swingup` | 128 | **212.70** | 1.66 | 601.80 ms | 506.20 ms | 95.60 ms |
| `dmc_cartpole_swingup` | 256 | **222.84** | 0.87 | 1148.82 ms | 1029.80 ms | 119.02 ms |
| `dmc_pendulum_swingup` | 64 | **285.37** | 4.46 | 224.27 ms | 170.31 ms | 53.96 ms |
| `dmc_pendulum_swingup` | 128 | **217.79** | 1.70 | 587.73 ms | 488.85 ms | 98.88 ms |
| `dmc_pendulum_swingup` | 256 | **220.27** | 0.86 | 1162.19 ms | 987.50 ms | 174.69 ms |
| `dmc_reacher_easy` | 64 | **300.56** | 4.70 | 212.93 ms | 156.09 ms | 56.85 ms |
| `dmc_reacher_easy` | 128 | **219.71** | 1.72 | 582.58 ms | 513.58 ms | 69.00 ms |
| `dmc_reacher_easy` | 256 | **221.42** | 0.86 | 1156.16 ms | 973.37 ms | 182.80 ms |
| `dmc_acrobot_swingup` | 64 | **298.39** | 4.66 | 214.49 ms | 159.44 ms | 55.05 ms |
| `dmc_acrobot_swingup` | 128 | **216.38** | 1.69 | 591.54 ms | 499.90 ms | 91.64 ms |
| `dmc_acrobot_swingup` | 256 | **221.14** | 0.86 | 1157.64 ms | 989.81 ms | 167.83 ms |
| `dmc_finger_spin` | 64 | **285.34** | 4.46 | 224.29 ms | 173.54 ms | 50.75 ms |
| `dmc_finger_spin` | 128 | **217.19** | 1.70 | 589.36 ms | 520.07 ms | 69.28 ms |
| `dmc_finger_spin` | 256 | **222.82** | 0.87 | 1148.90 ms | 998.67 ms | 150.24 ms |
| `atari_pong` | 64 | **752.26** | 11.75 | 85.08 ms | 30.45 ms | 54.63 ms |
| `atari_pong` | 128 | **820.43** | 6.41 | 156.02 ms | 68.52 ms | 87.49 ms |
| `atari_pong` | 256 | **931.50** | 3.64 | 274.83 ms | 122.58 ms | 152.25 ms |
| `atari_breakout` | 64 | **742.00** | 11.59 | 86.25 ms | 34.60 ms | 51.66 ms |
| `atari_breakout` | 128 | **790.17** | 6.17 | 161.99 ms | 71.70 ms | 90.29 ms |
| `atari_breakout` | 256 | **942.96** | 3.68 | 271.49 ms | 107.22 ms | 164.27 ms |
| `atari_space_invaders` | 64 | **753.77** | 11.78 | 84.91 ms | 34.70 ms | 50.21 ms |
| `atari_space_invaders` | 128 | **797.04** | 6.23 | 160.59 ms | 63.86 ms | 96.73 ms |
| `atari_space_invaders` | 256 | **962.01** | 3.76 | 266.11 ms | 117.01 ms | 149.10 ms |
| `atari_seaquest` | 64 | **729.04** | 11.39 | 87.79 ms | 30.92 ms | 56.87 ms |
| `atari_seaquest` | 128 | **832.02** | 6.50 | 153.84 ms | 81.97 ms | 71.87 ms |
| `atari_seaquest` | 256 | **886.67** | 3.46 | 288.72 ms | 103.82 ms | 184.90 ms |
| `atari_alien` | 128 | **658.82** | 5.15 | 194.29 ms | 75.84 ms | 118.44 ms |
| `atari_alien` | 256 | **874.87** | 3.42 | 292.61 ms | 108.05 ms | 184.57 ms |
| `atari_qbert` | 64 | **765.82** | 11.97 | 83.57 ms | 25.57 ms | 58.00 ms |
| `atari_qbert` | 256 | **966.29** | 3.77 | 264.93 ms | 106.11 ms | 158.82 ms |
| `atari_asteroids` | 64 | **707.85** | 11.06 | 90.42 ms | 28.10 ms | 62.32 ms |
| `atari_asteroids` | 128 | **749.95** | 5.86 | 170.68 ms | 66.63 ms | 104.05 ms |
| `atari_asteroids` | 256 | **1016.07** | 3.97 | 251.95 ms | 104.06 ms | 147.89 ms |
| `atari_ms_pacman` | 128 | **815.72** | 6.37 | 156.92 ms | 57.83 ms | 99.08 ms |
| `atari_ms_pacman` | 256 | **812.36** | 3.17 | 315.13 ms | 156.34 ms | 158.79 ms |
| `atari_enduro` | 64 | **770.60** | 12.04 | 83.05 ms | 29.59 ms | 53.47 ms |
| `atari_enduro` | 128 | **798.08** | 6.23 | 160.39 ms | 77.84 ms | 82.54 ms |
| `atari_enduro` | 256 | **959.89** | 3.75 | 266.70 ms | 111.90 ms | 154.80 ms |
| `atari_hero` | 64 | **736.54** | 11.51 | 86.89 ms | 27.75 ms | 59.14 ms |
| `atari_hero` | 256 | **877.13** | 3.43 | 291.86 ms | 128.42 ms | 163.44 ms |
| `atari_freeway` | 64 | **734.04** | 11.47 | 87.19 ms | 30.99 ms | 56.19 ms |
| `atari_freeway` | 128 | **778.89** | 6.09 | 164.34 ms | 68.59 ms | 95.75 ms |
| `atari_freeway` | 256 | **943.23** | 3.68 | 271.41 ms | 123.30 ms | 148.11 ms |
| `atari_riverraid` | 64 | **654.05** | 10.22 | 97.85 ms | 31.23 ms | 66.62 ms |
| `atari_riverraid` | 128 | **847.39** | 6.62 | 151.05 ms | 66.67 ms | 84.38 ms |
| `atari_riverraid` | 256 | **1002.82** | 3.92 | 255.28 ms | 114.71 ms | 140.57 ms |
