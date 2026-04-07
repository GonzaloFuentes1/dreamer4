# Benchmark de Recolección Escalado (1 GPU)

Evaluación de latencia y rendimiento variando grandes volúmenes de workers y GPUs.                                                                             
Mide la simulación empleando infraestructura Multi-worker ([64, 128, 256]) distribuida en 1 GPUs.
                                                                
| Ambiente (Task) | Envs Simultáneos | SPS Total | FPS por Env | MS Totales por Batch | Render (MS C++) | Inferencia (MS GPU) |
|:---|:---:|---:|---:|---:|---:|---:|
| `dmc_cheetah_run` | 64 | **926.63** | 14.48 | 69.07 ms | 10.92 ms | 58.14 ms |
| `dmc_cheetah_run` | 128 | **1025.41** | 8.01 | 124.83 ms | 1.50 ms | 123.32 ms |
| `dmc_cheetah_run` | 256 | **1033.16** | 4.04 | 247.78 ms | 5.38 ms | 242.40 ms |
| `dmc_walker_walk` | 64 | **918.16** | 14.35 | 69.70 ms | 2.38 ms | 67.33 ms |
| `dmc_walker_walk` | 128 | **1010.26** | 7.89 | 126.70 ms | 4.73 ms | 121.97 ms |
| `dmc_walker_walk` | 256 | **1033.88** | 4.04 | 247.61 ms | 6.40 ms | 241.22 ms |
| `dmc_hopper_hop` | 64 | **919.95** | 14.37 | 69.57 ms | 1.54 ms | 68.03 ms |
| `dmc_hopper_hop` | 128 | **1028.30** | 8.03 | 124.48 ms | 1.70 ms | 122.77 ms |
| `dmc_hopper_hop` | 256 | **1043.62** | 4.08 | 245.30 ms | 3.78 ms | 241.52 ms |
| `dmc_cartpole_swingup` | 64 | **929.14** | 14.52 | 68.88 ms | 1.02 ms | 67.86 ms |
| `dmc_cartpole_swingup` | 128 | **1029.56** | 8.04 | 124.33 ms | 1.42 ms | 122.91 ms |
| `dmc_cartpole_swingup` | 256 | **1052.13** | 4.11 | 243.32 ms | 2.39 ms | 240.92 ms |
| `dmc_pendulum_swingup` | 64 | **931.54** | 14.56 | 68.70 ms | 0.85 ms | 67.86 ms |
| `dmc_pendulum_swingup` | 128 | **1033.29** | 8.07 | 123.88 ms | 1.19 ms | 122.69 ms |
| `dmc_pendulum_swingup` | 256 | **1047.82** | 4.09 | 244.32 ms | 1.59 ms | 242.73 ms |
| `dmc_reacher_easy` | 64 | **931.44** | 14.55 | 68.71 ms | 0.97 ms | 67.74 ms |
| `dmc_reacher_easy` | 128 | **1029.47** | 8.04 | 124.34 ms | 1.27 ms | 123.07 ms |
| `dmc_reacher_easy` | 256 | **1048.80** | 4.10 | 244.09 ms | 1.72 ms | 242.36 ms |
| `dmc_acrobot_swingup` | 64 | **927.89** | 14.50 | 68.97 ms | 0.95 ms | 68.02 ms |
| `dmc_acrobot_swingup` | 128 | **1027.20** | 8.02 | 124.61 ms | 1.23 ms | 123.38 ms |
| `dmc_acrobot_swingup` | 256 | **1046.58** | 4.09 | 244.61 ms | 2.43 ms | 242.17 ms |
| `dmc_finger_spin` | 64 | **924.58** | 14.45 | 69.22 ms | 1.18 ms | 68.04 ms |
| `dmc_finger_spin` | 128 | **1029.30** | 8.04 | 124.36 ms | 1.44 ms | 122.91 ms |
| `dmc_finger_spin` | 256 | **1047.09** | 4.09 | 244.49 ms | 1.88 ms | 242.60 ms |
| `atari_pong` | 64 | **688.48** | 10.76 | 92.96 ms | 4.72 ms | 88.24 ms |
| `atari_pong` | 128 | **811.49** | 6.34 | 157.73 ms | 12.95 ms | 144.78 ms |
| `atari_pong` | 256 | **908.29** | 3.55 | 281.85 ms | 18.42 ms | 263.42 ms |
| `atari_breakout` | 64 | **650.47** | 10.16 | 98.39 ms | 5.30 ms | 93.09 ms |
| `atari_breakout` | 128 | **802.71** | 6.27 | 159.46 ms | 30.68 ms | 128.78 ms |
| `atari_breakout` | 256 | **925.36** | 3.61 | 276.65 ms | 17.60 ms | 259.05 ms |
| `atari_space_invaders` | 64 | **648.35** | 10.13 | 98.71 ms | 10.77 ms | 87.94 ms |
| `atari_space_invaders` | 128 | **814.47** | 6.36 | 157.16 ms | 11.08 ms | 146.08 ms |
| `atari_space_invaders` | 256 | **909.50** | 3.55 | 281.47 ms | 19.83 ms | 261.64 ms |
| `atari_seaquest` | 64 | **646.99** | 10.11 | 98.92 ms | 9.55 ms | 89.37 ms |
| `atari_seaquest` | 128 | **824.09** | 6.44 | 155.32 ms | 35.69 ms | 119.63 ms |
| `atari_seaquest` | 256 | **946.81** | 3.70 | 270.38 ms | 40.38 ms | 230.00 ms |
| `atari_alien` | 64 | **654.97** | 10.23 | 97.71 ms | 25.66 ms | 72.06 ms |
| `atari_alien` | 128 | **789.04** | 6.16 | 162.22 ms | 35.97 ms | 126.25 ms |
| `atari_alien` | 256 | **900.31** | 3.52 | 284.35 ms | 46.51 ms | 237.84 ms |
| `atari_qbert` | 64 | **651.24** | 10.18 | 98.27 ms | 18.06 ms | 80.21 ms |
| `atari_qbert` | 128 | **958.31** | 7.49 | 133.57 ms | 22.85 ms | 110.72 ms |
| `atari_qbert` | 256 | **918.05** | 3.59 | 278.85 ms | 23.92 ms | 254.93 ms |
| `atari_asteroids` | 64 | **636.05** | 9.94 | 100.62 ms | 30.58 ms | 70.05 ms |
| `atari_asteroids` | 128 | **819.85** | 6.41 | 156.13 ms | 27.13 ms | 129.00 ms |
| `atari_asteroids` | 256 | **900.21** | 3.52 | 284.38 ms | 22.15 ms | 262.23 ms |
| `atari_ms_pacman` | 64 | **736.04** | 11.50 | 86.95 ms | 28.89 ms | 58.06 ms |
| `atari_ms_pacman` | 128 | **955.65** | 7.47 | 133.94 ms | 31.21 ms | 102.73 ms |
| `atari_ms_pacman` | 256 | **910.15** | 3.56 | 281.27 ms | 20.34 ms | 260.93 ms |
| `atari_enduro` | 64 | **643.81** | 10.06 | 99.41 ms | 27.42 ms | 71.98 ms |
| `atari_enduro` | 128 | **863.38** | 6.75 | 148.25 ms | 22.18 ms | 126.08 ms |
| `atari_enduro` | 256 | **868.90** | 3.39 | 294.63 ms | 49.39 ms | 245.24 ms |
| `atari_hero` | 64 | **646.58** | 10.10 | 98.98 ms | 31.61 ms | 67.37 ms |
| `atari_hero` | 128 | **809.88** | 6.33 | 158.05 ms | 17.13 ms | 140.92 ms |
| `atari_hero` | 256 | **968.64** | 3.78 | 264.29 ms | 47.87 ms | 216.41 ms |
| `atari_freeway` | 64 | **642.48** | 10.04 | 99.61 ms | 7.37 ms | 92.24 ms |
| `atari_freeway` | 128 | **803.45** | 6.28 | 159.31 ms | 37.37 ms | 121.94 ms |
| `atari_freeway` | 256 | **954.73** | 3.73 | 268.14 ms | 42.14 ms | 226.00 ms |
| `atari_riverraid` | 64 | **657.54** | 10.27 | 97.33 ms | 28.78 ms | 68.56 ms |
| `atari_riverraid` | 128 | **957.53** | 7.48 | 133.68 ms | 9.81 ms | 123.87 ms |
| `atari_riverraid` | 256 | **899.86** | 3.52 | 284.49 ms | 16.61 ms | 267.88 ms |
