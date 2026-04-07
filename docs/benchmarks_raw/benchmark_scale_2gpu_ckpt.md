# Benchmark de Recolección Escalado (2 GPU) - Preentrenado

Evaluación de latencia y rendimiento variando grandes volúmenes de workers y GPUs, **cargando pesos de un modelo real**.

Mide la simulación empleando infraestructura Multi-worker ([64, 128, 256]) distribuida en 2 GPUs.
                                                                
| Ambiente (Task) | Envs Simultáneos | SPS Total | FPS por Env | MS Totales por Batch | Render (MS C++) | Inferencia (MS GPU) |
|:---|:---:|---:|---:|---:|---:|---:|
| `dmc_cheetah_run` | 64 | **196.01** | 3.06 | 326.51 ms | 242.68 ms | 83.83 ms |
| `dmc_cheetah_run` | 128 | **193.87** | 1.51 | 660.24 ms | 522.51 ms | 137.72 ms |
| `dmc_cheetah_run` | 256 | **255.57** | 1.00 | 1001.70 ms | 807.92 ms | 193.78 ms |
| `dmc_walker_walk` | 64 | **203.99** | 3.19 | 313.75 ms | 227.32 ms | 86.43 ms |
| `dmc_walker_walk` | 128 | **203.99** | 1.59 | 627.47 ms | 513.44 ms | 114.03 ms |
| `dmc_walker_walk` | 256 | **263.23** | 1.03 | 972.53 ms | 793.93 ms | 178.60 ms |
| `dmc_hopper_hop` | 64 | **198.60** | 3.10 | 322.26 ms | 223.10 ms | 99.16 ms |
| `dmc_hopper_hop` | 128 | **196.86** | 1.54 | 650.20 ms | 486.77 ms | 163.44 ms |
| `dmc_hopper_hop` | 256 | **263.13** | 1.03 | 972.91 ms | 816.43 ms | 156.49 ms |
| `dmc_cartpole_swingup` | 64 | **205.62** | 3.21 | 311.25 ms | 230.95 ms | 80.30 ms |
| `dmc_cartpole_swingup` | 128 | **203.31** | 1.59 | 629.58 ms | 525.83 ms | 103.75 ms |
| `dmc_cartpole_swingup` | 256 | **256.36** | 1.00 | 998.60 ms | 800.39 ms | 198.21 ms |
| `dmc_pendulum_swingup` | 64 | **204.10** | 3.19 | 313.57 ms | 228.32 ms | 85.26 ms |
| `dmc_pendulum_swingup` | 128 | **199.21** | 1.56 | 642.55 ms | 511.13 ms | 131.42 ms |
| `dmc_pendulum_swingup` | 256 | **256.82** | 1.00 | 996.81 ms | 828.71 ms | 168.10 ms |
| `dmc_reacher_easy` | 64 | **203.55** | 3.18 | 314.41 ms | 220.43 ms | 93.98 ms |
| `dmc_reacher_easy` | 128 | **195.34** | 1.53 | 655.28 ms | 516.20 ms | 139.08 ms |
| `dmc_reacher_easy` | 256 | **253.86** | 0.99 | 1008.43 ms | 807.41 ms | 201.02 ms |
| `dmc_acrobot_swingup` | 64 | **197.75** | 3.09 | 323.65 ms | 234.96 ms | 88.69 ms |
| `dmc_acrobot_swingup` | 128 | **196.11** | 1.53 | 652.69 ms | 514.33 ms | 138.37 ms |
| `dmc_acrobot_swingup` | 256 | **258.36** | 1.01 | 990.87 ms | 811.71 ms | 179.16 ms |
| `dmc_finger_spin` | 64 | **204.41** | 3.19 | 313.10 ms | 241.41 ms | 71.69 ms |
| `dmc_finger_spin` | 128 | **196.62** | 1.54 | 650.99 ms | 511.64 ms | 139.34 ms |
| `dmc_finger_spin` | 256 | **253.88** | 0.99 | 1008.35 ms | 816.23 ms | 192.12 ms |
| `atari_pong` | 64 | **243.05** | 3.80 | 263.33 ms | 196.74 ms | 66.58 ms |
| `atari_pong` | 128 | **311.12** | 2.43 | 411.42 ms | 293.87 ms | 117.55 ms |
| `atari_pong` | 256 | **508.59** | 1.99 | 503.35 ms | 345.72 ms | 157.63 ms |
| `atari_breakout` | 64 | **239.42** | 3.74 | 267.32 ms | 193.45 ms | 73.87 ms |
| `atari_breakout` | 128 | **305.88** | 2.39 | 418.46 ms | 320.28 ms | 98.18 ms |
| `atari_breakout` | 256 | **509.35** | 1.99 | 502.60 ms | 350.71 ms | 151.89 ms |
| `atari_space_invaders` | 64 | **237.68** | 3.71 | 269.26 ms | 199.19 ms | 70.08 ms |
| `atari_space_invaders` | 128 | **318.77** | 2.49 | 401.55 ms | 307.40 ms | 94.15 ms |
| `atari_space_invaders` | 256 | **494.91** | 1.93 | 517.26 ms | 370.04 ms | 147.23 ms |
| `atari_seaquest` | 64 | **240.39** | 3.76 | 266.24 ms | 191.70 ms | 74.53 ms |
| `atari_seaquest` | 128 | **320.40** | 2.50 | 399.50 ms | 308.08 ms | 91.42 ms |
| `atari_seaquest` | 256 | **504.01** | 1.97 | 507.93 ms | 340.97 ms | 166.96 ms |
| `atari_alien` | 64 | **240.15** | 3.75 | 266.50 ms | 202.14 ms | 64.36 ms |
| `atari_alien` | 128 | **314.20** | 2.45 | 407.38 ms | 311.68 ms | 95.71 ms |
| `atari_alien` | 256 | **482.50** | 1.88 | 530.57 ms | 338.34 ms | 192.23 ms |
| `atari_qbert` | 64 | **259.65** | 4.06 | 246.49 ms | 193.75 ms | 52.74 ms |
| `atari_qbert` | 128 | **301.85** | 2.36 | 424.05 ms | 281.21 ms | 142.85 ms |
| `atari_qbert` | 256 | **487.12** | 1.90 | 525.54 ms | 313.51 ms | 212.03 ms |
| `atari_asteroids` | 64 | **240.58** | 3.76 | 266.02 ms | 187.36 ms | 78.67 ms |
| `atari_asteroids` | 128 | **311.62** | 2.43 | 410.75 ms | 305.01 ms | 105.74 ms |
| `atari_asteroids` | 256 | **518.98** | 2.03 | 493.28 ms | 304.80 ms | 188.48 ms |
| `atari_ms_pacman` | 64 | **235.73** | 3.68 | 271.50 ms | 183.18 ms | 88.32 ms |
| `atari_ms_pacman` | 128 | **306.02** | 2.39 | 418.27 ms | 313.13 ms | 105.15 ms |
| `atari_ms_pacman` | 256 | **527.81** | 2.06 | 485.02 ms | 347.41 ms | 137.61 ms |
| `atari_enduro` | 64 | **231.26** | 3.61 | 276.74 ms | 203.89 ms | 72.85 ms |
| `atari_enduro` | 128 | **290.28** | 2.27 | 440.96 ms | 302.59 ms | 138.37 ms |
| `atari_enduro` | 256 | **481.49** | 1.88 | 531.69 ms | 360.26 ms | 171.42 ms |
| `atari_hero` | 64 | **252.02** | 3.94 | 253.95 ms | 186.12 ms | 67.83 ms |
| `atari_hero` | 128 | **298.68** | 2.33 | 428.55 ms | 309.07 ms | 119.48 ms |
| `atari_hero` | 256 | **509.53** | 1.99 | 502.42 ms | 324.95 ms | 177.47 ms |
| `atari_freeway` | 64 | **251.79** | 3.93 | 254.18 ms | 197.41 ms | 56.77 ms |
| `atari_freeway` | 128 | **288.72** | 2.26 | 443.33 ms | 277.18 ms | 166.15 ms |
| `atari_freeway` | 256 | **482.97** | 1.89 | 530.06 ms | 326.88 ms | 203.18 ms |
| `atari_riverraid` | 64 | **224.03** | 3.50 | 285.68 ms | 190.79 ms | 94.89 ms |
| `atari_riverraid` | 128 | **301.75** | 2.36 | 424.19 ms | 276.89 ms | 147.30 ms |
| `atari_riverraid` | 256 | **484.43** | 1.89 | 528.45 ms | 319.52 ms | 208.93 ms |
