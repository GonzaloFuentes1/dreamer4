import os
from typing import Any

# ==============================================================================
# MAGIC EGL FIX: Pre-inicializar el render de MuJoCo *ANTES* de importar torchrl
# porque torch/torchrl corrompe los bindings del driver gráfico EGL al cargar.
# ==============================================================================
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
if 'DISPLAY' in os.environ:
    del os.environ['DISPLAY']
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ["RL_WARNINGS"] = "False"
os.environ.setdefault('EGL_LOG_LEVEL', 'fatal')   # suprime "failed to create dri2 screen"


# El launcher `launch_phase0_dist.py` ahora setea EGL_DEVICE_ID correctamente
# para coincidir con la GPU de SLURM. No debemos borrarlo! Ya no interfiere.

try:
    from dm_control import mujoco
    mujoco.Physics.from_xml_string('<mujoco></mujoco>')
except Exception:
    pass

import torch
from torchrl.envs import ParallelEnv, EnvCreator
from torchrl.envs.transforms import TransformedEnv, Compose, Resize, ToTensorImage
from tensordict.base import TensorDictBase


# ── Domain Randomization ─────────────────────────────────────────────────────
_FLOOR_GEOM_KEYWORDS = {"floor", "ground", "groundplane", "track"}

# Geoms a excluir de la randomización (son marcadores, etc.)
_SKIP_GEOM_KEYWORDS = {"target", "goal", "site"}


def _is_floor(name: str) -> bool:
    nl = name.lower()
    return any(kw in nl for kw in _FLOOR_GEOM_KEYWORDS)


def _is_skipped(name: str) -> bool:
    nl = name.lower()
    return any(kw in nl for kw in _SKIP_GEOM_KEYWORDS)


class DMControlColorRandomizer(Transform):
    """
    TorchRL Transform que randomiza colores de MuJoCo en cada reset.

    Args:
        agent_color_std:  Desviación estándar alrededor del color original
                          de cada geom del agente. 0 = sin randomización.
        floor_color_std:  Igual para el geom del suelo.
        randomize_light:  Si True, también mueve la posición de la luz.
        seed:             Semilla inicial del RNG (se re-siembra en set_seed).
    """

    def __init__(
        self,
        agent_color_std: float = 0.15,
        floor_color_std: float = 0.10,
        randomize_light: bool = False,
        seed: int | None = None,
    ):
        # Transform sin in_keys / out_keys (solo actúa como side-effect en reset)
        super().__init__(in_keys=[], out_keys=[])
        self.agent_color_std = agent_color_std
        self.floor_color_std = floor_color_std
        self.randomize_light = randomize_light

        self._rng = np.random.default_rng(seed)

        # Se rellenan la primera vez que se accede al physics model
        self._base_agent_colors: dict[str, np.ndarray] | None = None
        self._base_floor_colors: dict[str, np.ndarray] | None = None
        self._base_light_pos: np.ndarray | None = None

    # ------------------------------------------------------------------
    # TorchRL interface
    # ------------------------------------------------------------------

    def _reset_env_preprocess(self, tensordict: TensorDictBase | None) -> TensorDictBase | None:
        """Se ejecuta ANTES de que el env base procese el reset — momento perfecto
        para mutar el physics model."""
        try:
            physics = self._get_physics()
            if physics is not None:
                self._lazy_cache_base_colors(physics)
                self._apply_color_randomization(physics)
        except Exception:
            # Nunca debe crashear la fase de colección por colorización
            pass
        return tensordict

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return tensordict_reset

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    # ------------------------------------------------------------------
    # Acceso al physics model de MuJoCo
    # ------------------------------------------------------------------

    def _get_physics(self):
        """Navega parent → base_env → _env → physics."""
        env = self.parent
        if env is None:
            return None
        # TransformedEnv.base_env
        base = getattr(env, "base_env", None)
        if base is None:
            return None
        # torchrl DMControlEnv guarda el env dm_control en _env
        dm_env = getattr(base, "_env", None)
        if dm_env is None:
            return None
        # dm_control Wrapper expone .physics
        return getattr(dm_env, "physics", None)

    # ------------------------------------------------------------------
    # Caching de colores originales
    # ------------------------------------------------------------------

    def _lazy_cache_base_colors(self, physics) -> None:
        if self._base_agent_colors is not None:
            return  # ya inicializado

        named_geom = physics.named.model.geom
        agent_colors: dict[str, np.ndarray] = {}
        floor_colors: dict[str, np.ndarray] = {}

        geom_names = list(named_geom.axes.row.names)
        for name in geom_names:
            if _is_skipped(name):
                continue
            rgba = np.array(named_geom[name]["rgba"], dtype=np.float32)
            if rgba[3] < 0.01:        # transparente → ignorar
                continue
            if _is_floor(name):
                floor_colors[name] = rgba.copy()
            else:
                agent_colors[name] = rgba.copy()

        self._base_agent_colors = agent_colors
        self._base_floor_colors = floor_colors

        if self.randomize_light:
            try:
                self._base_light_pos = np.array(
                    physics.named.model.light["light0"]["pos"], dtype=np.float32
                )
            except Exception:
                self.randomize_light = False

    # ------------------------------------------------------------------
    # Aplicar randomización
    # ------------------------------------------------------------------

    def _apply_color_randomization(self, physics) -> None:
        named_geom = physics.named.model.geom

        # Agente
        if self.agent_color_std > 0 and self._base_agent_colors:
            for name, base_rgba in self._base_agent_colors.items():
                noise = self._rng.normal(0.0, self.agent_color_std, size=3).astype(np.float32)
                new_rgb = np.clip(base_rgba[:3] + noise, 0.0, 1.0)
                rgba = np.array(named_geom[name]["rgba"], dtype=np.float32)
                rgba[:3] = new_rgb
                named_geom[name]["rgba"][:] = rgba

        # Suelo
        if self.floor_color_std > 0 and self._base_floor_colors:
            for name, base_rgba in self._base_floor_colors.items():
                noise = self._rng.normal(0.0, self.floor_color_std, size=3).astype(np.float32)
                new_rgb = np.clip(base_rgba[:3] + noise, 0.0, 1.0)
                rgba = np.array(named_geom[name]["rgba"], dtype=np.float32)
                rgba[:3] = new_rgb
                named_geom[name]["rgba"][:] = rgba

        # Luz
        if self.randomize_light and self._base_light_pos is not None:
            try:
                named_light = physics.named.model.light
                noise = self._rng.normal(0.0, 0.5, size=3).astype(np.float32)
                named_light["light0"]["pos"][:] = self._base_light_pos + noise
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Reproducibilidad
    # ------------------------------------------------------------------

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        self._rng = np.random.default_rng(seed)
        return seed


def parse_dmc_task(task: str):
    """
    dmc_cheetah_run -> cheetah, run
    cheetah-run -> cheetah, run
    """
    if task.startswith("dmc_"):
        parts = task.replace("dmc_", "").split("_")
    else:
        parts = task.split("-")
        
    domain = parts[0]
    task_name = "_".join(parts[1:])
    return domain, task_name

def parse_atari_task(task: str):
    """
    atari_pong -> PongNoFrameskip-v4
    """
    parts = task.replace("atari_", "").split("_")
    name = "".join([p.capitalize() for p in parts])
    return f"{name}NoFrameskip-v4"

def _make_single_env(task: str, img_size: int = 64, frame_skip: int = 1, color_randomization: bool = False, seed: int = 42):
    """
    Contexto constructor anónimo para cada hilo de Multiprocessing local
    (Propuesta 1)
    """
    import os
    # --- LA MAGIA NUEVA TAMBIÉN EN LOS WORKERS ---
    # Respetamos el EGL_DEVICE_ID heredado!
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    if 'DISPLAY' in os.environ:
        del os.environ['DISPLAY']
    os.environ.setdefault('EGL_LOG_LEVEL', 'fatal')

    # Environment config is already safely global now.

    # Tasks usually come as dmc_cheetah_run or just cheetah-run
    # If it is not atari_ it is assumed to be dmc
    is_dmc = task.startswith("dmc_") or ("-" in task and not task.startswith("atari_"))

    if is_dmc:
        from torchrl.envs.libs.dm_control import DMControlEnv
        domain, task_name = parse_dmc_task(task)

        env = DMControlEnv(
            env_name=domain,
            task_name=task_name,
            from_pixels=True,
            pixels_only=False,
            frame_skip=frame_skip,
        )
    else:
        # Fallback Atari
        from torchrl.envs.libs.gym import GymEnv
        atari_name = parse_atari_task(task)
        env = GymEnv(
            env_name=atari_name,
            frame_skip=frame_skip,
            from_pixels=True,
            pixels_only=False
        )

    transforms = []
    if color_randomization and is_dmc:
        transforms.append(DMControlColorRandomizer(seed=seed))
    transforms += [
        ToTensorImage(in_keys=["pixels"]),
        Resize(w=img_size, h=img_size, in_keys=["pixels"]),
    ]

    return TransformedEnv(env, Compose(*transforms))


def make_torchrl_env(
    task: str,
    num_envs: int = 16,
    seed: int = 42,
    img_size: int = 64,
    frame_skip: int = 1,
    color_randomization: bool = False,
) -> ParallelEnv:
    """
    TorchRL ParallelEnv con constructores directos.

    Args:
        color_randomization: Si True, randomiza colores de MuJoCo en cada reset
                             (domain randomization visual — solo DMControl).
    """
    # EnvCreator empaqueta la inicialización de manera segura entre procesos
    create_env_fn = EnvCreator(
        lambda: _make_single_env(
            task,
            img_size=img_size,
            frame_skip=frame_skip,
            color_randomization=color_randomization,
            seed=seed,
        )
    )
    
    parallel_env = ParallelEnv(
        num_workers=num_envs,
        create_env_fn=create_env_fn,
        mp_start_method="spawn"
    )
    
    parallel_env.set_seed(seed)
    
    print(f"[TorchRL Wrapper] Construido {num_envs} envs nativos en paralelo para {task}.")
    
    return parallel_env
