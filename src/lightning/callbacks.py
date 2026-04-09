# dreamer4/lightning/callbacks.py
from __future__ import annotations

from typing import Any

import torch
import pytorch_lightning as pl

from viz import log_tokenizer_viz_wandb, run_dynamics_eval
from losses import dynamics_pretrain_loss, make_tau_schedule


class TokenizerVizCallback(pl.Callback):
    """
    Logs a visualisation panel to W&B every `viz_every` training steps.
    Reads the last batch / prediction cached by TokenizerLightningModule.training_step.
    Only runs on global_rank == 0.
    """

    def __init__(self, viz_every: int, max_items: int = 4, max_T: int = 8):
        self.viz_every = int(viz_every)
        self.max_items = int(max_items)
        self.max_T     = int(max_T)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        step = trainer.global_step
        if not trainer.is_global_zero:
            return
        if self.viz_every <= 0 or step % self.viz_every != 0:
            return

        x        = getattr(pl_module, "_last_batch", None)
        pred     = getattr(pl_module, "_last_pred",  None)
        mae_mask = getattr(pl_module, "_last_mask",  None)

        if x is None or pred is None or mae_mask is None:
            return

        with torch.no_grad():
            log_tokenizer_viz_wandb(
                x_btchw=x.detach(),
                pred_btnd=pred.detach(),
                mae_mask_btNp1=mae_mask.detach(),
                patch=pl_module._patch,
                step=step,
                max_items=self.max_items,
                max_T=self.max_T,
            )


class DynamicsEvalCallback(pl.Callback):
    """
    Runs autoregressive evaluation and logs PSNR + video panels to W&B
    every `eval_every` training steps.  Only runs on global_rank == 0.
    """

    def __init__(self, eval_every: int):
        self.eval_every = int(eval_every)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        step = trainer.global_step
        if not trainer.is_global_zero:
            return
        if self.eval_every <= 0 or step % self.eval_every != 0:
            return

        frames   = getattr(pl_module, "_last_frames",   None)
        actions  = getattr(pl_module, "_last_actions",  None)
        act_mask = getattr(pl_module, "_last_act_mask", None)

        if frames is None:
            return

        eval_cfg = getattr(pl_module.cfg, "dynamics", None)
        if eval_cfg is None:
            eval_cfg = getattr(pl_module.cfg, "finetune", None)
        if eval_cfg is None:
            return

        B_eval = min(frames.shape[0], int(eval_cfg.eval_batch_size))
        sched  = make_tau_schedule(
            k_max=int(eval_cfg.k_max),
            schedule=str(eval_cfg.get("eval_schedule", "shortcut")),
            d=float(eval_cfg.get("eval_d", 0.25)),
        )

        with torch.no_grad():
            run_dynamics_eval(
                encoder=pl_module._encoder,
                decoder=pl_module._decoder,
                dyn=pl_module.dyn,
                frames=frames[:B_eval].detach(),
                actions=None if actions  is None else actions[:B_eval].detach(),
                act_mask=None if act_mask is None else act_mask[:B_eval].detach(),
                H=pl_module._H,
                W=pl_module._W,
                C=pl_module._C,
                patch=pl_module._patch,
                packing_factor=pl_module.packing_factor,
                k_max=int(eval_cfg.k_max),
                ctx_length=int(eval_cfg.eval_ctx),
                horizon=int(eval_cfg.get("eval_horizon", 16)),
                sched=sched,
                max_items=int(eval_cfg.get("eval_max_items", 4)),
                step=step,
            )


class ActionShuffleMetricCallback(pl.Callback):
    """
    Periodically computes the shuffled-action loss ratio as a diagnostic.
    A ratio > 1 means the model uses actions (shuffled actions hurt more).
    Only runs on global_rank == 0 and only when use_actions=True.
    Heavy (runs 2 extra forward passes), so use a large `log_every`.
    """

    def __init__(self, log_every: int = 1000):
        self.log_every = int(log_every)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        step = trainer.global_step
        if not trainer.is_global_zero:
            return
        if self.log_every <= 0 or step % self.log_every != 0:
            return

        dc      = pl_module.cfg.dynamics
        actions = getattr(pl_module, "_last_actions",  None)
        z1      = getattr(pl_module, "_last_z1",       None)
        act_mask= getattr(pl_module, "_last_act_mask", None)
        B_self  = getattr(pl_module, "_last_B_self",   0)

        if actions is None or z1 is None:
            return

        with torch.no_grad():
            loss_real, _ = dynamics_pretrain_loss(
                pl_module.dyn,
                z1=z1,
                actions=actions,
                act_mask=act_mask,
                k_max=int(dc.k_max),
                B_self=B_self,
                step=step,
                bootstrap_start=int(dc.bootstrap_start),
            )
            perm = torch.randperm(actions.shape[0], device=actions.device)
            loss_shuffled, _ = dynamics_pretrain_loss(
                pl_module.dyn,
                z1=z1,
                actions=actions[perm],
                act_mask=act_mask,
                k_max=int(dc.k_max),
                B_self=B_self,
                step=step,
                bootstrap_start=int(dc.bootstrap_start),
            )

        ratio = loss_shuffled / loss_real.clamp_min(1e-12)

        import wandb
        wandb.log({"stats/action_shuffle_loss_ratio": float(ratio.item())}, step=step)


class AgentEvalCallback(pl.Callback):
    """
    Runs online evaluation of the current policy in DMControl every `eval_every`
    training steps and logs episode returns per task to W&B.

    Uses the live model weights from the AgentModule (no checkpoint needed).
    Only runs on global_rank == 0 to avoid spawning many envs.

    Requires dm_control to be installed (Phase-0 dependency).
    """

    def __init__(
        self,
        tasks: list,
        eval_every: int = 2000,
        n_episodes: int = 1,
        episode_len: int = 500,
        context_len: int = 16,
        img_size: int = 128,
        camera_id: int = 0,
        frame_skip: int = 2,
    ):
        self.tasks       = list(tasks)
        self.eval_every  = int(eval_every)
        self.n_episodes  = int(n_episodes)
        self.episode_len = int(episode_len)
        self.context_len = int(context_len)
        self.img_size    = int(img_size)
        self.camera_id   = int(camera_id)
        self.frame_skip  = int(frame_skip)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        step = trainer.global_step
        if not trainer.is_global_zero:
            return
        if self.eval_every <= 0 or step % self.eval_every != 0:
            return

        import os
        import math
        import numpy as np
        import wandb

        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["EGL_LOG_LEVEL"] = "fatal"   # suprime los "failed to create dri2 screen"

        if pl_module.device.type == "cuda":
            # Si slurm asignó por ej `CUDA_VISIBLE_DEVICES="4,5"`, sacamos el primero visible real
            # En configuraciones pl.Trainer dist, a veces dependemos del local_rank / gpu_id
            gpu_id = os.environ.get("EGL_DEVICE_ID", os.environ.get("CUDA_VISIBLE_DEVICES", str(pl_module.device.index or 0))).split(",")[0]
            os.environ["EGL_DEVICE_ID"] = str(gpu_id)

        # Evita que intente abrir una ventana X11 real
        os.environ["DISPLAY"] = ""

        try:
            from dm_control import mujoco
            mujoco.Physics.from_xml_string('<mujoco></mujoco>')
        except Exception:
            pass

        try:
            from dm_control import suite
        except ImportError:
            return  # dm_control no disponible, skip silencioso

        from model import temporal_patchify, pack_bottleneck_to_spatial

        device  = pl_module.device
        ac      = pl_module.cfg.agent
        n_agent = int(ac.n_agent)
        k_max   = int(ac.k_max)
        emax    = int(round(math.log2(k_max)))

        encoder       = pl_module._encoder
        dyn           = pl_module._dyn
        task_embedder = pl_module._task_embedder
        policy_head   = pl_module.policy_head
        pf            = pl_module.packing_factor
        n_spatial     = pl_module.n_spatial
        patch         = pl_module._patch
        action_dim    = int(ac.action_dim)

        returns_per_task = {}

        for task_idx, task in enumerate(self.tasks):
            try:
                domain, task_name = task.split("-", 1)
                task_name = task_name.replace("-", "_")
                env = suite.load(domain_name=domain, task_name=task_name,
                                 task_kwargs={"random": 0},
                                 visualize_reward=False)
            except Exception:
                continue

# Task embedding 
            task_inp = torch.zeros(1, 512, device=device)
            try:
                import json, os
                tasks_file = os.path.join(os.path.dirname(__file__), "..", "..", "tasks.json")
                if os.path.exists(tasks_file):
                    with open(tasks_file, "r") as jf:
                        t_meta = json.load(jf)
                        if task in t_meta and "text_embedding" in t_meta[task]:
                            task_inp = torch.tensor(t_meta[task]["text_embedding"], device=device).unsqueeze(0)
            except Exception as e:
                print(f"[EvalCallback] Aviso: no se pudo cargar lang_emb ({e})")
            ep_returns = []

            for _ in range(self.n_episodes):
                ts = env.reset()
                z_buf    = None
                act_buf  = None
                task_buf = None
                prev_act = torch.zeros(1, 1, action_dim, device=device)
                ep_return = 0.0
                step_count = 0

                while not ts.last() and step_count < self.episode_len:
                    # Render frame
                    frame_np = env.physics.render(
                        height=self.img_size, width=self.img_size,
                        camera_id=self.camera_id
                    )
                    frame = torch.from_numpy(frame_np.copy()).permute(2, 0, 1)  # (3,H,W)
                    frame_f = frame.float().div(255.0).to(device)
                    frame_in = frame_f.unsqueeze(0).unsqueeze(0)  # (1,1,3,H,W)

                    with torch.no_grad():
                        patches = temporal_patchify(frame_in, patch)
                        z_new, _ = encoder(patches)
                        z_new = pack_bottleneck_to_spatial(z_new, n_spatial=n_spatial, k=pf)  # (1,1,ns,ds)

                        task_new = task_embedder(task_inp, B=1, T=1)  # (1,1,n_agent,d)

                        z_buf    = z_new    if z_buf    is None else torch.cat([z_buf,    z_new],    dim=1)
                        act_buf  = prev_act if act_buf  is None else torch.cat([act_buf,  prev_act], dim=1)
                        task_buf = task_new if task_buf is None else torch.cat([task_buf, task_new], dim=1)

                        # Trim context
                        if z_buf.shape[1] > self.context_len:
                            z_buf    = z_buf[:,    -self.context_len:]
                            act_buf  = act_buf[:,  -self.context_len:]
                            task_buf = task_buf[:, -self.context_len:]

                        T_ctx = z_buf.shape[1]
                        step_t = torch.full((1, T_ctx), emax,       device=device, dtype=torch.long)
                        sig_t  = torch.full((1, T_ctx), k_max - 1,  device=device, dtype=torch.long)

                        _, h_t = dyn(act_buf, step_t, sig_t, z_buf,
                                     act_mask=torch.ones_like(act_buf), agent_tokens=task_buf)
                        h_flat = h_t[:, -1].flatten(1)  # (1, state_dim)

                        act_mask = torch.ones(1, action_dim, device=device)
                        _, action = policy_head.sample(h_flat, act_mask=act_mask)
                        action_np = action.squeeze(0).cpu().numpy()

                    # Rescale [-1,1] → env action spec
                    spec = env.action_spec()
                    lo, hi = spec.minimum, spec.maximum
                    n_active = len(lo)
                    action_env = ((action_np[:n_active] + 1.0) / 2.0 * (hi - lo) + lo).clip(lo, hi)

                    # frame_skip: advance physics N steps, accumulate reward
                    for _ in range(self.frame_skip):
                        ts = env.step(action_env)
                        ep_return += float(ts.reward or 0.0)
                        if ts.last():
                            break
                    # Zero out inactive dims so dynamics sees the same format as training
                    action_np[n_active:] = 0.0
                    prev_act = torch.from_numpy(action_np).to(device).view(1, 1, -1)
                    step_count += 1

                ep_returns.append(ep_return)

            mean_ret = float(np.mean(ep_returns))
            returns_per_task[task] = mean_ret

        # Log to W&B
        log_dict = {f"eval/{t}/episode_return": v for t, v in returns_per_task.items()}
        if returns_per_task:
            log_dict["eval/mean_episode_return"] = float(
                np.mean(list(returns_per_task.values()))
            )
        wandb.log(log_dict, step=step)
