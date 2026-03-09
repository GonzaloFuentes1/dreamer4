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

        dc = pl_module.cfg.dynamics
        B_eval = min(frames.shape[0], int(dc.eval_batch_size))
        sched  = make_tau_schedule(k_max=int(dc.k_max), schedule=str(dc.eval_schedule), d=float(dc.eval_d))

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
                k_max=int(dc.k_max),
                ctx_length=int(dc.eval_ctx),
                horizon=int(dc.eval_horizon),
                sched=sched,
                max_items=int(dc.eval_max_items),
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
