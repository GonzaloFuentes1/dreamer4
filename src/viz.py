# viz.py
# Visualisation helpers extracted from train_phase1a_tokenizer.py and train_phase1b_dynamics.py.
# All functions are @torch.no_grad() and log to W&B.

from typing import Any, Dict, Optional

import torch
import wandb

from model import (
    Encoder,
    Decoder,
    Dynamics,
    temporal_patchify,
    temporal_unpatchify,
    pack_bottleneck_to_spatial,
    unpack_spatial_to_bottleneck,
)
from losses import make_tau_schedule


# ---------------------------------------------------------------------------
# Tokenizer visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def log_tokenizer_viz_wandb(
    *,
    x_btchw: torch.Tensor,           # (B,T,C,H,W) float in [0,1]
    pred_btnd: torch.Tensor,          # (B,T,Np,Dp) float in [0,1]
    mae_mask_btNp1: torch.Tensor,     # (B,T,Np,1) bool True=masked
    patch: int,
    step: int,
    max_items: int = 8,
    max_T: int = 6,
    tag: str = "tokenizer/viz",
):
    B, T, C, H, W = x_btchw.shape
    Tv = min(T, max_T)
    Bv = min(B, max_items)

    target_btnd = temporal_patchify(x_btchw[:, :Tv], patch)

    masked_input_btnd = torch.where(mae_mask_btNp1[:, :Tv], torch.zeros_like(target_btnd), target_btnd)
    recon_masked_btnd = torch.where(mae_mask_btNp1[:, :Tv], pred_btnd[:, :Tv], target_btnd)
    recon_full_btnd   = pred_btnd[:, :Tv]

    target_img = temporal_unpatchify(target_btnd,       H, W, C, patch)
    masked_img = temporal_unpatchify(masked_input_btnd, H, W, C, patch)
    rmask_img  = temporal_unpatchify(recon_masked_btnd, H, W, C, patch)
    rfull_img  = temporal_unpatchify(recon_full_btnd,   H, W, C, patch)

    def tile_time(x: torch.Tensor) -> torch.Tensor:
        x = x[:, :Tv]
        return x.permute(0, 2, 3, 1, 4).contiguous().view(x.shape[0], C, H, Tv * W)

    tgt = tile_time(target_img[:Bv])
    msk = tile_time(masked_img[:Bv])
    rm  = tile_time(rmask_img[:Bv])
    rf  = tile_time(rfull_img[:Bv])

    panel = torch.cat([tgt, msk, rm, rf], dim=2)               # (Bv,C,4H,Tv*W)
    big   = torch.cat([panel[i] for i in range(Bv)], dim=1)    # (C,Bv*4H,Tv*W)

    big = (big.clamp(0, 1) * 255.0).to(torch.uint8)
    big_hwc = big.permute(1, 2, 0).cpu().numpy()

    wandb.log(
        {
            tag: wandb.Image(big_hwc, caption="rows=target/masked/recon_masked/recon_full"),
            "tokenizer/masked_frac": float(mae_mask_btNp1[:, :Tv].float().mean().item()),
        },
        step=step,
    )


# ---------------------------------------------------------------------------
# Dynamics evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_one_timestep_packed(
    dyn: Dynamics,
    *,
    past_packed: torch.Tensor,
    k_max: int,
    sched: Dict[str, Any],
    actions: Optional[torch.Tensor] = None,
    act_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    import math
    device = past_packed.device
    dtype  = past_packed.dtype
    B, t = past_packed.shape[:2]
    n_spatial, d_spatial = past_packed.shape[2], past_packed.shape[3]

    K  = int(sched["K"])
    e  = int(sched["e"])
    tau     = sched["tau"]
    tau_idx = sched["tau_idx"]
    dt = float(sched["dt"])

    z = torch.randn((B, 1, n_spatial, d_spatial), device=device, dtype=dtype)

    emax = int(round(math.log2(k_max)))
    step_idxs_full   = torch.full((B, t + 1), emax, device=device, dtype=torch.long)
    step_idxs_full[:, -1] = e
    signal_idxs_full = torch.full((B, t + 1), k_max - 1, device=device, dtype=torch.long)

    if act_mask is not None and act_mask.dim() == 1:
        act_mask = act_mask.view(1, 1, -1)

    for i in range(K):
        tau_i = float(tau[i])
        sig_i = int(tau_idx[i])

        signal_idxs_full[:, -1] = sig_i
        packed_seq = torch.cat([past_packed, z], dim=1)

        actions_in  = None if actions  is None else actions[:, : t + 1]
        actmask_in  = None if act_mask is None else act_mask[:, : t + 1]

        x1_hat_full, _ = dyn(
            actions_in, step_idxs_full, signal_idxs_full, packed_seq,
            act_mask=actmask_in, agent_tokens=None,
        )
        x1_hat = x1_hat_full[:, -1:, :, :]

        denom = max(1e-4, 1.0 - tau_i)
        b = (x1_hat.float() - z.float()) / denom
        z = (z.float() + b * dt).to(dtype)

    return z[:, 0]


@torch.no_grad()
def sample_autoregressive_packed_sequence(
    dyn: Dynamics,
    *,
    z_gt_packed: torch.Tensor,
    ctx_length: int,
    horizon: int,
    k_max: int,
    sched: Dict[str, Any],
    actions: Optional[torch.Tensor] = None,
    act_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, T = z_gt_packed.shape[:2]
    L = min(T, ctx_length + horizon)
    ctx_length = min(ctx_length, L - 1)
    horizon = min(horizon, L - ctx_length)

    outs = [z_gt_packed[:, t] for t in range(ctx_length)]

    for _ in range(horizon):
        past = torch.stack(outs, dim=1)
        z_next = sample_one_timestep_packed(
            dyn, past_packed=past, k_max=k_max, sched=sched,
            actions=actions, act_mask=act_mask,
        )
        outs.append(z_next)

    return torch.stack(outs, dim=1)


@torch.no_grad()
def decode_packed_to_frames(
    decoder: Decoder,
    *,
    z_packed: torch.Tensor,
    H: int, W: int, C: int, patch: int,
    packing_factor: int,
) -> torch.Tensor:
    z_btLd    = unpack_spatial_to_bottleneck(z_packed, k=packing_factor)
    patches   = decoder(z_btLd)
    frames    = temporal_unpatchify(patches, H, W, C, patch)
    return frames.clamp(0, 1)


@torch.no_grad()
def log_dynamics_eval_wandb(
    *,
    gt: torch.Tensor,
    pred: torch.Tensor,
    ctx_length: int,
    step: int,
    tag: str,
    max_items: int = 4,
    gap_px: int = 16,
):
    B, T, C, H, W = gt.shape
    Bv = min(B, max_items)

    def tile_time(x: torch.Tensor) -> torch.Tensor:
        x = x[:Bv]
        B_, T_, C_, H_, W_ = x.shape
        ctx = int(max(0, min(ctx_length, T_)))
        y = x.permute(0, 2, 3, 1, 4).contiguous().view(B_, C_, H_, T_ * W_)
        if gap_px > 0 and 0 < ctx < T_:
            split = ctx * W_
            left  = y[..., :split]
            right = y[..., split:]
            gap   = torch.zeros((B_, C_, H_, gap_px), device=y.device, dtype=y.dtype)
            y = torch.cat([left, gap, right], dim=-1)
        return y

    gt_t = tile_time(gt)
    pr_t = tile_time(pred)

    panel = torch.cat([gt_t, pr_t], dim=2)
    big   = torch.cat([panel[i] for i in range(Bv)], dim=1)

    big = (big.clamp(0, 1) * 255.0).to(torch.uint8)
    big_hwc = big.permute(1, 2, 0).cpu().numpy()

    wandb.log(
        {f"{tag}/viz": wandb.Image(big_hwc, caption=f"rows=GT/Pred | ctx={ctx_length} | T={T}")},
        step=step,
    )


@torch.no_grad()
def run_dynamics_eval(
    *,
    encoder: Encoder,
    decoder: Decoder,
    dyn: Dynamics,
    frames: torch.Tensor,
    actions: Optional[torch.Tensor],
    act_mask: Optional[torch.Tensor],
    H: int, W: int, C: int, patch: int,
    packing_factor: int,
    k_max: int,
    ctx_length: int,
    horizon: int,
    sched: Dict[str, Any],
    max_items: int,
    step: int,
):
    dyn_was_training = dyn.training
    dyn.eval()

    B, T = frames.shape[:2]
    T_eval = min(T, ctx_length + horizon)
    ctx_length = min(ctx_length, T_eval - 1)
    horizon = min(horizon, T_eval - ctx_length)

    frames_eval = frames[:, :T_eval]
    patches = temporal_patchify(frames_eval, patch)
    z_btLd, _ = encoder(patches)
    assert z_btLd.shape[2] % packing_factor == 0
    n_spatial = z_btLd.shape[2] // packing_factor
    z_gt_packed = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=packing_factor)

    actions_eval  = None if actions  is None else actions[:, :T_eval]
    act_mask_eval = None if act_mask is None else (
        act_mask[:, :T_eval] if act_mask.dim() == 3 else act_mask
    )

    z_pred_packed = sample_autoregressive_packed_sequence(
        dyn,
        z_gt_packed=z_gt_packed,
        ctx_length=ctx_length,
        horizon=horizon,
        k_max=k_max,
        sched=sched,
        actions=actions_eval,
        act_mask=act_mask_eval,
    )

    pred_frames = decode_packed_to_frames(
        decoder,
        z_packed=z_pred_packed,
        H=H, W=W, C=C, patch=patch,
        packing_factor=packing_factor,
    )

    floor = frames_eval.clone()
    if horizon > 0:
        floor[:, ctx_length:ctx_length + horizon] = frames_eval[
            :, ctx_length - 1 : ctx_length
        ].expand(-1, horizon, -1, -1, -1)

    gt_h    = frames_eval[:, ctx_length : ctx_length + horizon]
    pred_h  = pred_frames[:, ctx_length : ctx_length + horizon]
    floor_h = floor[:, ctx_length : ctx_length + horizon]

    mse_pred  = (pred_h.float()  - gt_h.float()).pow(2).mean()
    mse_floor = (floor_h.float() - gt_h.float()).pow(2).mean()

    psnr_pred  = 10.0 * torch.log10(1.0 / mse_pred.clamp_min(1e-12))
    psnr_floor = 10.0 * torch.log10(1.0 / mse_floor.clamp_min(1e-12))

    mse_ratio = mse_pred / mse_floor.clamp_min(1e-12)
    psnr_gain = psnr_pred - psnr_floor

    per_t_pred  = (pred_h.float()  - gt_h.float()).pow(2).mean(dim=(0, 2, 3, 4))
    per_t_floor = (floor_h.float() - gt_h.float()).pow(2).mean(dim=(0, 2, 3, 4))

    if horizon > 0:
        i0 = 0
        im = (horizon - 1) // 2
        i1 = horizon - 1

        wandb.log(
            {
                "eval/mse_pred":                     float(mse_pred.item()),
                "eval/mse_floor":                    float(mse_floor.item()),
                "eval/mse_ratio_pred_over_floor":    float(mse_ratio.item()),
                "eval/psnr_pred":                    float(psnr_pred.item()),
                "eval/psnr_floor":                   float(psnr_floor.item()),
                "eval/psnr_gain_over_floor_db":      float(psnr_gain.item()),
                "eval/mse_pred_t1":                  float(per_t_pred[i0].item()),
                "eval/mse_pred_tmid":                float(per_t_pred[im].item()),
                "eval/mse_pred_tend":                float(per_t_pred[i1].item()),
                "eval/mse_floor_t1":                 float(per_t_floor[i0].item()),
                "eval/mse_floor_tmid":               float(per_t_floor[im].item()),
                "eval/mse_floor_tend":               float(per_t_floor[i1].item()),
            },
            step=step,
        )

    log_dynamics_eval_wandb(
        gt=frames_eval,
        pred=pred_frames,
        ctx_length=ctx_length,
        step=step,
        tag="eval",
        max_items=max_items,
    )

    if dyn_was_training:
        dyn.train()
