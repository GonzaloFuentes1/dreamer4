# losses.py
# Standalone loss functions and sampling helpers extracted from train_phase1b_dynamics.py.
# model.py already contains recon_loss_from_mae and lpips_on_mae_recon — imported here
# for convenience so callers can import everything loss-related from one place.

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from model import recon_loss_from_mae, lpips_on_mae_recon  # re-export


# ---------------------------------------------------------------------------
# LPIPS wrapper with lazy init (safe for PyTorch Lightning's device lifecycle)
# ---------------------------------------------------------------------------

class LPIPSLoss(nn.Module):
    """
    Wraps the lpips library with lazy initialisation so it can be instantiated
    in LightningModule.__init__ (before the device is known) and set up later
    in LightningModule.setup() once the correct device is available.
    """

    def __init__(self, net: str = "alex", weight: float = 0.2, subsample_frac: float = 0.5):
        super().__init__()
        self.weight = float(weight)
        self.subsample_frac = float(subsample_frac)
        self._net = net
        self._fn = None  # lazy; call setup(device) before forward

    def setup(self, device: torch.device):
        try:
            import lpips as lpips_lib
        except ImportError as e:
            raise ImportError("pip install lpips") from e
        self._fn = lpips_lib.LPIPS(net=self._net).to(device)
        self._fn.eval()
        self._fn.requires_grad_(False)

    def forward(
        self,
        pred_btnd: torch.Tensor,
        target_btnd: torch.Tensor,
        mae_mask_btNp1: torch.Tensor,
        *,
        H: int,
        W: int,
        C: int,
        patch: int,
    ) -> torch.Tensor:
        assert self._fn is not None, "Call LPIPSLoss.setup(device) before forward()"
        lp = lpips_on_mae_recon(
            self._fn,
            pred_btnd,
            target_btnd,
            mae_mask_btNp1,
            H=H,
            W=W,
            C=C,
            patch=patch,
            subsample_frac=self.subsample_frac,
        )
        return self.weight * lp


# ---------------------------------------------------------------------------
# Shortcut / tau sampling helpers (moved from train_phase1b_dynamics.py verbatim)
# ---------------------------------------------------------------------------

def _emax_from_kmax(k_max: int) -> int:
    emax = int(round(math.log2(k_max)))
    assert (1 << emax) == k_max, "k_max must be power of two"
    return emax


def _sample_step_excluding_dmin(
    device: torch.device, B: int, T: int, k_max: int
) -> tuple[torch.Tensor, torch.Tensor]:
    emax = _emax_from_kmax(k_max)
    step_idx = torch.randint(low=0, high=max(1, emax), size=(B, T), device=device, dtype=torch.long)
    d = 1.0 / (1 << step_idx).to(torch.float32)
    return d, step_idx


def _sample_tau_for_step(
    device: torch.device, B: int, T: int, k_max: int, step_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    K = (1 << step_idx).to(torch.long)
    u = torch.rand((B, T), device=device, dtype=torch.float32)
    j_idx = torch.floor(u * K.to(torch.float32)).to(torch.long)
    tau = j_idx.to(torch.float32) / K.to(torch.float32)
    scale = torch.div(torch.tensor(k_max, device=device), K, rounding_mode="floor")
    tau_idx = j_idx * scale
    return tau, tau_idx


def _is_pow2(n: int) -> bool:
    return (n > 0) and ((n & (n - 1)) == 0)


def make_tau_schedule(*, k_max: int, schedule: str, d: Optional[float] = None) -> Dict:
    """
    Returns a schedule dict used by the autoregressive sampler.
      K = number of integration steps (grid size)
      e = log2(K)  (step_idx)
      scale = k_max // K
      tau_idx[i] = discrete signal index at step i
      tau[i] = i/K
      dt = 1/K
    """
    assert _is_pow2(k_max), "k_max must be power of two"
    if schedule == "finest":
        K = k_max
    elif schedule == "shortcut":
        assert d is not None, "shortcut schedule requires d"
        inv = int(round(1.0 / float(d)))
        assert _is_pow2(inv), "d must be 1/(power of two)"
        assert inv <= k_max, "d must be >= 1/k_max"
        assert (k_max % inv) == 0, "k_max must be divisible by 1/d"
        K = inv
    else:
        raise ValueError(f"unknown schedule: {schedule}")

    e = int(round(math.log2(K)))
    scale = k_max // K
    tau = [i / K for i in range(K)] + [1.0]
    tau_idx = [i * scale for i in range(K)] + [k_max]
    return dict(K=K, e=e, scale=scale, tau=tau, tau_idx=tau_idx, dt=1.0 / K, schedule=schedule, d=1.0 / K)


# ---------------------------------------------------------------------------
# Main dynamics pretraining loss (moved from train_phase1b_dynamics.py verbatim)
# ---------------------------------------------------------------------------

def dynamics_pretrain_loss(
    dynamics: torch.nn.Module,
    *,
    z1: torch.Tensor,                    # (B,T,Sz,Dz) packed clean targets
    actions: Optional[torch.Tensor],     # (B,T,A) or None
    act_mask: Optional[torch.Tensor],    # (B,T,A) or (A,) or None
    k_max: int,
    B_self: int,
    step: int,
    bootstrap_start: int,
    agent_tokens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = z1.device
    B, T = z1.shape[:2]
    assert 0 <= B_self < B
    B_emp = B - B_self
    emax = _emax_from_kmax(k_max)

    act_mask_full = act_mask
    act_mask_self = None if act_mask_full is None else act_mask_full[B_emp:]

    step_idx_emp = torch.full((B_emp, T), emax, device=device, dtype=torch.long)
    if B_self > 0:
        d_self, step_idx_self = _sample_step_excluding_dmin(device, B_self, T, k_max)
        step_idx_full = torch.cat([step_idx_emp, step_idx_self], dim=0)
    else:
        d_self = torch.zeros((0, T), device=device, dtype=torch.float32)
        step_idx_self = torch.zeros((0, T), device=device, dtype=torch.long)
        step_idx_full = step_idx_emp

    sigma_full, sigma_idx_full = _sample_tau_for_step(device, B, T, k_max, step_idx_full)
    sigma_emp = sigma_full[:B_emp]
    sigma_self = sigma_full[B_emp:]
    sigma_idx_self = sigma_idx_full[B_emp:]

    z0_full = torch.randn_like(z1)
    z_tilde_full = (1.0 - sigma_full)[..., None, None] * z0_full + sigma_full[..., None, None] * z1
    z_tilde_self = z_tilde_full[B_emp:]

    w_emp = 0.9 * sigma_emp + 0.1
    w_self = 0.9 * sigma_self + 0.1

    z1_hat_full, _ = dynamics(
        actions, step_idx_full, sigma_idx_full, z_tilde_full,
        act_mask=act_mask_full, agent_tokens=agent_tokens,
    )
    z1_hat_emp = z1_hat_full[:B_emp]
    z1_hat_self = z1_hat_full[B_emp:]

    flow_per = (z1_hat_emp.float() - z1[:B_emp].float()).pow(2).mean(dim=(2, 3))  # (B_emp,T)
    loss_emp = (flow_per * w_emp).mean()

    boot_mse = torch.zeros((), device=device, dtype=torch.float32)
    loss_self = torch.zeros((), device=device, dtype=torch.float32)

    do_boot = (B_self > 0) and (step >= bootstrap_start)
    if do_boot:
        d_half = d_self / 2.0
        step_idx_half = step_idx_self + 1
        sigma_plus = sigma_self + d_half
        sigma_idx_plus = sigma_idx_self + (
            torch.tensor(k_max, device=device, dtype=torch.float32) * d_half
        ).to(torch.long)

        act_self = actions[B_emp:] if actions is not None else None
        ag_self = agent_tokens[B_emp:] if agent_tokens is not None else None

        z1_hat_half1, _ = dynamics(
            act_self, step_idx_half, sigma_idx_self, z_tilde_self,
            act_mask=act_mask_self, agent_tokens=ag_self,
        )
        b_prime = (z1_hat_half1.float() - z_tilde_self.float()) / (
            1.0 - sigma_self
        ).clamp_min(1e-6)[..., None, None]
        z_prime = z_tilde_self.float() + b_prime * d_half[..., None, None]

        z1_hat_half2, _ = dynamics(
            act_self, step_idx_half, sigma_idx_plus, z_prime.to(z_tilde_self.dtype),
            act_mask=act_mask_self, agent_tokens=ag_self,
        )
        b_doubleprime = (z1_hat_half2.float() - z_prime.float()) / (
            1.0 - sigma_plus
        ).clamp_min(1e-6)[..., None, None]

        vhat_sigma = (z1_hat_self.float() - z_tilde_self.float()) / (
            1.0 - sigma_self
        ).clamp_min(1e-6)[..., None, None]
        vbar_target = ((b_prime + b_doubleprime) / 2.0).detach()

        boot_per = (1.0 - sigma_self).pow(2) * (vhat_sigma - vbar_target).pow(2).mean(dim=(2, 3))
        loss_self = (boot_per * w_self).mean()
        boot_mse = boot_per.mean()

    loss = ((loss_emp * (B - B_self)) + (loss_self * B_self)) / B

    aux = {
        "flow_mse": flow_per.mean().detach(),
        "bootstrap_mse": boot_mse.detach(),
        "loss_emp": loss_emp.detach(),
        "loss_self": loss_self.detach(),
        "sigma_mean": sigma_full.mean().detach(),
    }
    return loss, aux
