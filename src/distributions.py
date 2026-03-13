# distributions.py
# SymExp TwoHot distribution (Dreamer 3, eq. 10) and PMPO policy loss (eq. 11).
#
# SymExp TwoHot
#   Encodes scalar targets as soft one-hot over 255 bins uniformly spaced in
#   *symlog* space, bridging symlog(-20) … symlog(20).  The expected value is
#   recovered as symexp(Σ p_i * bins_i).  This is exactly the Dreamer 3
#   parameterisation and handles rewards/values spanning many orders of magnitude.
#
# PMPO
#   Equation 11: balances positive and negative advantage sets equally (α=0.5)
#   and regularises toward a behavioral prior with reverse-KL (β=0.3).

import math
from typing import Optional

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# symlog / symexp
# ─────────────────────────────────────────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * log(1 + |x|)  — maps unbounded reals to a compact range."""
    return x.sign() * (x.abs() + 1.0).log()


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return x.sign() * (x.abs().exp() - 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# TwoHot helpers
# ─────────────────────────────────────────────────────────────────────────────

_NUM_BINS:    int   = 255
_SYMLOG_LOW:  float = -20.0
_SYMLOG_HIGH: float =  20.0


def make_twohot_bins(device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return (255,) of bin boundaries **in symlog space**, uniformly spaced."""
    return torch.linspace(_SYMLOG_LOW, _SYMLOG_HIGH, _NUM_BINS, device=device, dtype=dtype)


def twohot_encode(targets: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    targets : (*,)  in natural scale
    bins    : (N,)  in symlog scale  (from make_twohot_bins)
    returns : (*, N)  soft one-hot: at most 2 adjacent bins have non-zero weight
    """
    N  = bins.shape[0]
    bw = (_SYMLOG_HIGH - _SYMLOG_LOW) / (N - 1)       # uniform bin width

    t_sl   = symlog(targets).clamp(_SYMLOG_LOW, _SYMLOG_HIGH)    # (*)
    below  = ((t_sl - _SYMLOG_LOW) / bw).long().clamp(0, N - 2)  # (*)
    above  = below + 1

    w_above = (t_sl - bins[below]) / (bins[above] - bins[below] + 1e-8)
    w_below = 1.0 - w_above

    out = torch.zeros(*targets.shape, N, device=targets.device, dtype=targets.dtype)
    out.scatter_(-1, below.unsqueeze(-1), w_below.unsqueeze(-1))
    out.scatter_(-1, above.unsqueeze(-1), w_above.unsqueeze(-1))
    return out


def twohot_decode(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    logits : (*, N)  raw network output
    bins   : (N,)    in symlog scale
    returns: (*)     expected value in natural scale: symexp(Σ p_i * bins_i)
    """
    probs      = F.softmax(logits.float(), dim=-1)               # (*, N)
    symlog_val = (probs * bins.to(dtype=probs.dtype)).sum(-1)    # (*,)
    return symexp(symlog_val)


def twohot_loss(
    logits:  torch.Tensor,   # (*, N)
    targets: torch.Tensor,   # (*)     in natural scale
    bins:    torch.Tensor,   # (N,)    in symlog scale
) -> torch.Tensor:
    """Cross-entropy against twohot-encoded targets. Returns (*,) per-element loss."""
    labels  = twohot_encode(targets, bins).detach()     # (*, N)  no grad
    log_p   = F.log_softmax(logits.float(), dim=-1)     # (*, N)
    return -(labels * log_p).sum(-1)                    # (*)


# ─────────────────────────────────────────────────────────────────────────────
# PMPO policy loss  (eq. 11)
# ─────────────────────────────────────────────────────────────────────────────

def pmpo_loss(
    log_probs:       torch.Tensor,   # (N,) or (B, H)  log π_θ(a|s)
    advantages:      torch.Tensor,   # same shape       A_t = R_λ - v_t
    log_probs_prior: torch.Tensor,   # same shape       log π_prior(a|s)
    alpha: float = 0.5,
    beta:  float = 0.3,
) -> torch.Tensor:
    """
    PMPO loss (eq. 11). All tensors are flattened internally. Returns scalar.

    L = (1-α)/|D-| Σ_{D-}(-log π)  −  α/|D+| Σ_{D+}(-log π)  +  β/N Σ KL[π∥π_prior]

    D+ = {i | A_i ≥ 0}  (reinforce on good actions)
    D- = {i | A_i <  0}  (suppressed bad actions)
    KL[π∥π_prior] ≈ log π_θ − log π_prior  (reverse KL, same sampled action)
    """
    lp     = log_probs.flatten()                    # (N,)
    adv    = advantages.flatten()                   # (N,)
    lp_pr  = log_probs_prior.flatten().detach()     # (N,) — always stop-grad

    pos_mask = (adv >= 0.0)
    neg_mask = ~pos_mask

    pos_n = pos_mask.float().sum().clamp(min=1.0)
    neg_n = neg_mask.float().sum().clamp(min=1.0)

    pos_loss = -(lp * pos_mask.float()).sum() / pos_n * alpha
    neg_loss =  (lp * neg_mask.float()).sum() / neg_n * (1.0 - alpha)
    kl_loss  = beta * (lp - lp_pr).mean()           # reverse KL approximation

    return pos_loss + neg_loss + kl_loss
