# src/loss_norm.py
# RMS loss normalisation as described in Dreamer 4:
#   "normalize all loss terms by running estimates of their root-mean-square (RMS)"
#
# Each loss term is divided by its running RMS before being summed, which
# automatically balances heterogeneous losses (e.g. flow_mse vs BC vs reward)
# without manual coefficients that require tuning.

import torch
import torch.nn as nn


class RunningRMS(nn.Module):
    """
    Tracks a running RMS of a scalar loss and normalises by it.

    State is stored as an nn.Buffer so it is:
      - persisted inside Lightning checkpoints (survives resume)
      - placed on the correct device automatically
      - excluded from the optimiser parameter list

    Args:
        decay    : EMA decay for the squared loss estimate (default 0.99).
        eps      : floor for the RMS to avoid division by zero / instability
                   at the very start of training (default 1.0 — keeps the raw
                   loss magnitude until the running estimate stabilises).
        max_update: cap on the loss value fed into the EMA (default 50.0).
                   Outlier spikes inflate the EMA only up to this value, so a
                   single bad batch cannot suppress the loss term for hundreds
                   of subsequent steps.  The actual loss passed to the
                   optimiser is unaffected.
    """

    def __init__(self, decay: float = 0.99, eps: float = 1.0, max_update: float = 50.0):
        super().__init__()
        self.decay      = float(decay)
        self.eps        = float(eps)
        self.max_update = float(max_update)
        # EMA of loss² ; sqrt(sq) == RMS
        self.register_buffer("sq", torch.ones(1))

    @torch.no_grad()
    def _update(self, loss: torch.Tensor) -> None:
        # Clamp the value used for the EMA update to prevent outlier spikes
        # from permanently inflating the normaliser.  The loss fed to the
        # optimiser (the return value of normalize()) is NOT clamped.
        val = loss.detach().float().mean().clamp(max=self.max_update)
        self.sq.mul_(self.decay).add_((1.0 - self.decay) * val * val)

    def normalize(self, loss: torch.Tensor) -> torch.Tensor:
        """Update running estimate and return loss / rms (same dtype as loss)."""
        self._update(loss)
        rms = self.sq.sqrt().clamp(min=self.eps).to(loss.dtype)
        return loss / rms
