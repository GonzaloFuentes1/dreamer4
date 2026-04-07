# agent.py
# Phase-2 and Phase-3 network heads for Dreamer 4.
#
# All heads receive h_t — the agent-token outputs of the frozen Dynamics
# transformer — flattened to (B, n_agent * d_model_dyn).
#
#   PolicyHead   — actor; Gaussian over continuous actions; MTP heads for Phase 2
#   RewardHead   — symexp TwoHot; MTP heads for Phase 2
#   ValueHead    — symexp TwoHot; single head for Phase 3
#
# Phase 2  (train_phase2_finetuning.py) trains PolicyHead + RewardHead + Dynamics.
# Phase 3  (train_phase3_imagination.py) initialises PolicyHead from the Phase-2 checkpoint,
#          freezes it as a prior, and trains a new PolicyHead + ValueHead via PMPO.

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import (
    make_twohot_bins,
    twohot_decode,
    twohot_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared trunk
# ─────────────────────────────────────────────────────────────────────────────

class _Trunk(nn.Module):
    """Two-layer MLP with SiLU activations — shared across MTP heads."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# PolicyHead
# ─────────────────────────────────────────────────────────────────────────────

class PolicyHead(nn.Module):
    """
    Actor that reads from h_t (agent-token output of frozen Dynamics).

    Input : h_flat = h_t.flatten(1)  (B, n_agent * d_model_dyn)
    Output: Gaussian over action_dim  (B, action_dim)

    Has `mtp_length` separate (mu, log_std) output pairs for multi-token
    prediction during Phase 2 behavior cloning.  Phase 3 always uses offset=0.
    """

    def __init__(
        self,
        state_dim:  int,           # n_agent * d_model_dyn
        action_dim: int = 16,
        hidden_dim: int = 512,
        mtp_length: int = 8,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.mtp_length = int(mtp_length)

        self.trunk = _Trunk(state_dim, hidden_dim)
        self.mu_heads      = nn.ModuleList([nn.Linear(hidden_dim, action_dim) for _ in range(mtp_length)])
        self.log_std_heads = nn.ModuleList([nn.Linear(hidden_dim, action_dim) for _ in range(mtp_length)])

        for head in self.mu_heads:
            nn.init.zeros_(head.bias)
            nn.init.normal_(head.weight, std=0.01)
        for head in self.log_std_heads:
            nn.init.constant_(head.bias, -1.0)   # init std ≈ 0.37
            nn.init.normal_(head.weight, std=0.01)

    def forward(
        self, h_flat: torch.Tensor, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_std), each (B, action_dim)."""
        feat    = self.trunk(h_flat)
        mu      = self.mu_heads[offset](feat)
        log_std = self.log_std_heads[offset](feat).clamp(-5.0, 2.0)
        return mu, log_std

    def sample(
        self,
        h_flat:   torch.Tensor,
        act_mask: Optional[torch.Tensor] = None,
        offset:   int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterized sample.
        Returns (action_raw, action) — action = tanh(action_raw) * act_mask ∈ [-1,1].
        action_raw is the pre-tanh sample needed for log_prob().
        """
        mu, log_std = self.forward(h_flat, offset)
        std        = log_std.exp()
        action_raw = mu + std * torch.randn_like(mu)
        action     = torch.tanh(action_raw)
        if act_mask is not None:
            action = action * act_mask
        return action_raw, action

    def log_prob(
        self,
        h_flat:     torch.Tensor,
        action_raw: torch.Tensor,
        act_mask:   Optional[torch.Tensor] = None,
        offset:     int = 0,
    ) -> torch.Tensor:
        """(B,)  sum of log-probs over active action dimensions."""
        mu, log_std = self.forward(h_flat, offset)
        std  = log_std.exp()
        lp   = torch.distributions.Normal(mu, std).log_prob(action_raw)  # (B, A)
        if act_mask is not None:
            lp = lp * act_mask
        return lp.sum(-1)

    def entropy(
        self,
        h_flat:   torch.Tensor,
        act_mask: Optional[torch.Tensor] = None,
        offset:   int = 0,
    ) -> torch.Tensor:
        """(B,)  differential entropy summed over active dimensions."""
        _, log_std = self.forward(h_flat, offset)
        H = 0.5 * (1.0 + math.log(2.0 * math.pi * math.e)) + log_std
        if act_mask is not None:
            H = H * act_mask
        return H.sum(-1)

    def bc_loss(
        self,
        h_flat_bt:   torch.Tensor,           # (B, T, state_dim)
        actions_bt:  torch.Tensor,           # (B, T, action_dim)  target in [-1,1]
        act_mask_bt: Optional[torch.Tensor], # (B, T, action_dim)  or None
    ) -> torch.Tensor:
        """
        MTP behavior-cloning loss: from h_t predict a_{t+n} for n in [0, mtp_length).
        Averages over MTP heads, valid timesteps, and batch.
        """
        B, T, _ = h_flat_bt.shape
        L       = self.mtp_length
        T_valid = max(1, T - L)

        feat  = self.trunk(h_flat_bt[:, :T_valid].flatten(0, 1))      # (B*T_v, hidden) — computed once

        total = torch.tensor(0.0, device=h_flat_bt.device)
        for n in range(L):
            end = T_valid + n
            if end > T:
                break
            a_n  = actions_bt[:, n:n + T_valid].flatten(0, 1)        # (B*T_v, A)
            mask = None if act_mask_bt is None else act_mask_bt[:, n:n + T_valid].flatten(0, 1)

            a_raw   = torch.atanh(a_n.clamp(-0.999, 0.999))
            mu      = self.mu_heads[n](feat)
            log_std = self.log_std_heads[n](feat).clamp(-5.0, 2.0)
            std     = log_std.exp()
            lp      = torch.distributions.Normal(mu, std).log_prob(a_raw)
            if mask is not None:
                lp = lp * mask
            total = total - lp.sum(-1).mean()
        return total / L


# ─────────────────────────────────────────────────────────────────────────────
# RewardHead
# ─────────────────────────────────────────────────────────────────────────────

class RewardHead(nn.Module):
    """
    Reward head: h_flat → symexp TwoHot logits (255 bins).
    Has mtp_length output heads for Phase-2 multi-token prediction.
    """

    def __init__(
        self,
        state_dim:  int,
        hidden_dim: int = 256,
        mtp_length: int = 8,
        num_bins:   int = 255,
    ):
        super().__init__()
        self.mtp_length = int(mtp_length)
        self.num_bins   = int(num_bins)
        self.trunk      = _Trunk(state_dim, hidden_dim)
        self.heads      = nn.ModuleList([nn.Linear(hidden_dim, num_bins) for _ in range(mtp_length)])
        self.register_buffer("bins", make_twohot_bins(), persistent=False)

    def forward(self, h_flat: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Returns reward logits (B, num_bins)."""
        return self.heads[offset](self.trunk(h_flat))

    def predict(self, h_flat: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Expected reward (B,) in natural scale."""
        logits = self.forward(h_flat, offset)
        return twohot_decode(logits, self.bins.to(logits.device))

    def mtp_loss(
        self,
        h_flat_bt:  torch.Tensor,   # (B, T, state_dim)
        rewards_bt: torch.Tensor,   # (B, T)
    ) -> torch.Tensor:
        """MTP reward loss averaged over heads and valid timesteps."""
        B, T, _ = h_flat_bt.shape
        L       = self.mtp_length
        T_valid = max(1, T - L)

        feat  = self.trunk(h_flat_bt[:, :T_valid].flatten(0, 1))    # (B*T_v, hidden) — computed once
        bins  = self.bins.to(h_flat_bt.device)

        total = torch.tensor(0.0, device=h_flat_bt.device)
        for n in range(L):
            if n + T_valid > T:
                break
            r_n    = rewards_bt[:, n:n + T_valid].flatten()        # (B*T_v,)
            logits = self.heads[n](feat)
            total  = total + twohot_loss(logits, r_n, bins).mean()
        return total / L


# ─────────────────────────────────────────────────────────────────────────────
# ValueHead
# ─────────────────────────────────────────────────────────────────────────────

class ValueHead(nn.Module):
    """
    Value head: h_flat → symexp TwoHot logits → E[V(s)].
    Single head (no MTP).
    """

    def __init__(
        self,
        state_dim:  int,
        hidden_dim: int = 512,
        num_bins:   int = 255,
    ):
        super().__init__()
        self.num_bins = int(num_bins)
        self.trunk    = _Trunk(state_dim, hidden_dim)
        self.head     = nn.Linear(hidden_dim, num_bins)
        self.register_buffer("bins", make_twohot_bins(), persistent=False)

    def forward(self, h_flat: torch.Tensor) -> torch.Tensor:
        """Returns value logits (B, num_bins)."""
        return self.head(self.trunk(h_flat))

    def predict(self, h_flat: torch.Tensor) -> torch.Tensor:
        """Expected value (B,) in natural scale."""
        logits = self.forward(h_flat)
        return twohot_decode(logits, self.bins.to(logits.device))

    def loss(self, h_flat: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Twohot cross-entropy loss. targets (B,) in natural scale."""
        logits = self.forward(h_flat)
        return twohot_loss(logits, targets, self.bins.to(logits.device)).mean()
