"""Unit tests for the loss helpers in src/losses.py and src/model.py (torch only)."""


import pytest
import torch
from torch import nn

from losses import (
    _emax_from_kmax,
    _is_pow2,
    _sample_step_excluding_dmin,
    _sample_tau_for_step,
    dynamics_pretrain_loss,
    make_tau_schedule,
)
from model import recon_loss_from_mae

# ---------------------------------------------------------------------------
# Power-of-two helpers and tau schedules
# ---------------------------------------------------------------------------

def test_is_pow2():
    assert [_is_pow2(n) for n in (1, 2, 3, 4, 6, 8, 64)] == [True, True, False, True, False, True, True]
    assert not _is_pow2(0)


def test_emax_from_kmax():
    assert _emax_from_kmax(1) == 0
    assert _emax_from_kmax(8) == 3
    assert _emax_from_kmax(64) == 6
    with pytest.raises(AssertionError):
        _emax_from_kmax(6)


def test_make_tau_schedule_finest():
    sched = make_tau_schedule(k_max=8, schedule="finest")
    assert sched["K"] == 8
    assert sched["e"] == 3
    assert sched["scale"] == 1
    assert sched["dt"] == pytest.approx(1 / 8)
    assert sched["tau"] == [i / 8 for i in range(8)] + [1.0]
    assert sched["tau_idx"] == list(range(8)) + [8]


def test_make_tau_schedule_shortcut():
    sched = make_tau_schedule(k_max=8, schedule="shortcut", d=0.25)
    assert sched["K"] == 4
    assert sched["e"] == 2
    assert sched["scale"] == 2
    assert sched["tau"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert sched["tau_idx"] == [0, 2, 4, 6, 8]
    # tau_idx is always tau scaled to the finest grid.
    for t, ti in zip(sched["tau"], sched["tau_idx"]):
        assert ti == round(t * 8)


def test_make_tau_schedule_validates_arguments():
    with pytest.raises(AssertionError):
        make_tau_schedule(k_max=6, schedule="finest")
    with pytest.raises(AssertionError):
        make_tau_schedule(k_max=8, schedule="shortcut")  # d is required
    with pytest.raises(AssertionError):
        make_tau_schedule(k_max=8, schedule="shortcut", d=1 / 16)  # finer than k_max
    with pytest.raises(AssertionError):
        make_tau_schedule(k_max=8, schedule="shortcut", d=1 / 3)  # not a power of two
    with pytest.raises(ValueError):
        make_tau_schedule(k_max=8, schedule="unknown")


# ---------------------------------------------------------------------------
# Step-size and signal-level sampling
# ---------------------------------------------------------------------------

def test_sample_step_excludes_the_finest_step_size():
    torch.manual_seed(0)
    k_max = 16
    d, step_idx = _sample_step_excluding_dmin(torch.device("cpu"), 8, 12, k_max)
    emax = _emax_from_kmax(k_max)
    assert d.shape == step_idx.shape == (8, 12)
    assert step_idx.min().item() >= 0
    assert step_idx.max().item() <= emax - 1  # d_min = 1/k_max is never sampled
    torch.testing.assert_close(d, 1.0 / (2.0 ** step_idx.float()))
    assert d.min().item() > 1.0 / k_max


def test_sample_tau_lies_on_the_grid_of_the_chosen_step_size():
    torch.manual_seed(0)
    k_max = 16
    B, T = 6, 10
    _, step_idx = _sample_step_excluding_dmin(torch.device("cpu"), B, T, k_max)
    tau, tau_idx = _sample_tau_for_step(torch.device("cpu"), B, T, k_max, step_idx)
    K = (1 << step_idx).float()
    assert tau.shape == tau_idx.shape == (B, T)
    assert torch.all(tau >= 0) and torch.all(tau < 1)
    # tau is a multiple of 1/K, and tau_idx is the same level on the k_max grid.
    torch.testing.assert_close(tau * K, torch.round(tau * K), atol=1e-5, rtol=0)
    torch.testing.assert_close(tau_idx.float(), torch.round(tau * k_max), atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# Reconstruction loss
# ---------------------------------------------------------------------------

def test_recon_loss_only_counts_masked_patches():
    B, T, Np, Dp = 1, 2, 3, 4
    pred = torch.zeros(B, T, Np, Dp)
    target = torch.zeros(B, T, Np, Dp)
    target[0, 0, 1] = 2.0   # masked patch with error 2   -> squared error 4
    target[0, 1, 2] = 10.0  # unmasked patch: must be ignored
    mask = torch.zeros(B, T, Np, 1, dtype=torch.bool)
    mask[0, 0, 1] = True
    mask[0, 0, 0] = True  # masked patch with zero error
    loss = recon_loss_from_mae(pred, target, mask)
    # Two masked patches, each with Dp elements: sum = 4 * Dp = 16 ; denominator = 2 * Dp = 8.
    assert loss.item() == pytest.approx(2.0)


def test_recon_loss_with_empty_mask_is_zero_and_finite():
    pred = torch.randn(2, 3, 5, 4)
    target = torch.randn(2, 3, 5, 4)
    mask = torch.zeros(2, 3, 5, 1, dtype=torch.bool)
    loss = recon_loss_from_mae(pred, target, mask)
    assert loss.item() == 0.0


def test_recon_loss_is_computed_in_fp32_for_half_inputs():
    pred = torch.full((1, 1, 2, 2), 1.0, dtype=torch.bfloat16)
    target = torch.zeros(1, 1, 2, 2, dtype=torch.bfloat16)
    mask = torch.ones(1, 1, 2, 1, dtype=torch.bool)
    loss = recon_loss_from_mae(pred, target, mask)
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# dynamics_pretrain_loss with a stub dynamics network
# ---------------------------------------------------------------------------

class _StubDynamics(nn.Module):
    """Mimics the call signature of model.Dynamics: returns (z1_hat, aux)."""

    def __init__(self, scale: float = 0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.calls = 0

    def forward(self, actions, step_idx, sigma_idx, z_tilde, act_mask=None, agent_tokens=None):
        self.calls += 1
        return z_tilde * self.scale, None


def _batch(B=4, T=3, Sz=2, Dz=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, Sz, Dz, generator=g)


def test_pretrain_loss_without_bootstrap_is_weighted_flow_mse():
    torch.manual_seed(0)
    z1 = _batch()
    dyn = _StubDynamics(scale=0.0)  # predicts zeros
    loss, aux = dynamics_pretrain_loss(
        dyn, z1=z1, actions=None, act_mask=None, k_max=8, B_self=0, step=0, bootstrap_start=0,
    )
    assert dyn.calls == 1
    flow = aux["flow_mse"].item()
    # Zero prediction: flow_mse is the mean squared target. The weights lie in [0.1, 1].
    assert flow == pytest.approx(z1.pow(2).mean().item(), rel=1e-5)
    assert 0.1 * flow - 1e-6 <= loss.item() <= flow + 1e-6
    assert aux["bootstrap_mse"].item() == 0.0
    assert aux["loss_self"].item() == 0.0
    assert 0.0 <= aux["sigma_mean"].item() < 1.0


def test_pretrain_loss_bootstrap_is_skipped_before_bootstrap_start():
    torch.manual_seed(0)
    dyn = _StubDynamics(scale=0.5)
    loss, aux = dynamics_pretrain_loss(
        dyn, z1=_batch(), actions=None, act_mask=None, k_max=8, B_self=2, step=0, bootstrap_start=10,
    )
    assert dyn.calls == 1
    assert aux["bootstrap_mse"].item() == 0.0
    assert torch.isfinite(loss)


def test_pretrain_loss_with_bootstrap_runs_extra_forward_passes_and_has_gradients():
    torch.manual_seed(0)
    dyn = _StubDynamics(scale=0.5)
    loss, aux = dynamics_pretrain_loss(
        dyn, z1=_batch(), actions=None, act_mask=None, k_max=8, B_self=2, step=10, bootstrap_start=10,
    )
    assert dyn.calls == 3  # main pass + two half-step passes for the bootstrap target
    assert torch.isfinite(loss)
    assert aux["bootstrap_mse"].item() >= 0.0
    assert set(aux) == {"flow_mse", "bootstrap_mse", "loss_emp", "loss_self", "sigma_mean"}
    loss.backward()
    assert dyn.scale.grad is not None and torch.isfinite(dyn.scale.grad)


def test_pretrain_loss_requires_at_least_one_empirical_sample():
    dyn = _StubDynamics()
    with pytest.raises(AssertionError):
        dynamics_pretrain_loss(
            dyn, z1=_batch(B=4), actions=None, act_mask=None, k_max=8, B_self=4, step=0, bootstrap_start=0,
        )


def test_pretrain_loss_rejects_non_power_of_two_k_max():
    dyn = _StubDynamics()
    with pytest.raises(AssertionError):
        dynamics_pretrain_loss(
            dyn, z1=_batch(), actions=None, act_mask=None, k_max=6, B_self=0, step=0, bootstrap_start=0,
        )
