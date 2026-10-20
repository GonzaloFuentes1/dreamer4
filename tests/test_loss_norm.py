"""Unit tests for src/loss_norm.py (running-RMS loss normalisation)."""

import math

import pytest
import torch

from loss_norm import RunningRMS


def test_initial_state_and_buffer_is_checkpointed():
    norm = RunningRMS()
    assert list(norm.parameters()) == []
    state = norm.state_dict()
    assert "sq" in state
    assert state["sq"].item() == 1.0


def test_first_update_matches_hand_computation():
    norm = RunningRMS(decay=0.99, eps=1.0, max_update=50.0)
    loss = torch.tensor(5.0)
    out = norm.normalize(loss)
    sq = 0.99 * 1.0 + 0.01 * 25.0
    assert out.item() == pytest.approx(5.0 / math.sqrt(sq), rel=1e-6)


def test_rms_is_floored_at_eps():
    norm = RunningRMS(decay=0.99, eps=1.0)
    out = norm.normalize(torch.tensor(0.0))
    # sq decays below 1, but the divisor is floored at eps = 1, so the loss is unchanged.
    assert out.item() == 0.0
    out = norm.normalize(torch.tensor(0.5))
    assert out.item() == pytest.approx(0.5)


def test_outlier_is_capped_in_the_running_estimate_only():
    norm = RunningRMS(decay=0.99, eps=1.0, max_update=50.0)
    big = torch.tensor(1e6)
    out = norm.normalize(big)
    # EMA input is clamped to max_update; the returned loss is still the full-size loss divided by rms.
    sq = 0.99 * 1.0 + 0.01 * 50.0 ** 2
    assert norm.sq.item() == pytest.approx(sq, rel=1e-6)
    assert out.item() == pytest.approx(1e6 / math.sqrt(sq), rel=1e-6)


def test_normalised_loss_keeps_gradient_and_dtype():
    norm = RunningRMS()
    x = torch.tensor(3.0, requires_grad=True)
    out = norm.normalize(x * 2.0)
    out.backward()
    assert x.grad is not None and x.grad.item() > 0
    assert out.dtype == torch.float32


def test_sq_converges_towards_the_squared_loss():
    norm = RunningRMS(decay=0.9)
    for _ in range(200):
        norm.normalize(torch.tensor(4.0))
    assert norm.sq.item() == pytest.approx(16.0, rel=1e-3)
