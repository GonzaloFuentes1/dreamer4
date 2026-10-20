"""Unit tests for src/distributions.py (symlog/symexp, SymExp TwoHot, PMPO)."""

import math

import pytest
import torch

from distributions import (
    make_twohot_bins,
    pmpo_loss,
    symexp,
    symlog,
    twohot_decode,
    twohot_encode,
    twohot_loss,
)

# ---------------------------------------------------------------------------
# symlog / symexp
# ---------------------------------------------------------------------------

def test_symlog_symexp_are_inverse():
    x = torch.tensor(
        [-1e6, -1234.5, -10.0, -1.0, -1e-3, 0.0, 1e-3, 0.5, 1.0, 7.0, 1e3, 1e6],
        dtype=torch.float64,
    )
    torch.testing.assert_close(symexp(symlog(x)), x, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(symlog(symexp(symlog(x))), symlog(x), rtol=1e-9, atol=1e-9)


def test_symlog_symexp_inverse_on_random_values():
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(1000, generator=g, dtype=torch.float64) * 100.0)
    torch.testing.assert_close(symexp(symlog(x)), x, rtol=1e-9, atol=1e-9)


def test_symlog_known_values():
    assert symlog(torch.tensor(0.0)).item() == 0.0
    assert symlog(torch.tensor(math.e - 1.0)).item() == pytest.approx(1.0, abs=1e-6)
    assert symexp(torch.tensor(1.0)).item() == pytest.approx(math.e - 1.0, abs=1e-6)


def test_symlog_is_odd_monotonic_and_compressive():
    x = torch.linspace(-100.0, 100.0, 2001, dtype=torch.float64)
    y = symlog(x)
    torch.testing.assert_close(symlog(-x), -y)
    assert torch.all(y[1:] > y[:-1])
    # Large magnitudes are compressed to roughly log scale.
    assert symlog(torch.tensor(1e6)).item() < 15.0


# ---------------------------------------------------------------------------
# TwoHot bins
# ---------------------------------------------------------------------------

def test_twohot_bins_layout():
    bins = make_twohot_bins()
    assert bins.shape == (255,)
    assert bins[0].item() == pytest.approx(-20.0)
    assert bins[-1].item() == pytest.approx(20.0)
    diffs = bins[1:] - bins[:-1]
    assert torch.all(diffs > 0)
    torch.testing.assert_close(diffs, torch.full_like(diffs, 40.0 / 254.0), rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# TwoHot encode / decode
# ---------------------------------------------------------------------------

TARGETS = [-1e4, -100.0, -5.5, -1.0, -0.3, 0.0, 0.25, 0.7, 2.0, 100.0, 1e4]


def test_twohot_encode_is_a_distribution_with_at_most_two_bins():
    bins = make_twohot_bins()
    enc = twohot_encode(torch.tensor(TARGETS), bins)
    assert enc.shape == (len(TARGETS), 255)
    assert torch.all(enc >= 0)
    torch.testing.assert_close(enc.sum(-1), torch.ones(len(TARGETS)), rtol=1e-5, atol=1e-5)
    assert torch.all((enc > 0).sum(-1) <= 2)


def test_twohot_encode_weights_the_two_adjacent_bins():
    bins = make_twohot_bins()
    target = torch.tensor([1.0])
    enc = twohot_encode(target, bins)[0]
    s = symlog(target).item()
    hi = int(torch.searchsorted(bins, torch.tensor(s)).item())
    lo = hi - 1
    assert bins[lo].item() <= s <= bins[hi].item()
    nonzero = torch.nonzero(enc).flatten().tolist()
    assert set(nonzero) <= {lo, hi}
    # Linear interpolation weights in symlog space.
    w_hi = (s - bins[lo].item()) / (bins[hi].item() - bins[lo].item())
    assert enc[hi].item() == pytest.approx(w_hi, abs=1e-4)
    assert enc[lo].item() == pytest.approx(1.0 - w_hi, abs=1e-4)


def test_twohot_decode_of_encode_recovers_the_target():
    """The expectation of the two-hot encoding, mapped back with symexp, is the target."""
    bins = make_twohot_bins()
    targets = torch.tensor(TARGETS)
    enc = twohot_encode(targets, bins)
    logits = torch.log(enc)  # log(0) = -inf -> exactly zero probability after softmax
    decoded = twohot_decode(logits, bins)
    torch.testing.assert_close(decoded, targets, rtol=1e-3, atol=1e-3)


def test_twohot_decode_expectation_in_symlog_space():
    bins = make_twohot_bins()
    # Probability mass split between two known bins.
    logits = torch.full((255,), float("-inf"))
    logits[100] = math.log(0.25)
    logits[150] = math.log(0.75)
    expected_symlog = 0.25 * bins[100].item() + 0.75 * bins[150].item()
    decoded = twohot_decode(logits, bins)
    assert decoded.item() == pytest.approx(
        symexp(torch.tensor(expected_symlog)).item(), rel=1e-4
    )


def test_twohot_decode_uniform_logits_is_zero():
    # The bins are symmetric around zero, so a uniform distribution decodes to 0.
    bins = make_twohot_bins()
    decoded = twohot_decode(torch.zeros(4, 255), bins)
    torch.testing.assert_close(decoded, torch.zeros(4), atol=1e-4, rtol=0)


def test_twohot_encode_clamps_out_of_range_targets():
    bins = make_twohot_bins()
    enc = twohot_encode(torch.tensor([1e12, -1e12]), bins)
    torch.testing.assert_close(enc.sum(-1), torch.ones(2), rtol=1e-5, atol=1e-5)
    assert enc[0, -1].item() == pytest.approx(1.0, abs=1e-4)
    assert enc[1, 0].item() == pytest.approx(1.0, abs=1e-4)


def test_twohot_encode_preserves_leading_shape():
    bins = make_twohot_bins()
    enc = twohot_encode(torch.randn(3, 4, 5), bins)
    assert enc.shape == (3, 4, 5, 255)


# ---------------------------------------------------------------------------
# TwoHot loss
# ---------------------------------------------------------------------------

def test_twohot_loss_shape_and_nonnegative():
    bins = make_twohot_bins()
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(6, 255, generator=g)
    targets = torch.randn(6, generator=g) * 10.0
    loss = twohot_loss(logits, targets, bins)
    assert loss.shape == (6,)
    assert torch.all(loss >= 0)


def test_twohot_loss_is_lower_for_correct_logits_than_uniform():
    bins = make_twohot_bins()
    targets = torch.tensor([-3.0, 0.0, 0.4, 25.0])
    # Finite logits (log(0) = -inf would give 0 * -inf = nan in the cross-entropy).
    perfect = torch.log(twohot_encode(targets, bins).clamp_min(1e-9))
    uniform = torch.zeros(4, 255)
    loss_perfect = twohot_loss(perfect, targets, bins)
    loss_uniform = twohot_loss(uniform, targets, bins)
    assert torch.all(loss_perfect < loss_uniform)
    # Uniform logits give exactly log(N) cross-entropy against any distribution.
    torch.testing.assert_close(
        loss_uniform, torch.full((4,), math.log(255)), rtol=1e-4, atol=1e-4
    )


def test_twohot_loss_gradient_descent_reduces_loss():
    bins = make_twohot_bins()
    targets = torch.tensor([2.0, -7.0])
    logits = torch.zeros(2, 255, requires_grad=True)
    opt = torch.optim.SGD([logits], lr=5.0)
    first = twohot_loss(logits, targets, bins).mean().item()
    for _ in range(25):
        opt.zero_grad()
        twohot_loss(logits, targets, bins).mean().backward()
        opt.step()
    last = twohot_loss(logits, targets, bins).mean().item()
    assert last < first
    decoded = twohot_decode(logits.detach(), bins)
    torch.testing.assert_close(decoded, targets, rtol=0.15, atol=0.15)


def test_twohot_loss_does_not_backprop_into_targets():
    bins = make_twohot_bins()
    targets = torch.tensor([1.5], requires_grad=True)
    logits = torch.zeros(1, 255, requires_grad=True)
    twohot_loss(logits, targets, bins).sum().backward()
    assert logits.grad is not None
    assert targets.grad is None or torch.all(targets.grad == 0)


# ---------------------------------------------------------------------------
# PMPO
# ---------------------------------------------------------------------------

def test_pmpo_loss_matches_hand_computation_without_kl():
    lp = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    adv = torch.tensor([1.0, -1.0, 2.0, -2.0])
    # D+ = {0, 2}: mean log-prob = -2 ; D- = {1, 3}: mean log-prob = -3.
    # loss = -alpha * (-2) + (1 - alpha) * (-3) = 1.0 - 1.5 = -0.5   (alpha = 0.5)
    loss = pmpo_loss(lp, adv, lp.clone(), alpha=0.5, beta=0.3)
    assert loss.item() == pytest.approx(-0.5, abs=1e-6)


def test_pmpo_loss_alpha_weights_the_two_sets():
    lp = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    adv = torch.tensor([1.0, -1.0, 2.0, -2.0])
    loss = pmpo_loss(lp, adv, lp.clone(), alpha=0.8, beta=0.0)
    expected = -0.8 * (-2.0) + 0.2 * (-3.0)
    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_pmpo_loss_kl_term_is_reverse_kl_estimate():
    lp = torch.tensor([-1.0, -2.0])
    adv = torch.tensor([1.0, -1.0])
    prior = torch.tensor([-1.5, -1.0])
    base = pmpo_loss(lp, adv, lp.clone(), alpha=0.5, beta=0.3)
    with_kl = pmpo_loss(lp, adv, prior, alpha=0.5, beta=0.3)
    expected_kl = 0.3 * ((lp - prior).mean().item())
    assert (with_kl - base).item() == pytest.approx(expected_kl, abs=1e-6)


def test_pmpo_loss_zero_advantage_counts_as_positive():
    lp = torch.tensor([-1.0, -3.0])
    adv = torch.tensor([0.0, 0.0])
    loss = pmpo_loss(lp, adv, lp.clone(), alpha=0.5, beta=0.0)
    # All samples are in D+: -alpha * mean(lp) = 0.5 * 2 = 1.0, and D- is empty.
    assert loss.item() == pytest.approx(1.0, abs=1e-6)


def test_pmpo_loss_handles_empty_negative_set_and_accepts_2d_inputs():
    lp = -torch.rand(3, 4)
    adv = torch.rand(3, 4)  # all non-negative
    loss = pmpo_loss(lp, adv, lp.clone())
    assert torch.isfinite(loss)
    assert loss.shape == ()


def test_pmpo_loss_gradients_push_up_good_actions_and_push_down_bad_ones():
    lp = torch.tensor([-1.0, -2.0], requires_grad=True)
    adv = torch.tensor([1.0, -1.0])
    prior = torch.tensor([-1.0, -2.0], requires_grad=True)
    pmpo_loss(lp, adv, prior, alpha=0.5, beta=0.0).backward()
    # Minimising the loss increases log-prob of D+ (negative gradient) and
    # decreases it for D- (positive gradient).
    assert lp.grad[0].item() < 0
    assert lp.grad[1].item() > 0
    # The prior is always treated as constant.
    assert prior.grad is None
