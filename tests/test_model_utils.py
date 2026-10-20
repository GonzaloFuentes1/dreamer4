"""Unit tests for the tensor utilities in src/model.py and the heads in src/agent.py (torch only)."""

import math

import pytest
import torch

from agent import PolicyHead, RewardHead, ValueHead
from model import (
    Modality,
    TokenLayout,
    add_sinusoidal_positions,
    pack_bottleneck_to_spatial,
    sinusoid_table,
    temporal_patchify,
    temporal_unpatchify,
    unpack_spatial_to_bottleneck,
)

# ---------------------------------------------------------------------------
# Patchify / unpatchify
# ---------------------------------------------------------------------------

def test_patchify_shape_and_roundtrip():
    B, T, C, H, W, patch = 2, 3, 3, 16, 8, 4
    video = torch.rand(B, T, C, H, W)
    patches = temporal_patchify(video, patch)
    assert patches.shape == (B, T, (H // patch) * (W // patch), C * patch * patch)
    restored = temporal_unpatchify(patches, H, W, C, patch)
    torch.testing.assert_close(restored, video)


def test_patchify_groups_pixels_by_patch():
    # A 1-channel 4x4 frame with patch=2 -> 4 patches of 4 pixels, in raster order of patches.
    frame = torch.arange(16, dtype=torch.float32).view(1, 1, 1, 4, 4)
    patches = temporal_patchify(frame, 2)[0, 0]
    assert patches.shape == (4, 4)
    assert patches[0].tolist() == [0.0, 1.0, 4.0, 5.0]
    assert patches[1].tolist() == [2.0, 3.0, 6.0, 7.0]
    assert patches[2].tolist() == [8.0, 9.0, 12.0, 13.0]
    assert patches[3].tolist() == [10.0, 11.0, 14.0, 15.0]


def test_patchify_requires_divisible_size():
    with pytest.raises(AssertionError):
        temporal_patchify(torch.rand(1, 1, 3, 10, 10), 4)


# ---------------------------------------------------------------------------
# Bottleneck packing
# ---------------------------------------------------------------------------

def test_pack_unpack_bottleneck_roundtrip():
    B, T, n_spatial, k, D = 2, 3, 4, 2, 5
    z = torch.randn(B, T, n_spatial * k, D)
    packed = pack_bottleneck_to_spatial(z, n_spatial=n_spatial, k=k)
    assert packed.shape == (B, T, n_spatial, k * D)
    torch.testing.assert_close(unpack_spatial_to_bottleneck(packed, k=k), z)


def test_pack_bottleneck_checks_length():
    with pytest.raises(AssertionError):
        pack_bottleneck_to_spatial(torch.randn(1, 1, 7, 3), n_spatial=4, k=2)


# ---------------------------------------------------------------------------
# Positional embeddings and token layout
# ---------------------------------------------------------------------------

def test_sinusoid_table_values_and_shape():
    table = sinusoid_table(5, 8)
    assert table.shape == (5, 8)
    assert table.dtype == torch.float32
    # Position 0: sin(0) = 0 on even dims, cos(0) = 1 on odd dims.
    assert table[0, 0::2].abs().max().item() == 0.0
    torch.testing.assert_close(table[0, 1::2], torch.ones(4))
    assert table.abs().max().item() <= 1.0


def test_add_sinusoidal_positions_preserves_shape_and_dtype():
    tokens = torch.zeros(2, 3, 4, 8, dtype=torch.float32)
    out = add_sinusoidal_positions(tokens, scale_pos_embeds=True)
    assert out.shape == tokens.shape and out.dtype == tokens.dtype
    # Scaled positions are 1/sqrt(D) times the unscaled ones.
    unscaled = add_sinusoidal_positions(tokens, scale_pos_embeds=False)
    torch.testing.assert_close(out * math.sqrt(8), unscaled, rtol=1e-5, atol=1e-6)


def test_token_layout_slices_and_modality_ids():
    layout = TokenLayout(
        n_latents=3,
        segments=((Modality.ACTION, 1), (Modality.SPATIAL, 2), (Modality.AGENT, 1)),
    )
    assert layout.S() == 7
    ids = layout.modality_ids()
    assert ids.shape == (7,)
    assert ids.tolist() == [
        int(Modality.LATENT)] * 3 + [int(Modality.ACTION)] + [int(Modality.SPATIAL)] * 2 + [int(Modality.AGENT)]
    slices = layout.slices()
    assert slices[Modality.LATENT] == slice(0, 3)
    assert slices[Modality.ACTION] == slice(3, 4)
    assert slices[Modality.SPATIAL] == slice(4, 6)
    assert slices[Modality.AGENT] == slice(6, 7)


# ---------------------------------------------------------------------------
# Agent heads (src/agent.py)
# ---------------------------------------------------------------------------

STATE_DIM, ACTION_DIM = 12, 5


def test_policy_head_sample_is_bounded_and_respects_the_action_mask():
    torch.manual_seed(0)
    head = PolicyHead(STATE_DIM, action_dim=ACTION_DIM, hidden_dim=16, mtp_length=3)
    h = torch.randn(6, STATE_DIM)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0]).expand(6, -1)
    raw, action = head.sample(h, act_mask=mask)
    assert raw.shape == action.shape == (6, ACTION_DIM)
    assert action.abs().max().item() <= 1.0
    assert torch.all(action[:, 2:4] == 0)


def test_policy_head_log_prob_and_entropy_ignore_masked_dimensions():
    torch.manual_seed(0)
    head = PolicyHead(STATE_DIM, action_dim=ACTION_DIM, hidden_dim=16, mtp_length=2)
    h = torch.randn(4, STATE_DIM)
    raw, _ = head.sample(h)
    full = head.log_prob(h, raw)
    none_active = head.log_prob(h, raw, act_mask=torch.zeros(4, ACTION_DIM))
    assert full.shape == (4,)
    torch.testing.assert_close(none_active, torch.zeros(4))
    assert head.entropy(h, act_mask=torch.zeros(4, ACTION_DIM)).abs().max().item() == 0.0
    # Entropy is an affine function of log_std: it differs from sum(log_std) by a constant.
    _, log_std = head.forward(h)
    offset = head.entropy(h) - log_std.sum(-1)
    torch.testing.assert_close(offset, offset[0].expand(4), rtol=1e-5, atol=1e-5)


def test_policy_head_bc_loss_is_finite_and_differentiable():
    torch.manual_seed(0)
    L = 3
    head = PolicyHead(STATE_DIM, action_dim=ACTION_DIM, hidden_dim=16, mtp_length=L)
    h = torch.randn(2, 8, STATE_DIM)
    actions = torch.rand(2, 8, ACTION_DIM) * 2 - 1
    loss = head.bc_loss(h, actions, None)
    assert loss.shape == () and torch.isfinite(loss)
    loss.backward()
    assert head.mu_heads[0].weight.grad is not None


def test_reward_head_predict_and_mtp_loss():
    torch.manual_seed(0)
    head = RewardHead(STATE_DIM, hidden_dim=16, mtp_length=3)
    h = torch.randn(5, STATE_DIM)
    assert head.forward(h).shape == (5, 255)
    assert head.predict(h).shape == (5,)
    h_bt = torch.randn(2, 8, STATE_DIM)
    rewards = torch.rand(2, 8)
    loss = head.mtp_loss(h_bt, rewards)
    assert loss.shape == () and torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert head.heads[0].weight.grad is not None
    # `bins` is a non-persistent buffer: it must not appear in checkpoints.
    assert "bins" not in head.state_dict()


def test_value_head_can_fit_a_constant_target():
    torch.manual_seed(0)
    head = ValueHead(STATE_DIM, hidden_dim=32)
    h = torch.randn(16, STATE_DIM)
    target = torch.full((16,), 3.0)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    first = head.loss(h, target).item()
    for _ in range(150):
        opt.zero_grad()
        head.loss(h, target).backward()
        opt.step()
    last = head.loss(h, target).item()
    assert last < first
    assert head.predict(h).mean().item() == pytest.approx(3.0, abs=0.5)
