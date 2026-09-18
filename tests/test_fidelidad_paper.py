"""
Regresiones contra el paper (Hafner, Yan, Lillicrap 2025, arXiv 2509.24527).

Cada test cita la seccion que fija el comportamiento. No cubren todo el modelo:
cubren las cuatro cosas que estaban mal y que pueden volver a romperse en
silencio, que es el criterio para que un test valga la pena aqui.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import (  # noqa: E402
    Dynamics,
    Encoder,
    Modality,
    MultiheadSelfAttention,
    SpaceSelfAttentionModality,
    TokenLayout,
    build_attn_kw,
)

D_MODEL, N_HEADS = 32, 4


def _dyn(space_mode, attn_kw=None):
    torch.manual_seed(0)
    return Dynamics(
        d_model=D_MODEL, d_bottleneck=8, d_spatial=8, n_spatial=4,
        n_register=2, n_agent=1, n_heads=N_HEADS, depth=2, k_max=8,
        time_every=2, space_mode=space_mode, attn_kw=attn_kw,
    ).eval()


def _h_t(space_mode, z):
    dyn = _dyn(space_mode)
    with torch.no_grad():
        _, h = dyn(
            torch.zeros(1, 3, 16),
            torch.zeros(1, 3, dtype=torch.long),
            torch.zeros(1, 3, dtype=torch.long),
            z,
            agent_tokens=torch.zeros(1, 3, 1, D_MODEL),
        )
    return h


# --- Seccion 3.3: los agent tokens ven el mundo, el mundo no los ve a ellos ---

def test_el_agente_ve_la_observacion():
    """
    "the agent tokens attend to themselves and all other modalities" (3.3).

    Con `wm_agent_isolated` h_t era identico bit a bit cambiara lo que cambiara
    la imagen: la politica de las fases 2 y 3 estaba ciega.
    """
    torch.manual_seed(1); z_a = torch.randn(1, 3, 4, 8)
    torch.manual_seed(2); z_b = torch.randn(1, 3, 4, 8)
    delta = (_h_t("wm_agent_read", z_a) - _h_t("wm_agent_read", z_b)).abs().max()
    assert delta > 1e-6, "el agent token no reacciona a la observacion"


def test_el_mundo_no_lee_al_agente():
    """
    "no other modalities can attend back to the agent tokens. This is crucial
    for avoiding causal confusion of the world model" (3.3).
    """
    layout = TokenLayout(
        n_latents=0,
        segments=((Modality.ACTION, 1), (Modality.IMAGE, 3), (Modality.AGENT, 2)),
    )
    ids = layout.modality_ids()
    allow = SpaceSelfAttentionModality(D_MODEL, N_HEADS, ids, 0, "wm_agent_read", 0.0).attn_mask[0, 0]
    es_agente = ids == int(Modality.AGENT)

    assert allow[es_agente][:, ~es_agente].all(), "el agente deberia ver el resto"
    assert not allow[~es_agente][:, es_agente].any(), "el resto NO deberia ver al agente"


# --- Seccion 3.1: cuello de botella continuo con tanh ---

def test_bottleneck_acotado_por_tanh():
    """"a linear projection to a smaller channel dimension followed by a tanh" (3.1)."""
    enc = Encoder(patch_dim=12, d_model=D_MODEL, n_latents=2, n_patches=4,
                  n_heads=N_HEADS, depth=2, d_bottleneck=8, time_every=2)
    z, _ = enc(torch.randn(2, 3, 4, 12))
    assert z.abs().max() <= 1.0 + 1e-6


# --- Seccion 3.4: transformer eficiente ---

@pytest.mark.parametrize("kw", [
    {}, {"qk_norm": True}, {"soft_cap": 50.0}, {"rope": True}, {"n_kv_heads": 2},
    {"qk_norm": True, "soft_cap": 50.0, "rope": True, "n_kv_heads": 2},
])
def test_variantes_de_atencion_conservan_la_forma(kw):
    x = torch.randn(2, 6, D_MODEL)
    assert MultiheadSelfAttention(D_MODEL, N_HEADS, 0.0, **kw)(x).shape == x.shape


def test_sin_gqa_se_conserva_el_qkv_fusionado():
    """Los checkpoints ya entrenados dependen del nombre `qkv`."""
    assert MultiheadSelfAttention(D_MODEL, N_HEADS, 0.0).fused_qkv
    assert "qkv.weight" in dict(MultiheadSelfAttention(D_MODEL, N_HEADS, 0.0).named_parameters())


def test_soft_cap_acota_logits_grandes():
    a = MultiheadSelfAttention(D_MODEL, N_HEADS, 0.0, soft_cap=1.0)
    assert torch.isfinite(a(torch.randn(2, 6, D_MODEL) * 100)).all()


def test_rope_desactiva_el_sinusoidal_aditivo():
    """Con RoPE la posicion va dentro de la atencion; sumarla ademas la duplica."""
    enc = Encoder(patch_dim=12, d_model=D_MODEL, n_latents=2, n_patches=4,
                  n_heads=N_HEADS, depth=2, d_bottleneck=8, time_every=2,
                  attn_kw={"rope": True})
    assert enc.use_rope


def test_build_attn_kw_por_defecto_no_cambia_nada():
    """Config sin las claves nuevas => comportamiento anterior, checkpoints intactos."""
    assert build_attn_kw({}) == {}
