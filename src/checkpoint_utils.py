# dreamer4/checkpoint_utils.py
# Shared helpers for loading frozen checkpoints across training phases.
from __future__ import annotations

import torch


def load_frozen_tokenizer(ckpt_path: str, device: torch.device):
    """
    Load a Phase-1a tokenizer checkpoint and return frozen (encoder, decoder, tok_args).

    Supports both Lightning checkpoints (keys: state_dict, hyper_parameters) and legacy
    raw-dict checkpoints (keys: model, args).  Handles both continuous Encoder and
    DiscreteEncoder variants.

    Returns:
        encoder  — frozen, on device, requires_grad=False
        decoder  — frozen, on device, requires_grad=False
        tok_args — dict of tokenizer hyperparameters read from the checkpoint
    """
    from model import Encoder, Decoder, Tokenizer

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        # Lightning checkpoint: hyper_parameters holds the config.
        hp = ckpt.get("hyper_parameters", {})
        tc = hp.get("cfg", {}).get("tokenizer", {})
        tok_args = dict(tc) if tc else {}

        full_sd = ckpt["state_dict"]
        model_sd = {k[len("model."):]: v for k, v in full_sd.items() if k.startswith("model.")}
        model_sd = {k.replace("_orig_mod.", ""): v for k, v in model_sd.items()}
    else:
        # Legacy format: plain dict with "model" and "args".
        tok_args = dict(ckpt.get("args", {}))
        model_sd = ckpt["model"]
        model_sd = {k.replace("_orig_mod.", ""): v for k, v in model_sd.items()}

    H         = int(tok_args.get("H", 128))
    W         = int(tok_args.get("W", 128))
    C         = int(tok_args.get("C", 3))
    patch     = int(tok_args.get("patch", 4))
    n_patches = (H // patch) * (W // patch)
    d_patch   = patch * patch * C

    _enc_kwargs = dict(
        patch_dim=d_patch,
        d_model=int(tok_args.get("d_model", 256)),
        n_latents=int(tok_args.get("n_latents", 16)),
        n_patches=n_patches,
        n_heads=int(tok_args.get("n_heads", 4)),
        depth=int(tok_args.get("depth", 8)),
        dropout=0.0,
        mlp_ratio=float(tok_args.get("mlp_ratio", 4.0)),
        time_every=int(tok_args.get("time_every", 1)),
        latents_only_time=bool(tok_args.get("latents_only_time", True)),
        mae_p_min=0.0,
        mae_p_max=0.0,
        scale_pos_embeds=bool(tok_args.get("scale_pos_embeds", True)),
    )

    if tok_args.get("discrete", False):
        from model import DiscreteEncoder
        enc = DiscreteEncoder(
            **_enc_kwargs,
            n_categories=int(tok_args.get("d_bottleneck", 32)),
            temperature=float(tok_args.get("temperature", 1.0)),
        )
    else:
        enc = Encoder(
            **_enc_kwargs,
            d_bottleneck=int(tok_args.get("d_bottleneck", 32)),
        )

    dec = Decoder(
        d_bottleneck=int(tok_args.get("d_bottleneck", 32)),
        d_model=int(tok_args.get("d_model", 256)),
        n_heads=int(tok_args.get("n_heads", 4)),
        depth=int(tok_args.get("depth", 8)),
        n_latents=int(tok_args.get("n_latents", 16)),
        n_patches=n_patches,
        d_patch=d_patch,
        dropout=0.0,
        mlp_ratio=float(tok_args.get("mlp_ratio", 4.0)),
        time_every=int(tok_args.get("time_every", 1)),
        latents_only_time=bool(tok_args.get("latents_only_time", True)),
        scale_pos_embeds=bool(tok_args.get("scale_pos_embeds", True)),
    )

    tok = Tokenizer(enc, dec)
    tok.load_state_dict(model_sd, strict=True)
    tok = tok.to(device).eval()
    for p in tok.parameters():
        p.requires_grad_(False)

    return tok.encoder, tok.decoder, tok_args
