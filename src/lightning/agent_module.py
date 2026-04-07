# dreamer4/lightning/agent_module.py
# Phase 3 — AgentLightningModule: imagination training via PMPO (eq. 11).
#
# Frozen   : Encoder, Dynamics, TaskEmbedder, RewardHead  (from Phase-2 ckpt)
# Frozen   : PolicyPrior  (frozen copy of Phase-2 PolicyHead, for KL term)
# Trainable: PolicyHead (RL-finetuned), ValueHead (new)
#
# Algorithm per step (Algorithm 1, Phase 3):
#   1. Encode real observations → packed latents z
#   2. Imagination rollout H steps:
#        a. Run ODE (shortcut) → z_{t+1}
#        b. Clean forward through frozen Dynamics → h_{t+1} (agent token)
#        c. PolicyHead samples  a_{t+1} = tanh(Normal(μ,σ))
#   3. Annotate imagined trajectory with r̂ (frozen RewardHead) and v̂ (ValueHead)
#   4. Compute λ-returns  R^λ_t  (eq. 10)
#   5. PMPO actor loss  (eq. 11): balanced D+/D- + KL to behavioral prior
#   6. Value loss: twohot cross-entropy to R^λ targets

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig

from model import (
    Dynamics,
    TaskEmbedder,
    Encoder,
    temporal_patchify,
    pack_bottleneck_to_spatial,
)
from losses import make_tau_schedule
from viz import sample_one_timestep_packed
from agent import PolicyHead, ValueHead, RewardHead
from distributions import pmpo_loss


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_finetune_checkpoint(
    ckpt_path: str,
    device: torch.device,
    ac_cfg,
) -> Tuple[Encoder, Dynamics, TaskEmbedder, PolicyHead, RewardHead, dict, int]:
    """
    Load a Phase-2 (finetune) Lightning checkpoint.

    Returns (encoder, dyn, task_embedder, policy_head, reward_head, tok_args, resolved_pf).
    All modules are placed on `device` and frozen.
    """
    from model import Tokenizer, Decoder

    ckpt     = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp       = ckpt.get("hyper_parameters", {})
    cfg_ft   = hp.get("cfg", {})
    ft_cfg   = cfg_ft.get("finetune", {}) or {}

    # ---- Reconstruct tok_args for the encoder ----
    # Prefer tok_args saved directly by FinetuneLightningModule.on_save_checkpoint
    # (contains the exact values from the tokenizer checkpoint).  Fall back to
    # reconstructing from finetune hyper-params for checkpoints produced before
    # on_save_checkpoint was added.
    saved_tok_args = ckpt.get("tok_args", {})
    tok_args = {
        "H":                  saved_tok_args.get("H",                  ft_cfg.get("H",                  128)),
        "W":                  saved_tok_args.get("W",                  ft_cfg.get("W",                  128)),
        "C":                  saved_tok_args.get("C",                  ft_cfg.get("C",                  3)),
        "patch":              saved_tok_args.get("patch",              ft_cfg.get("patch",              4)),
        "n_latents":          saved_tok_args.get("n_latents",          ft_cfg.get("n_latents",          16)),
        "d_bottleneck":       saved_tok_args.get("d_bottleneck",       ft_cfg.get("d_bottleneck",       32)),
        "d_model":            saved_tok_args.get("d_model",            ft_cfg.get("d_enc_model",        256)),
        "n_heads":            saved_tok_args.get("n_heads",            ft_cfg.get("n_enc_heads",        4)),
        "depth":              saved_tok_args.get("depth",              ft_cfg.get("enc_depth",          8)),
        "mlp_ratio":          saved_tok_args.get("mlp_ratio",          ft_cfg.get("mlp_ratio",          4.0)),
        "time_every":         saved_tok_args.get("time_every",         ft_cfg.get("time_every",         1)),
        "latents_only_time":  saved_tok_args.get("latents_only_time",  ft_cfg.get("latents_only_time",  True)),
        # Discrete tokenizer fields (False/1.0 are safe defaults for continuous checkpoints)
        "discrete":           saved_tok_args.get("discrete",           False),
        "temperature":        saved_tok_args.get("temperature",        1.0),
    }

    pf_cfg     = int(ft_cfg.get("packing_factor", ac_cfg.packing_factor))
    pf         = pf_cfg
    n_latents  = tok_args["n_latents"]
    d_bottleneck = int(tok_args["d_bottleneck"])
    n_patches  = (tok_args["H"] // tok_args["patch"]) ** 2
    d_patch    = tok_args["patch"] ** 2 * tok_args["C"]

    sd = ckpt.get("state_dict", {})
    # Strip _orig_mod. prefix added by torch.compile()
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    # Some historical checkpoints may have inconsistent cfg.packing_factor metadata.
    # Infer from saved Dynamics weight shape when available.
    sp_w = sd.get("dyn.spatial_proj.weight")
    if sp_w is not None and hasattr(sp_w, "shape") and len(sp_w.shape) == 2:
        in_features = int(sp_w.shape[1])
        if d_bottleneck > 0 and in_features % d_bottleneck == 0:
            pf_from_weights = in_features // d_bottleneck
            if pf_from_weights != pf_cfg:
                print(
                    f"[agent_module] packing_factor mismatch (cfg={pf_cfg}, ckpt_weights={pf_from_weights}); "
                    f"using ckpt_weights={pf_from_weights}"
                )
            pf = int(pf_from_weights)

    n_spatial  = n_latents // pf
    d_spatial  = d_bottleneck * pf

    # ---- Encoder ----
    _enc_kwargs = dict(
        patch_dim=d_patch,
        d_model=tok_args["d_model"],
        n_latents=n_latents,
        n_patches=n_patches,
        n_heads=tok_args["n_heads"],
        depth=tok_args["depth"],
        dropout=0.0,
        mlp_ratio=tok_args["mlp_ratio"],
        time_every=tok_args["time_every"],
        latents_only_time=tok_args["latents_only_time"],
        mae_p_min=0.0,
        mae_p_max=0.0,
    )
    if tok_args.get("discrete", False):
        from model import DiscreteEncoder
        enc = DiscreteEncoder(
            **_enc_kwargs,
            n_categories=int(tok_args["d_bottleneck"]),
            temperature=float(tok_args.get("temperature", 1.0)),
        )
    else:
        enc = Encoder(
            **_enc_kwargs,
            d_bottleneck=int(tok_args["d_bottleneck"]),
        )
    enc_sd = {k[len("_encoder."):]: v for k, v in sd.items() if k.startswith("_encoder.")}
    if enc_sd:
        enc.load_state_dict(enc_sd, strict=True)
    else:
        print(
            "[agent_module] WARNING: no '_encoder.*' keys found in finetune checkpoint. "
            "Encoder will be randomly initialised — re-run Phase 2 to generate a fixed checkpoint."
        )

    # ---- Dynamics ----
    dyn = Dynamics(
        d_model=ac_cfg.d_model_dyn,
        d_bottleneck=tok_args["d_bottleneck"],
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=ac_cfg.n_register,
        n_agent=ac_cfg.n_agent,
        n_heads=ac_cfg.n_heads,
        depth=ac_cfg.dyn_depth,
        k_max=ac_cfg.k_max,
        dropout=0.0,
        mlp_ratio=ac_cfg.mlp_ratio,
        time_every=ac_cfg.time_every,
        space_mode=ac_cfg.space_mode,
    )
    dyn_sd = {k[4:]: v for k, v in sd.items() if k.startswith("dyn.")}
    if dyn_sd:
        dyn.load_state_dict(dyn_sd, strict=True)

    # ---- TaskEmbedder ----
    te = TaskEmbedder(
        d_model=ac_cfg.d_model_dyn,
        n_agent=ac_cfg.n_agent,
        use_ids=bool(ac_cfg.use_task_ids),
        n_tasks=int(ac_cfg.n_tasks),
        d_task=512,
    )
    te_sd = {k[len("task_embedder."):]: v for k, v in sd.items() if k.startswith("task_embedder.")}
    if te_sd:
        te.load_state_dict(te_sd, strict=True)

    # ---- PolicyHead (used as prior) ----
    state_dim  = ac_cfg.n_agent * ac_cfg.d_model_dyn
    ph = PolicyHead(
        state_dim=state_dim,
        action_dim=int(ft_cfg.get("action_dim", ac_cfg.action_dim)),
        hidden_dim=int(ft_cfg.get("hidden_dim", ac_cfg.hidden_dim)),
        mtp_length=int(ft_cfg.get("mtp_length", ac_cfg.mtp_length)),
    )
    ph_sd = {k[len("policy_head."):]: v for k, v in sd.items() if k.startswith("policy_head.")}
    if ph_sd:
        ph.load_state_dict(ph_sd, strict=True)

    # ---- RewardHead ----
    rh = RewardHead(
        state_dim=state_dim,
        hidden_dim=int(ft_cfg.get("hidden_dim", ac_cfg.hidden_dim)) // 2,
        mtp_length=int(ft_cfg.get("mtp_length", ac_cfg.mtp_length)),
    )
    rh_sd = {k[len("reward_head."):]: v for k, v in sd.items() if k.startswith("reward_head.")}
    if rh_sd:
        rh.load_state_dict(rh_sd, strict=True)

    # Move all to device
    for m in (enc, dyn, te, ph, rh):
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)

    return enc, dyn, te, ph, rh, tok_args, pf


# ─────────────────────────────────────────────────────────────────────────────
# AgentLightningModule
# ─────────────────────────────────────────────────────────────────────────────

class AgentLightningModule(pl.LightningModule):
    """
    Phase 3: policy + value optimisation via imagination training.

    The dynamics transformer is fully frozen (loaded from Phase-2 checkpoint).
    Only PolicyHead and ValueHead have trainable parameters.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.policy_head: Optional[PolicyHead] = None
        self.value_head:  Optional[ValueHead]  = None

        self._tok_args:      Optional[Dict[str, Any]] = None
        self._patch:         int = 4
        self._H:             int = 128
        self._W:             int = 128
        self._C:             int = 3
        self.n_spatial:      int = 0
        self.d_spatial:      int = 0
        self.packing_factor: int = 2

    # ------------------------------------------------------------------
    # Lightning lifecycle
    # ------------------------------------------------------------------

    def setup(self, stage: str):
        ac = self.cfg.agent

        # ---- Load Phase-2 checkpoint (frozen) -------------------------
        enc, dyn, task_embedder, policy_prior, reward_head, tok_args, resolved_pf = (
            _load_finetune_checkpoint(ac.finetune_ckpt, self.device, ac)
        )

        # Store outside DDP / optimizer scope  (same pattern as DynamicsModule)
        object.__setattr__(self, "_encoder",       enc)
        object.__setattr__(self, "_dyn",           dyn)
        object.__setattr__(self, "_task_embedder", task_embedder)
        object.__setattr__(self, "_policy_prior",  policy_prior)  # frozen prior
        object.__setattr__(self, "_reward_head",   reward_head)

        self._tok_args       = tok_args
        self._patch          = int(tok_args.get("patch",        4))
        self._H              = int(tok_args.get("H",          128))
        self._W              = int(tok_args.get("W",          128))
        self._C              = int(tok_args.get("C",            3))
        n_latents            = int(tok_args.get("n_latents",   16))
        d_bottleneck         = int(tok_args.get("d_bottleneck", 32))
        pf                   = int(resolved_pf)

        assert n_latents % pf == 0
        self.n_spatial      = n_latents // pf
        self.d_spatial      = d_bottleneck * pf
        self.packing_factor = pf

        if self.policy_head is not None:
            return  # already initialised

        state_dim = ac.n_agent * ac.d_model_dyn

        # ---- PolicyHead: initialise from prior, then RL-finetune ------
        self.policy_head = copy.deepcopy(policy_prior)
        for p in self.policy_head.parameters():
            p.requires_grad_(True)

        # Phase 3 uses only offset=0. Freeze extra MTP heads so DDP does not
        # flag them as unused parameters when running with strategy='ddp'.
        for n in range(1, int(self.policy_head.mtp_length)):
            for p in self.policy_head.mu_heads[n].parameters():
                p.requires_grad_(False)
            for p in self.policy_head.log_std_heads[n].parameters():
                p.requires_grad_(False)

        # ---- ValueHead: new ----------------------------------------
        self.value_head = ValueHead(
            state_dim=state_dim,
            hidden_dim=int(ac.hidden_dim),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_frames(self, obs_u8: torch.Tensor) -> torch.Tensor:
        """(B,T,3,H,W) uint8 → (B,T,n_spatial,d_spatial)."""
        frames    = obs_u8.float() / 255.0
        patches   = temporal_patchify(frames, self._patch)
        z_btLd    = self._encoder(patches)[0]   # [0] works for both Encoder and DiscreteEncoder
        return pack_bottleneck_to_spatial(z_btLd, n_spatial=self.n_spatial, k=self.packing_factor)

    @torch.no_grad()
    def _get_h_t(
        self,
        z_packed: torch.Tensor,    # (B, T, n_spatial, d_spatial)
        acts:     torch.Tensor,    # (B, T, 16)
        masks:    torch.Tensor,    # (B, T, 16)
        task_toks: torch.Tensor,   # (B, T, n_agent, d_model)
    ) -> torch.Tensor:
        """
        Run a single clean (τ=1) forward through frozen Dynamics.
        Returns h_t (B, T, n_agent * d_model) — agent-token output, flattened.
        """
        B, T = z_packed.shape[:2]
        emax       = int(round(math.log2(self.cfg.agent.k_max)))
        step_clean = torch.full((B, T), emax,                      device=z_packed.device, dtype=torch.long)
        sig_clean  = torch.full((B, T), self.cfg.agent.k_max - 1,  device=z_packed.device, dtype=torch.long)

        _, h_t = self._dyn(
            acts, step_clean, sig_clean, z_packed,
            act_mask=masks,
            agent_tokens=task_toks,
        )
        return h_t.flatten(2)   # (B, T, n_agent * d_model)

    def _task_tokens(self, batch: Dict, B: int, T: int) -> torch.Tensor:
        ac = self.cfg.agent
        if ac.use_task_ids:
            inp = batch["emb_id"].long()[:B]
        else:
            inp = batch["lang_emb"].float()[:B]
        return self._task_embedder(inp, B=B, T=T)

    # ------------------------------------------------------------------
    # Imagination rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _imagine_rollout(
        self,
        z_ctx:     torch.Tensor,   # (B, T_ctx, n_spatial, d_spatial)
        acts_ctx:  torch.Tensor,   # (B, T_ctx, 16)
        mask_ctx:  torch.Tensor,   # (B, T_ctx, 16)
        task_toks: torch.Tensor,   # (B, T_ctx, n_agent, d_model)
        horizon:   int,
        sched:     dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unroll H imagination steps from the last real context frame.

        Returns
        -------
        h_states    : (B, H, state_dim)  agent-token states at each imagined step
        h_final     : (B, state_dim)     state after last imagined step (bootstrap)
        actions_raw : (B, H, A)          pre-tanh Normal samples (for log_prob)
        task_mask   : (B, A)             per-task action mask (constant per episode)
        """
        ac          = self.cfg.agent
        B, T_ctx    = z_ctx.shape[:2]
        max_ctx     = int(ac.max_ctx_len)
        task_mask   = mask_ctx[:, -1]       # (B, A) constant across time — t=0 is always zero due to action shift

        z_buf    = z_ctx.clone()
        act_buf  = acts_ctx.clone()
        mask_buf = mask_ctx.clone()
        task_buf = task_toks.clone()

        h_states_list    = []
        actions_raw_list = []

        for _ in range(horizon):
            # 1. Get h_t from clean dynamics forward at CURRENT context
            h_flat = self._get_h_t(z_buf, act_buf, mask_buf, task_buf)[:, -1]   # (B, state_dim)

            # 2. Sample action from policy (no grads — actions stored for recompute)
            a_raw, a_tanh = self.policy_head.sample(h_flat, act_mask=task_mask)

            h_states_list.append(h_flat)
            actions_raw_list.append(a_raw)

            # 3. Extend action buffers with new action
            T_buf        = z_buf.shape[1]
            act_ext  = torch.cat([act_buf,  a_tanh[:, None, :]],       dim=1)
            mask_ext = torch.cat([mask_buf, task_mask[:, None, :]],    dim=1)

            # Extend task tokens (replicate last token)
            task_ext = torch.cat([task_buf, task_buf[:, -1:]], dim=1)

            # 4. ODE: generate z_{t+1}
            z_next = sample_one_timestep_packed(
                self._dyn,
                past_packed=z_buf,
                k_max=int(ac.k_max),
                sched=sched,
                actions=act_ext,
                act_mask=mask_ext,
            )  # (B, n_spatial, d_spatial)

            # 5. Append and slide window if needed
            z_new    = torch.cat([z_buf, z_next[:, None]], dim=1)
            if z_new.shape[1] > max_ctx:
                z_new    = z_new[:,    -max_ctx:]
                act_ext  = act_ext[:,  -max_ctx:]
                mask_ext = mask_ext[:, -max_ctx:]
                task_ext = task_ext[:, -max_ctx:]

            z_buf    = z_new
            act_buf  = act_ext
            mask_buf = mask_ext
            task_buf = task_ext

        # Get final state for value bootstrap
        h_final = self._get_h_t(z_buf, act_buf, mask_buf, task_buf)[:, -1]

        h_states    = torch.stack(h_states_list,    dim=1)   # (B, H, state_dim)
        actions_raw = torch.stack(actions_raw_list, dim=1)   # (B, H, A)
        return h_states, h_final, actions_raw, task_mask

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        ac = self.cfg.agent

        obs_u8   = batch["obs"]
        act      = batch["act"]
        act_mask = batch["act_mask"]

        act = (act * act_mask).clamp(-1.0, 1.0)

        z_packed = self._encode_frames(obs_u8[:, :-1])   # (B, T, n_s, d_s)
        B, T     = z_packed.shape[:2]

        act_shifted  = torch.zeros_like(act);       act_shifted[:, 1:]  = act[:, :-1]
        mask_shifted = torch.zeros_like(act_mask);  mask_shifted[:, 1:] = act_mask[:, :-1]

        T_ctx     = min(int(ac.ctx_length), T)
        B_img     = min(B, int(ac.imagination_batch_size))
        horizon   = int(ac.imagination_horizon)

        sched = make_tau_schedule(
            k_max=int(ac.k_max),
            schedule=str(ac.eval_schedule),
            d=float(ac.eval_d),
        )

        task_toks = self._task_tokens(batch, B_img, T_ctx)  # (B_img, T_ctx, n_agent, d)

        # ── Imagination rollout (no grad) ─────────────────────────────────
        h_states, h_final, acts_raw_stored, task_mask = self._imagine_rollout(
            z_packed[:B_img, :T_ctx],
            act_shifted[:B_img, :T_ctx],
            mask_shifted[:B_img, :T_ctx],
            task_toks,
            horizon,
            sched,
        )
        # h_states  : (B_img, H, state_dim) — all stop-grad
        # acts_raw_stored: (B_img, H, A)

        h_states   = h_states.detach()
        h_final    = h_final.detach()
        task_mask  = task_mask.detach()
        acts_raw_d = acts_raw_stored.detach()

        BH        = B_img * horizon
        h_bh      = h_states.flatten(0, 1)                            # (B*H, state_dim)
        mask_bh   = task_mask[:, None, :].expand(-1, horizon, -1).flatten(0, 1)  # (B*H, A)

        # ── Reward annotation (frozen RewardHead) ────────────────────────
        rew_imag = self._reward_head.predict(h_bh).view(B_img, horizon).detach()

        # ── Value predictions (trainable ValueHead) ──────────────────────
        v_logits  = self.value_head.forward(h_bh).view(B_img, horizon, -1)
        values    = self.value_head.predict(h_bh).view(B_img, horizon)   # natural scale
        v_boot    = self.value_head.predict(h_final).detach()            # (B_img,)

        # ── λ-returns (eq. 10)  R^λ_t = r_t + γ[(1-λ)v_{t+1} + λ R^λ_{t+1}] ─
        gamma   = float(ac.gamma)
        lambda_ = float(ac.lambda_)

        R_lambda              = torch.zeros_like(rew_imag)
        R_lambda[:, -1]       = rew_imag[:, -1] + gamma * v_boot
        for t in reversed(range(horizon - 1)):
            R_lambda[:, t] = (
                rew_imag[:, t]
                + gamma * ((1.0 - lambda_) * values[:, t + 1].detach()
                           + lambda_        * R_lambda[:, t + 1])
            )

        # ── Policy: recompute log-probs WITH gradients ────────────────────
        log_probs = self.policy_head.log_prob(
            h_bh, acts_raw_d.flatten(0, 1), mask_bh
        ).view(B_img, horizon)

        # Behavioral prior log-probs (stop-grad applied inside pmpo_loss)
        with torch.no_grad():
            lp_prior = self._policy_prior.log_prob(
                h_bh, acts_raw_d.flatten(0, 1), mask_bh
            ).view(B_img, horizon)

        # ── PMPO actor loss (eq. 11) ──────────────────────────────────────
        advantages  = (R_lambda - values.detach())
        actor_loss  = pmpo_loss(
            log_probs, advantages, lp_prior,
            alpha=float(ac.alpha),
            beta=float(ac.beta),
        )

        # ── Value loss: twohot cross-entropy to R^λ ───────────────────────
        critic_loss = self.value_head.loss(h_bh, R_lambda.detach().flatten())

        loss = actor_loss + float(ac.critic_coef) * critic_loss

        self.log_dict({
            "loss/total":          loss,
            "loss/actor":          actor_loss,
            "loss/critic":         critic_loss,
            "stats/value_mean":    values.mean(),
            "stats/R_lambda_mean": R_lambda.mean(),
            "stats/rew_imag_mean": rew_imag.mean(),
            "stats/adv_std":       advantages.std(),
        }, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)

        return loss

    # ------------------------------------------------------------------
    # Optimizer — only PolicyHead + ValueHead
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        ac     = self.cfg.agent
        params = [
            p
            for p in (
                list(self.policy_head.parameters())
                + list(self.value_head.parameters())
            )
            if p.requires_grad
        ]
        return torch.optim.AdamW(
            params,
            lr=float(ac.lr),
            weight_decay=float(ac.weight_decay),
            betas=(0.9, 0.999),
        )
