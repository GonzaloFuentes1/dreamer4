import json
import math
import os
import torch
import torch.nn as nn
from torchrl.envs.utils import RandomPolicy


def make_random_policy(action_spec):
    return RandomPolicy(action_spec=action_spec)


class DreamerV4AgentPolicy(nn.Module):
    def __init__(self, action_spec, num_envs=16, context_len=16,
                 dev="cuda", ckpt_path=None, task_name=None):
        super().__init__()
        self.action_spec  = action_spec
        self.num_envs     = num_envs
        self.context_len  = context_len
        self.d_action     = 16   # DreamerV4 internal action dim (always 16)
        self.target_dev   = torch.device(dev if torch.cuda.is_available() else "cpu")
        self.ckpt_path    = ckpt_path

        if hasattr(action_spec, "space") and hasattr(action_spec.space, "n"):
            self.is_discrete  = True
            self.env_d_action = action_spec.space.n
        else:
            self.is_discrete  = False
            self.env_d_action = action_spec.shape[-1]

        # Task language embedding (512-d) loaded from tasks.json
        self.task_emb = torch.zeros(512)
        tasks_file = os.path.join(os.path.dirname(__file__), "..", "..", "tasks.json")
        if task_name and os.path.exists(tasks_file):
            try:
                with open(tasks_file) as f:
                    meta = json.load(f)
                if task_name in meta and "text_embedding" in meta[task_name]:
                    self.task_emb = torch.tensor(meta[task_name]["text_embedding"])
                    print(f"[Exploration] lang_emb cargado para {task_name}")
            except Exception as e:
                print(f"[Exploration] Aviso: no se pudo cargar lang_emb ({e})")

        # Inference buffers
        self.z_buf    = None
        self.act_buf  = None
        self.task_buf = None

        # Architecture — loaded from checkpoint if available
        self.has_real_arch = False
        self._patch        = 4    # updated by _load_from_checkpoint
        self._n_spatial    = 8    # updated by _load_from_checkpoint
        self._pf           = 2    # packing factor, updated by _load_from_checkpoint
        self._k_max        = 8    # updated by _load_from_checkpoint

        if self.ckpt_path:
            try:
                self._load_from_checkpoint(self.ckpt_path)
            except Exception as e:
                print(f"[Exploration] No se pudo cargar el checkpoint, usando acciones aleatorias: {e}")

        self.prev_act = torch.zeros(self.num_envs, 1, self.d_action, device=self.target_dev)

    # ------------------------------------------------------------------
    # Checkpoint loading — mirrors agent_module._load_finetune_checkpoint
    # ------------------------------------------------------------------

    def _load_from_checkpoint(self, ckpt_path: str):
        """
        Load full inference stack from a Phase-3 agent OR Phase-2 finetune checkpoint.

        Phase-3 checkpoints only store policy_head + value_head weights; the
        frozen encoder/dynamics weights live in the Phase-2 finetune checkpoint
        referenced by hyper_parameters.cfg.agent.finetune_ckpt.

        Phase-2 checkpoints store everything directly.
        """
        from model import Encoder, Dynamics, TaskEmbedder, pack_bottleneck_to_spatial
        from agent import PolicyHead

        dev = self.target_dev
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp   = ckpt.get("hyper_parameters", {})
        cfg  = hp.get("cfg", {})
        sd   = ckpt.get("state_dict", {})

        # Decide which checkpoint holds the frozen components
        if any(k.startswith("_encoder.") for k in sd):
            # Phase-2 finetune checkpoint — has everything
            ft_ckpt_path = ckpt_path
            ft_ckpt      = ckpt
            ft_sd        = sd
            ft_cfg       = cfg.get("finetune", cfg.get("dynamics", {}))
        else:
            # Phase-3 agent checkpoint — frozen components are in finetune_ckpt
            ft_ckpt_path = (cfg.get("agent", {}) or {}).get("finetune_ckpt", None)
            if not ft_ckpt_path or not os.path.exists(ft_ckpt_path):
                raise FileNotFoundError(
                    f"Phase-3 checkpoint does not contain encoder weights and "
                    f"finetune_ckpt not found: {ft_ckpt_path}"
                )
            ft_ckpt = torch.load(ft_ckpt_path, map_location="cpu", weights_only=False)
            ft_sd   = ft_ckpt.get("state_dict", {})
            ft_cfg  = ft_ckpt.get("hyper_parameters", {}).get("cfg", {}).get("finetune", {})

        # ---- Read tok_args saved by FinetuneLightningModule ----
        saved_tok = ft_ckpt.get("tok_args", {})
        H            = int(saved_tok.get("H",           ft_cfg.get("H",            128)))
        patch        = int(saved_tok.get("patch",       ft_cfg.get("patch",        4)))
        n_latents    = int(saved_tok.get("n_latents",   ft_cfg.get("n_latents",    16)))
        d_bottleneck = int(saved_tok.get("d_bottleneck",ft_cfg.get("d_bottleneck", 32)))
        d_model_tok  = int(saved_tok.get("d_model",     ft_cfg.get("d_enc_model",  256)))
        n_heads_tok  = int(saved_tok.get("n_heads",     ft_cfg.get("n_enc_heads",  4)))
        depth_tok    = int(saved_tok.get("depth",       ft_cfg.get("enc_depth",    8)))
        mlp_ratio    = float(saved_tok.get("mlp_ratio", ft_cfg.get("mlp_ratio",   4.0)))
        time_every   = int(saved_tok.get("time_every",  ft_cfg.get("time_every",   1)))
        n_patches    = (H // patch) ** 2
        patch_dim    = patch * patch * 3

        # Infer packing factor from Dynamics spatial_proj weight shape
        d_model_dyn  = int(ft_cfg.get("d_model_dyn", 512))
        pf = 2  # default
        sp_w = ft_sd.get("dyn.spatial_proj.weight")
        if sp_w is not None and d_bottleneck > 0:
            in_features = int(sp_w.shape[1])
            if in_features % d_bottleneck == 0:
                pf = in_features // d_bottleneck

        n_spatial = n_latents // pf
        d_spatial = d_bottleneck * pf

        self._patch     = patch
        self._n_spatial = n_spatial
        self._pf        = pf
        self._k_max     = int(ft_cfg.get("k_max", 8))

        # ---- Encoder ----
        self.encoder = Encoder(
            patch_dim=patch_dim,
            d_model=d_model_tok,
            n_latents=n_latents,
            n_patches=n_patches,
            n_heads=n_heads_tok,
            depth=depth_tok,
            d_bottleneck=d_bottleneck,
            dropout=0.0,
            mlp_ratio=mlp_ratio,
            time_every=time_every,
            latents_only_time=bool(saved_tok.get("latents_only_time", True)),
            mae_p_min=0.0,
            mae_p_max=0.0,
        )
        enc_sd = {k[len("_encoder."):]: v for k, v in ft_sd.items() if k.startswith("_encoder.")}
        if enc_sd:
            self.encoder.load_state_dict(enc_sd, strict=True)

        # ---- Dynamics ----
        self.dyn = Dynamics(
            d_model=d_model_dyn,
            d_bottleneck=d_bottleneck,
            d_spatial=d_spatial,
            n_spatial=n_spatial,
            n_register=int(ft_cfg.get("n_register", 4)),
            n_agent=int(ft_cfg.get("n_agent", 1)),
            n_heads=int(ft_cfg.get("n_heads", 4)),
            depth=int(ft_cfg.get("dyn_depth", 8)),
            k_max=self._k_max,
            dropout=0.0,
            mlp_ratio=float(ft_cfg.get("mlp_ratio", 4.0)),
            time_every=int(ft_cfg.get("time_every", 4)),
            space_mode=str(ft_cfg.get("space_mode", "wm_agent_isolated")),
            action_dim=self.d_action,
        )
        dyn_sd = {k[4:]: v for k, v in ft_sd.items() if k.startswith("dyn.")}
        if dyn_sd:
            self.dyn.load_state_dict(dyn_sd, strict=True)

        # ---- TaskEmbedder ----
        n_agent = int(ft_cfg.get("n_agent", 1))
        self.task_embedder = TaskEmbedder(
            d_model=d_model_dyn,
            n_agent=n_agent,
            use_ids=bool(ft_cfg.get("use_task_ids", False)),
            n_tasks=int(ft_cfg.get("n_tasks", 128)),
            d_task=512,
        )
        te_sd = {k[len("task_embedder."):]: v for k, v in ft_sd.items() if k.startswith("task_embedder.")}
        if te_sd:
            self.task_embedder.load_state_dict(te_sd, strict=True)

        # ---- PolicyHead — prefer Phase-3 weights, fall back to Phase-2 ----
        state_dim = n_agent * d_model_dyn
        self.policy_head = PolicyHead(
            state_dim=state_dim,
            action_dim=self.d_action,
            hidden_dim=int(ft_cfg.get("hidden_dim", 512)),
            mtp_length=int(ft_cfg.get("mtp_length", 8)),
        )
        ph_sd = {k[len("policy_head."):]: v for k, v in sd.items() if k.startswith("policy_head.")}
        if not ph_sd:
            ph_sd = {k[len("policy_head."):]: v for k, v in ft_sd.items() if k.startswith("policy_head.")}
        if ph_sd:
            self.policy_head.load_state_dict(ph_sd, strict=True)

        # Move all to device and freeze
        for m in (self.encoder, self.dyn, self.task_embedder, self.policy_head):
            m.to(dev).eval()
            for p in m.parameters():
                p.requires_grad_(False)

        self.has_real_arch = True
        print(f"[Exploration] Arquitectura cargada — H={H} patch={patch} "
              f"n_spatial={n_spatial} pf={pf} d_model_dyn={d_model_dyn} "
              f"desde {os.path.basename(ckpt_path)}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, tensordict):
        from model import temporal_patchify, pack_bottleneck_to_spatial

        B = tensordict.batch_size[0]

        if "pixels" in tensordict.keys():
            frames = tensordict["pixels"].to(self.target_dev)
        else:
            # Fallback zero frame — size derived from loaded encoder
            H = self._patch * int(math.sqrt(
                self.encoder.n_patches)) if self.has_real_arch else 64
            frames = torch.zeros((B, 3, H, H), device=self.target_dev, dtype=torch.uint8)

        frames_f  = frames.float().div(255.0) if frames.dtype == torch.uint8 else frames.float()
        frames_in = frames_f.unsqueeze(1)   # (B, 1, C, H, W)

        if self.has_real_arch:
            with torch.no_grad():
                patches = temporal_patchify(frames_in, patch=self._patch)
                z_new, _ = self.encoder(patches)
                z_new = pack_bottleneck_to_spatial(z_new, n_spatial=self._n_spatial, k=self._pf)

                task_inp = self.task_emb.to(self.target_dev).unsqueeze(0).expand(B, -1)
                task_new = self.task_embedder(task_inp, B=B, T=1)

                if self.z_buf is None or self.z_buf.shape[0] != B:
                    self.z_buf    = z_new
                    self.act_buf  = self.prev_act[:B]
                    self.task_buf = task_new
                else:
                    self.z_buf    = torch.cat([self.z_buf,    z_new],    dim=1)
                    self.act_buf  = torch.cat([self.act_buf,  self.prev_act[:B]], dim=1)
                    self.task_buf = torch.cat([self.task_buf, task_new], dim=1)

                if self.z_buf.shape[1] > self.context_len:
                    self.z_buf    = self.z_buf[:,    -self.context_len:]
                    self.act_buf  = self.act_buf[:,  -self.context_len:]
                    self.task_buf = self.task_buf[:, -self.context_len:]

                T_ctx  = self.z_buf.shape[1]
                emax   = int(round(math.log2(self._k_max)))
                step_t = torch.full((B, T_ctx), emax,            device=self.target_dev, dtype=torch.long)
                sig_t  = torch.full((B, T_ctx), self._k_max - 1, device=self.target_dev, dtype=torch.long)

                _, h_t  = self.dyn(self.act_buf, step_t, sig_t, self.z_buf,
                                   act_mask=None, agent_tokens=self.task_buf)
                h_flat  = h_t[:, -1].flatten(1)
                act_mask = torch.ones(B, self.d_action, device=self.target_dev)
                _, action = self.policy_head.sample(h_flat, act_mask=act_mask)
        else:
            action = torch.randn(B, self.d_action, device=self.target_dev)

        self.prev_act[:B] = action.view(B, 1, -1)

        if self.is_discrete:
            sliced_action = action[:, :self.env_d_action].argmax(dim=-1).cpu()
        else:
            sliced_action = action[:, :self.env_d_action].cpu()

        tensordict.set("action", sliced_action)
        return tensordict


def make_agent_policy(ckpt_path: str, action_spec, observation_spec=None,
                      device="cuda", num_envs=16, task_name=None):
    return DreamerV4AgentPolicy(
        action_spec=action_spec,
        num_envs=num_envs,
        dev=device,
        ckpt_path=ckpt_path,
        task_name=task_name,
    )


def get_collect_policy(config, env_action_spec, env_obs_spec=None,
                       device="cuda", task_name=None):
    policy_type = config.get("policy", "random")
    if policy_type == "random":
        return make_random_policy(action_spec=env_action_spec)
    elif policy_type == "agent":
        ckpt = config.get("agent_ckpt", None)
        return make_agent_policy(
            ckpt_path=ckpt,
            action_spec=env_action_spec,
            observation_spec=env_obs_spec,
            device=device,
            num_envs=config.get("num_envs_per_task", 16),
            task_name=task_name,
        )
    else:
        raise ValueError(f"Política seleccionada desconocida: {policy_type}")
