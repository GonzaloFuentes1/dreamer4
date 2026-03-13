# dreamer4 — Arquitectura y flujo de entrenamiento

> Renderiza con VS Code + extensión `bierner.markdown-mermaid`, o pega los bloques en https://mermaid.live

---

## 1. Pipeline de entrenamiento (5 fases, ciclo recursivo)

```mermaid
flowchart TD
    subgraph P0["Phase 0 — Colección de episodios (ciclo N)"]
        COL["collect_phase0_data.py\ncollect_phase0.yaml"]
        ENV["DMControl\n(30 tareas)"]
        RPOL["Política random\n(bootstrap, ciclo 0)"]
        APOL["Política entrenada\n(agent.ckpt, ciclos ≥1)"]
        DEMOS["demos/{task}.pt\n{episode, action, reward}"]
        SHARDS["frames/{task}/*_shard*.pt\n{frames: uint8}"]
        TJSON["tasks.json\n{action_dim, text_embedding}"]
        COL --> ENV
        RPOL --> COL
        APOL -. "ciclos ≥1" .-> COL
        ENV --> DEMOS
        ENV --> SHARDS
        ENV --> TJSON
    end

    subgraph DATA["📦 Datos (entran al pipeline)"]
        FS["ShardedFrameDataset\nsrc/sharded_frame_dataset.py"]
        WMD["WMDataset\nsrc/wm_dataset.py\n{obs, act, rew, lang_emb}"]
        FDM["FrameDataModule\nlightning/frame_datamodule.py"]
        WMDM["WMDataModule\nlightning/wm_datamodule.py"]
        SHARDS --> FS --> FDM
        DEMOS --> WMD
        TJSON --> WMD
        SHARDS --> WMD --> WMDM
    end

    subgraph P1A["Fase 1a — Tokenizer"]
        TT["train_phase1a_tokenizer.py"]
        TLM["TokenizerLightningModule\nlightning/tokenizer_module.py"]
        TOK["Tokenizer\n(Encoder + Decoder)\nsrc/model.py"]
        FDM --> TLM
        TT --> TLM --> TOK
        TLM --> TCKPT["tokenizer.ckpt"]
    end

    subgraph P1B["Fase 1b — Dynamics"]
        TD["train_phase1b_dynamics.py"]
        DLM["DynamicsLightningModule\nlightning/dynamics_module.py"]
        DYN["Dynamics\n(BlockCausalTransformer)\nsrc/model.py"]
        TCKPT -. "frozen Encoder" .-> DLM
        WMDM --> DLM
        TD --> DLM --> DYN
        DLM --> DCKPT["dynamics.ckpt"]
    end

    subgraph P2["Fase 2 — Finetune (BC + Reward)"]
        TF["train_phase2_finetuning.py"]
        FLM["FinetuneLightningModule\nlightning/finetune_module.py"]
        PH2["PolicyHead  MTP L=8\nsrc/agent.py"]
        RH["RewardHead  MTP L=8\nsrc/agent.py"]
        TCKPT -. "frozen Encoder" .-> FLM
        DCKPT -. "init Dynamics\n+ TaskEmbedder" .-> FLM
        WMDM --> FLM
        TF --> FLM
        FLM --> PH2
        FLM --> RH
        FLM --> FCKPT["finetune.ckpt"]
    end

    subgraph P3["Fase 3 — Agent (PMPO)"]
        TA["train_phase3_imagination.py"]
        ALM["AgentLightningModule\nlightning/agent_module.py"]
        PH3["PolicyHead trainable\n+ policy_prior frozen"]
        VH["ValueHead\nsrc/agent.py"]
        FCKPT -. "frozen Enc+Dyn\n+TaskEmb+Prior+Rew" .-> ALM
        WMDM --> ALM
        TA --> ALM
        ALM --> PH3
        ALM --> VH
        ALM --> ACKPT["agent.ckpt"]
    end

    %% Ciclo recursivo: agent.ckpt alimenta la siguiente colección
    ACKPT -. "ciclo N+1\n(política entrenada)" .-> APOL

    style P0   fill:#fff0f0,stroke:#c88
    style DATA fill:#f0f4ff,stroke:#aab
    style P1A  fill:#fff7e6,stroke:#dd8
    style P1B  fill:#fffbe6,stroke:#cc8
    style P2   fill:#e6fff0,stroke:#8c8
    style P3   fill:#fce6ff,stroke:#99a
```

---

## 2. Dependencias entre módulos

```mermaid
flowchart LR
    subgraph SRC["src/  — módulos de núcleo"]
        direction TB
        DIST["distributions.py\nsymlog · symexp\ntwohot · pmpo_loss"]
        AGT["agent.py\nPolicyHead\nRewardHead\nValueHead"]
        MDL["model.py\nEncoder · Decoder · Tokenizer\nDynamics · TaskEmbedder\nBlockCausalTransformer\nActionEncoder"]
        LOSS["losses.py\nLPIPSLoss\ndynamics_pretrain_loss\nmake_tau_schedule"]
        VIZ["viz.py\nsample_one_timestep_packed\nlog_tokenizer_viz_wandb\nrun_dynamics_eval"]
        WMDS["wm_dataset.py\nWMDataset"]
        SHRD["sharded_frame_dataset.py\nShardedFrameDataset"]
        DIST --> AGT
        MDL --> LOSS
        MDL --> VIZ
        LOSS --> VIZ
        SHRD --> WMDS
    end

    subgraph LTN["src/lightning/  — módulos Lightning"]
        direction TB
        TMOD["tokenizer_module.py\nTokenizerLightningModule"]
        DMOD["dynamics_module.py\nDynamicsLightningModule"]
        FMOD["finetune_module.py\nFinetuneLightningModule"]
        AMOD["agent_module.py\nAgentLightningModule"]
        FDMOD["frame_datamodule.py\nFrameDataModule"]
        WMOD["wm_datamodule.py\nWMDataModule"]
        CB["callbacks.py\nTokenizerVizCallback\nDynamicsEvalCallback\nActionShuffleMetricCallback"]
    end

    MDL --> TMOD
    MDL --> DMOD
    MDL --> FMOD
    MDL --> AMOD
    LOSS --> TMOD
    LOSS --> DMOD
    LOSS --> FMOD
    LOSS --> CB
    AGT --> FMOD
    AGT --> AMOD
    DIST --> AMOD
    VIZ --> AMOD
    VIZ --> CB
    WMDS --> WMOD
    SHRD --> FDMOD
    FDMOD --> TMOD
    WMOD --> DMOD
    WMOD --> FMOD
    WMOD --> AMOD
```

---

## 3. Flujo de datos interno — Fase 2 y Fase 3

```mermaid
sequenceDiagram
    participant B as Batch<br/>{obs,act,rew,lang_emb}
    participant ENC as Encoder<br/>(frozen)
    participant DYN as Dynamics<br/>(BlockCausalTransformer)
    participant TE as TaskEmbedder
    participant POL as PolicyHead
    participant REW as RewardHead
    participant VAL as ValueHead
    participant DIST as distributions.py

    Note over B,DIST: ── Fase 2: training_step (Finetune) ──
    B->>ENC: obs (B,T,C,H,W)
    ENC-->>DYN: z_spatial (B,T,n_spat,d)
    B->>TE: lang_emb / task_id
    TE-->>DYN: task_tokens (B,T,n_agent,d)
    DYN-->>POL: h_t (B,T,n_agent,d_model) → h_flat
    DYN-->>REW: h_flat
    POL->>DIST: bc_loss MTP L=8 (eq. 9)
    REW->>DIST: twohot_loss MTP L=8 (eq. 9)
    Note over POL,DIST: loss = wm_loss + bc_loss + rew_loss → finetune.ckpt

    Note over B,DIST: ── Fase 3: _imagine_rollout (Agent) ──
    B->>ENC: obs contexto real
    ENC-->>DYN: z_packed
    loop H pasos imaginados
        DYN-->>POL: h_t → sample(act_raw, act)
        POL-->>DYN: acción → sample_one_timestep_packed ODE
        DYN-->>REW: h_t → predict reward (frozen)
        DYN-->>VAL: h_t → predict value
    end
    VAL->>DIST: λ-returns γ=0.997
    POL->>DIST: pmpo_loss α=0.5 β=0.3 (eq. 11)
    VAL->>DIST: twohot_loss (eq. 10)
    Note over POL,DIST: loss = pmpo + value_loss → agent.ckpt
```
