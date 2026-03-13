#!/usr/bin/env python
# collect_phase0_data.py — Phase 0: episode collection from DMControl.
#
# Generates the raw data needed to start the dreamer4 training pipeline.
# Can be run in two modes:
#
#   Random policy (bootstrap — no prior model):
#     python collect_phase0_data.py \
#       collect.out_data_dir=./data/cycle0/demos \
#       collect.out_frames_dir=./data/cycle0/frames
#
#   Trained agent policy (subsequent cycles):
#     python collect_phase0_data.py \
#       collect.policy=agent \
#       collect.agent_ckpt=./logs/agent_ckpts/last.ckpt \
#       collect.out_data_dir=./data/cycle1/demos \
#       collect.out_frames_dir=./data/cycle1/frames
#
#   Collect only specific tasks:
#     python collect_phase0_data.py \
#       "collect.tasks=[walker-walk,cheetah-run]"
#
# Recursive offline training cycle:
#   Cycle 0:  collect (random) → train Phase1a → Phase1b → Phase2 → Phase3
#   Cycle N:  collect (agent from cycleN-1) → fine-tune all phases with new data
#             (union of all cycles' data_dirs in WMDataModule config)

import sys
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from task_set import TASK_SET


# ─────────────────────────────────────────────────────────────────────────────
# Parallel worker  (module-level so it's picklable with spawn)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_worker(args: tuple) -> tuple:
    """Subprocess worker: loads its own policy copy and collects one task."""
    (task, task_idx, cc_dict, policy_mode, ckpt_path,
     context_len, packing_factor, action_noise,
     device_id, verbose, src_dir) = args

    # Re-establish src path in the freshly spawned process
    sys.path.insert(0, src_dir)

    import torch
    from omegaconf import OmegaConf
    from collector import collect_task, AgentPolicy

    cc = OmegaConf.create(cc_dict)

    if policy_mode == "random":
        policy = "random"
    else:
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        policy = AgentPolicy(
            ckpt_path=ckpt_path,
            task_name=task,
            context_len=context_len,
            packing_factor=packing_factor,
            action_noise=action_noise,
            device=device,
            verbose=verbose,
        )
        policy.set_task(task_idx=task_idx)

    ok = collect_task(task=task, cfg=cc, policy=policy, task_idx=task_idx)
    return task, ok


@hydra.main(config_path="configs", config_name="collect_phase0", version_base=None)
def main(cfg: DictConfig) -> None:
    import random
    import numpy as np
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    cc = cfg.collect

    # ── Resolve task list ──────────────────────────────────────────────────
    if cc.tasks is None:
        tasks = list(TASK_SET)
    else:
        tasks = list(OmegaConf.to_container(cc.tasks, resolve=True))

    print(f"[Phase 0] mode={cc.policy}  tasks={len(tasks)}  "
          f"eps_per_task={cc.n_episodes_per_task}  ep_len={cc.episode_len}")
    print(f"[Phase 0] out_data_dir  → {cc.out_data_dir}")
    print(f"[Phase 0] out_frames_dir→ {cc.out_frames_dir}")

    # ── Vectorized collection (batched GPU inference, policy=agent only) ────
    if cc.get("vectorized", False):
        if cc.policy != "agent":
            raise ValueError("collect.vectorized=true requires collect.policy=agent")
        if not cc.agent_ckpt or cc.agent_ckpt == "???":
            raise ValueError(
                "collect.agent_ckpt must be set when collect.policy=agent. "
                "Example: collect.agent_ckpt=./logs/agent_ckpts/last.ckpt"
            )
        from vec_collector import VectorizedAgentPolicy, collect_vectorized
        n_envs = int(cc.get("n_envs_per_batch", 4))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Phase 0] Vectorized mode: n_envs={n_envs}  device={device}")
        vpolicy = VectorizedAgentPolicy(
            ckpt_path=str(cc.agent_ckpt),
            n_envs=n_envs,
            context_len=int(cc.context_len),
            packing_factor=int(cc.packing_factor),
            action_noise=float(cc.action_noise),
            device=device,
            verbose=bool(cc.verbose),
        )
        task_idxs = list(range(len(tasks)))
        results   = collect_vectorized(tasks, task_idxs, cc, vpolicy)
        n_ok   = sum(1 for v in results.values() if v)
        n_skip = len(tasks) - n_ok
        print(f"\n[Phase 0] Done.  collected={n_ok}  skipped={n_skip}")
        return

    # ── Build policy (sequential / multi-process modes) ──────────────────
    if cc.policy == "random":
        policy = "random"
        policy_obj = None
    elif cc.policy == "agent":
        if not cc.agent_ckpt or cc.agent_ckpt == "???":
            raise ValueError(
                "collect.agent_ckpt must be set when collect.policy=agent. "
                "Example: collect.agent_ckpt=./logs/agent_ckpts/last.ckpt"
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Phase 0] Loading agent policy from {cc.agent_ckpt} on {device}")

        from collector import AgentPolicy
        # Build one shared policy; set_task() / reset() are called per episode
        # We pass task_idx=0 here; collect_task() will override it
        policy_obj = AgentPolicy(
            ckpt_path=str(cc.agent_ckpt),
            task_name=tasks[0],
            context_len=int(cc.context_len),
            packing_factor=int(cc.packing_factor),
            action_noise=float(cc.action_noise),
            device=device,
            verbose=bool(cc.verbose),
        )
        policy = policy_obj
    else:
        raise ValueError(f"Unknown collect.policy={cc.policy!r}. Expected 'random' or 'agent'.")

    # ── Collect ─────────────────────────────────────────────────────────────
    from collector import collect_task

    n_ok = 0
    n_skip = 0
    n_workers = min(int(cc.get("num_workers", 1)), len(tasks))

    if n_workers <= 1:
        # ── Sequential (default for 1 worker or solo GPU) ──────────────────
        for task_idx, task in enumerate(tasks):
            print(f"\n[Phase 0] ── {task}  ({task_idx+1}/{len(tasks)}) ──")
            if policy != "random":
                policy_obj.set_task(task_idx=task_idx)
            ok = collect_task(task=task, cfg=cc, policy=policy, task_idx=task_idx)
            if ok:
                n_ok += 1
            else:
                n_skip += 1
    else:
        # ── Parallel workers (spawn — safe for MuJoCo + CUDA) ──────────────
        print(f"[Phase 0] Parallel mode: {n_workers} workers for {len(tasks)} tasks")

        # Resolve paths to absolute so spawned workers (different cwd) find them
        src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
        cc_dict = dict(OmegaConf.to_container(cc, resolve=True))
        cc_dict["out_data_dir"]  = os.path.abspath(str(cc.out_data_dir))
        cc_dict["out_frames_dir"] = os.path.abspath(str(cc.out_frames_dir))
        cc_dict["tasks_json"]    = os.path.abspath(str(cc.tasks_json))
        ckpt_path_abs = os.path.abspath(str(cc.agent_ckpt)) if cc.policy == "agent" else None

        n_gpus = torch.cuda.device_count()
        worker_args = [
            (
                task, task_idx, cc_dict,
                cc.policy, ckpt_path_abs,
                int(cc.context_len), int(cc.packing_factor), float(cc.action_noise),
                task_idx % max(n_gpus, 1),  # round-robin GPU assignment
                bool(cc.verbose), src_dir,
            )
            for task_idx, task in enumerate(tasks)
        ]

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futs = {executor.submit(_collect_worker, a): a[0] for a in worker_args}
            for fut in as_completed(futs):
                task_name, ok = fut.result()
                print(f"[Phase 0] ✓ {task_name} done  ok={ok}")
                if ok:
                    n_ok += 1
                else:
                    n_skip += 1

    print(f"\n[Phase 0] Done.  collected={n_ok}  skipped={n_skip}")
    print(f"[Phase 0] tasks.json  → {cc.tasks_json}")
    print(f"\nNext step — start Phase 1a:")
    print(f"  python train_phase1a_tokenizer.py \\")
    print(f"    data.frames_dirs=[{cc.out_frames_dir}]")
    print(f"\nAnd Phase 1b / 2 / 3:")
    print(f"  python train_phase1b_dynamics.py \\")
    print(f"    data.data_dirs=[{cc.out_data_dir}] \\")
    print(f"    data.frames_dirs=[{cc.out_frames_dir}] \\")
    print(f"    dynamics.tokenizer_ckpt=./logs/tokenizer_ckpts/last.ckpt")


if __name__ == "__main__":
    main()
