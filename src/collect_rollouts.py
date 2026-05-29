#!/usr/bin/env python
"""Collect frozen-base rollouts with per-step privileged state + success label.

Feeds the STL pre-gate: we need a mix of successful and failed trajectories on
the mid-band tasks (push-v3, plate-slide-v3) with the per-step MetaWorld state,
so we can test whether an STL robustness signal ρ separates good from bad.

Stock SmolVLAPolicy (fresh flow noise each chunk), RANDOM init (one seed per
episode) — we WANT diversity, not the fixed-init determinism the ILC kill-test
used. Reuses the faithful harness rollout pattern from dead/07-ilc-vla.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HARNESS = Path("/home/user/Desktop/vla_projects/RESEARCH/active/_harness-faithful-smolvla")
sys.path.append(str(HARNESS))
from harness_common import build_env_and_policy  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.envs.utils import preprocess_observation, add_envs_task  # noqa: E402

def collect(task, n_episodes, start_seed, ckpt, device="cuda"):
    vec, policy, pre, post, env_pre, env_post = build_env_and_policy(
        ckpt, task=task, batch_size=1, device=device
    )
    policy.eval()
    max_steps = vec.call("_max_episode_steps")[0]
    episodes = []
    for ep in range(n_episodes):
        seed = start_seed + ep
        np.random.seed(seed)
        torch.manual_seed(seed)
        obs, info = vec.reset(seed=[seed])
        policy.reset()
        target = np.asarray(
            vec.envs[0]._env.unwrapped._target_pos, dtype=np.float64
        )[:3].copy()

        # obs['agent_pos'] is the 4-dim policy proprio [EE_xyz, gripper]; object
        # & goal live only in the underlying MetaWorld env. Read non-mutating
        # getters at the TOP of each step (current, pre-action state) so the
        # Gymnasium auto-reset on termination never contaminates the log.
        # Derived obj_to_target = ||obj-goal|| is verified == MetaWorld's own
        # info['obj_to_target'].
        mw = vec.envs[0]._env.unwrapped
        ee, grip, obj, goal, acts = [], [], [], [], []

        def read_success(info_dict):
            found = False
            cands = [info_dict]
            fi = info_dict.get("final_info") if isinstance(info_dict, dict) else None
            if fi is not None:
                cands += list(fi) if isinstance(fi, (list, tuple, np.ndarray)) else [fi]
            for d in cands:
                if isinstance(d, dict) and d.get("is_success") is not None:
                    found = found or bool(np.asarray(d["is_success"]).any())
            return found

        success = False
        for step in range(max_steps):
            ap = np.asarray(obs["agent_pos"][0], dtype=np.float64)
            ee.append(np.asarray(mw.get_endeff_pos(), dtype=np.float64)[:3].tolist())
            grip.append(float(ap[3]))
            obj.append(np.asarray(mw._get_pos_objects(), dtype=np.float64)[:3].tolist())
            goal.append(np.asarray(mw._target_pos, dtype=np.float64)[:3].tolist())

            p = preprocess_observation(obs)
            p = add_envs_task(vec, p)
            p = env_pre(p)
            p = pre(p)
            with torch.inference_mode():
                action = policy.select_action(p)
            action = post(action)
            at = env_post({ACTION: action})
            action_np = at[ACTION].to("cpu").numpy()[0].copy()
            acts.append(action_np.tolist())  # action taken at the just-logged state
            obs, reward, terminated, truncated, info = vec.step(np.array([action_np]))
            success = success or read_success(info)
            if terminated[0] or truncated[0]:
                break
        episodes.append({
            "ep": ep, "seed": seed, "success": success, "len": len(ee),
            "target": target.tolist(),
            "ee": ee, "grip": grip, "obj": obj, "goal": goal, "acts": acts,
        })
        print(f"{task} ep{ep:02d} seed={seed} success={success} len={len(ee)}",
              flush=True)
    n_succ = sum(e["success"] for e in episodes)
    print(f"=== {task}: {n_succ}/{len(episodes)} success ===", flush=True)
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="push-v3,plate-slide-v3")
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--start_seed", type=int, default=5000)
    ap.add_argument("--ckpt", default=str(HARNESS / "ckpt_smolvla_metaworld"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(HARNESS / "results/base_rollouts.json"))
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    t0 = time.time()
    data = {}
    for task in tasks:
        print(f"\n=== collecting {task} ===", flush=True)
        data[task] = collect(task, args.n_episodes, args.start_seed,
                              args.ckpt, args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"tasks": tasks, "n_episodes": args.n_episodes,
                   "start_seed": args.start_seed, "data": data,
                   "wall_s": time.time() - t0}, f)
    print(f"\nSaved -> {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
