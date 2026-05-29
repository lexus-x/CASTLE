#!/usr/bin/env python
"""Gate B2 pilot: does the STL-ρ-routed residual beat (a) base and (b) a
random-routed residual of the SAME budget?

Three conditions, evaluated on SHARED episode seeds (paired, fair):
  base   — frozen VLA, no residual.
  rho    — residual applied only at steps that VIOLATE the online progress spec
           (object not at goal AND no progress over the last W steps).
  random — residual applied at random steps, matched to rho's firing rate.

Pass = rho > base AND rho > random with non-overlapping CIs. If rho ≈ random,
the routing is not load-bearing → "two increments", not a contribution.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HARNESS = Path("/home/user/Desktop/vla_projects/RESEARCH/active/_harness-faithful-smolvla")
sys.path.append(str(HARNESS))
from harness_common import build_env_and_policy  # noqa: E402
from residual_head import featurize, load as load_res  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.envs.utils import preprocess_observation, add_envs_task  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0, c - h), 100 * min(1, c + h))


def read_success(info):
    cands = [info]
    fi = info.get("final_info") if isinstance(info, dict) else None
    if fi is not None:
        cands += list(fi) if isinstance(fi, (list, tuple, np.ndarray)) else [fi]
    for d in cands:
        if isinstance(d, dict) and d.get("is_success") is not None:
            if bool(np.asarray(d["is_success"]).any()):
                return True
    return False


def rollout(vec, policy, procs, model, condition, seed, max_steps, cfg, device):
    pre, post, env_pre, env_post = procs
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed + 777)  # independent of env init
    obs, info = vec.reset(seed=[seed])
    policy.reset()
    mw = vec.envs[0]._env.unwrapped
    success = False
    d_hist = []
    n_gate = n_step = 0
    for step in range(max_steps):
        ee = np.asarray(mw.get_endeff_pos(), dtype=np.float64)[:3]
        obj = np.asarray(mw._get_pos_objects(), dtype=np.float64)[:3]
        goal = np.asarray(mw._target_pos, dtype=np.float64)[:3]
        grip = float(np.asarray(obs["agent_pos"][0])[3])
        d_og = float(np.linalg.norm(obj - goal))
        d_eo = float(np.linalg.norm(ee - obj))
        p = preprocess_observation(obs)
        p = add_envs_task(vec, p)
        p = env_pre(p)
        p = pre(p)
        with torch.inference_mode():
            a = policy.select_action(p)
        a = post(a)
        base_a = env_post({ACTION: a})[ACTION].to("cpu").numpy()[0].copy()

        if condition == "base":
            gate = 0
        elif condition == "always":
            gate = 1
        elif condition == "rho":
            # SELECTIVE online STL violation: past a grace window the gripper
            # still has not engaged the object (d_eo large) and the task is not
            # done -> the base is failing the "approach-before-act" sub-goal,
            # the failure mode the pre-gate found predictive on plate-slide.
            gate = 1 if (step > cfg["grace"] and d_eo > cfg["eps_app"]
                         and d_og > cfg["eps_goal"]) else 0
        else:  # random
            gate = 1 if rng.random() < cfg["rand_rate"] else 0
        n_step += 1
        n_gate += gate

        if gate and model is not None:
            feat = featurize(ee, obj, goal, grip)
            g_a = model.predict(feat, device=device)[0]
            final = base_a + cfg["alpha"] * (g_a - base_a)
        else:
            final = base_a
        final = np.clip(final, -1.0, 1.0)

        d_hist.append(d_og)
        obs, r, term, trunc, info = vec.step(np.array([final]))
        success = success or read_success(info)
        if term[0] or trunc[0]:
            break
    return success, n_gate / max(n_step, 1)


def run_condition(vec, policy, procs, model, condition, seeds, max_steps, cfg, device):
    succ, gates = [], []
    for sd in seeds:
        s, gf = rollout(vec, policy, procs, model, condition, sd, max_steps, cfg, device)
        succ.append(s)
        gates.append(gf)
    k, n = int(sum(succ)), len(succ)
    lo, hi = wilson(k, n)
    return {"succ_pct": 100 * k / n, "k": k, "n": n, "ci": [lo, hi],
            "mean_gate_frac": float(np.mean(gates))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", default=str(HARNESS / "ckpt_smolvla_metaworld"))
    ap.add_argument("--residual", default=None)
    ap.add_argument("--n_eval", type=int, default=40)
    ap.add_argument("--eval_seed", type=int, default=7000)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--W", type=int, default=15)
    ap.add_argument("--prog_eps", type=float, default=0.003)
    ap.add_argument("--eps_goal", type=float, default=0.07)
    ap.add_argument("--eps_app", type=float, default=0.05)
    ap.add_argument("--grace", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res_path = args.residual or str(HARNESS / f"results/residual_{args.task}.pt")
    model, meta = load_res(res_path, device=args.device)
    print(f"[{args.task}] residual params={meta['n_params']} "
          f"val_mse={meta.get('best_val_mse'):.4f}")

    vec, policy, pre, post, env_pre, env_post = build_env_and_policy(
        args.ckpt, task=args.task, batch_size=1, device=args.device)
    policy.eval()
    procs = (pre, post, env_pre, env_post)
    max_steps = vec.call("_max_episode_steps")[0]
    seeds = [args.eval_seed + i for i in range(args.n_eval)]
    cfg = dict(W=args.W, prog_eps=args.prog_eps, eps_goal=args.eps_goal,
               eps_app=args.eps_app, grace=args.grace,
               alpha=args.alpha, rand_rate=0.0)

    print(f"eval {args.n_eval} paired episodes (seeds {seeds[0]}..{seeds[-1]})")
    base = run_condition(vec, policy, procs, model, "base", seeds, max_steps, cfg, args.device)
    print(f"  base   : {base['succ_pct']:.1f}% {base['ci']}")
    always = run_condition(vec, policy, procs, model, "always", seeds, max_steps, cfg, args.device)
    print(f"  always : {always['succ_pct']:.1f}% {always['ci']}  (residual every step — capability ctrl)")
    rho = run_condition(vec, policy, procs, model, "rho", seeds, max_steps, cfg, args.device)
    print(f"  rho    : {rho['succ_pct']:.1f}% {rho['ci']}  gate_frac={rho['mean_gate_frac']:.3f}")
    cfg["rand_rate"] = rho["mean_gate_frac"]  # match the budget
    rand = run_condition(vec, policy, procs, model, "random", seeds, max_steps, cfg, args.device)
    print(f"  random : {rand['succ_pct']:.1f}% {rand['ci']}  gate_frac={rand['mean_gate_frac']:.3f}")

    # routing is load-bearing only if rho beats base AND the always-on and
    # random-routed controls (else the method reduces to a plain BC-residual).
    def beats(a, b):
        return a["succ_pct"] > b["succ_pct"] and a["ci"][0] > b["ci"][1]
    routing_load_bearing = beats(rho, always) and beats(rho, rand)
    helps_over_base = beats(rho, base)
    if helps_over_base and routing_load_bearing:
        verdict = "PASS (rho beats base, always-on, and random — routing load-bearing)"
    elif helps_over_base:
        verdict = "PARTIAL (residual helps, but routing NOT load-bearing vs always/random)"
    else:
        verdict = "NO-PASS (no boost over base)"
    print(f"\n  VERDICT [{args.task}]: {verdict}")
    out = args.out or str(HARNESS / f"results/pilot_{args.task}.json")
    json.dump({"task": args.task, "cfg": cfg, "base": base, "always": always,
               "rho": rho, "random": rand, "verdict": verdict}, open(out, "w"), indent=2)
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
