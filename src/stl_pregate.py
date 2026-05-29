#!/usr/bin/env python
"""STL pre-gate (CASTLE's never-run, base-independent kill-test).

Question: does the STL robustness ρ separate successful from failed base
trajectories (AUROC > ~0.7)? If yes, STL is a usable routing signal and we
proceed to the scoop-check + residual pilot. If no, fall back to the
statistical verifier (or kill).

We report each ρ component at full horizon AND at a 50% prefix. The prefix
columns are the honest ones: a high full-horizon AUROC on `rho_goal` is nearly
tautological (ρ_goal ≈ the success metric), whereas a high *prefix* AUROC means
ρ predicts failure EARLY — which is what makes it useful for routing.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from stl_verifier import robustness, COMPONENTS, DEFAULTS


def auroc(scores, labels):
    """AUROC with tie handling. Higher score ⇒ predict positive (success)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def evaluate_task(eps, params):
    labels = [bool(e["success"]) for e in eps]
    full = {c: [] for c in COMPONENTS}
    pref = {c: [] for c in COMPONENTS}
    for e in eps:
        rf = robustness(e, params, prefix_frac=1.0)
        rp = robustness(e, params, prefix_frac=0.5)
        for c in COMPONENTS:
            full[c].append(rf[c])
            pref[c].append(rp[c])
    out = {"n": len(eps), "n_succ": int(sum(labels)), "labels": labels}
    out["auroc_full"] = {c: auroc(full[c], labels) for c in COMPONENTS}
    out["auroc_prefix50"] = {c: auroc(pref[c], labels) for c in COMPONENTS}
    # sanity: raw min_d_goal as a (negated) success detector
    min_dg = [robustness(e, params)["min_d_goal"] for e in eps]
    out["auroc_neg_min_dgoal"] = auroc([-x for x in min_dg], labels)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default=str(Path(__file__).parent /
                    "results/base_rollouts.json"))
    ap.add_argument("--eps_goal", type=float, default=DEFAULTS["eps_goal"])
    ap.add_argument("--eps_app", type=float, default=DEFAULTS["eps_app"])
    ap.add_argument("--pass_auroc", type=float, default=0.70)
    ap.add_argument("--out", default=str(Path(__file__).parent /
                    "results/stl_pregate.json"))
    args = ap.parse_args()
    params = dict(eps_goal=args.eps_goal, eps_app=args.eps_app)

    data = json.load(open(args.rollouts))["data"]
    report = {"params": params, "tasks": {}}
    print(f"STL PRE-GATE  (eps_goal={args.eps_goal} eps_app={args.eps_app}, "
          f"pass>{args.pass_auroc})\n")
    overall_best_prefix = []
    for task, eps in data.items():
        r = evaluate_task(eps, params)
        report["tasks"][task] = r
        print(f"=== {task}  (n={r['n']}, success={r['n_succ']}/{r['n']}) ===")
        print(f"  {'component':<12}{'AUROC_full':>12}{'AUROC_pref50':>14}")
        for c in COMPONENTS:
            print(f"  {c:<12}{r['auroc_full'][c]:>12.3f}{r['auroc_prefix50'][c]:>14.3f}")
        print(f"  {'(neg_min_dgoal sanity)':<12}{r['auroc_neg_min_dgoal']:>22.3f}")
        # the meaningful early signals: approach + ordering at prefix
        best_pref = max([r["auroc_prefix50"][c] for c in
                         ("rho_app", "rho_order", "rho_full")
                         if not np.isnan(r["auroc_prefix50"][c])] or [float("nan")])
        best_full = max([r["auroc_full"][c] for c in COMPONENTS
                         if not np.isnan(r["auroc_full"][c])] or [float("nan")])
        overall_best_prefix.append(best_pref)
        print(f"  -> best full AUROC={best_full:.3f}  "
              f"best early(prefix) AUROC={best_pref:.3f}\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2, default=float)

    full_pass = all(
        max(r["auroc_full"].values()) > args.pass_auroc
        for r in report["tasks"].values()
    )
    early_pass = all(b > args.pass_auroc for b in overall_best_prefix
                     if not np.isnan(b))
    print("VERDICT:")
    print(f"  separates good/bad (full horizon, the plan's bar): "
          f"{'PASS' if full_pass else 'FAIL'}")
    print(f"  early/predictive (prefix-50%, the routing-useful bar): "
          f"{'PASS' if early_pass else 'WEAK — STL is post-hoc, residual routing may not help'}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
