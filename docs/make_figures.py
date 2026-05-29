#!/usr/bin/env python
"""Generate the README figures from the result JSONs. Pure plotting, no GPU."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

C_BASE, C_MID, C_GOOD, C_BAD = "#94a3b8", "#2563eb", "#16a34a", "#dc2626"


def fig_survey():
    d = json.load(open(os.path.join(RES, "base_survey_metaworld.json")))
    rows = [r for r in d["survey"] if "error" not in r]
    rows.sort(key=lambda r: r["pooled_success_pct"])
    tasks = [r["task"].replace("-v3", "") for r in rows]
    vals = [r["pooled_success_pct"] for r in rows]
    lo = [max(0.0, r["pooled_success_pct"] - r["wilson95_lo"]) for r in rows]
    hi = [max(0.0, r["wilson95_hi"] - r["pooled_success_pct"]) for r in rows]
    cols = [C_MID if r["midband"] else C_BASE for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axhspan(30, 75, color=C_MID, alpha=0.08)
    ax.barh(tasks, vals, xerr=[lo, hi], color=cols, ecolor="#475569", capsize=3)
    ax.axvline(30, ls="--", c="#64748b", lw=1)
    ax.axvline(75, ls="--", c="#64748b", lw=1)
    ax.set_xlabel("frozen SmolVLA base success rate (%)  ·  n=90, 3 seeds")
    ax.set_title("Gate A: finding a MEASURABLE base\n"
                 "blue = 30–75% mid-band (a boost is detectable); grey = floored / ceilinged")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_survey.png"), dpi=130)
    print("wrote fig_survey.png")


def fig_pregate():
    d = json.load(open(os.path.join(RES, "stl_pregate.json")))
    tasks = list(d["tasks"].keys())
    comps = ["rho_app", "rho_goal", "rho_full"]
    labels = ["ρ_approach\n(gripper→obj)", "ρ_goal\n(=success, trivial)", "ρ_full\n(conjunction)"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(10, 4.2), sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    x = np.arange(len(comps))
    for ax, task in zip(axes, tasks):
        r = d["tasks"][task]
        full = [r["auroc_full"][c] for c in comps]
        pref = [r["auroc_prefix50"][c] for c in comps]
        ax.bar(x - 0.2, full, 0.38, label="full horizon", color=C_MID)
        ax.bar(x + 0.2, pref, 0.38, label="50% prefix (early)", color=C_GOOD)
        ax.axhline(0.7, ls="--", c=C_BAD, lw=1, label="pass bar 0.70")
        ax.axhline(0.5, ls=":", c="#94a3b8", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(task.replace("-v3", ""))
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("AUROC (ρ separates success vs failure)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("STL pre-gate: does robustness ρ separate good/bad base trajectories?",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_pregate.png"), dpi=130)
    print("wrote fig_pregate.png")


def fig_pilot():
    d = json.load(open(os.path.join(RES, "pilot_plate-slide-v3.json")))
    conds = ["base", "always", "rho", "random"]
    names = ["base\n(frozen VLA)", "always\n(residual ∀ step)",
             "ρ-routed\n(STL gate)", "random\n(matched rate)"]
    vals = [d[c]["succ_pct"] for c in conds]
    lo = [max(0.0, d[c]["succ_pct"] - d[c]["ci"][0]) for c in conds]
    hi = [max(0.0, d[c]["ci"][1] - d[c]["succ_pct"]) for c in conds]
    cols = [C_BASE, "#a78bfa", C_GOOD, "#f59e0b"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(names, vals, yerr=[lo, hi], color=cols, ecolor="#475569", capsize=4)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("plate-slide success (%)  ·  n=20 paired episodes")
    ax.set_ylim(0, 105)
    ax.set_title("Pilot (PRELIMINARY, n=20): residual lifts base (+25pp), but ρ-routing\n"
                 "≈ always-on ≈ random within noise → routing not yet load-bearing")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_pilot.png"), dpi=130)
    print("wrote fig_pilot.png")


if __name__ == "__main__":
    fig_survey()
    fig_pregate()
    fig_pilot()
    print("done")
