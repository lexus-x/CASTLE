#!/usr/bin/env python
"""Train the residual head by behaviour-cloning the base's OWN successful actions.

Training data = (privileged-state feature, base action) pairs drawn ONLY from
the base policy's SUCCESSFUL rollouts on a task. No expert, no RL. The learned
head g(state) is, at deployment, applied as a residual a' = base + gate·(g−base)
where the gate is set by STL-ρ violation (see pilot_eval.py).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from residual_head import ResidualHead, featurize, save


def build_dataset(eps, success_only=True):
    F, A = [], []
    for e in eps:
        if success_only and not e["success"]:
            continue
        n = min(len(e["ee"]), len(e["acts"]))
        for t in range(n):
            F.append(featurize(e["ee"][t], e["obj"][t], e["goal"][t],
                               e["grip"][t])[0])
            A.append(np.asarray(e["acts"][t], dtype=np.float32))
    return np.asarray(F, dtype=np.float32), np.asarray(A, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default=str(Path(__file__).parent /
                    "results/base_rollouts_act.json"))
    ap.add_argument("--task", required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.load(open(args.rollouts))["data"]
    eps = data[args.task]
    n_succ = sum(e["success"] for e in eps)
    F, A = build_dataset(eps, success_only=True)
    print(f"[{args.task}] {n_succ} success eps -> {len(F)} (state,action) pairs, "
          f"action_dim={A.shape[1]}")
    if len(F) < 50:
        print("WARNING: very little training data")

    # train/val split
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(F))
    nval = max(1, int(0.15 * len(F)))
    vi, ti = idx[:nval], idx[nval:]
    dev = args.device
    Ft = torch.tensor(F, device=dev)
    At = torch.tensor(A, device=dev)

    model = ResidualHead(action_dim=A.shape[1]).to(dev)
    with torch.no_grad():
        model.f_mean.copy_(Ft[ti].mean(0))
        model.f_std.copy_(Ft[ti].std(0))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.MSELoss()
    tib = torch.tensor(ti, device=dev)
    vib = torch.tensor(vi, device=dev)
    best_val = float("inf")
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        pred = model(Ft[tib])
        loss = lossf(pred, At[tib])
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                vl = lossf(model(Ft[vib]), At[vib]).item()
            best_val = min(best_val, vl)
            print(f"  epoch {ep+1:4d}  train_mse={loss.item():.4f}  val_mse={vl:.4f}")

    out = args.out or str(Path(__file__).parent /
                          f"results/residual_{args.task}.pt")
    save(model, out, {"task": args.task, "action_dim": int(A.shape[1]),
                      "n_success_eps": int(n_succ), "n_pairs": int(len(F)),
                      "n_params": model.n_params(), "best_val_mse": best_val})
    print(f"params={model.n_params()}  saved -> {out}")


if __name__ == "__main__":
    main()
