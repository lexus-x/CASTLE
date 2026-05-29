#!/usr/bin/env python
"""Lightweight residual action head for the STL-routed corrector.

Conditioned on the privileged MetaWorld state (object/goal positions) that the
frozen VLA never sees — this is the honest mechanism by which the residual can
*add* capability rather than merely re-rank the base's own samples (a shield
can only do the latter, which we measured underperforming base).

Trained by behaviour-cloning the base policy's actions on its OWN SUCCESSFUL
rollouts (no expert, no RL). At test time the residual is applied only at
STL-ρ-flagged steps; elsewhere the base acts unchanged.
"""
import numpy as np
import torch
import torch.nn as nn


def featurize(ee, obj, goal, grip):
    """18-d state feature from privileged positions (batch or single)."""
    ee = np.asarray(ee, dtype=np.float32).reshape(-1, 3)
    obj = np.asarray(obj, dtype=np.float32).reshape(-1, 3)
    goal = np.asarray(goal, dtype=np.float32).reshape(-1, 3)
    grip = np.asarray(grip, dtype=np.float32).reshape(-1, 1)
    og = obj - goal
    eo = ee - obj
    d_og = np.linalg.norm(og, axis=1, keepdims=True)
    d_eo = np.linalg.norm(eo, axis=1, keepdims=True)
    f = np.concatenate([ee, obj, goal, grip, og, eo, d_og, d_eo], axis=1)
    return f.astype(np.float32)  # (N, 18)


FEAT_DIM = 18


class ResidualHead(nn.Module):
    def __init__(self, action_dim=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEAT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        # feature normalization (filled at fit time)
        self.register_buffer("f_mean", torch.zeros(FEAT_DIM))
        self.register_buffer("f_std", torch.ones(FEAT_DIM))

    def forward(self, feat):
        x = (feat - self.f_mean) / (self.f_std + 1e-6)
        return self.net(x)

    @torch.no_grad()
    def predict(self, feat_np, device="cpu"):
        t = torch.as_tensor(feat_np, dtype=torch.float32, device=device)
        return self.forward(t).cpu().numpy()

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())


def save(model, path, meta):
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = ResidualHead(action_dim=ckpt["meta"].get("action_dim", 4))
    m.load_state_dict(ckpt["state_dict"])
    m.to(device).eval()
    return m, ckpt["meta"]
