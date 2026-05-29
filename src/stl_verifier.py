#!/usr/bin/env python
"""STL verifier for object-to-target MetaWorld tasks (push, plate-slide).

The CASTLE component: compile a task into a Signal Temporal Logic (STL) spec and
compute its quantitative robustness ρ over a trajectory (Donzé–Maler semantics).
ρ>0 ⇒ satisfied, ρ<0 ⇒ violated; |ρ| is the margin. A per-timestep ρ(t) lets the
downstream residual be routed to the violating segments.

Spec for "move the object to the target" (hand-written for the pre-gate; the
instruction→spec compiler is a later step):

    φ = F(d_goal ≤ εg)                      # eventually object reaches the goal
      ∧ F(d_app  ≤ εa)                      # eventually gripper reaches the object
      ∧ (d_app ≤ εa) U (d_goal ≤ εg)        # approach BEFORE the object is at goal
      ∧ G(d_goal ≤ d_goal(0) + δ)           # never knock the object far past start

where d_goal(t)=‖obj(t)−goal(t)‖ (== MetaWorld obj_to_target, verified) and
d_app(t)=‖ee(t)−obj(t)‖.
"""
import numpy as np

# success/contact scales (m). MetaWorld push/slide success ≈ 0.05–0.07.
DEFAULTS = dict(eps_goal=0.07, eps_app=0.05, knock_delta=0.05)


def _traj_arrays(ep):
    ee = np.asarray(ep["ee"], dtype=np.float64)
    obj = np.asarray(ep["obj"], dtype=np.float64)
    goal = np.asarray(ep["goal"], dtype=np.float64)
    d_goal = np.linalg.norm(obj - goal, axis=1)  # == obj_to_target
    d_app = np.linalg.norm(ee - obj, axis=1)      # == tcp_to_obj
    return d_goal, d_app


def _until(rob_p, rob_q):
    """Quantitative robustness of (p U q) at t=0 over the whole horizon.
    ρ(p U q) = max_{t'} min( ρ_q(t'), min_{t''<t'} ρ_p(t'') )."""
    n = len(rob_q)
    pref_min_p = np.minimum.accumulate(rob_p)  # min_{0..t'} ρ_p
    best = -np.inf
    for tp in range(n):
        hold = pref_min_p[tp - 1] if tp > 0 else np.inf
        best = max(best, min(rob_q[tp], hold))
    return float(best)


def robustness(ep, params=None, prefix_frac=1.0):
    """Return STL component robustness + a per-step goal margin.
    prefix_frac<1 evaluates only the first fraction of the trajectory
    (tests whether ρ is EARLY-predictive, not just a post-hoc success detector)."""
    p = {**DEFAULTS, **(params or {})}
    d_goal, d_app = _traj_arrays(ep)
    n = max(1, int(round(len(d_goal) * prefix_frac)))
    d_goal, d_app = d_goal[:n], d_app[:n]

    m_goal = p["eps_goal"] - d_goal          # ≥0 iff object within εg of goal
    m_app = p["eps_app"] - d_app             # ≥0 iff gripper within εa of object
    knock_bound = d_goal[0] + p["knock_delta"]
    m_safe = knock_bound - d_goal            # ≥0 iff object not knocked far past start

    rho_goal = float(np.max(m_goal))         # F(reach goal)
    rho_app = float(np.max(m_app))           # F(approach)
    rho_safe = float(np.min(m_safe))         # G(no knock-away)
    rho_order = _until(m_app, m_goal)        # approach U at-goal
    rho_full = min(rho_goal, rho_app, rho_safe, rho_order)
    return {
        "rho_goal": rho_goal,
        "rho_app": rho_app,
        "rho_safe": rho_safe,
        "rho_order": rho_order,
        "rho_full": rho_full,
        "step_goal_margin": m_goal,          # per-step ρ for routing the residual
        "min_d_goal": float(np.min(d_goal)),
        "min_d_app": float(np.min(d_app)),
    }


COMPONENTS = ["rho_goal", "rho_app", "rho_safe", "rho_order", "rho_full"]
