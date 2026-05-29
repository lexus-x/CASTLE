# CASTLE — Roadmap (proposal-grade → paper-grade)

Ordered by leverage. The first item is the **make-or-break**; everything below
assumes it survives.

## P0 — Prove the routing is load-bearing (the central claim)
The pilot showed the residual helps (+25 pp) but ρ-routing ≈ always-on ≈ random
because the gate fired 74% of steps. Until this is fixed there is no *temporal-
logic* contribution, only a residual policy.
- **Selective gate:** fire ρ only on severe/persistent violations (target <30%
  of steps). Sweep grace/εₐ and a "regression" trigger (object moving *away*).
- **Powered comparison:** base / always-on / ρ-routed / random at **≥3 seeds,
  n≥40 paired** → non-overlapping Wilson CIs.
- **Win condition (either is publishable):** (a) ρ-routed > always-on on success;
  or (b) ρ-routed *matches* always-on with far fewer interventions → an
  efficiency/safety story (less perturbation of a working policy).

## P1 — Remove the privileged-state crutch (real-robot credibility)
The residual currently reads ground-truth object/goal positions the VLA can't
see. Retrain it on **image/proprio features only** (or a learned state encoder)
so the method transfers off the sim oracle. Report the success delta vs the
privileged version — quantifies how much of the gain was the oracle.

## P2 — Instruction → STL auto-compiler (turns a crutch into the centerpiece)
Specs are currently hand-written per task. An LLM that compiles a free-form
instruction into an STL spec (and grounds its predicates in detectable state)
(a) removes the "you hand-tuned the spec" reviewer attack and (b) makes
*language → temporal-logic → control* the headline novelty. Report spec
sensitivity and failure cases.

## P3 — Second, near-independent contribution: runtime safety monitor
ρ(t) is a free byproduct. Benchmark it as a **constraint-violation detector**
(violation-detection AUROC; violation-rate ↓ under CASTLE vs base). This hedges
the paper: even if the success lift is modest, a strong safety-monitor result
stands on its own.

## P4 — Scale & reviewer-proof baselines
- All 5 mid-band tasks (drawer, push, window, plate-slide, peg) × ≥3 seeds.
- Head-to-head: **A2C2** (trained per-step correction) and a **SafeDec-style
  shield** — the two nearest prior works — under matched compute/params.
- Ablations: gate selectivity, residual α, residual capacity, verifier variant
  (STL vs the statistical `P(success|obs)` fallback).

## P5 — Presentation
- **Video:** side-by-side GIF of a base failure vs the CASTLE-corrected rollout
  (render MetaWorld frames during eval, stitch with imageio). Most persuasive
  single asset for a README / talk.
- Per-task convergence and ρ(t) timelines (show *where* the gate fires).

## Non-goals (scoped out, by design)
- Touching the VLA's weights (the frozen-policy premise is the point).
- Contact-rich precision tasks as the primary claim (peg-insert) — keep as a
  stress-test, not the headline; the residual/oracle story is cleanest in the
  free-space-ish mid-band tasks.
