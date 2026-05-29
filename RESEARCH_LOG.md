# CASTLE — Research Log (decision trail, verdicts, open questions)

A faithful record of the reasoning that produced this project, written so a
reviewer (or future-us) can see *why* every turn was taken. Format per entry:
**Question → What we found → Decision.** Newest reasoning at the bottom.

---

## 0. Premise — "will any of this VLA work actually give a good result?"

**Q:** Is there a publishable VLA method in the portfolio?
**Found:** The novel-VLA-*method* lane is a graveyard. An 18-agent two-paradigm
sweep (import-a-science-law; import-a-hot-AI-technique) found **every** clean
method idea already shipped by others within weeks (MeanFlow-VLA, EBT-Policy,
discrete-diffusion-VLA, recurrent-depth-VLA, …). VLA is the most over-crowded
robotics-ML subfield; method-novelty has a half-life of weeks.
**Decision:** Stop hunting single-technique method imports. The only un-scooped
contribution types are **benchmarks/diagnostics** and, possibly, **novel
*combinations*** (a less picked-over space than single imports).

## 1. ILC-VLA — the last "alive" method — verified dead

**Q:** Is training-free Iterative Learning Control on a frozen VLA a paper?
**Found (from raw results, not the writeup):** On the full matrix it fails its
own pre-registered kill-test. reach-v3: a *partial* correction (5.05 cm → 3.9 cm,
plateaus) that only converts because reach's 5 cm success threshold is loose —
threshold-amplified. peg-insert (contact): **0/20**, no convergence (the rigid
LTI plant assumption breaks at contact). button/drawer already pass → no room.
Scoop risk re-rated 🔴 high (ILC × VLA is obvious control plumbing).
**Decision:** Close out ILC NO-GO. **Meta-lesson that reframes everything: the
blocker was never idea-novelty — it was that the eval base couldn't *show* a
gain.** (Same trap killed AEGIS at 0% and made JANUS look like noise.)

## 2. User intent — "I want a VLA *method*, a novel combination with a multi-aspect boost"

**Q:** Can a *novel combination* be a paper even if each part exists?
**Found:** Yes in principle — but in VLA it must clear three bars: (1)
**scoop-survival** (each part + the join still empty), (2) **load-bearing
synergy** (beats the sum of parts, or it's "two increments"), (3)
**confound-free, measurable boost** (matched params/compute/seeds, on a base
that can *show* it). "Guaranteed novel + boost" is impossible and is the JANUS
trap; we replace *guarantee* with *high-evidence components + cheap kill-gates*.
**Decision:** Pursue one combination behind explicit, cheap kill-gates.

## 3. CASTLE — from "shield" to "verifier"

**Q:** Is CASTLE (a train-free temporal-logic shield on a frozen VLA) the method?
**Found:** As a *pure shield* it shares the **re-ranker ceiling** — a shield can
only *reject* the VLA's own samples, never *add* capability. We had measured
exactly this: K=8 consensus/proofreading re-ranking scored **16%/14% < 20% base**
on reach. A shield cannot rescue an incompetent base. **However**, CASTLE's STL
robustness ρ is an excellent *verifier*: compiled from the instruction (zero
calibration data), interpretable, continuous, and in a verified-empty cell
(temporal-logic × VLA).
**Decision:** Demote CASTLE from *shield* to *router*. Combine it with a
capability-*adding* residual head → "**STL-ρ-routed residual correction**."

## 4. Gate A — fix the base (the precondition)

**Q:** Is there a frozen-base regime where a boost is even measurable?
**Found:** Survey of 12 MetaWorld tasks (n=90, 3 seeds): **5 tasks in the 30–75%
mid-band** (drawer 67, push 58, window 50, plate-slide 47, peg 40). **reach-v3
(28%) is a floored outlier** — the very task every prior dead-verdict used.
**Decision:** ✅ Gate A PASS on MetaWorld alone. **LIBERO dropped** (unnecessary;
sim not installed). Primary tasks: push + plate-slide (clean, stable). The
"undertrained base" contingency that killed ILC/CASTLE is **lifted**.

## 5. STL pre-gate — does ρ separate good from bad? (base-independent)

**Q:** Is the STL signal real, before we train anything?
**Found (AUROC of ρ vs eventual success):** `ρ_goal` 0.97–0.99 is
near-tautological (≈ the success metric — discounted). The honest signal,
`ρ_approach`: **plate-slide 0.91 full / 0.87 at the 50% prefix** (strong AND
early → usable for routing); **push 0.78 full / 0.43 early** (weak — push is
non-prehensile, "approach" is the wrong sub-goal).
**Decision:** STL clears the bar; it is **task-dependent** — specs must match
each task's sub-goal structure. Proceed with plate-slide primary.

## 6. Scoop check — novelty

**Q:** Is STL-ρ-routed residual on a frozen VLA already published?
**Found:** **UNCLAIMED.** Nearest: SafeDec (STL-shields, no residual); A2C2
(trained per-step correction, no spec); VLA-SCT (training-free, no temporal
logic). The triple-combination is empty.
**Decision:** ✅ Gate B1 PASS. ⚠️ external-LLM audit — verify arXiv IDs resolve
before citing.

## 7. Pilot — the make-or-break (PRELIMINARY, n=20, plate-slide)

**Q:** Does the ρ-routed residual beat base AND the always-on / random controls?
**Found:** base **65%** → always-on **85%** → ρ-routed **90%** → random **80%**
(13/17/18/16 of 20). The residual **clearly adds capability** (+25 pp raw,
base→ρ). **But ρ-routed ≈ always-on ≈ random within n=20 noise, and the gate
fires 74% of steps (not selective).**
**Decision:** ⚠️ **Capability confirmed; routing NOT yet load-bearing.** Right
now the method ≈ "apply the residual everywhere," which weakens the
temporal-logic novelty toward a plain residual-policy. This is the gap to close.

---

## Current verdict

- **For a proposal:** ✅ **defensible.** Verified-novel idea + real preliminary
  evidence (measurable testbed, working verifier signal, +25 pp residual lift) +
  falsifiable plan with kill-gates.
- **For a finished paper:** ❌ **not yet.** The central novelty claim — that
  *temporal-logic routing* is what does the work — is unconfirmed. If a selective
  gate cannot make ρ-routing beat always-on, the contribution collapses to a
  residual-policy and the novelty is at risk.

## Open questions → proposed solutions

1. **Is routing load-bearing?** → Make the gate genuinely *selective* (fire
   <30% of steps, on severe/persistent violations) and re-run base/always/ρ/
   random at **≥3 seeds, n≥40** for non-overlapping CIs. If selective ρ matches
   always-on with far fewer interventions, that itself is an efficiency/safety win.
2. **Is the win just privileged state?** → Retrain the residual on *image
   features* (no privileged obj/goal) so it could transfer to a real robot.
3. **Spec generality.** → Replace hand-written specs with an **instruction→STL
   compiler** (LLM), and report per-task spec sensitivity.
4. **Reviewer baselines.** → Reproduce A2C2 and a SafeDec-style shield head-to-head.
5. **Second contribution.** → Report ρ as a standalone **runtime safety monitor**
   (violation-detection AUROC + violation-rate ↓) — near-independent of the
   success result, hedges the paper.
