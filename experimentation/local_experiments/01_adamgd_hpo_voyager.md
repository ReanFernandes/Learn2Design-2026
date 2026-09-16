# 01 — HPO for AdamGD on ConstrainedVoyagerProblem

**Goal:** check whether AdamGD's default hyperparameters are actually good, and
whether a light BO wrapper beats plain random search at tuning them — single
fidelity, cheap proxy problem, before touching real `UIFOProblem` seeds.

## Why

The FAQ warns the loss landscape is steep and suggests `lr <= 0.01`, but the
shipped `AdamGD` example defaults to `lr=0.1`. Worth checking systematically
rather than guessing.

## Setup

- Problem: `ConstrainedVoyagerProblem` (fast proxy, same loss semantics as UIFO)
- Tuned: `learning_rate`, `b1`, `b2` (via `1-b1`, `1-b2` on log scale — standard
  for Adam betas). Gradient-clip norm is hardcoded in `AdamGD`, left untouched.
- Score per candidate: mean over 3 seeds of `min(loss | is_feasible)`, or a
  fixed penalty (10.0) if no feasible eval was logged — mirrors the real
  scoring rule's feasible-only requirement.
- Budget: 30s per inner run (calibrated, see below), 12 trials per method
  (BO: 5 Sobol init + 7 GP-guided).
- Methods compared: uniform random search vs. BoTorch `SingleTaskGP` +
  `qLogExpectedImprovement`, same trial budget.
- Script: [`01_adamgd_hpo_voyager.py`](01_adamgd_hpo_voyager.py)

## Search space calibration

Before the sweep, checked how much budget AdamGD needs to reach a *feasible*
point at all, across `lr in {0.01, 0.05, 0.1}` and budgets `{15, 30, 45}s`:

`lr=0.01` and `lr=0.05` **never reached feasibility at any budget tested**
(0/N feasible evals). Only `lr=0.1` did, converging almost immediately
(~4.51 loss at 15s, barely moving by 45s). So the originally planned
`lr ∈ [1e-3, 3e-1]` log-uniform range was mostly a dead zone for this
fidelity — narrowed to `lr ∈ [0.05, 0.3]`.

## Results

| Method | Trials | Best config | Best mean loss |
|---|---|---|---|
| Untuned default (reference) | — | `lr=0.1, b1=0.9, b2=0.999` | ~4.51 |
| Random search | 12 | `lr=0.258, b1=0.928, b2=0.958` | **3.423** |
| BO (GP + qLogEI) | 12 (5 init + 7 guided) | `lr=0.113, b1=0.500, b2=0.900` | 3.623 |

5/12 random-search trials and 2/12 BO trials landed on zero feasible evals
(hit the 10.0 penalty) — roughly 20-40% of this space is a flat infeasible
wall.

## Takeaway

- **HPO is worth it**: both methods beat the untuned default by ~20%.
- **BO did not beat random search here**, on this single run. Likely cause:
  the response surface has a sharp feasible/infeasible cliff (see above) that
  a smooth-kernel GP fit on only 12 points models poorly — undermining BO's
  main advantage.
- 12 vs. 12 trials is a small sample; this single run isn't strong evidence
  BO is worse in general here, just that it didn't help this time.
- Next round candidates: feasibility-aware acquisition instead of a fixed
  penalty constant, more trials, or repeat with multiple outer seeds before
  concluding either way.
