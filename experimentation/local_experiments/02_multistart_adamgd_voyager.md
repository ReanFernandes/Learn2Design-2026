# 02 — Batched multi-start AdamGD vs. single-start, on CPU

**Goal:** use `jax.vmap` to run N parallel AdamGD trajectories per step
instead of one, under the *same fixed wall-clock budget* — does splitting the
budget across parallel starts help or hurt?

## Setup

- `MultiStartAdamGD` ([multistart_adam_gd.py](multistart_adam_gd.py)): N
  independent trajectories, independent optax state and independent
  gradient-clip norm per trajectory (vmapped `init`/`update`), evaluated in
  one `obj.vmap_value_and_grad` call per step.
- Hyperparameters fixed to the 01 experiment's tuned best (`lr=0.258,
  b1=0.928, b2=0.958`) for all conditions, isolating the multi-start question.
- `ConstrainedVoyagerProblem`, 60s wall-clock budget, 5 seeds.
- Pre-check: CPU (Apple M4) batch-scaling calibration
  ([scratch_batch_scaling_cpu.py](scratch_batch_scaling_cpu.py)) showed real
  per-candidate throughput gains from batching (~5x from batch=1 to batch=16)
  in isolation — motivated trying this at all on a GPU-less machine.

## Results

| Config | Feasible runs | Mean loss | Mean evals | Evals/trajectory |
|---|---|---|---|---|
| `AdamGD` (single) | **5/5** | **3.429** | 643 | 643 |
| `MultiStartAdamGD` (n=8) | 3/5 | 4.300 | 528 | ~66 |
| `MultiStartAdamGD` (n=16) | 1/5 | 4.729 | 483 | ~30 |

More parallel starts monotonically hurt both feasibility rate and loss.

## Why

Experiment 01's calibration showed feasibility on this problem typically
needs on the order of 100+ *consecutive* Adam steps along one trajectory
(momentum/adaptive scaling accumulate over consecutive steps down one
direction) — it isn't just "land anywhere good," it's "sustain a descent long
enough to cross the constraint wall." Splitting a fixed wall-clock budget
across N parallel trajectories divides total steps by N *per trajectory*:
n=16 trajectories got ~30 steps each, well under that threshold, so most
seeds never got even one of their 16 trajectories across. n=8 (~66
steps/trajectory) was borderline. Single-start (643 steps) had plenty.

This is a different regime from the "avoid bad local optima" intuition that
usually motivates multi-start. It also clarifies something from the PRX
paper discussion: `Urania`'s parallelism ran thousands of workers **each to
its own full local convergence** over a huge aggregate compute budget — it
wasn't dividing one small fixed time window across simultaneous shallow
runs. Those are different kinds of "parallel," and this experiment shows
why conflating them backfires here.

## Caveat

The vmapped optax `init`/`update` are not covered by
`obj.warmup_vmap_value_and_grad()` (which only warms the raw objective call),
so they likely JIT-compile on first use *inside* the timed budget — probably
a small, not the dominant, effect given how cleanly the per-trajectory-step
account explains the monotonic trend, but worth pre-warming explicitly in a
cleaner rerun.

## Takeaway

Naive fixed-budget-division multi-start is actively harmful on this problem.
If exploiting vmap parallelism is still worth pursuing, it likely needs to
look more like `Urania`'s pool: N trajectories run to (or near) their own
natural convergence and compute is spent running *more independent full
attempts*, not fewer steps each — i.e. parallelism should buy more total
attempts at fixed depth, not more attempts at reduced depth. Worth
revisiting at the real 4-hour/H100 budget, where per-step cost is far lower
and 100+ steps per trajectory may be affordable even at large N.
