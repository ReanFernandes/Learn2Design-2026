# H100 Testing Plan

Goal: validate `BatchedRestartAdamGD` (batched trajectories + tuned Adam +
restart-on-plateau) on real target hardware, replacing every CPU-dev-scale
assumption from tonight's session with real numbers. Local runs so far are
correctness checks only (does it run, no NaN/crash) -- **no performance
claims from this repo's local results should carry over.**

## Phase 0 — Environment

```bash
git clone <your fork>
cd Learn2Design-2026
pip install -e ".[cuda13]"   # NOT the CPU install used in local dev
python -c "import jax; print(jax.devices())"   # must show GPU, not CpuDevice
python learn2design/scripts/smoke_test.py
```

## Phase 1 — Re-run our own calibration, on real hardware

Don't trust the documented H100 chart blindly for *our specific* batch
widths and problem — reproduce it here first, cheap and fast:

```bash
python experimentation/local_experiments/scratch_batch_scaling_cpu.py
```

(Script is backend-agnostic — same code, just runs on whatever device JAX
finds. Despite the filename, no CPU-specific code in it.) Compare against
the documented chart (`media/uifo_batch_scaling_all_operations.png`) —
confirms whether batch~15-19 is really the cheap sweet spot on *your*
specific GPU/driver setup before designing around it.

**Result (A100-SXM4 40GB, ml2ran08, 2026-09-16):** Confirmed sub-linear
batch cost, no saturation yet at batch=32 — worth checking wider before
picking a width for Phase 3.

```
batch=  1 | per_call=  43.7ms | per_candidate= 43.70ms | candidates/sec=  22.9
batch=  2 | per_call=  63.4ms | per_candidate= 31.69ms | candidates/sec=  31.6
batch=  4 | per_call=  72.1ms | per_candidate= 18.02ms | candidates/sec=  55.5
batch=  8 | per_call=  84.2ms | per_candidate= 10.52ms | candidates/sec=  95.0
batch= 16 | per_call= 111.4ms | per_candidate=  6.96ms | candidates/sec= 143.6
batch= 32 | per_call= 161.9ms | per_candidate=  5.06ms | candidates/sec= 197.6
```

Note: the first attempt at this produced garbage (non-monotonic, batch=2
a 200x outlier) — root cause was `warmup_vmap_value_and_grad()` defaulting
to `batch_size=2` in dfbench regardless of the batch actually being timed,
so every batch size except 2 was paying a fresh JIT-compile cost inside
the timed loop. Fixed in `scratch_batch_scaling_cpu.py` by passing
`batch_size=batch_size` explicitly to the warmup call.

## Phase 2 — Correctness re-check on GPU

Same smoke test we ran locally, just to confirm the restart logic behaves
identically on GPU (it should — pure JAX, no device-specific code):

```python
from dfbench import Objective
from dfbench.problems import ConstrainedVoyagerProblem
from experimentation.h100_experiments.batched_restart_adam_gd import BatchedRestartAdamGD

obj = Objective(ConstrainedVoyagerProblem(), max_time=30, save=["is_feasible"])
algo = BatchedRestartAdamGD()
algo.optimize(obj, random_seed=1, n_starts=16, patience=200)
print(obj.eval_count, obj.best_loss, obj.best_is_feasible)
```

## Phase 3 — Batch-width sweep, short-medium budget (fast iteration)

This is where the real decision happens — Phase 1's numbers tell you the
*cost* of each batch width; this tells you the *payoff*. Sweep `n_starts`
across a range informed by Phase 1 (e.g. 1, 4, 8, 16, 24, 32), fixed
`patience`, on `ConstrainedVoyagerProblem`, budget ~5-10 min each, a few
seeds. Compare best-feasible-loss and time-to-best. This replaces the
CPU-only sweep from tonight (`local_experiments/02_multistart_adamgd_voyager.md`)
— expect a different (likely more favorable to batching) result here.

## Phase 4 — Patience/perturbation sweep

Fix the best `n_starts` from Phase 3, sweep `patience` (how long to wait
before restarting) and `perturbation_scale`. Same budget scale as Phase 3.

## Phase 5 — Validate on real UIFOProblem, multiple topology seeds

Voyager is a proxy — confirm the winning config from Phases 3-4 actually
transfers:

```python
from dfbench.problems import UIFOProblem
for seed in [0, 1, 2, 3, 4]:
    obj = Objective(UIFOProblem(size=3, topology_seed=seed), max_time=600, save=["is_feasible"])
    # run BatchedRestartAdamGD, record best feasible loss per seed
```

## Phase 6 — Full-scale confirmation run

Once Phases 3-5 converge on a config, run it at the real competition scale
(4h, single topology) to confirm nothing changes qualitatively at full
budget that didn't show up at the shorter Phase 3-5 scales.

## Phase 7 (secondary, time-permitting) — Re-tune hyperparameters on H100

`TUNED_LR`/`B1`/`B2` in `batched_restart_adam_gd.py` came from CPU/Voyager/
short-budget HPO (`local_experiments/01_adamgd_hpo_voyager.md`) — a
reasonable starting prior, not validated at target scale. Worth a lighter
re-run of that HPO harness here once the base algorithm is confirmed
working, if time allows.

---

## Later: remote SSH collaboration

Once you're set up and running on the cluster, we'll figure out SSH access
so experiments can be sent over directly rather than relayed manually —
details TBD once we get there.
