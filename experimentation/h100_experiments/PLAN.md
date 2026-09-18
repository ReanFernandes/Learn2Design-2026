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

**Result (A100-SXM4 40GB, 2026-09-16/17), extended sweep, both problems:**

`ConstrainedVoyagerProblem` — saturates far higher than expected, OOMs at
batch=256 (needs 36.7GB), but diminishing returns kick in well before that:

```
batch=  1 | candidates/sec=  21.7
batch=  2 | candidates/sec=  30.4
batch=  4 | candidates/sec=  55.8
batch=  8 | candidates/sec=  94.2
batch= 16 | candidates/sec= 142.3
batch= 32 | candidates/sec= 196.6
batch= 64 | candidates/sec= 229.8   (32->64: 2x batch, only 1.17x throughput)
batch=128 | candidates/sec= 260.4   (64->128: 2x batch, only 1.13x throughput)
batch=256 | OOM (needs 36.71GiB)
```

`UIFOProblem(size=3)` — the real target problem — OOMs far earlier, at
batch=16 (needs 48GB):

```
batch=  1 | candidates/sec=  4.3
batch=  2 | candidates/sec=  6.4
batch=  4 | candidates/sec=  7.2
batch=  8 | candidates/sec=  7.8
batch= 16 | OOM (needs 48GB)
```

**Conclusion**: the documented H100 chart (`media/uifo_batch_scaling_all_
operations.png`) benchmarks UIFO, not Voyager — confirmed these are not
comparable. Voyager's throughput ceiling (~260/s) is ~33x UIFO's (~7.8/s),
and its usable batch range (up to 128) is ~16x UIFO's (up to 8). Nothing
tuned on Voyager (batch width, and probably learning rate/patience/restart
behavior too) should be assumed to transfer to UIFO. **Phase 3+'s batch-
width sweep should target UIFO directly, in the 1-8 range** — Voyager's
range is irrelevant to it. Voyager is kept only for Phase 2's cheap
crash/correctness check, not for tuning decisions.

Two bugs hit and fixed along the way, worth remembering:
- `warmup_vmap_value_and_grad()` defaults to `batch_size=2` in dfbench
  regardless of the batch actually being timed, so every batch size
  except 2 paid a fresh JIT-compile cost inside the timed loop, producing
  a garbage non-monotonic curve with batch=2 as a 200x outlier. Fixed by
  passing `batch_size=batch_size` explicitly to the warmup call.
- `git pull origin main 2>&1 | tail -5 && ...` gates on `tail`'s exit
  code, not `git pull`'s — a failed pull (e.g. from two concurrent jobs
  racing on the same shared `/cephfs` checkout) silently fell through to
  running whatever stale script was already on disk, with no error. Fixed
  by adding `set -o pipefail` before the pipeline in the job command.

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

**Result (A100, 2026-09-18):** Clean — no crash, no NaN.

```
eval_count=6384
best_loss=2.87669753585832
best_is_feasible=True
```

Better than the CPU/HPO best (3.423) in a 30s budget, though this is a
correctness check on Voyager, not a Phase 3-grade payoff comparison.

Found and fixed the same warmup bug here too:
`warmup_vmap_value_and_grad()` (no arg) defaulted to `batch_size=2`
regardless of `n_starts` — lower-impact than in the calibration script
(one wasted compile at run start, not per timed call, since the shape is
constant for the rest of the run) but still wrong. Fixed by passing
`batch_size=n_starts`.

Also hit `ModuleNotFoundError: No module named 'experimentation'` running
the script by file path (`python foo/bar.py` puts `foo/bar`'s own dir on
`sys.path`, not the repo root) — fixed by invoking as a module instead
(`python -m experimentation.h100_experiments.phase2_correctness_check`,
run from the repo root).

## Phase 3 — Batch-width sweep, short-medium budget (fast iteration)

**Revised after Phase 1**: sweep directly on `UIFOProblem(size=3)`, not
Voyager — Phase 1 showed Voyager's usable batch range (up to 128) and
Voyager-tuned choices don't transfer to UIFO, which OOMs at batch=16.
Sweep `n_starts` in **1, 2, 4, 8** (the actual usable range on a 40GB
A100), fixed `patience`, a few topology seeds, budget ~5-10 min each.
Compare best-feasible-loss and time-to-best. This also subsumes what was
originally planned as a separate Phase 5 Voyager->UIFO transfer check —
since we're on UIFO from the start now, there's nothing to transfer.

## Phase 4 — Patience/perturbation sweep

Fix the best `n_starts` from Phase 3, sweep `patience` (how long to wait
before restarting) and `perturbation_scale`. Same budget scale as Phase 3,
same problem (`UIFOProblem`).

## Phase 5 — Multi-seed validation

Confirm the winning config from Phases 3-4 holds across topology seeds,
not just the one used for tuning:

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
