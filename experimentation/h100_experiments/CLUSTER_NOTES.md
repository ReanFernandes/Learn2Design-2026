# LAMARR Cluster — Session State & Hard-Won Gotchas

Written just before a context compaction, to make sure nothing operationally
important gets lost. Read this first if continuity seems shaky.

## Access

- SSH host alias is `LAMARR` — **uppercase, case-sensitive**. `ssh lamarr`
  fails to resolve; `ssh LAMARR` works. → `gwkilab.cs.tu-dortmund.de`, user
  `fernande`. Key-based auth via agent, no password/2FA needed.
- Local tmux session `cluster` holds the SSH connection. Inside that, a
  **remote** tmux session `l2d_h100` was created on `gwkilab` itself
  (double-tmux, so work survives disconnects on either end). Reattach with
  `tmux attach -t l2d_h100` once inside gwkilab.
- Not H100 — corrected early on. Real hardware: A100-SXM4, either **40GB**
  (`ml2ran01-08`) or **80GB** (`ml2ran09-12`). SLURM schedules to whichever's
  free by default; add `--nodelist=ml2ran[09-12]` to target the 80GB ones
  specifically if VRAM matters for a given test.
- SLURM + Pyxis/enroot containers. GPU partitions: `GPU1/2/4/8` (by GPU
  count). We've only used `GPU1` (single-GPU jobs) so far — also the cheapest
  under the fair-share system (multi-GPU jobs cost more priority than the
  same work split into sequential single-GPU jobs).
- Official docs: https://gitlab.tu-dortmund.de/lamarr/lamarr-public/cluster
  — confirms scripted `srun` submission is normal/expected (not `sbatch`,
  which "doesn't work well with the container environment"), pip/conda
  install inside containers is standard, fair-share favors many small jobs
  over few big ones, stop jobs manually when done.

## Filesystem — the part that will bite you if forgotten

- **`/home/fernande` is FULL on the login node** (NFS from server `kimo`,
  3.2GB/3.2GB, 0 bytes free). Don't try to write there, ever, until that's
  resolved separately.
- **`/home/fernande` does not exist at all on compute nodes** — different
  storage system, not mounted. The recurring `couldn't chdir to
  /home/fernande: ... going to /tmp instead` warning is harmless/expected
  default behavior — **do not "fix" it**.
- **`--container-mount-home` actively BREAKS container startup here** (hard
  failure, container never starts) — confirmed by direct test. Never use
  this flag on this cluster.
- **Real persistent storage: `/cephfs/users/fernande`** (268TB cluster-wide,
  45TB free, confirmed writable). This is the actual working directory root
  — use it for everything code/results.
- `/raid` exists (28TB local disk) but is **not user-namespaced by default**
  (`/raid/fernande` doesn't auto-exist, `mkdir` it yourself if needed) and is
  **node-local** (not shared across nodes) + auto-purged after inactivity.
  Scratch/temp only, never rely on it persisting or being visible elsewhere.
- Named containers (`--container-name`) are **node-local** — landing on a
  different node re-imports the image even with the same name reused.

## What's set up and confirmed working right now

- Fork (push here, never upstream): https://github.com/ReanFernandes/Learn2Design-2026
  — `gh` authenticated locally as `ReanFernandes`.
- Repo cloned: `/cephfs/users/fernande/Learn2Design-2026` (180M; `dataset.h5`
  verified byte-identical to local, 74,920,439 bytes).
- `uv` installed: `/cephfs/users/fernande/bin/uv` (0.12.15). **Not on PATH**
  — its shell-rc-file self-registration failed (same home-dir issue above,
  harmless). Reference by full path or export PATH manually each time.
- venv: `/cephfs/users/fernande/Learn2Design-2026/.venv`, Python 3.12.3,
  installed with the `[cuda13]` extra (container ships CUDA 13.0, driver
  580.167.08 — confirmed via `nvcc --version`).
- **JAX confirmed on real GPU**: `jax.default_backend()` → `'gpu'`,
  `jax.devices()` → `[CudaDevice(id=0)]`.
- ~10 small test jobs run total, all clean/auto-exited, `squeue --me` empty
  after every check. No lingering resources at any point.

## The working `srun` template (verified end to end)

```bash
srun --mem=32GB --export=ALL -c 8 --gres=gpu:1 \
     --container-name=l2d_test \
     --container-image=nvcr.io/ml2r/interactive_cuda \
     -p GPU1 --mail-type=NONE \
     bash -c "cd /cephfs/users/fernande/Learn2Design-2026 && .venv/bin/python <script>"
```
Never add `--container-mount-home`. Never pass `--mail-user` (omit it,
`--mail-type=NONE` is sufficient — SLURM didn't complain).

## User's standing instruction

Cluster is genuinely busy (47 jobs on GPU1 when checked, near a submission
deadline for other users) — **keep footprint minimal**: small resource
requests, short/self-terminating commands over long interactive `--pty`
sessions where possible, verify before assuming, never anything destructive.
This is a standing constraint, not a one-time request.

## Where we are in the plan (see PLAN.md in this same folder)

Phase 0 (environment) is done. Next is Phase 1: re-run
`experimentation/local_experiments/scratch_batch_scaling_cpu.py` (backend-
agnostic despite the filename) on this real A100 to get real batch-scaling
numbers — the documented H100 chart doesn't directly apply since this is
different hardware (different VRAM, different tensor core generation).
`batched_restart_adam_gd.py` (same folder) is the algorithm to test, already
verified for correctness (not performance) on local CPU.
