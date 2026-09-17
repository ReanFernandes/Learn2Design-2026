# LAMARR Cluster — Session State & Hard-Won Gotchas

Written to survive gaps in continuity (context compaction, or the user
disconnecting to use a different VPN for another project). Read this first
if continuity seems shaky, or right after reconnecting post-VPN-switch.

## Reconnecting after a VPN-forced disconnect (2026-09-17)

The user needs to occasionally VPN into a different cluster for unrelated
work, which kills the SSH connection to LAMARR. **This does not affect
anything on the cluster side** — the chain is Mac → SSH → `gwkilab` login
node → remote tmux session `l2d_h100` → `srun` job on a compute node. The
SLURM job and the remote tmux session both live on `gwkilab` itself and
don't care whether the Mac's SSH connection is up. To resume:

1. `ssh LAMARR` again in the local `cluster` tmux pane (the old SSH client
   process died with the connection; the local tmux session itself survives).
2. `tmux attach -t l2d_h100` to get back into the remote session.
3. `tmux list-windows -t l2d_h100` — as of 2026-09-17 there are two windows:
   `0` (Voyager batch-scaling sweep) and `1` named `uifo_batchscale` (UIFO
   batch-scaling sweep). Check `squeue -u fernande` for job state — jobs
   `68565` (Voyager) and `68566` (UIFO) were queued (`PD`, priority) on
   `GPU1` as of the disconnect.
4. Output is also durably logged (see below) in case a pane's scrollback
   is ever lost — check the log files first before worrying about it.

**Important gotcha discovered while setting this up**: don't try to control
tmux (new windows, etc.) by sending prefix keystrokes (`C-b` then a command)
*through* the attached pane via `tmux send-keys -t cluster ...` — this
proved unreliable (the keystrokes landed as literal text at a shell/srun
prompt instead of being intercepted as a tmux prefix, for reasons not fully
understood) and once accidentally cancelled a queued job via a stray
Ctrl-C sent while trying to "reset" a garbled prompt. **Instead, issue tmux
control commands directly over their own `ssh LAMARR "tmux ..."` call** —
e.g. `ssh LAMARR "tmux new-window -t l2d_h100 -n foo"` or
`ssh LAMARR "tmux send-keys -t l2d_h100:0 '...' Enter"`. This works
reliably and doesn't touch/interfere with whatever's already in the pane.

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

## Getting local code changes onto the cluster (git, not rsync/scp)

`/cephfs` is **not mounted on the login node** — only inside compute-node
jobs — so `rsync`/`scp` straight to the login node fails with a confusing
"No such file or directory". Git is the working transport instead:

- Locally: a `fork` remote was added (`https://github.com/ReanFernandes/
  Learn2Design-2026.git`) — **push there, never to `origin`** (the
  organizers' upstream repo, still configured locally as `origin` from the
  original clone).
- On the cluster, the repo was cloned directly from the fork, so its
  `origin` remote already *is* the fork — a plain `git pull origin main`
  inside a job pulls in whatever was last pushed from local.
- `.gitignore` has a `*scratch*` pattern that silently excludes
  `scratch_batch_scaling_cpu.py` unless force-added (`git add -f`) — easy
  to forget and get a confusing "no such file" on the cluster after an
  otherwise-successful-looking push.

## Logging job output durably

Interactive `srun` output only exists in the tmux pane's scrollback unless
redirected. For anything worth keeping, pipe through `tee` to a file under
`/cephfs/users/fernande/Learn2Design-2026/experimentation/h100_experiments/logs/`
(created with `mkdir -p` as part of the job command, since it only exists
inside a compute-node job context) — e.g. `... | tee .../logs/foo.log`.
Current logs there: `voyager_batchscale.log`, `uifo_batchscale.log`.

## User's standing instruction

Cluster is genuinely busy — job counts on GPU1 climbed from 47 to 78+ over
the course of one day, near a submission deadline for other users —
**keep footprint minimal**: small resource requests, short/self-terminating
commands over long interactive `--pty` sessions where possible, verify
before assuming, never anything destructive. This is a standing constraint,
not a one-time request.

**Correction learned the hard way**: an empty per-partition queue (e.g.
`GPU8` showing 0 jobs) does NOT mean idle GPUs — partitions `GPU1/2/4/8`
share the same physical nodes, just under different fair-share accounting.
Check actual free capacity with `scontrol show node <name>` (compare
`AllocTRES gres/gpu=N` against `CfgTRES`), not `squeue -p <partition>`.
As of 2026-09-17, cluster-wide only 2 of ~104 total GPUs were genuinely
free (one each on `ml2ran01`, `ml2ran07`) — the rest were at 8/8 allocated
or draining for maintenance (`ml2ran02`, `ml2ran06`). Partition-hopping
doesn't get you a shorter queue; `GPU1`'s queue is where the real capacity
already is.

## Where we are in the plan (see PLAN.md in this same folder)

Phase 0 (environment) done. Phase 1 (batch-scaling calibration) mostly
done: found and fixed a real bug in the calibration script (see PLAN.md's
Phase 1 section — `warmup_vmap_value_and_grad()` defaults to `batch_size=2`
regardless of what's being timed, so every other batch size was paying a
fresh JIT-compile cost inside the timed loop; fixed by passing
`batch_size=batch_size` explicitly). Clean numbers obtained for
`ConstrainedVoyagerProblem` batch 1-32 (sub-linear, no saturation yet).

Also discovered the documented H100 chart (`media/uifo_batch_scaling_all_
operations.png`) benchmarks `UIFOProblem` (the real competition problem),
not `ConstrainedVoyagerProblem` (a much smaller reference-design baseline)
— the two aren't comparable; Voyager can hold a much bigger batch before
hitting the same memory wall. Extended the sweep script
(`scratch_batch_scaling_cpu.py`, now takes a `voyager`/`uifo` CLI arg,
sweeps up to 1024 with graceful OOM handling) and queued both variants
(jobs `68565` Voyager, `68566` UIFO, both on `GPU1`, both logging to the
`logs/` dir above) to find where each actually saturates.

**Open design question, leaning toward "yes"**: given batch-scaling numbers
don't transfer between Voyager and UIFO, probably nothing tuned on Voyager
(learning rate, patience, restart behavior, batch-width payoff) transfers
either. Current lean: keep Voyager only for Phase 2's cheap crash/
correctness check, move Phase 3+ (the tuning sweeps that actually matter)
straight to `UIFOProblem` rather than tuning-then-validating. Not yet
decided/acted on — revisit once both sweeps report back.
