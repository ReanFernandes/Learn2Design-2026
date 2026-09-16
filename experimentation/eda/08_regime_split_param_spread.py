"""Script 07 found near-maximal (not random) spread on most parameters for
DABFEEGBB-..., which we suspected was regime-mixing: different runs landing
in qualitatively different solution modes, each with its own parameter
values, pooled together into one misleadingly-high spread number.

Test that directly: look at this topology's own loss distribution for a
natural split point, then recompute normalized spread SEPARATELY within
each regime. If regime-mixing was the explanation, within-regime spread
should drop substantially for most parameters.
"""

import matplotlib.pyplot as plt
import numpy as np
from dfbench import Objective
from dfbench.problems import UIFOProblem
from eda_utils import load_params_for_topology

TOPOLOGY = "DABFEEGBB-SLSLLSLSSDLS"
UNIFORM_REFERENCE_STD = 1 / np.sqrt(12)

params, loss = load_params_for_topology(TOPOLOGY)
print(f"entries: {len(loss)}")

# Look at this topology's own loss shape first, rather than guessing a split.
fig0, ax0 = plt.subplots(figsize=(6, 4))
ax0.hist(loss, bins=80)
ax0.set_xlabel("loss")
ax0.set_ylabel("count")
ax0.set_title(f"loss distribution, {TOPOLOGY}")
fig0.tight_layout()
hist_path = "/Users/reanfernandes/Learn2Design-2026/experimentation/eda/regime_loss_histogram.png"
fig0.savefig(hist_path, dpi=130)
print(f"saved: {hist_path}")

# Report a few candidate split points' resulting group sizes so we can pick
# a defensible one after actually looking at the histogram.
for threshold in [-0.1, 0.0, 0.2, 0.5, 1.0]:
    n_below = (loss < threshold).sum()
    print(f"threshold={threshold:+.1f}: n_below={n_below:5d}  n_above={len(loss) - n_below:5d}")

# Clear gap in the histogram sits around -0.1: a small isolated "good regime"
# cluster (~240 entries) vs the dominant "common regime" (~2640 entries).
THRESHOLD = -0.1
good = params[loss < THRESHOLD]
common = params[loss >= THRESHOLD]
print(f"\ngood regime: {len(good)} entries, common regime: {len(common)} entries")

problem = UIFOProblem(size=3, topology=TOPOLOGY)
obj = Objective(problem)
bounds = np.asarray(obj.bounds)
bound_range = bounds[1] - bounds[0]

pooled_std = params.std(axis=0) / bound_range
good_std = good.std(axis=0) / bound_range
common_std = common.std(axis=0) / bound_range

print(f"\nmean normalized std -- pooled: {pooled_std.mean():.4f}  "
      f"good-regime-only: {good_std.mean():.4f}  common-regime-only: {common_std.mean():.4f}")
print(f"(uniform-random reference: {UNIFORM_REFERENCE_STD:.4f})")

n_pooled_above_ref = (pooled_std > UNIFORM_REFERENCE_STD).sum()
n_common_above_ref = (common_std > UNIFORM_REFERENCE_STD).sum()
print(f"\nparams with spread ABOVE uniform-random reference:")
print(f"  pooled (all runs mixed): {n_pooled_above_ref}/{len(pooled_std)}")
print(f"  common-regime only:      {n_common_above_ref}/{len(common_std)}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(np.sort(pooled_std), marker=".", linestyle="none", markersize=3, label="pooled (all runs)")
ax.plot(np.sort(common_std), marker=".", linestyle="none", markersize=3, label="common-regime only")
ax.plot(np.sort(good_std), marker=".", linestyle="none", markersize=3, label="good-regime only (n=241)")
ax.axhline(UNIFORM_REFERENCE_STD, color="red", linestyle="--", label="uniform-random reference")
ax.set_xlabel("parameter rank (sorted independently per curve)")
ax.set_ylabel("normalized std")
ax.set_title(f"Parameter spread: pooled vs. within-regime\n{TOPOLOGY}")
ax.legend()
fig.tight_layout()
out_path = "/Users/reanfernandes/Learn2Design-2026/experimentation/eda/regime_split_spread_comparison.png"
fig.savefig(out_path, dpi=130)
print(f"\nsaved: {out_path}")
