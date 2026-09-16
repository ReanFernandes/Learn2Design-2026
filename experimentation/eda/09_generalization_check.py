"""Does the finding from 08 -- "common regime stays near-maximal spread,
good regime has ~half its params pinned near-zero" -- replicate on OTHER
heavily-repeated topologies, or was it a one-off feature of DABFEEGBB-...?

Check the next two most-repeated topologies. First look at each one's own
loss histogram (don't assume the same threshold transfers), then rerun the
same regime-split spread computation.
"""

import matplotlib.pyplot as plt
import numpy as np
from eda_utils import load_params_for_topology, regime_split_normalized_std

UNIFORM_REFERENCE_STD = 1 / np.sqrt(12)
PINNED_CUTOFF = 0.05  # "near-zero spread" threshold for the pinned-fraction summary

# Neither of these showed DABFEEGBB-...'s clean isolated low-loss cluster on
# inspection -- no visible gap to eyeball a threshold from. Use a data-driven
# split instead: bottom 10th percentile of each topology's OWN loss values,
# so the method doesn't depend on a visible gap being present.
CANDIDATE_TOPOLOGIES = [
    "DFACGAFAC-LSSSSLLSSDSS",  # 2nd most-repeated, 1945 entries
    "AHCGDCFAH-SLLSSHSLLLLS",  # 3rd most-repeated, 1585 entries
]
PERCENTILE = 10

fig, axes = plt.subplots(1, len(CANDIDATE_TOPOLOGIES), figsize=(6 * len(CANDIDATE_TOPOLOGIES), 4.5))
thresholds = {}
for ax, topology in zip(axes, CANDIDATE_TOPOLOGIES):
    _params, loss = load_params_for_topology(topology)
    threshold = float(np.percentile(loss, PERCENTILE))
    thresholds[topology] = threshold
    ax.hist(loss, bins=80)
    ax.axvline(threshold, color="red", linestyle="--", label=f"{PERCENTILE}th pct = {threshold:.3f}")
    ax.set_xlabel("loss")
    ax.set_title(f"{topology}\n(n={len(loss)})")
    ax.legend()
fig.tight_layout()
hist_path = "/Users/reanfernandes/Learn2Design-2026/experimentation/eda/generalization_loss_histograms.png"
fig.savefig(hist_path, dpi=130)
print(f"saved: {hist_path}\n")

for topology in CANDIDATE_TOPOLOGIES:
    threshold = thresholds[topology]
    result = regime_split_normalized_std(topology, threshold)
    n_good, n_common = result["n_good"], result["n_common"]
    print(f"{topology}  (threshold={threshold})")
    print(f"  good regime: {n_good} entries, common regime: {n_common} entries")

    if n_good < 5:
        print("  too few good-regime entries to trust a spread estimate, skipping\n")
        continue

    pooled_mean = result["pooled_std"].mean()
    good_mean = result["good_std"].mean()
    common_mean = result["common_std"].mean()
    n_pinned_good = (result["good_std"] < PINNED_CUTOFF).sum()
    n_pinned_common = (result["common_std"] < PINNED_CUTOFF).sum()
    n_total = len(result["pooled_std"])

    print(f"  mean normalized std -- pooled: {pooled_mean:.4f}  good: {good_mean:.4f}  common: {common_mean:.4f}")
    print(f"  params with spread < {PINNED_CUTOFF} (near-zero/'pinned') -- "
          f"good: {n_pinned_good}/{n_total}  common: {n_pinned_common}/{n_total}")
    print()
