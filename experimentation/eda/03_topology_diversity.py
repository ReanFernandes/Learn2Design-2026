"""EDA: how concentrated is the dataset on a small number of topologies?
This determines whether a config->params meta-model trained on this data has
any hope of generalizing to genuinely novel hidden topologies, or whether
it'll just be very good at variations of a handful of dominant ones.
"""

import matplotlib.pyplot as plt
import numpy as np
from eda_utils import load_metadata_df

df = load_metadata_df()

counts = df["topology_string"].value_counts()  # one row per unique topology, sorted desc
n_unique = len(counts)
n_entries = len(df)

print(f"unique topologies: {n_unique}")
print(f"total entries: {n_entries}")
print()

# Concentration: what fraction of ALL entries come from just the top-K topologies?
for k in [1, 10, 50, 100]:
    share = counts.head(k).sum() / n_entries
    print(f"top {k:3d} topologies account for {share * 100:5.1f}% of all entries")
print()

# The other end: how many topologies appear only once (a single attempt ever)?
n_singletons = (counts == 1).sum()
print(
    f"singleton topologies (exactly 1 entry): {n_singletons} "
    f"({n_singletons / n_unique * 100:.1f}% of unique topologies, "
    f"{n_singletons / n_entries * 100:.1f}% of all entries)"
)
print()
print("top 10 most-repeated topologies:")
print(counts.head(10))

# Rank-frequency plot on log-log axes -- the standard diagnostic for a
# long-tailed / power-law-like concentration pattern.
fig, ax = plt.subplots(figsize=(6, 4.5))
ranks = np.arange(1, n_unique + 1)
ax.loglog(ranks, counts.values, marker=".", linestyle="none", markersize=3)
ax.set_xlabel("topology rank (1 = most repeated)")
ax.set_ylabel("entry count for that topology")
ax.set_title("Topology repetition: rank-frequency (log-log)")
fig.tight_layout()
out_path = "/Users/reanfernandes/Learn2Design-2026/experimentation/eda/topology_rank_frequency.png"
fig.savefig(out_path, dpi=130)
print(f"\nsaved: {out_path}")
