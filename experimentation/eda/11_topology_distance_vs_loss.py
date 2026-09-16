"""Does ANY notion of distance between topologies predict how different
their outcomes are? Prerequisite question before reasoning about whether a
hidden topology is "near" or "far" from what we've explored.

Fresh-only entries (initialized_from empty), topologies with >=10 fresh
attempts for a trustworthy raw mean (no shrinkage here -- shrinking
singletons toward the population mean would artificially make far-apart
pairs look similar, corrupting exactly the comparison we're testing).

Two candidate distances:
  - Hamming: how many character positions differ between topology strings
  - Role distance: Euclidean distance in the 6-dim component-role-count space
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eda_utils import add_topology_features, load_metadata_df

df = add_topology_features(load_metadata_df())
df3 = df[df["size"] == 3].copy()
df3["is_fresh"] = df3["initialized_from"] == ""
fresh = df3[df3["is_fresh"]]

role_cols = [c for c in df3.columns if c.startswith("n_")]

fresh_counts = fresh["topology_string"].value_counts()
stable = fresh_counts[fresh_counts >= 10].index
print(f"topologies with >=10 FRESH attempts: {len(stable)}")

topo_level = (
    fresh[fresh["topology_string"].isin(stable)]
    .groupby("topology_string")
    .agg(**{c: (c, "first") for c in role_cols}, mean_loss=("loss", "mean"), n=("loss", "count"))
)
print(f"mean fresh attempts per stable topology: {topo_level['n'].mean():.1f}")


def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


strings = topo_level.index.tolist()
role_arr = topo_level[role_cols].to_numpy()
losses = topo_level["mean_loss"].to_numpy()

rows = []
for i, j in itertools.combinations(range(len(strings)), 2):
    h = hamming(strings[i], strings[j])
    role_dist = np.linalg.norm(role_arr[i] - role_arr[j])
    loss_diff = abs(losses[i] - losses[j])
    rows.append((h, role_dist, loss_diff))

pairs = pd.DataFrame(rows, columns=["hamming", "role_dist", "loss_diff"])
print(f"\ntotal pairs: {len(pairs)}")
print(pairs.describe())

print(f"\ncorrelation(hamming, loss_diff)   = {pairs['hamming'].corr(pairs['loss_diff']):.3f}")
print(f"correlation(role_dist, loss_diff) = {pairs['role_dist'].corr(pairs['loss_diff']):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, col, label in zip(axes, ["hamming", "role_dist"], ["Hamming distance", "role-count distance"]):
    binned = pairs.groupby(pd.cut(pairs[col], bins=10))["loss_diff"].agg(["mean", "sem", "count"])
    x = [interval.mid for interval in binned.index]
    ax.errorbar(x, binned["mean"], yerr=binned["sem"], marker="o", capsize=3)
    ax.set_xlabel(label)
    ax.set_ylabel("mean |loss difference|")
    ax.set_title(f"loss difference vs. {label}")
fig.tight_layout()
out_path = "/Users/reanfernandes/Learn2Design-2026/experimentation/eda/topology_distance_vs_loss.png"
fig.savefig(out_path, dpi=130)
print(f"\nsaved: {out_path}")
