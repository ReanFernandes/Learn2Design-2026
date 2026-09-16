"""Does topology STRUCTURE (component-role composition) predict how well a
topology can do, before we ever look at specific parameter values?

Aggregate to one row per topology (not per entry) to avoid the top-100
over-representation bias from before. Restrict to topologies with >=10
attempts (same stable subset as script 05) so per-topology loss estimates
aren't dominated by 1-2 sample noise.

Note: n_beamsplitter + n_directional_beamsplitter = 9 always (size=3, fixed
interior slot count), and n_laser + n_squeezer + n_detector + n_homodyne = 12
always (fixed boundary slot count) -- these features aren't independent, more
of one structurally means less of another. Keep that in mind interpreting
correlations.
"""

import matplotlib.pyplot as plt
import pandas as pd
from eda_utils import add_topology_features, load_metadata_df

pd.set_option("display.width", 160)

df = add_topology_features(load_metadata_df())
df3 = df[df["size"] == 3].copy()
role_cols = [c for c in df3.columns if c.startswith("n_")]

counts = df3["topology_string"].value_counts()
stable_topologies = counts[counts >= 10].index
stable = df3[df3["topology_string"].isin(stable_topologies)]

# One row per topology: role counts (fixed per topology) + loss aggregates.
topo_level = stable.groupby("topology_string").agg(
    **{c: (c, "first") for c in role_cols},
    mean_loss=("loss", "mean"),
    best_loss=("loss", "min"),
    n_attempts=("loss", "count"),
)

print(f"topologies in this analysis: {len(topo_level)}")
print()

print("correlation of each component-role count with topology mean_loss:")
print(topo_level[role_cols + ["mean_loss"]].corr()["mean_loss"].drop("mean_loss").sort_values())
print()
print("(same, vs best_loss -- caveat: topologies with more attempts get an easier")
print(" 'best of N' by chance alone, independent of true structural quality)")
print(topo_level[role_cols + ["best_loss"]].corr()["best_loss"].drop("best_loss").sort_values())
print()

# Simplest, most direct view: mean loss by squeezer count and by
# directional-beamsplitter count.
print("mean_loss by n_squeezer:")
print(topo_level.groupby("n_squeezer")["mean_loss"].agg(["mean", "count"]))
print()
print("mean_loss by n_directional_beamsplitter:")
print(topo_level.groupby("n_directional_beamsplitter")["mean_loss"].agg(["mean", "count"]))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flat, role_cols):
    grouped = topo_level.groupby(col)["mean_loss"].agg(["mean", "sem", "count"])
    grouped = grouped[grouped["count"] >= 3]  # drop buckets too thin to trust
    ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["sem"], marker="o", capsize=3)
    ax.set_xlabel(col)
    ax.set_ylabel("topology mean_loss")
    ax.set_title(col)
fig.suptitle("Topology mean_loss vs. each component-role count (size=3, >=10-attempt topologies)")
fig.tight_layout()
out_path = "/Users/reanfernandes/Learn2Design-2026/experimentation/eda/structure_vs_loss.png"
fig.savefig(out_path, dpi=130)
print(f"\nsaved: {out_path}")
