"""The other end of the spectrum from script 12: not "what's the closest
pair," but "how far apart are configurations typically, across the whole
dataset, and relative to the theoretical maximum possible distance."
"""

import matplotlib.pyplot as plt
import numpy as np
from eda_utils import load_metadata_df, pairwise_semantic_distance

df = load_metadata_df()
df3 = df[df["size"] == 3]
unique_topologies = sorted(df3["topology_string"].unique())
n = len(unique_topologies)

print(f"unique size=3 topologies: {n}")
dist = pairwise_semantic_distance(unique_topologies)

# Theoretical max: every position maximally different (cost 2 each).
n_interior, n_boundary = 9, 12
theoretical_max = (n_interior + n_boundary) * 2
print(f"theoretical max distance (every position a type-change): {theoretical_max}")

iu = np.triu_indices(n, k=1)  # upper triangle, no self-pairs, no double-counting
all_dists = dist[iu]
print(f"\ntotal pairs: {len(all_dists):,}")
print(f"min={all_dists.min()}  mean={all_dists.mean():.2f}  median={np.median(all_dists):.1f}  "
      f"max={all_dists.max()}  (out of theoretical max {theoretical_max})")
print(f"as fraction of theoretical max: mean={all_dists.mean()/theoretical_max*100:.1f}%  "
      f"max_observed={all_dists.max()/theoretical_max*100:.1f}%")

# Where do the two dominant beats-Voyager topologies sit relative to each other?
a, b = "FFFHGHCAH-SSLLLLLLSSDS", "DABFEEGBB-SLSLLSLSSDLS"
if a in unique_topologies and b in unique_topologies:
    ia, ib = unique_topologies.index(a), unique_topologies.index(b)
    print(f"\ndistance between the two dominant beats-Voyager topologies: {dist[ia, ib]} "
          f"({dist[ia, ib] / theoretical_max * 100:.1f}% of theoretical max)")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(all_dists, bins=range(0, theoretical_max + 2), color="#4C72B0")
ax.axvline(all_dists.mean(), color="red", linestyle="--", label=f"mean={all_dists.mean():.1f}")
ax.axvline(theoretical_max, color="black", linestyle=":", label=f"theoretical max={theoretical_max}")
ax.set_xlabel("semantic distance")
ax.set_ylabel("count (pairs)")
ax.set_title("Full pairwise distance distribution, all 11,678 topologies")
ax.legend()
fig.tight_layout()
out_path = "/Users/reanfernandes/Learn2Design-2026/experimentation/eda/distance_distribution.png"
fig.savefig(out_path, dpi=130)
print(f"\nsaved: {out_path}")
