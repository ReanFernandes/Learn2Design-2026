"""Build the "genotype network" / semantic Hamming graph over ALL unique
size=3 topologies (not just the 155 well-sampled ones) -- find the actual
closest neighbor pairs that exist in the dataset, check how well-sampled
(fresh-only) each side is, and see whether semantic distance predicts loss
difference better than plain Hamming did (finding 13).
"""

import numpy as np
import pandas as pd
from eda_utils import load_metadata_df, pairwise_semantic_distance

df = load_metadata_df()
df3 = df[df["size"] == 3].copy()
df3["is_fresh"] = df3["initialized_from"] == ""
fresh = df3[df3["is_fresh"]]

unique_topologies = sorted(df3["topology_string"].unique())
print(f"unique size=3 topologies: {len(unique_topologies)}")

print("computing pairwise semantic distance matrix...")
dist = pairwise_semantic_distance(unique_topologies)
n = len(unique_topologies)
np.fill_diagonal(dist, 999)  # exclude self-pairs from "closest neighbor" search
print(f"distance matrix: {dist.shape}, dtype={dist.dtype}")

min_dist = dist.min()
print(f"\nminimum nonzero distance found anywhere in the full 12k-topology graph: {min_dist}")
print("distance value counts (off-diagonal, sampled):")
vals, counts = np.unique(dist, return_counts=True)
for v, c in list(zip(vals, counts))[:8]:
    print(f"  distance={v:4d}: {c // 2:>10,} pairs")  # //2: matrix is symmetric

# Fresh-only mean loss + count per topology, for checking sample reliability.
fresh_stats = fresh.groupby("topology_string")["loss"].agg(["mean", "count"])
fresh_stats = fresh_stats.reindex(unique_topologies)  # align to matrix order, NaN for none

# Closest neighbor pairs (smallest distance), with both sides having >=3 fresh attempts.
MIN_FRESH = 3
rows, cols = np.where(dist == min_dist)
pairs = [(unique_topologies[i], unique_topologies[j]) for i, j in zip(rows, cols) if i < j]
print(f"\npairs at the minimum distance ({min_dist}): {len(pairs)}")

usable = []
for a, b in pairs:
    na = fresh_stats.loc[a, "count"] if pd.notna(fresh_stats.loc[a, "count"]) else 0
    nb = fresh_stats.loc[b, "count"] if pd.notna(fresh_stats.loc[b, "count"]) else 0
    if na >= MIN_FRESH and nb >= MIN_FRESH:
        usable.append((a, b, na, nb, fresh_stats.loc[a, "mean"], fresh_stats.loc[b, "mean"]))

print(f"of those, pairs with >={MIN_FRESH} fresh attempts on BOTH sides: {len(usable)}")
for a, b, na, nb, la, lb in usable[:15]:
    print(f"  {a} (n={na:.0f}, loss={la:.3f})  <->  {b} (n={nb:.0f}, loss={lb:.3f})  |diff|={abs(la-lb):.3f}")
