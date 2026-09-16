"""Not "how far apart are two topologies" but "how much does EACH individual
position vary across the whole dataset." Explains why mean pairwise distance
(62% of max) sits well below the observed max (100%) -- some positions must
be far more conserved than others.

Caveat: "rarely varies in the explored sample" could mean genuine physical
importance (deviating hurts, so search avoided it) OR just an artifact of
the topology-generation process's own sampling bias. Can't fully separate
those from this data alone -- report both possibilities.
"""

import numpy as np
import pandas as pd
from eda_utils import load_metadata_df

df = load_metadata_df()
df3 = df[df["size"] == 3]
unique_topologies = sorted(df3["topology_string"].unique())
n = len(unique_topologies)
print(f"unique size=3 topologies: {n}")

interior_positions = [f"{r}{c}" for r in range(1, 4) for c in range(1, 4)]  # row-major, matches string order
boundary_positions = ["01", "02", "03", "10", "14", "20", "24", "30", "34", "41", "42", "43"]

interior_chars = np.array([[t.split("-")[0][p] for t in unique_topologies] for p in range(9)])
boundary_chars = np.array([[t.split("-")[1][p] for t in unique_topologies] for p in range(12)])


def entropy_bits(chars: np.ndarray) -> float:
    _, counts = np.unique(chars, return_counts=True)
    p = counts / counts.sum()
    return -(p * np.log2(p)).sum()


rows = []
for label, chars, max_entropy in [
    *[(f"interior[{pos}]", interior_chars[i], np.log2(8)) for i, pos in enumerate(interior_positions)],
    *[(f"boundary[{pos}]", boundary_chars[i], np.log2(4)) for i, pos in enumerate(boundary_positions)],
]:
    h = entropy_bits(chars)
    vals, counts = np.unique(chars, return_counts=True)
    dominant_idx = counts.argmax()
    rows.append(
        {
            "position": label,
            "entropy_bits": h,
            "max_entropy_bits": max_entropy,
            "conservation_score": 1 - h / max_entropy,  # 1 = always same char, 0 = fully uniform/free
            "dominant_char": vals[dominant_idx],
            "dominant_fraction": counts[dominant_idx] / counts.sum(),
        }
    )

result = pd.DataFrame(rows).sort_values("conservation_score", ascending=False)
pd.set_option("display.width", 160)
print()
print(result.to_string(index=False))
