"""EDA Stage 1 continued: what does the `loss` distribution actually look
like? Summary stats (mean/std/quantiles) hide shape -- a histogram doesn't.

We already learned `complexity` was secretly two populations mixed together
(size=3 vs size=4). Check the same thing for `loss` while we're at it.
"""

import matplotlib.pyplot as plt
from eda_utils import load_metadata_df

df = load_metadata_df()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].hist(df["loss"], bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
axes[0].axvline(0, color="black", linestyle="--", linewidth=1, label="loss=0 (reference)")
axes[0].set_xlabel("loss")
axes[0].set_ylabel("count")
axes[0].set_title("All entries")
axes[0].legend()

for size_value, color in [(3, "#4C72B0"), (4, "#DD8452")]:
    subset = df[df["size"] == size_value]["loss"]
    axes[1].hist(
        subset,
        bins=60,
        alpha=0.6,
        label=f"size={size_value} (n={len(subset)})",
        color=color,
    )
axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
axes[1].set_xlabel("loss")
axes[1].set_title("Split by grid size")
axes[1].legend()

fig.tight_layout()
out_path = "/Users/reanfernandes/Learn2Design-2026/experimentation/eda/loss_histogram.png"
fig.savefig(out_path, dpi=130)
print(f"saved: {out_path}")
