"""Does loss vary more BETWEEN different topologies, or WITHIN repeated
attempts at the SAME topology? Standard one-way ANOVA-style variance
decomposition: SS_total = SS_between + SS_within.

eta_squared = SS_between / SS_total is the fraction of loss variance
explained by "which topology you got" -- high = topology sets the ceiling,
low = run quality dominates over topology choice.
"""

from eda_utils import load_metadata_df

df = load_metadata_df()
df3 = df[df["size"] == 3].copy()  # keep grid sizes separate, as before

grand_mean = df3["loss"].mean()
group_mean = df3.groupby("topology_string")["loss"].transform("mean")
group_count = df3.groupby("topology_string")["loss"].transform("count")

ss_total = ((df3["loss"] - grand_mean) ** 2).sum()
ss_between = ((group_mean - grand_mean) ** 2).sum()
ss_within = ((df3["loss"] - group_mean) ** 2).sum()

eta_squared = ss_between / ss_total

print(f"size=3 entries used: {len(df3)}")
print(f"ss_total={ss_total:.1f}  ss_between={ss_between:.1f}  ss_within={ss_within:.1f}")
print(f"(check: between+within = {ss_between + ss_within:.1f}, should ~= ss_total)")
print()
print(f"eta_squared (fraction of loss variance explained by topology) = {eta_squared:.3f}")
print()

# Repeat restricted to topologies with a decent number of attempts, so the
# per-topology mean/std estimates aren't dominated by 1-2 sample noise.
repeated = df3[group_count >= 10]
grand_mean_r = repeated["loss"].mean()
group_mean_r = repeated.groupby("topology_string")["loss"].transform("mean")
ss_total_r = ((repeated["loss"] - grand_mean_r) ** 2).sum()
ss_between_r = ((group_mean_r - grand_mean_r) ** 2).sum()
eta_squared_r = ss_between_r / ss_total_r
n_topologies_r = repeated["topology_string"].nunique()

print(f"restricted to topologies with >=10 attempts ({n_topologies_r} topologies, {len(repeated)} entries):")
print(f"eta_squared = {eta_squared_r:.3f}")
print()

# A few concrete examples: per-topology loss range for well-repeated topologies.
summary = (
    df3[group_count >= 10]
    .groupby("topology_string")["loss"]
    .agg(["count", "mean", "std", "min", "max"])
    .sort_values("count", ascending=False)
    .head(8)
)
print("per-topology loss spread, most-repeated topologies:")
print(summary.round(3))
