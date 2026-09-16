"""Do the heavily-repeated topologies span a wide variety of component
makeups, or are they narrow variations on a theme? Compare the entry-
weighted view (what a naive model would actually train on) against the
unique-topology view (the true diversity that exists) using component-role
features instead of raw topology-string identity.
"""

import pandas as pd
from eda_utils import add_topology_features, load_metadata_df

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

df = add_topology_features(load_metadata_df())
role_cols = [c for c in df.columns if c.startswith("n_")]

# Restrict to size=3 for now -- role counts aren't comparable across grid
# sizes (a size=4 topology has more slots of every kind by construction).
df3 = df[df["size"] == 3]

unique3 = df3.drop_duplicates("topology_string")
top100_strings = df3["topology_string"].value_counts().head(100).index
top100 = df3[df3["topology_string"].isin(top100_strings)].drop_duplicates("topology_string")

print(f"size=3 entries: {len(df3)}, unique topologies: {len(unique3)}, top-100 topologies: {len(top100)}")
print()

comparison = pd.DataFrame(
    {
        "entry-weighted (mean)": df3[role_cols].mean(),
        "unique-topology (mean)": unique3[role_cols].mean(),
        "top-100 only (mean)": top100[role_cols].mean(),
    }
)
print("mean component-role counts, three ways to weight the same data:")
print(comparison.round(2))
print()

# Spread, not just center: does the top-100 cluster cover a narrower RANGE
# of each role count than the full unique pool does?
spread = pd.DataFrame(
    {
        "unique-topology (std)": unique3[role_cols].std(),
        "top-100 only (std)": top100[role_cols].std(),
        "unique (min-max)": [
            f"{unique3[c].min()}-{unique3[c].max()}" for c in role_cols
        ],
        "top-100 (min-max)": [f"{top100[c].min()}-{top100[c].max()}" for c in role_cols],
    }
)
print("spread comparison:")
print(spread.round(2))
