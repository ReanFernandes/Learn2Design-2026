"""What characterizes the entries that actually beat Voyager (loss < 0),
across the WHOLE dataset -- not one topology's replicates, but everything.
Is it a few topologies doing this repeatedly, or broadly spread? And does
component-role composition look different for this subset than the general
population (building on finding 7 -- structure predicts loss)?
"""

import pandas as pd
from eda_utils import add_topology_features, load_metadata_df

pd.set_option("display.width", 160)

df = add_topology_features(load_metadata_df())
df3 = df[df["size"] == 3]
role_cols = [c for c in df3.columns if c.startswith("n_")]

beats = df3[df3["loss"] < 0]
print(f"entries beating Voyager (loss<0): {len(beats)} / {len(df3)} ({len(beats) / len(df3) * 100:.2f}%)")
print(f"unique topologies represented: {beats['topology_string'].nunique()} / {df3['topology_string'].nunique()}")
print()

print("top topologies by count within the beats-Voyager subset:")
print(beats["topology_string"].value_counts().head(10))
print()

print("component-role composition: full population vs beats-Voyager subset")
comparison = pd.DataFrame(
    {
        "full population (mean)": df3[role_cols].mean(),
        "beats Voyager (mean)": beats[role_cols].mean(),
    }
)
print(comparison.round(3))
print()

print("detector vs homodyne, as fraction of the (single) readout slot:")
for name, subset in [("full population", df3), ("beats Voyager", beats)]:
    n_det = (subset["n_detector"] == 1).sum()
    n_hom = (subset["n_homodyne"] == 1).sum()
    n_neither = len(subset) - n_det - n_hom
    print(f"  {name:16s}: detector={n_det / len(subset) * 100:5.1f}%  "
          f"homodyne={n_hom / len(subset) * 100:5.1f}%  neither={n_neither / len(subset) * 100:5.1f}%")
