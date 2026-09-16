"""EDA Stage 1: load the flat metadata fields into a pandas DataFrame and
get basic descriptive statistics. No slicing of the big variable-length
pools (bounded_params, power_values) yet -- just the small scalar columns
that live directly on the `entries` table.
"""

from eda_utils import load_metadata_df

df = load_metadata_df()

print(f"shape: {df.shape}  (rows, columns)")
print()
print("dtypes:")
print(df.dtypes)
print()
print("first 5 rows:")
print(df.head())
print()
print("describe() on the numeric columns:")
print(df[["loss", "complexity"]].describe())
print()

# `complexity` looked right-skewed above. Before trusting that shape, check
# whether it's actually two populations mixed together -- stratify by a
# plausible grouping variable (grid size) and re-describe.
print("complexity, stratified by size (3x3 vs 4x4 UIFO grid):")
print(df.groupby("size")["complexity"].describe())
