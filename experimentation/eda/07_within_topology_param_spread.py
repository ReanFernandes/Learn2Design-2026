"""For a single, heavily-repeated topology, compare its ~2,878 independently
optimized endpoints parameter-by-parameter. A parameter that lands in nearly
the same place across independent runs is "pinned" by the physics (likely
important); one that varies across most of its allowed range is flexible
(many values work about equally well, or the topology has multiple distinct
solution regimes).

Spread is normalized by each parameter's own bound range so reflectivity
([0,1]) and length ([0.1,4000]) are comparable on the same 0-1ish scale.
For a uniform random spread across the full range, normalized std ~= 0.289
(1/sqrt(12)) -- that's the "no constraint at all" reference point.
"""

import matplotlib.pyplot as plt
import numpy as np
from dfbench import Objective
from dfbench.problems import UIFOProblem
from eda_utils import load_params_for_topology

TOPOLOGY = "DABFEEGBB-SLSLLSLSSDLS"  # most-repeated topology, 2878 entries
UNIFORM_REFERENCE_STD = 1 / np.sqrt(12)

params, _loss = load_params_for_topology(TOPOLOGY)  # (n_entries, n_params)
print(f"topology: {TOPOLOGY}")
print(f"replicate entries: {params.shape[0]}, params per entry: {params.shape[1]}")

problem = UIFOProblem(size=3, topology=TOPOLOGY)
obj = Objective(problem)
bounds = np.asarray(obj.bounds)  # (2, n_params)
pairs = obj.optimization_pairs
assert bounds.shape[1] == params.shape[1] == len(pairs)

bound_range = bounds[1] - bounds[0]
raw_std = params.std(axis=0)
normalized_std = raw_std / bound_range

def pair_label(pair) -> str:
    # Tied parameters (e.g. inter-grid-cell spaces) map to a list of several
    # (component, property) pairs sharing one optimization variable.
    if isinstance(pair[0], str):
        comp, prop = pair
        return f"{comp}.{prop}"
    comps = "+".join(c for c, _ in pair)
    prop = pair[0][1]
    return f"[{comps}].{prop}"


def pair_property(pair) -> str:
    return pair[1] if isinstance(pair[0], str) else pair[0][1]


labels = [pair_label(p) for p in pairs]

order = np.argsort(normalized_std)
print("\nmost 'pinned' (lowest normalized spread across 2878 independent runs):")
for i in order[:10]:
    print(f"  {labels[i]:30s} normalized_std={normalized_std[i]:.4f}  raw_std={raw_std[i]:.4g}")
print("\nmost 'flexible' (highest normalized spread):")
for i in order[-10:]:
    print(f"  {labels[i]:30s} normalized_std={normalized_std[i]:.4f}  raw_std={raw_std[i]:.4g}")

# Aggregate by property type (reflectivity, tuning, length, power, db, mass)
# to see if certain KINDS of parameters are systematically more pinned.
props = np.array([pair_property(p) for p in pairs])
print("\nmean normalized spread by property type:")
for prop in sorted(set(props)):
    mask = props == prop
    print(f"  {prop:15s} n={mask.sum():3d}  mean_normalized_std={normalized_std[mask].mean():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(np.sort(normalized_std), marker=".", linestyle="none", markersize=3)
axes[0].axhline(UNIFORM_REFERENCE_STD, color="red", linestyle="--", label="uniform-random reference")
axes[0].set_xlabel("parameter rank (sorted, most pinned first)")
axes[0].set_ylabel("normalized std across 2878 independent optimized runs")
axes[0].set_title(f"Parameter spread, sorted\n{TOPOLOGY}")
axes[0].legend()

prop_means = {
    prop: normalized_std[props == prop].mean() for prop in sorted(set(props))
}
axes[1].bar(prop_means.keys(), prop_means.values())
axes[1].axhline(UNIFORM_REFERENCE_STD, color="red", linestyle="--", label="uniform-random reference")
axes[1].set_ylabel("mean normalized std")
axes[1].set_title("Spread by property type")
axes[1].legend()
axes[1].tick_params(axis="x", rotation=30)

fig.tight_layout()
out_path = "/Users/reanfernandes/Learn2Design-2026/experimentation/eda/within_topology_param_spread.png"
fig.savefig(out_path, dpi=130)
print(f"\nsaved: {out_path}")
