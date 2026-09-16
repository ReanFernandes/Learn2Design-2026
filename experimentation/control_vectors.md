# What a control vector is

A control vector (`bounded_params`, ~174-330 numbers depending on topology
and grid size) is the flat array of continuous physical knobs for one fixed
topology. Every number's meaning is topology-specific — same index means a
different physical thing in a different topology — so alignment must go
through `objective.optimization_pairs` (component, property) or
`optimization_pairs = topology_from_string(...)`, never raw position.

## Two kinds of parts

**Always-present mirrors** — exist in every topology regardless of the
string's letters, and always contribute the same 3 knobs each:

| Property | Bounds | Meaning |
|---|---|---|
| `reflectivity` | [0, 1] | fraction of light reflected vs. transmitted |
| `tuning` | [-360°, 360°] | mirror position/phase offset |
| `mass` | [0.01, 200] kg | suspended mass |

- Every interior cell has 4 surrounding mirrors (`ml`/`mr`/`mt`/`mb`) → size²×4 mirrors.
- Every boundary cell has 1 mirror → 4×size boundary mirrors.

**Chosen parts** — selected by the topology string's letters, contribute
different knobs (or none) depending on which type was picked:

| Component (from string) | Knobs contributed |
|---|---|
| `beamsplitter` (interior, A-D) | reflectivity, tuning, mass (same as a mirror) |
| `directional_beamsplitter` (interior, E-H) | **none** — ideal fixed 4-port router |
| `laser` (boundary, L) | `power` [0, 200] W |
| `squeezer` (boundary, S) | `db` [0, 10] (squeeze strength), `angle` [-360°,360°] (squeeze phase) |
| `detector` / `balanced_homodyne` (boundary, D/H) | **none** — passive readout |
| `space` (any connection) | `length` [0.1, 4000] — some tied together across parallel inter-cell edges |

## Verified example

Topology `ABBGDGBEA-SLLLSDSLSLSL` (194 params): 108 (36 interior mirrors × 3)
+ 36 (12 boundary mirrors × 3) + 18 (6 beamsplitter centers × 3, since 3 of
9 centers are directional and contribute 0) + 6 (6 lasers × 1) + 10
(5 squeezers × 2) + 0 (1 detector) + 16 (tied space lengths) = **194. Exact.**

## Continuous vs. categorical

Every value in `bounded_params` is continuous, real-valued — no categorical
values live in the control vector. The only discrete/categorical choices in
the whole UIFO formulation are topology-level (component type + orientation
per cell, source/readout type per boundary) — fixed via the topology string
*before* optimization starts, never part of what's being continuously
optimized. This is why gradient-based methods apply cleanly: nothing inside
the vector is non-differentiable.

Three nuances worth knowing, none of which change "continuous" but all of
which affect how the values should be treated:

- **`tuning`/`angle` are periodic, not just bounded** — degrees in
  [-360°, 360°]. Two values near opposite ends (e.g. -359° and 359°) are
  physically close but far apart in raw Euclidean distance. Matters for any
  distance-based method (GP kernel, nearest-neighbor, clustering) applied
  directly to the raw vector.
- **`reflectivity`'s real usable bound isn't exactly [0, 1]** — verified in
  `base_problem.py`: clamped to `[1e-12, 1 - 1e-12]` specifically to avoid
  NaN gradients at the physical extremes (perfectly reflecting/transmitting
  mirror is a differentiation singularity).
- **Not every index is one-to-one with a physical property** — tied
  parameters (see the worked example above) are many-to-one: one scalar
  value shared across several component/property pairs.

## How to access this for any topology

```python
from dfbench.problems import UIFOProblem
from dfbench import Objective

obj = Objective(UIFOProblem(size=3, topology="..."))
obj.bounds              # (2, n_params) low/high per index
obj.optimization_pairs  # (component, property) per index, or a list of
                         # pairs for tied parameters sharing one variable
```

To see the physical grid layout a string encodes (which cell/boundary is
what), use [experimentation/eda/print_topology.py](eda/print_topology.py).
