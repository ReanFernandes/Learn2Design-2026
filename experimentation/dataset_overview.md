# What the dataset gives us

`dataset/dataset.h5` — 29,650 optimized UIFO setups from a 360,000 GPU-hour
exploration campaign, distilled from the larger GraviTune dataset.

## What one entry is

`(topology_string, converged control vector, loss, sensitivity curve, power
diagnostics)`. Every `bounded_params` is a genuine optimized endpoint, not a
random/pre-optimization vector — verified directly: re-running the simulator
on a stored vector reproduces the saved loss to 7 decimal places
(`evaluate_entry.py`). `initialized_from` tracks lineage *between* optimized
endpoints (re-optimizing an earlier result further), not before/after
snapshots of one run.

## Scale and structure

- 12,437 unique topologies; 28,863 entries at grid size=3, 787 at size=4
  (kept separate in any analysis — different param counts, not comparable).
- `complexity` == `param_length` exactly, for every entry (it's just the
  parameter count, not a separate metric).
- Full HDF5 layout, loading examples: [dataset/README.md](../dataset/README.md).

## Coverage is heavily concentrated — matters for any modeling use

Top 100 of 12,437 unique topologies (0.8%) account for **52.7%** of all
entries. 94.9% of unique topologies are singletons (one attempt ever),
together only 39.8% of the data. A model trained naively here will be
dominated by ~100 topologies and have near-zero signal for the vast
majority it's never seen more than once — see
[eda/README.md](eda/README.md) findings 5-6 for the structural detail
(the concentration centers near "typical" component composition, not an
outlier corner — extrapolation risk is concentrated at structurally
extreme, mostly-singleton topologies).

## Quality: good, not state-of-the-art

Best single entry across all 29,650 rows: loss = **-0.399**. Round 1's #1
competitor averaged **0.020** across 10 (different) hidden topologies —
numerically better than this dataset's single best entry. Not a strict
comparison (different topologies), but a useful sanity check: this is a
strong corpus for pretraining/warm-starts, not a ceiling to match.

## What it's for

Explicitly sanctioned by the competition (`docs/submission.md`): pretraining
surrogates, initial-point models, warm-starts — no penalty for spending the
whole budget on surrogate inference as long as one feasible result is
logged. Full findings log (multimodal loss shape, structure-vs-loss
correlations, within-topology parameter pinning, generalization checks):
[eda/README.md](eda/README.md).
