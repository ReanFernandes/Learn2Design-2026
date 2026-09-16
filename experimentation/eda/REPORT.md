# EDA Report: Can the Dataset Guide Algorithm Design?

## Conclusion

**No, not with confidence.** The dataset can't support generalizable claims
about the topology search space. Treat any structural pattern found here as
a weak, unconfirmed hint — never a basis for algorithm design — unless
independently corroborated by something outside the dataset itself. Move to
direct empirical algorithm testing rather than further dataset mining.

## How we got there

**Scale is the root problem.** 29,650 entries span 12,437 unique topologies
— out of ~6.6 trillion valid size-3 configurations. Coverage is ~2
billionths of the space, and not evenly spread: top 100 topologies (0.8% of
unique ones) account for 52.7% of entries; 94.9% of unique topologies are
singletons.

**The data itself is real and correctly understood**: every `bounded_params`
is a genuine converged optimum (re-simulating reproduces the saved loss to 7
decimals), and loss is a meaningful, consistent yardstick — computed live
against a simulated Voyager reference, same scale across all topologies.

**One structural signal survived scrutiny**: more `directional_beamsplitter`
correlates with better loss (non-linear — only visible binned, not in raw
correlation). Trusted because it was confirmed two independent ways
(marginal correlation + pairwise distance) *and* matches an independently
known physical mechanism (non-Michelson routing, from the source PRX
paper). This is the one finding worth carrying forward.

**Most other apparent structure didn't survive closer scrutiny:**
- A "narrow, precision-pinned good regime" finding reversed once fresh vs.
  reoptimized entries were separated — mostly a lineage artifact, not a
  landscape property.
- Beating Voyager is dominated by 2 of 11,678 topologies (93% of every
  instance) — and those two sit about as far apart from each other,
  structurally, as two random topologies do. No shared recipe.

**Sparsity is severe, and we confirmed it directly, not just argued it.**
Built the full pairwise distance graph over all 11,678 topologies — only 3
usable near-neighbor pairs exist (out of ~68 million possible) with enough
samples to trust. Not enough close comparisons in the data to learn local
structure from.

**Topology generation itself carries no hidden bias.** Every interior
position is perfectly uniform (entropy at the theoretical max); the one
boundary skew found is fully explained by the encoding's own "exactly one
detector" rule, not by any preference in how topologies were sampled.

---
Full findings log (16 entries, one per script): [README.md](README.md)
