# Submission analysis (living doc)

Analysis of public Round 1 competition data (`competition_data/round1/`), used
to calibrate targets and reverse-engineer plausible top-performer strategies
from aggregate stats. Update this as we learn more — append, don't rewrite.

## Round 1 leaderboard calibration

43 participants, retired topologies (final eval uses fresh hidden ones —
treat absolute numbers as approximate, relative calibration as solid).

| Target | Loss | Notes |
|---|---|---|
| Win (rank 1) | 0.020 | `jagarwal` |
| 2nd | 0.071 | 3.5x worse than 1st — biggest gap on the board |
| 3rd (prize cutoff) | 0.171 | |
| Top 10 | 0.365 | |
| Median | 0.569 | |

**Organizer baselines placed in the real distribution** — "beat the
baseline" is a low bar:

| Baseline | Loss | Rank/43 |
|---|---|---|
| `NAAdamGD` (best organizer baseline) | 0.504 | ~19th (median-ish) |
| `AdamGD` | 0.639 | ~26th |
| `PyCMACMAES` | 3.554 | ~40th |
| `RandomSearch` | 4.749 | ~43rd |

Half the field already beats the organizers' own best baseline. Targets:
not-broken (<3.5), respectable (<0.5), competitive (<0.365), prize (<0.17),
win (~0.02, needs something distinctive per the #1/#2 gap).

## Throughput fingerprint analysis

`mean_evaluations_per_second` across the leaderboard is cleanly bimodal:

- **Cluster A: ~3.6-4.4 evals/sec** (~52k-59k total evals/4h) — 24 participants
- **Cluster B: ~9.2-11.6 evals/sec** (~132k-167k total evals/4h) — 19 participants

**8 of the top 9 ranks are in Cluster B** (only `seament`, #3, breaks the
pattern).

Matched against the documented H100 batch-scaling curve
([media/uifo_batch_scaling_all_operations.png](../media/uifo_batch_scaling_all_operations.png)):
- `value_and_grad` at batch=1 costs ~250ms/call → **4 evals/sec** baseline.
- Throughput speedup reaches ~2.4-2.9x somewhere in batch~5-19 →
  **~9.6-11.6 evals/sec**.

Cluster A ≈ unbatched single-point gradient descent. Cluster B ≈ modest
`vmap` batching, roughly batch~5-19. Hypothesis, not confirmed (submissions
are private) — but the throughput match is tight and the rank correlation
is strong.

**`jagarwal` vs `giannischalk`**: near-identical throughput (9.73 vs 9.71/s)
and eval count (~140k), yet 3.5x different score. Batching looks close to
necessary to reach the top of the board, but clearly not sufficient — the
per-batch update/selection rule is most of what separates #1 from #2.

### Ties to our own experiments

Experiment 02 (`local_experiments/02_multistart_adamgd_voyager.md`) found batching *hurt*
badly on our CPU (linear-ish cost per batch element starved per-trajectory
step count below the feasibility threshold). The H100 chart shows batching
cost is sub-linear there (~2.9x cost for ~15x candidates) — a fundamentally
better trade. Real leaderboard throughput data is now external evidence that
the strategy which hurt us on CPU plausibly wins on the actual eval hardware.

## Open questions / next to check

- What batch size specifically, within 5-19, do Cluster B's top performers
  land on? Curve is noisy/non-monotonic, can't pin exactly from throughput
  alone.
- Cluster A isn't uniformly bad (`seament` #3, `campi` #10, `anasjn` #11 are
  in it) — what explains those outliers doing well unbatched?
- `mean_time_to_best_minutes`: `jagarwal` finds its best at 79/240 min,
  `giannischalk` at 201/240 — both in Cluster B, very different convergence
  timing. Worth understanding once we have our own batched H100-realistic
  candidate to compare against.
- Revisit `MultiStartAdamGD` under a synthetic sub-linear batch-cost model
  (or real GPU access, when available) to see if the CPU finding actually
  reverses as hypothesized.

---
*Last updated: after the Round 1 leaderboard + H100 batch-scaling
cross-reference discussion.*
