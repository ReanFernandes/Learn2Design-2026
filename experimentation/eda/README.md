# Dataset EDA — findings so far

EDA on `dataset/dataset.h5` (29,650 entries), toward the goal of eventually
meta-learning a config→params mapping that generalizes to unseen hidden
topologies, not just memorizing this dataset's own configs. Scripts run in
order 01→04; each is small and self-contained (`eda_utils.py` has the
shared loading/parsing helpers).

## Findings

1. **`complexity` == `param_length` exactly**, for every entry — it's just
   the parameter count, not a separate derived metric.

2. **`complexity`'s apparent skew is actually two non-overlapping
   populations mixed together**: size=3 entries span 174-207, size=4 span
   291-330, zero overlap. General lesson: `groupby` before trusting a
   skewed marginal shape. (`01_load_and_describe.py`)

3. **`loss` is genuinely multimodal**, not skewed-unimodal: a small
   below-zero cluster (beats reference), a dominant spike at ~0.4-0.5, a
   long thin shelf from 1-3.5, and a secondary bump at ~4-4.5. Plausibly
   mirrors the PRX paper's "phase transition" loss curves — most runs
   plateau at the same common regime, a minority push through to a
   qualitatively better one. (`02_loss_histogram.py`)

4. **This dataset's best entry (loss=-0.399) is numerically worse than
   Round 1's #1 competitor's average score (0.020)** across 10 topologies
   (not topology-matched, so not a strict comparison — but a useful
   sanity check that this is a good, not state-of-the-art, corpus).

5. **Topology coverage is heavily concentrated**: top 100 of 12,437 unique
   topologies (0.8%) account for 52.7% of all entries. 94.9% of unique
   topologies are singletons (one entry ever), totaling only 39.8% of the
   data. Rank-frequency plot shows three distinct regimes (mega-repeated
   top ~10, a plateau ~10-90, then a long singleton tail) — plausibly a
   signature of Boltzmann-weighted exploit-heavy sampling (à la `Urania`),
   not uniform exploration. (`03_topology_diversity.py`)

6. **Structural (component-role) diversity**: the dominant/heavily-repeated
   topologies have *similar average* composition to the full unique pool,
   but a *narrower range* — they consistently exclude structurally extreme
   configs (e.g. very squeezer-heavy or squeezer-sparse). So the dataset's
   redundant signal clusters near the structural center; extrapolation risk
   is concentrated at the structural extremes, which are mostly singletons.
   (`04_component_role_diversity.py`)

## Implications for the meta-learning goal

- Raw parameter index isn't meaningful across topologies (different
  component sets) — any importance/interaction analysis, or model, needs
  to key by **component role** (via `optimization_pairs`), not position.
- A model trained naively (i.i.d. on rows) will be dominated by the top-100
  cluster; would want to downweight/stratify against topology identity.
- Confidence should scale with how close a target topology's component-role
  counts are to the well-covered center vs. the thin, singleton-heavy tails.

7. **Topology structure does predict achievable loss, non-linearly.**
   `n_directional_beamsplitter` shows a fairly clean trend: more of them
   correlates with better (lower) mean loss, plateauing around 6 — plausibly
   because directional beamsplitters enable the asymmetric, non-Michelson
   routing (e.g. the PRX paper's side-pumped L-shape transducer) that plain
   beamsplitters can't. `n_squeezer` shows a "sweet spot" around 4-5 rather
   than a monotonic trend — too few underuses squeezing, too many plausibly
   accumulates optical loss. Both were invisible in raw Pearson correlation
   (|r| < 0.25 for everything) — linear correlation undersells non-monotonic
   relationships; always check the binned/grouped shape, not just the
   coefficient. (`06_structure_vs_loss.py`)

8. **Within-topology parameter spread across independent optimized runs**
   (`07_within_topology_param_spread.py`, on `DABFEEGBB-...`, 2878 replicate
   entries): almost all parameters show *near-maximal* spread (~0.48,
   close to the theoretical max of 0.5 for a bounded variable), not
   moderate/random spread (~0.29) -- a sign of bimodality, not unimportance.
   Only squeezer `angle` params were genuinely pinned (~0.20), matching the
   physical fact that squeezing angle needs precise phase-matching to help
   at all.

9. **Splitting by loss regime confirms and sharpens finding 8**
   (`08_regime_split_param_spread.py`). This one topology's own loss
   distribution reproduces the whole dataset's multimodal shape (small
   isolated cluster near -0.3, dominant cluster near 0.4-0.5, long tail).
   Within the dominant "common" regime alone, spread stays near-maximal
   across virtually all params (mean 0.436, barely different from pooled)
   -- reaching "decent" doesn't require precision, most combinations land
   there. Within the rare "good" regime (241/2878 runs, loss<-0.1) roughly
   **half the parameters drop to near-zero spread** while the other half
   stay just as free as in the common regime -- reaching "great" requires
   precisely coordinating about half of this topology's ~192 parameters.
   Plausible explanation for why the real leaderboard's rank1->rank2 gap
   (0.020->0.071) is so much starker than rank2->rank10.

   **CORRECTION (see finding 12): this "narrow target" reading is largely
   an artifact, not primarily a landscape property.** Of the 241 good-regime
   entries, only 6 are fresh (`initialized_from` empty) -- the other 235
   (97.5%) are reoptimized descendants of a small number of lucky fresh
   discoveries (fresh hit rate: 6/1872 = 0.3%). The apparent "pinning" is
   largely "these are mostly refinements of the same one-or-few ancestors,
   so they resemble each other" -- not strong evidence the regime is
   intrinsically hard to reach via independent search. Don't treat the
   "half the parameters are precision-critical" number as landscape truth.

10. **Finding 9 does NOT generalize as-is** (`09_generalization_check.py`,
    checked on the 2nd/3rd most-repeated topologies). Splitting each at its
    own bottom-10th-percentile loss shows *no* good-vs-common spread
    difference (0/197 and 0/193 params pinned, either regime) -- unlike
    `DABFEEGBB-...`, neither topology's loss histogram has an isolated
    low-loss cluster, just one continuous skewed mode. **Correct statement
    of finding 9**: precision-pinning in a "good regime" only shows up when
    a topology actually *has* a distinct second regime (visible as a
    separated cluster in its own loss histogram) -- not a general property
    of "the best runs for any topology." Most topologies checked so far
    don't have that second regime; `DABFEEGBB-...` (the single most-repeated
    topology in the dataset) does.

11. **Beating Voyager (loss<0) is dominated by 2 topologies out of 11,678**
    (`10_beats_voyager.py`). Only 34 unique topologies ever produced a
    beats-Voyager entry (0.29% of all unique topologies); just two of them
    (`FFFHGHCAH-...`, `DABFEEGBB-...`) account for 93% of every
    beats-Voyager result in the dataset. The subset leans toward more
    `directional_beamsplitter` and detector-over-homodyne readouts,
    consistent with findings 7/10's caveats -- but since both dominant
    topologies happen to share those traits, treat this as "two specific
    lucky discoveries mined hard," not a broad, independently-replicated
    structural trend. Reinforces finding 10's lesson: don't over-generalize
    from a concentrated handful of examples.

12. **Fresh vs. reoptimized composition matters a lot, and corrects finding
    9** (checked directly: `DABFEEGBB-...`'s good regime is 235/241
    reoptimized, only 6 fresh). Reoptimized entries (`initialized_from`
    non-empty) start from an already-good prior result plus noise, then
    reconverge -- not independent draws of "how good is this topology,"
    biased toward good outcomes by construction. Any per-topology
    loss/spread statistic should use fresh-only entries to fairly compare
    across topologies; mixing in reoptimized entries confounds "topology
    structural quality" with "how much exploitation attention did this
    topology happen to receive" from the campaign.

13. **Testing "near" directly: pairwise topology-distance vs. loss-difference**
    (`11_topology_distance_vs_loss.py`, 155 topologies with >=10 fresh
    attempts each, 11,935 pairs). **Hamming/string distance carries ~zero
    signal** -- flat at 0.46-0.48 mean |loss diff| across the entire
    well-populated range (thousands of pairs per bin); an apparent spike at
    very low distance is a single-pair artifact (n=1), not real.
    **Component-role-count distance shows a real, if modest, signal** --
    binned mean |loss diff| climbs from ~0.46 to ~0.56 as role-distance
    increases, despite a near-zero raw correlation (0.026) -- same
    "check the shape, not the coefficient" lesson as finding 7. Converges
    with finding 7 via an independent method (pairwise, not marginal).
    Concrete answer to "what does near mean here": aggregate structural
    composition is a meaningful similarity notion; exact positional/string
    similarity is not. Still only validated within the 155 already-explored
    topologies -- says nothing about whether it holds near the structural
    extremes or genuinely unexplored regions.

14. **Built the semantic "genotype network"** (`12_genotype_network.py`,
    `eda_utils.pairwise_semantic_distance` -- weighted edit distance: 0=same
    char, 1=same-type different orientation, 2=different component type)
    over all 11,678 unique size=3 topologies. **Confirms the sparsity
    problem concretely, not just theoretically**: only 1 pair exists at the
    minimum distance found (3, not 1 or 2); relaxing to distance<=10 with a
    modest >=3-fresh-attempts-both-sides bar still yields only **3 usable
    pairs out of up to ~68 million possible**. Of those, 2 are trustworthy
    (n=51/13 and n=7/59) and both show real, non-trivial loss differences
    (0.34, 1.20) -- suggestive the landscape doesn't stay smooth even under
    small structural edits, but n=2 is far too few to generalize. Concrete
    answer to "can we find natural swap-pairs in the dataset": essentially
    no, not in usable quantity -- would need to generate comparisons
    ourselves (matched-budget fresh optimization on deliberately chosen
    close pairs), not mine the historical data further.

15. **Full pairwise-distance distribution** (`13_distance_distribution.py`,
    all 68.2M pairs among the 11,678 unique size=3 topologies): a smooth,
    roughly symmetric bell curve centered at mean=26.2 (62% of the
    theoretical max of 42), no clustering/multimodality -- close to what
    near-independent random draws from a large combinatorial space would
    produce. Puts finding 14 in context: the near-neighbor pairs found
    there are genuine far-left-tail outliers, not "closest of a natural
    cluster." Notable specific number: the two dominant beats-Voyager
    topologies (finding 11) sit at distance 26 from **each other** --
    essentially the population average. Despite dominating the same rare
    outcome, they look like two independent, structurally unrelated
    solutions, not variations on a shared theme.

16. **Per-position variability: nothing is conserved beyond hard structural
    constraints** (`14_per_position_variability.py`, marginal entropy per
    string position across all 11,678 topologies, not pairwise). Interior
    positions: entropy ≈3.000 bits, the exact theoretical max for 8 symbols
    -- perfectly uniform, zero conservation, at every one of the 9
    positions. Boundary positions: a modest, uniform ~30% conservation
    score across all 12 -- but this is fully and mechanically explained by
    the encoding's own constraint (exactly one of 12 boundary slots is
    D/H, placed uniformly at random: verified D+H frequency per position
    is ~8.3% = 1/12 almost exactly); once conditioned out, laser vs.
    squeezer splits ~46/46 at every position. No position favors any
    component type beyond what the counting constraint forces. Reinforces
    the session's broader conclusion: the topology-*selection* process
    looks like uniform random sampling with no detectable bias -- any real
    structure-to-loss signal (e.g. finding 7) comes from the physics, not
    from hidden bias in how training topologies were generated.

## Open / next

- Which specific parameters are in the "precision-critical" pinned half for
  `DABFEEGBB-...`? (component/property labels, not just counts.)
- What distinguishes topologies that DO have a separate low-loss regime
  (like `DABFEEGBB-...`) from ones that don't (findings 9 vs 10)? Worth
  checking more topologies to see how common true bimodality actually is.

- Does `initialized_from` lineage (re-optimized vs. fresh) predict landing
  in the better loss cluster?
- Parameter-level analysis, role-aligned via `optimization_pairs`: given
  finding 7 shows structure matters, what do winning *parameter values*
  look like for a given role (e.g. does squeezer dB cluster near its upper
  bound in good solutions)? Not started yet — needed before any
  physics-motivated initialization strategy is possible.
