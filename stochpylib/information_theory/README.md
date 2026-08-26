# stochpylib.information_theory

Information-theoretic measures end to end: entropy families, divergence
measures, mutual-information quantities, channel capacity and transfer-entropy
estimators, and classical coding results — all 31 spec names across five
submodules, natively on numpy/scipy.

**Status:** implemented & tested (31/31 spec names).

## Files

- `entropy.py` — Shannon `Entropy` (discrete; accepts probability vectors or raw
  count/sample data with auto-normalisation), `JointEntropy`, `ConditionalEntropy`,
  `CrossEntropy`, q-generalised `TsallisEntropy`, alpha-generalised `RenyiEntropy`
  (converges to Shannon as alpha → 1; alpha=0 is the Hartley support in bits),
  histogram-based `DifferentialEntropy` for continuous data, and `MaxEntropy`
  (bounded optimisation subject to moment constraints).
- `divergences.py` — `KLDivergence` (`RelativeEntropy` alias),
  symmetric/bounded `JensenShannonDivergence`, 1-D `WassersteinDistance`,
  `HellingerDistance`, `TotalVariation`, `ChiSquaredDivergence`, and the
  alpha-family `AlphaDivergence` (KL-limit verified near alpha=1).
- `mutual_info.py` — `MutualInformation` (contingency table),
  `NormalizedMutualInformation`, `VariationOfInformation`, `ConditionalMutualInfo`,
  `InteractionInformation` (co-information, can be negative for XOR-like
  structure), and `MultiInformation` (total correlation).
- `channels.py` — `ChannelCapacity` for BSC/BEC/Z channels, `InformationGain`
  (tree-split gain computed from frequency counts, equals MI for a two-variable
  split), lagged-conditional-MI `TransferEntropy` with discretisation,
  `DirectedInformation`, and ordinal-pattern-encoded `SymbolicTransferEntropy`.
- `coding.py` — `ShannonLimit` (BSC capacity), optimal prefix-free `HuffmanCode`
  tree, `TypicalSet` AEP-membership test, and the `AEP` probability bounds.
- `_base.py` — shared validation/auto-normalisation helpers
  (`_validate_probs`, `_normalise`, `_joint_table`) behind every public measure.

## Conventions

- Entropies are reported in **bits** (log2 throughout); divergences that have a
  natural-log form (KL, alpha-divergence) use **nats** — each class docstring
  states its unit.
- Probability inputs may be raw counts or unnormalised weights: everything flows
  through `_validate_probs`, which auto-normalises when the total deviates from 1.
- Discrete estimators take binned/tabulated data; continuous entropy and the
  transfer-entropy family discretise by design (histogram bins / quantile bands)
  — this is an estimator choice, documented per function, not an approximation bug.

## Known limitations

- `DifferentialEntropy` is a plugin histogram estimator — biased for smooth
  densities on few samples; no kNN/Kozachenko estimator yet.
- `TransferEntropy` bias floor for independent data is measurable but nonzero
  (permutation-free discretised MI); use the shuffled-baseline comparison shown
  in the tests before interpreting small values.
- Channel capacities cover the three canonical discrete memoryless channels,
  not arbitrary channel matrices.

Spec: vault `Modules/information_theory.md` (private). Tests:
`tests/information_theory/tests.py` (55 cases). Bugs found while building this
module live in `development/Probleme.md` ([35] InformationGain counts fix,
[36] Renyi alpha=0 bits fix).
