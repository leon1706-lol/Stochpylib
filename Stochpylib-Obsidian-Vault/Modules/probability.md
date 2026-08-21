# stochpylib.probability

*Core probability engine*

**Design-completeness score:** 10/10 — Perfect coverage: Bayes, combinatorics, independence, conditional probability

Status: **implemented** — `stochpylib/probability/` (basics.py, combinatorics.py,
independence.py), tested in `tests/probability/tests.py` (25 tests incl. doctests, all passing).

## Submodules

### `probability.basics`

- `sample_space()`
- `event()`
- `P()`
- `complement()`
- `union()`
- `intersection()`
- `conditional_P()`
- `bayes_theorem()`
- `total_probability()`

### `probability.combinatorics`

- `factorial()`
- `permutation()`
- `combination()`
- `multinomial()`
- `stirling()`
- `bell_number()`
- `catalan_number()`
- `derangement()`

### `probability.independence`

- `is_independent()`
- `mutual_exclusion()`
- `pairwise_independence()`
- `conditional_independence()`

---

[[Module-Map]] · [[Ratings]] · [[ARCHITECTURE]]
