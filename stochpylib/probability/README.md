# stochpylib.probability

The core probability engine: sample spaces, events, conditional probability,
Bayes' theorem, exact-integer combinatorics, and independence testing — the
foundation every other module builds on, deliberately plain functions rather
than classes (no shared base-class behavior exists here).

**Status:** implemented & tested (21/21 spec names).

## Files

- `basics.py` — sample spaces, events, `P()`, conditional probability,
  `bayes_theorem()`, `total_probability()`.
- `combinatorics.py` — factorial through derangements, Stirling numbers of the
  second kind, Bell and Catalan numbers, multinomials. Exact integer arithmetic
  throughout (the derangement recurrence is pure-int; see
  `development/Probleme.md` [2] for why floating-point series are banned here).
- `independence.py` — independence, mutual exclusion, pairwise and conditional
  independence checks over finite sample spaces.

## Conventions

- Combinatorics return exact Python ints at any size — never floats rounded.
- Events are sets/frozensets of atomic outcomes; probabilities are plain floats.
- No randomness anywhere in this module: everything is exact or closed-form.

Spec: vault `Modules/probability.md` (private). Tests:
`tests/probability/tests.py`. Bugs found while building this module:
`development/Probleme.md` ([1] wrong doctest example data).
