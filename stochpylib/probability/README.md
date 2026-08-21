# stochpylib.probability — core probability engine

The first completed module. Three submodules:

- `basics.py` — sample spaces, events, `P()`, conditional probability, Bayes' theorem,
  total probability.
- `combinatorics.py` — factorial, permutations, combinations, multinomials, Stirling numbers,
  Bell/Catalan numbers, derangements. Exact integer arithmetic throughout.
- `independence.py` — independence, mutual exclusion, pairwise/conditional independence checks.

Spec: vault `Modules/probability.md` (private). Tests: `../tests/probability/tests.py`.
See `development/Probleme.md` for bugs found and fixed while building this.
