# Function: stirling

- Module: stochpylib/probability/combinatorics.py
- Defined at: line 50

## Docstring

Stirling numbers of the first or second kind.

``kind="second"`` (default) counts partitions of n labeled items into k non-empty,
unlabeled subsets. ``kind="first"`` (unsigned) counts permutations of n items with
exactly k cycles.

>>> stirling(4, 2)
7
>>> stirling(4, 2, kind="first")
11

## Calls

- ValueError
- [[stochpylib_probability_combinatorics.py__combination]] (from `combination`)
- [[stochpylib_probability_combinatorics.py__factorial]] (from `factorial`)
- range
