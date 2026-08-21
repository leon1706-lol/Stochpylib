# Function: derangement

- Module: stochpylib/probability/combinatorics.py
- Defined at: line 104

## Docstring

Number of permutations of n items with no fixed points.

Uses the exact recurrence D(n) = (n - 1) * (D(n - 1) + D(n - 2)) rather than the
floating-point series formula, to keep results exact for large n.

>>> derangement(4)
9

## Calls

- range
