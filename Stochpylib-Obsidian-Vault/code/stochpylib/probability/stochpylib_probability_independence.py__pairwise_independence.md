# Function: pairwise_independence

- Module: stochpylib/probability/independence.py
- Defined at: line 32

## Docstring

Whether every pair among a collection of events is independent.

>>> from stochpylib.probability.basics import sample_space, event
>>> s = sample_space([1, 2, 3, 4])
>>> pairwise_independence([event(1, 2), event(1, 2, 3)], s)
False

## Calls

- all
- combinations
- [[stochpylib_probability_independence.py__is_independent]] (from `is_independent`)
