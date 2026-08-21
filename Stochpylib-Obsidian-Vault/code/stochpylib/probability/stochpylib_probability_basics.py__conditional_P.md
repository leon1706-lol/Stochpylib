# Function: conditional_P

- Module: stochpylib/probability/basics.py
- Defined at: line 94

## Docstring

P(A | B) = P(A ∩ B) / P(B).

>>> s = sample_space([1, 2, 3, 4, 5, 6])
>>> round(conditional_P(event(2, 4, 6), event(2, 3, 4, 5), s), 4)
0.5

## Calls

- [[stochpylib_probability_basics.py__P]] (from `P`)
- ZeroDivisionError
- [[stochpylib_probability_basics.py__intersection]] (from `intersection`)
