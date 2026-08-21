# Function: conditional_independence

- Module: stochpylib/probability/independence.py
- Defined at: line 45

## Docstring

Whether A and B are independent given C: P(A ∩ B | C) == P(A | C) * P(B | C).

>>> from stochpylib.probability.basics import sample_space, event
>>> s = sample_space([1, 2, 3, 4, 5, 6])
>>> conditional_independence(event(1, 2), event(2, 3), event(1, 2, 3), s)
False

## Calls

- [[stochpylib_probability_basics.py__P]] (from `P`)
- ZeroDivisionError
- [[stochpylib_probability_basics.py__conditional_P]] (from `conditional_P`)
- [[stochpylib_probability_basics.py__intersection]] (from `intersection`)
- [[../../external/math]] (external `math.isclose`)
