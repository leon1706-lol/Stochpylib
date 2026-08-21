# Function: is_independent

- Module: stochpylib/probability/independence.py
- Defined at: line 9

## Docstring

Whether two events are independent: P(A ∩ B) == P(A) * P(B).

>>> from stochpylib.probability.basics import sample_space, event
>>> s = sample_space([1, 2, 3, 4])
>>> is_independent(event(1, 2), event(1, 2, 3), s)
False

## Calls

- [[stochpylib_probability_basics.py__P]] (from `P`)
- [[stochpylib_probability_basics.py__intersection]] (from `intersection`)
- [[../../external/math]] (external `math.isclose`)
