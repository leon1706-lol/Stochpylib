# Function: bayes_theorem

- Module: stochpylib/probability/basics.py
- Defined at: line 107

## Docstring

Bayes' theorem: P(A | B) = P(A) * P(B | A) / P(B).

Classic disease-screening example: 1% base rate, 99% true-positive rate, 5% false-positive
rate among the healthy 99% gives P(B) = 0.01*0.99 + 0.99*0.05 = 0.0594.

>>> p_b = total_probability((0.99, 0.01), (0.05, 0.99))
>>> round(bayes_theorem(0.01, 0.99, p_b), 4)
0.1667

## Calls

- ZeroDivisionError
