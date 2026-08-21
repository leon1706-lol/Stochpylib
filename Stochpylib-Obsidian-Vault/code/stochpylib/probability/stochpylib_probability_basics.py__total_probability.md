# Function: total_probability

- Module: stochpylib/probability/basics.py
- Defined at: line 122

## Docstring

Law of total probability: P(B) = sum_i P(B | A_i) * P(A_i).

Each argument is a ``(P(B | A_i), P(A_i))`` pair over a partition of the sample space.

>>> round(total_probability((0.99, 0.01), (0.05, 0.99)), 4)
0.0594

## Calls

- sum
