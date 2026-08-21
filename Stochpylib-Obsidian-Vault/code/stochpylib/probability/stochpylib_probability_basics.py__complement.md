# Function: complement

- Module: stochpylib/probability/basics.py
- Defined at: line 57

## Docstring

The complement of an event within a sample space.

>>> s = sample_space([1, 2, 3, 4, 5, 6])
>>> complement(event(1, 2), s) == frozenset({3, 4, 5, 6})
True

## Calls

- frozenset
- space.keys
