# Function: sample_space

- Module: stochpylib/probability/basics.py
- Defined at: line 15

## Docstring

Build a sample space mapping each outcome to its probability.

If ``weights`` is omitted, outcomes are assumed equally likely.

>>> sample_space(["H", "T"])
{'H': 0.5, 'T': 0.5}

## Calls

- ValueError
- dict
- len
- list
- [[../../external/math]] (external `math.isclose`)
- set
- sum
- weights.values
