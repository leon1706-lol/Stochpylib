"""Independence checks built on top of `probability.basics`."""

import math
from itertools import combinations

from stochpylib.probability.basics import P, conditional_P, intersection


def is_independent(evt_a, evt_b, space, tol=1e-9):
    """Whether two events are independent: P(A ∩ B) == P(A) * P(B).

    >>> from stochpylib.probability.basics import sample_space, event
    >>> s = sample_space([1, 2, 3, 4])
    >>> is_independent(event(1, 2), event(1, 2, 3), s)
    False
    """
    return math.isclose(
        P(intersection(evt_a, evt_b), space), P(evt_a, space) * P(evt_b, space), abs_tol=tol
    )


def mutual_exclusion(evt_a, evt_b):
    """Whether two events are mutually exclusive (disjoint).

    >>> from stochpylib.probability.basics import event
    >>> mutual_exclusion(event(1, 2), event(3, 4))
    True
    """
    return len(intersection(evt_a, evt_b)) == 0


def pairwise_independence(events, space, tol=1e-9):
    """Whether every pair among a collection of events is independent.

    >>> from stochpylib.probability.basics import sample_space, event
    >>> s = sample_space([1, 2, 3, 4])
    >>> pairwise_independence([event(1, 2), event(1, 2, 3)], s)
    False
    """
    return all(
        is_independent(a, b, space, tol=tol) for a, b in combinations(events, 2)
    )


def conditional_independence(evt_a, evt_b, evt_c, space, tol=1e-9):
    """Whether A and B are independent given C: P(A ∩ B | C) == P(A | C) * P(B | C).

    >>> from stochpylib.probability.basics import sample_space, event
    >>> s = sample_space([1, 2, 3, 4, 5, 6])
    >>> conditional_independence(event(1, 2), event(2, 3), event(1, 2, 3), s)
    False
    """
    p_c = P(evt_c, space)
    if p_c == 0:
        raise ZeroDivisionError("conditional_independence undefined when P(C) == 0")
    return math.isclose(
        conditional_P(intersection(evt_a, evt_b), evt_c, space),
        conditional_P(evt_a, evt_c, space) * conditional_P(evt_b, evt_c, space),
        abs_tol=tol,
    )
