"""Core probability primitives: sample spaces, events, and Bayes' theorem.

A sample space is represented as a ``dict[outcome, float]`` mapping each outcome to its
probability (weights must sum to 1). An event is a ``frozenset`` of outcomes drawn from a
sample space.
"""

import math

Outcome = object
SampleSpace = dict
Event = frozenset


def sample_space(outcomes, weights=None):
    """Build a sample space mapping each outcome to its probability.

    If ``weights`` is omitted, outcomes are assumed equally likely.

    >>> sample_space(["H", "T"])
    {'H': 0.5, 'T': 0.5}
    """
    outcomes = list(outcomes)
    if not outcomes:
        raise ValueError("sample_space requires at least one outcome")
    if weights is None:
        p = 1.0 / len(outcomes)
        return {o: p for o in outcomes}
    weights = dict(weights)
    if set(weights) != set(outcomes):
        raise ValueError("weights must cover exactly the given outcomes")
    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(f"weights must sum to 1, got {total}")
    return {o: weights[o] for o in outcomes}


def event(*outcomes):
    """Build an event as a frozenset of outcomes.

    >>> event("H", "T") == frozenset({"H", "T"})
    True
    """
    return frozenset(outcomes)


def P(evt, space):
    """Probability of an event under a sample space.

    >>> s = sample_space(["H", "T"])
    >>> P(event("H"), s)
    0.5
    """
    return sum(space[o] for o in evt if o in space)


def complement(evt, space):
    """The complement of an event within a sample space.

    >>> s = sample_space([1, 2, 3, 4, 5, 6])
    >>> complement(event(1, 2), s) == frozenset({3, 4, 5, 6})
    True
    """
    return frozenset(space.keys()) - frozenset(evt)


def union(*events):
    """Set-theoretic union of one or more events.

    >>> union(event(1, 2), event(2, 3)) == frozenset({1, 2, 3})
    True
    """
    result = frozenset()
    for evt in events:
        result |= frozenset(evt)
    return result


def intersection(*events):
    """Set-theoretic intersection of one or more events.

    >>> intersection(event(1, 2, 3), event(2, 3, 4)) == frozenset({2, 3})
    True
    """
    events = [frozenset(e) for e in events]
    if not events:
        return frozenset()
    result = events[0]
    for evt in events[1:]:
        result &= evt
    return result


def conditional_P(evt_a, evt_b, space):
    """P(A | B) = P(A ∩ B) / P(B).

    >>> s = sample_space([1, 2, 3, 4, 5, 6])
    >>> round(conditional_P(event(2, 4, 6), event(2, 3, 4, 5), s), 4)
    0.5
    """
    p_b = P(evt_b, space)
    if p_b == 0:
        raise ZeroDivisionError("conditional_P undefined when P(B) == 0")
    return P(intersection(evt_a, evt_b), space) / p_b


def bayes_theorem(p_a, p_b_given_a, p_b):
    """Bayes' theorem: P(A | B) = P(A) * P(B | A) / P(B).

    Classic disease-screening example: 1% base rate, 99% true-positive rate, 5% false-positive
    rate among the healthy 99% gives P(B) = 0.01*0.99 + 0.99*0.05 = 0.0594.

    >>> p_b = total_probability((0.99, 0.01), (0.05, 0.99))
    >>> round(bayes_theorem(0.01, 0.99, p_b), 4)
    0.1667
    """
    if p_b == 0:
        raise ZeroDivisionError("bayes_theorem undefined when P(B) == 0")
    return p_a * p_b_given_a / p_b


def total_probability(*conditional_and_prior_pairs):
    """Law of total probability: P(B) = sum_i P(B | A_i) * P(A_i).

    Each argument is a ``(P(B | A_i), P(A_i))`` pair over a partition of the sample space.

    >>> round(total_probability((0.99, 0.01), (0.05, 0.99)), 4)
    0.0594
    """
    return sum(p_b_given_a * p_a for p_b_given_a, p_a in conditional_and_prior_pairs)
