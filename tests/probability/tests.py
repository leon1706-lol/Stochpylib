import math

import pytest

from stochpylib.probability.basics import (
    P,
    bayes_theorem,
    complement,
    conditional_P,
    event,
    intersection,
    sample_space,
    total_probability,
    union,
)
from stochpylib.probability.combinatorics import (
    bell_number,
    catalan_number,
    combination,
    derangement,
    factorial,
    multinomial,
    permutation,
    stirling,
)
from stochpylib.probability.independence import (
    conditional_independence,
    is_independent,
    mutual_exclusion,
    pairwise_independence,
)


# --- basics ---

def test_sample_space_uniform():
    s = sample_space(["H", "T"])
    assert s == {"H": 0.5, "T": 0.5}


def test_sample_space_weighted():
    s = sample_space(["H", "T"], weights={"H": 0.6, "T": 0.4})
    assert s == {"H": 0.6, "T": 0.4}


def test_sample_space_rejects_bad_weights():
    with pytest.raises(ValueError):
        sample_space(["H", "T"], weights={"H": 0.6, "T": 0.6})


def test_P_full_space_is_one():
    s = sample_space([1, 2, 3, 4, 5, 6])
    assert P(event(1, 2, 3, 4, 5, 6), s) == pytest.approx(1.0)


def test_P_empty_event_is_zero():
    s = sample_space([1, 2, 3, 4, 5, 6])
    assert P(event(), s) == 0


def test_complement():
    s = sample_space([1, 2, 3, 4, 5, 6])
    assert complement(event(1, 2), s) == frozenset({3, 4, 5, 6})


def test_union_and_intersection():
    a, b = event(1, 2, 3), event(2, 3, 4)
    assert union(a, b) == frozenset({1, 2, 3, 4})
    assert intersection(a, b) == frozenset({2, 3})


def test_conditional_P_fair_die():
    s = sample_space([1, 2, 3, 4, 5, 6])
    # P(even | in {2,3,4,5}) = P({2,4}) / P({2,3,4,5}) = (2/6) / (4/6) = 0.5
    assert conditional_P(event(2, 4, 6), event(2, 3, 4, 5), s) == pytest.approx(0.5)


def test_conditional_P_rejects_zero_probability_condition():
    s = sample_space([1, 2, 3, 4, 5, 6])
    with pytest.raises(ZeroDivisionError):
        conditional_P(event(1), event(), s)


def test_total_probability_and_bayes_disease_screening():
    # Classic example: 1% prevalence, 99% sensitivity, 5% false-positive rate.
    p_b = total_probability((0.99, 0.01), (0.05, 0.99))
    assert p_b == pytest.approx(0.0594)
    p_disease_given_positive = bayes_theorem(0.01, 0.99, p_b)
    assert p_disease_given_positive == pytest.approx(0.0099 / 0.0594)


def test_bayes_theorem_rejects_zero_probability_evidence():
    with pytest.raises(ZeroDivisionError):
        bayes_theorem(0.5, 0.5, 0.0)


# --- combinatorics ---

def test_factorial():
    assert factorial(5) == 120
    assert factorial(0) == 1


def test_permutation_and_combination():
    assert permutation(5, 2) == 20
    assert combination(5, 2) == 10


def test_multinomial():
    assert multinomial(10, 2, 3, 5) == 2520


def test_multinomial_rejects_mismatched_groups():
    with pytest.raises(ValueError):
        multinomial(10, 2, 3)


def test_stirling_second_and_first_kind():
    assert stirling(4, 2) == 7
    assert stirling(4, 2, kind="first") == 11


def test_bell_number():
    assert bell_number(4) == 15
    assert bell_number(0) == 1


def test_catalan_number():
    assert catalan_number(4) == 14
    assert catalan_number(0) == 1


def test_derangement():
    assert derangement(4) == 9
    assert derangement(0) == 1
    assert derangement(1) == 0


# --- independence ---

def test_is_independent_true_for_independent_events():
    # Two fair coin flips: A = first is H, B = second is H.
    s = sample_space(["HH", "HT", "TH", "TT"])
    a = event("HH", "HT")
    b = event("HH", "TH")
    assert is_independent(a, b, s)


def test_is_independent_false_for_dependent_events():
    s = sample_space([1, 2, 3, 4])
    assert not is_independent(event(1, 2), event(1, 2, 3), s)


def test_mutual_exclusion():
    assert mutual_exclusion(event(1, 2), event(3, 4))
    assert not mutual_exclusion(event(1, 2), event(2, 3))


def test_pairwise_independence():
    s = sample_space(["HH", "HT", "TH", "TT"])
    a = event("HH", "HT")  # first flip is H
    b = event("HH", "TH")  # second flip is H
    c = event("HH")  # both flips are H — dependent on a and on b individually
    assert pairwise_independence([a, b], s)
    assert not pairwise_independence([a, b, c], s)


def test_conditional_independence_rejects_zero_probability_condition():
    s = sample_space([1, 2, 3, 4])
    with pytest.raises(ZeroDivisionError):
        conditional_independence(event(1), event(2), event(), s)


def test_doctests_pass():
    import doctest

    from stochpylib.probability import basics, combinatorics, independence

    for module in (basics, combinatorics, independence):
        results = doctest.testmod(module, raise_on_error=True)
        assert results.attempted > 0
