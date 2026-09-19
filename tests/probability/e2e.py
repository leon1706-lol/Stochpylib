"""End-to-end API sweep for stochpylib.probability: one realistic exercise per public
name. ``test_every_public_name_is_exercised`` fails the moment a name ships without one;
``test_exercise[<name>]`` runs each exercise as its own pytest case."""

import math

import pytest

from stochpylib import probability as mod

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


def _die():
    return mod.sample_space([1, 2, 3, 4, 5, 6])


@exercise("sample_space")
def _sample_space():
    sp = mod.sample_space(["H", "T"], weights={"H": 0.7, "T": 0.3})
    assert mod.P(mod.event("H"), sp) == pytest.approx(0.7)


@exercise("event")
def _event():
    assert mod.P(mod.event(2, 4, 6), _die()) == pytest.approx(0.5)


@exercise("P")
def _p():
    assert mod.P(mod.event(1), _die()) == pytest.approx(1 / 6)


@exercise("complement")
def _complement():
    sp = _die()
    assert mod.P(mod.complement(mod.event(1, 2), sp), sp) == pytest.approx(4 / 6)


@exercise("union")
def _union():
    assert mod.P(mod.union(mod.event(1), mod.event(2)), _die()) == pytest.approx(2 / 6)


@exercise("intersection")
def _intersection():
    assert mod.P(mod.intersection(mod.event(1, 2, 3), mod.event(2, 3, 4)), _die()) == pytest.approx(2 / 6)


@exercise("conditional_P")
def _conditional():
    assert mod.conditional_P(mod.event(2), mod.event(2, 4, 6), _die()) == pytest.approx(1 / 3)


@exercise("bayes_theorem")
def _bayes():
    p_pos = mod.total_probability((0.99, 0.01), (0.05, 0.99))
    assert mod.bayes_theorem(0.01, 0.99, p_pos) == pytest.approx(0.1667, abs=1e-3)


@exercise("total_probability")
def _total():
    assert mod.total_probability((0.5, 0.5), (0.2, 0.5)) == pytest.approx(0.35)


@exercise("factorial")
def _factorial():
    assert mod.factorial(20) == math.factorial(20)


@exercise("permutation")
def _permutation():
    assert mod.permutation(10, 3) == 720


@exercise("combination")
def _combination():
    assert mod.combination(52, 5) == 2598960


@exercise("multinomial")
def _multinomial():
    assert mod.multinomial(6, 2, 2, 2) == 90


@exercise("stirling")
def _stirling():
    assert mod.stirling(5, 2) == 15 and mod.stirling(5, 2, kind="first") == 50


@exercise("bell_number")
def _bell():
    assert mod.bell_number(5) == 52


@exercise("catalan_number")
def _catalan():
    assert mod.catalan_number(6) == 132


@exercise("derangement")
def _derangement():
    assert mod.derangement(5) == 44


@exercise("is_independent")
def _independent():
    sp = mod.sample_space([(a, b) for a in range(2) for b in range(2)])
    a = mod.event((0, 0), (0, 1))
    b = mod.event((0, 0), (1, 0))
    assert mod.is_independent(a, b, sp)


@exercise("mutual_exclusion")
def _exclusion():
    assert mod.mutual_exclusion(mod.event(1), mod.event(2))
    assert not mod.mutual_exclusion(mod.event(1, 2), mod.event(2))


@exercise("pairwise_independence")
def _pairwise():
    sp = mod.sample_space([(a, b) for a in range(2) for b in range(2)])
    a = mod.event((0, 0), (0, 1))
    b = mod.event((0, 0), (1, 0))
    assert mod.pairwise_independence([a, b], sp)


@exercise("conditional_independence")
def _cond_indep():
    outs = [(a, b, c) for a in range(2) for b in range(2) for c in range(2)]
    sp = mod.sample_space(outs)
    a = mod.event(*[o for o in outs if o[0] == 0])
    b = mod.event(*[o for o in outs if o[1] == 0])
    c = mod.event(*[o for o in outs if o[2] == 0])
    assert mod.conditional_independence(a, b, c, sp)


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
