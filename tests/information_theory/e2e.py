"""End-to-end API sweep for stochpylib.information_theory: one realistic exercise per
public name. ``test_every_public_name_is_exercised`` fails the moment a name ships without
one; ``test_exercise[<name>]`` runs each exercise as its own pytest case."""

import numpy as np
import pytest

from stochpylib import information_theory as mod

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


_RNG = np.random.default_rng(0)
X = _RNG.integers(0, 3, 600)
Y = (X + _RNG.integers(0, 2, 600)) % 3
Z = _RNG.integers(0, 2, 600)
P = [0.5, 0.25, 0.125, 0.125]
Q = [0.25, 0.25, 0.25, 0.25]


@exercise("Entropy")
def _entropy():
    assert mod.Entropy(base=2).fit(P).result_ == pytest.approx(1.75)


@exercise("JointEntropy")
def _joint():
    je = mod.JointEntropy().fit(X, Y).result_
    assert je >= mod.Entropy().fit(np.bincount(X)).result_ - 1e-9


@exercise("ConditionalEntropy")
def _conditional():
    assert 0 <= mod.ConditionalEntropy().fit(Y, X).result_ <= np.log2(3) + 1e-9


@exercise("CrossEntropy")
def _cross():
    assert mod.CrossEntropy(base=2).fit(P, Q).result_ == pytest.approx(2.0)


@exercise("TsallisEntropy")
def _tsallis():
    assert mod.TsallisEntropy(q=2.0).fit(Q).result_ == pytest.approx(0.75)


@exercise("RenyiEntropy")
def _renyi():
    assert mod.RenyiEntropy(alpha=2.0).fit(Q).result_ == pytest.approx(2.0)


@exercise("DifferentialEntropy")
def _differential():
    h = mod.DifferentialEntropy(n_bins=40).fit(_RNG.normal(0, 1, 5000)).result_
    assert abs(h - 0.5 * np.log2(2 * np.pi * np.e)) < 0.3   # bits


@exercise("MaxEntropy")
def _maxent():
    me = mod.MaxEntropy(support_size=8).fit()
    assert abs(me.result_ - np.log2(8)) < 0.01


@exercise("KLDivergence")
def _kl():
    assert mod.KLDivergence().fit(P, Q).result_ == pytest.approx(0.25)


@exercise("RelativeEntropy")
def _relative():
    assert mod.RelativeEntropy().fit(P, Q).result_ == pytest.approx(mod.KLDivergence().fit(P, Q).result_)


@exercise("JensenShannonDivergence")
def _js():
    assert 0 < mod.JensenShannonDivergence().fit(P, Q).result_ < 1


@exercise("WassersteinDistance")
def _wasserstein():
    assert mod.WassersteinDistance().fit(np.zeros(100), np.ones(100)).result_ == pytest.approx(1.0)


@exercise("HellingerDistance")
def _hellinger():
    assert 0 < mod.HellingerDistance().fit(P, Q).result_ < 1


@exercise("TotalVariation")
def _tv():
    assert mod.TotalVariation().fit(P, Q).result_ == pytest.approx(0.25)


@exercise("ChiSquaredDivergence")
def _chi2():
    assert mod.ChiSquaredDivergence().fit(P, Q).result_ > 0


@exercise("AlphaDivergence")
def _alpha():
    assert mod.AlphaDivergence(alpha=2.0).fit(P, Q).result_ > 0


@exercise("MutualInformation")
def _mi():
    assert mod.MutualInformation().fit(X, Y).result_ > 0.1


@exercise("NormalizedMutualInformation")
def _nmi():
    assert mod.NormalizedMutualInformation().fit(X, X).result_ == pytest.approx(1.0)


@exercise("VariationOfInformation")
def _vi():
    assert mod.VariationOfInformation().fit(X, X).result_ == pytest.approx(0.0, abs=1e-9)


@exercise("ConditionalMutualInfo")
def _cmi():
    assert mod.ConditionalMutualInfo().fit(X, Y, Z).result_ >= 0


@exercise("InteractionInformation")
def _interaction():
    assert np.isfinite(mod.InteractionInformation().fit(X, Y, Z).result_)


@exercise("MultiInformation")
def _multi():
    assert mod.MultiInformation().fit(X, Y, Z).result_ >= 0


@exercise("ChannelCapacity")
def _capacity():
    c = mod.ChannelCapacity("BSC", crossover_prob=0.1).fit().result_
    assert c == pytest.approx(1 - mod.Entropy(base=2).fit([0.1, 0.9]).result_)


@exercise("InformationGain")
def _ig():
    assert mod.InformationGain().fit(X, Y).result_ == pytest.approx(
        mod.MutualInformation().fit(X, Y).result_, abs=1e-9)


@exercise("TransferEntropy")
def _te():
    x = _RNG.normal(size=800)
    y = np.roll(x, 1) + 0.3 * _RNG.normal(size=800)
    assert mod.TransferEntropy(lag=1, n_bins=6).fit(x, y).result_ >= 0


@exercise("DirectedInformation")
def _di():
    x = _RNG.normal(size=500)
    assert mod.DirectedInformation(lag=1, n_bins=4).fit(x, np.roll(x, 1)).result_ >= 0


@exercise("SymbolicTransferEntropy")
def _ste():
    x = _RNG.normal(size=500)
    assert mod.SymbolicTransferEntropy(embedding_dim=3, lag=1).fit(x, np.roll(x, 1)).result_ >= 0


@exercise("ShannonLimit")
def _shannon():
    assert mod.ShannonLimit(crossover_prob=0.0).fit().result_ == pytest.approx(1.0)


@exercise("HuffmanCode")
def _huffman():
    hc = mod.HuffmanCode().fit(probs=P)
    assert hc.average_length_ == pytest.approx(1.75) and hc.is_optimal_


@exercise("TypicalSet")
def _typical():
    ts = mod.TypicalSet(epsilon=0.1).fit([0.5, 0.5])
    assert ts.is_typical([0, 1] * 10)


@exercise("AEP")
def _aep():
    aep = mod.AEP(epsilon=0.1).fit(P, block_length=50)
    assert aep.typical_set_size_lower_ > 0


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
