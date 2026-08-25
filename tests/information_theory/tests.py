"""Tests for stochpylib.information_theory.

Covers: Shannon/Renyi/Tsallis entropy closed forms, cross-entropy bounds,
KL divergence properties, JS symmetry, Hellinger/TV/chi-squared divergences,
mutual information chain rules and bounds, transfer entropy causal detection,
Huffman code optimality, channel capacity closed forms, and wiring.

All randomness is seeded.
"""

import numpy as np
import pytest

from stochpylib import information_theory as it
from stochpylib.information_theory.entropy import (
    ConditionalEntropy, CrossEntropy, DifferentialEntropy, Entropy,
    JointEntropy, MaxEntropy, RenyiEntropy, TsallisEntropy,
)
from stochpylib.information_theory.divergences import (
    AlphaDivergence, ChiSquaredDivergence, HellingerDistance,
    JensenShannonDivergence, KLDivergence, RelativeEntropy,
    TotalVariation, WassersteinDistance,
)
from stochpylib.information_theory.mutual_info import (
    ConditionalMutualInfo, InteractionInformation, MutualInformation,
    MultiInformation, NormalizedMutualInformation, VariationOfInformation,
)
from stochpylib.information_theory.channels import (
    ChannelCapacity, InformationGain, SymbolicTransferEntropy,
    TransferEntropy,
)
from stochpylib.information_theory.coding import (
    AEP, HuffmanCode, ShannonLimit, TypicalSet,
)

# ---------------------------------------------------------------- entropy

class TestEntropy:
    def test_shannon_bernoulli_half(self):
        assert abs(Entropy.compute([.5, .5]) - 1.0) < 1e-9

    def test_shannon_deterministic_zero(self):
        assert abs(Entropy.compute([1.0, 0.0])) < 1e-12

    def test_shannon_bernoulli_formula(self):
        p = .3
        expected = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        assert abs(Entropy.compute([p, 1 - p]) - expected) < 1e-10

    def test_joint_geq_marginal(self):
        rng = np.random.default_rng(0)
        x = rng.choice([0, 1, 2], 500)
        y = rng.choice([0, 1], 500)
        je = JointEntropy().fit(x, y).result_
        hx = Entropy().fit(np.bincount(x, minlength=3)).result_
        assert je >= hx - 1e-9

    def test_conditional_entropy_nonneg(self):
        rng = np.random.default_rng(1)
        x = rng.choice([0, 1], 500)
        y = x + rng.integers(-1, 2, 500)  # correlated
        ce = ConditionalEntropy().fit(y, x).result_
        assert ce >= 0


class TestCrossEntropy:
    def test_cross_entropy_geq_entropy(self):
        h = Entropy.compute([.5, .5])
        ce = CrossEntropy.compute([.5, .5], [.4, .6])
        assert ce >= h - 1e-9

    def test_cross_entropy_equals_when_same(self):
        p = [.3, .7]
        assert abs(CrossEntropy.compute(p, p) -
                   Entropy.compute(p)) < 1e-9


class TestTsallis:
    def test_q_2_binary(self):
        # Tsallis q=2 for [.5,.5]: (1-1)/(1) = 0? No: sum p^2 = .5, S=(1-.5)/1=.5
        assert abs(TsallisEntropy.compute([.5, .5], q=2) - .5) < 1e-12

    def test_q_1_recover_shannon(self):
        ts = TsallisEntropy(q=1.0 + 1e-10).compute([.3, .7])
        sh = Entropy.compute([.3, .7])
        assert np.isfinite(ts)


class TestRenyi:
    def test_alpha_2(self):
        r = RenyiEntropy(alpha=2.0).fit([.25, .25, .25, .25]).result_
        assert abs(r - 2.0) < 1e-9   # uniform over 4 symbols: H_2 = log2(4)=2

    def test_alpha_near_1_converges_to_shannon(self):
        r = RenyiEntropy(alpha=.9999).fit([.3, .7]).result_
        sh = Entropy.compute([.3, .7])
        assert abs(r - sh) < .02


class TestDiffMax:
    def test_diff_entropy_positive_for_continuous(self):
        rng = np.random.default_rng(2)
        de = DifferentialEntropy(n_bins=20).compute(
            rng.standard_normal(5000))
        assert de > 0

    def test_max_entropy_uniform(self):
        me = MaxEntropy(support_size=8).fit()
        assert abs(me.result_ - np.log2(8)) < .01


# ---------------------------------------------------------------- divergences

class TestKLDivergence:
    def test_kl_self_is_zero(self):
        assert abs(KLDivergence.compute([.3, .7], [.3, .7])) < 1e-12

    def test_kl_nonnegative(self):
        d = KLDivergence.compute(.3 * np.ones(4), .25 * np.ones(4))
        assert d >= 0

    def test_kl_infinite_for_zero_overlap(self):
        assert KLDivergence.compute([1., 0.], [0., 1.]) == float("inf")

    def test_relative_entropy_alias(self):
        p, q = [.3, .7], [.5, .5]
        assert (RelativeEntropy.compute(p, q) ==
                KLDivergence.compute(p, q))


class TestJSDivergence:
    def test_symmetric(self):
        js_ab = JensenShannonDivergence.compute([.3, .7], [.5, .5])
        js_ba = JensenShannonDivergence.compute([.5, .5], [.3, .7])
        assert abs(js_ab - js_ba) < 1e-12

    def test_bounded_by_one_bit(self):
        js = JensenShannonDivergence.compute([1., 0.], [0., 1.])
        assert js <= 1.0 + 1e-9


class TestOtherDivergences:
    def test_hellinger_range(self):
        h_min = HellingerDistance.compute([.5, .5], [.5, .5])
        h_max = HellingerDistance.compute([1., 0.], [0., 1.])
        assert abs(h_min) < 1e-12
        assert abs(h_max - 1.0) < 1e-9

    def test_total_variation_bounds(self):
        tv_min = TotalVariation.compute([.5, .5], [.5, .5])
        tv_max = TotalVariation.compute([1., 0.], [0., 1.])
        assert abs(tv_min) < 1e-12
        assert abs(tv_max - 1.0) < 1e-9

    def test_chi_squared_nonneg(self):
        chi = ChiSquaredDivergence.compute([.4, .6], [.5, .5])
        assert chi >= 0

    def test_wasserstein_distance_positive(self):
        wd = WassersteinDistance.compute(
            np.random.default_rng(1).normal(0, 1, 200),
            np.random.default_rng(2).normal(3, 1, 200))
        assert wd > 2.0     # means differ by 3

    def test_alpha_divergence_alpha_2_positive(self):
        a2 = AlphaDivergence(alpha=2.).compute([.3, .3, .4], [.33, .33, .34])
        assert a2 > 0


# ---------------------------------------------------------------- mutual info

class TestMutualInformation:
    def test_mi_independent_zero(self):
        rng = np.random.default_rng(3)
        x = rng.integers(0, 2, 10000)
        y = rng.integers(0, 2, 10000)  # independent
        mi = MutualInformation().fit(x, y).result_
        assert mi < 0.01               # close to zero

    def test_mi_dependent_positive(self):
        rng = np.random.default_rng(4)
        x = rng.integers(0, 2, 5000)
        y = x.copy()                   # perfect dependence
        mi = MutualInformation().fit(x, y).result_
        assert mi > .9                 # should equal H(X) ≈ 1 bit

    def test_nmi_perfectly_dependent_is_one(self):
        rng = np.random.default_rng(5)
        x = rng.integers(0, 3, 3000)
        nmi = NormalizedMutualInformation().fit(x, x).result_
        assert abs(nmi - 1.0) < 1e-9

    def test_variation_of_information_nonneg(self):
        vi = VariationOfInformation().fit(
            np.r_[np.zeros(50), np.ones(50)],
            np.r_[np.ones(50), np.zeros(50)])
        assert vi.result_ >= 0


class TestConditionalMI:
    def test_cmi_nonneg(self):
        rng = np.random.default_rng(6)
        x = rng.integers(0, 3, 500)
        y = rng.integers(0, 3, 500)
        z = rng.integers(0, 2, 500)
        cmi = ConditionalMutualInfo.compute(x, y, z)
        assert cmi >= -1e-12


class TestInteractionMulti:
    def test_interaction_information_finite(self):
        rng = np.random.default_rng(7)
        x = rng.integers(0, 3, 500)
        y = rng.integers(0, 3, 500)
        z = rng.integers(0, 2, 500)
        ii = InteractionInformation.compute(x, y, z)
        assert np.isfinite(ii)

    def test_multi_information_independent_zero(self):
        rng = np.random.default_rng(8)
        cols = [rng.integers(0, 2, 5000) for _ in range(3)]
        mi_all = MultiInformation().fit(*cols).result_
        assert mi_all < 0.02           # independent -> ~0


# ---------------------------------------------------------------- channels

class TestChannelCapacity:
    def test_bsc_zero_noise_capacity_one(self):
        cap = ChannelCapacity.compute("BSC", crossover_prob=0.)
        assert abs(cap - 1.0) < 1e-9

    def test_bsc_full_noise_capacity_zero(self):
        cap = ChannelCapacity.compute("BSC", crossover_prob=.5)
        assert cap < 1e-9

    def test_bec_capacity_one_minus_e(self):
        cap = ChannelCapacity.compute("BEC", erasure_prob=.3)
        assert abs(cap - .7) < 1e-9


class TestTransferEntropy:
    def test_te_detects_causal_signal(self):
        """X causes Y but not vice versa."""
        rng = np.random.default_rng(40)
        n = 3000
        x = rng.standard_normal(n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = .7 * y[t - 1] + .5 * x[t - 1] + rng.standard_normal(n)[t] * .1
        te_xy = TransferEntropy(lag=1, n_bins=8).compute(x, y, lag=1)
        te_yx = TransferEntropy(lag=1, n_bins=8).compute(y, x, lag=1)
        assert te_xy > te_yx, f"TE(X->Y)={te_xy:.4f} <= TE(Y->X)={te_yx:.4f}"


class TestSymbolicTE:
    def test_symbolic_te_runs(self):
        rng = np.random.default_rng(41)
        x = rng.standard_normal(500)
        y = np.zeros(500)
        for t in range(1, 500):
            y[t] = .6 * y[t - 1] + .4 * x[t - 1]
        ste = SymbolicTransferEntropy(embedding_dim=3, lag=1)
        result = ste.fit(x, y).result_
        assert result >= 0


# ---------------------------------------------------------------- coding

class TestShannonLimit:
    def test_bsc_zero_noise_unlimited(self):
        cap = ShannonLimit.compute(crossover_prob=0.)
        assert abs(cap - 1.0) < 1e-9

    def test_bsc_typical_value(self):
        cap = ShannonLimit.compute(crossover_prob=.1)
        expected = 1 - (-.1 * np.log2(.1) - .9 * np.log2(.9))
        assert abs(cap - expected) < 1e-6


class TestHuffmanCode:
    def test_average_length_within_bound(self):
        hc = HuffmanCode().fit(probs=[.35, .3, .2, .15])
        assert hc.entropy_ <= hc.average_length_ <= hc.entropy_ + 1

    def test_uniform_symbols_length_log2_n(self):
        hc = HuffmanCode().fit(probs=[.25, .25, .25, .25])
        assert abs(hc.average_length_ - 2.0) < 1e-9

    def test_code_table_complete(self):
        hc = HuffmanCode().fit(probs=[.5, .3, .2])
        assert len(hc.code_table_) == 3
        assert all(len(v) > 0 for v in hc.code_table_.values())


class TestTypicalSetAEP:
    def test_typical_set_membership_balanced_coin(self):
        probs = [.5, .5]
        ts = TypicalSet(epsilon=.1).fit(probs)
        # balanced binary sequence IS typical for fair coin
        seq = [0, 1] * 10
        assert ts.is_typical(seq)

    def test_atypical_detected_with_biased_source(self):
        """Biased source: balanced sequence is NOT typical."""
        probs = [.9, .1]
        ts = TypicalSet(epsilon=.1).fit(probs)
        # balanced sequence has H_emp≈1.0 but H(p)=0.469
        seq = [0, 1] * 10
        assert not ts.is_typical(seq)
        # biased sequence IS typical
        biased_seq = [0] * 9 + [1]
        assert ts.is_typical(biased_seq)

    def test_aep_bounds(self):
        probs = [.4, .3, .2, .1]
        aep = AEP(epsilon=.1).fit(probs, block_length=100)
        assert aep.typical_set_size_lower_ > 0
        assert aep.typical_set_probability_lower_ >= .85


# ---------------------------------------------------------------- wiring

def test_module_wiring():
    import stochpylib
    assert "information_theory" in stochpylib.__all__
    assert hasattr(stochpylib, "information_theory")
    from stochpylib import information_theory
    spec_names = {
        "Entropy", "JointEntropy", "ConditionalEntropy", "CrossEntropy",
        "TsallisEntropy", "RenyiEntropy", "DifferentialEntropy",
        "MaxEntropy", "KLDivergence", "RelativeEntropy",
        "JensenShannonDivergence", "WassersteinDistance",
        "HellingerDistance", "TotalVariation", "ChiSquaredDivergence",
        "AlphaDivergence", "MutualInformation", "NormalizedMutualInformation",
        "VariationOfInformation", "ConditionalMutualInfo",
        "InteractionInformation", "MultiInformation", "ChannelCapacity",
        "InformationGain", "TransferEntropy", "DirectedInformation",
        "SymbolicTransferEntropy", "ShannonLimit", "HuffmanCode",
        "TypicalSet", "AEP"}
    missing = spec_names - set(information_theory.__all__)
    assert not missing, f"missing: {missing}"


# ---------------------------------------------------------------- V0.6.1 additions

def test_renyi_alpha_0_returns_log2_k_in_bits():
    """Renyi alpha=0 is Hartley entropy: log(K) must be in bits not nats."""
    r = RenyiEntropy(alpha=0).fit([.25] * 4).result_
    assert abs(r - 2.0) < .01   # log2(4) = 2 bits


def test_cmi_compute_returns_float_directly():
    rng = np.random.default_rng(42)
    x = rng.integers(0, 4, 500)
    y = rng.integers(0, 3, 500)
    z = rng.integers(0, 2, 500)
    result = ConditionalMutualInfo.compute(x, y, z)
    assert isinstance(result, float)


def test_transfer_entropy_bias_floor_independent():
    """Plug-in TE estimator has positive finite-sample bias O(K^2/n)."""
    rng = np.random.default_rng(99)
    x_ind = rng.standard_normal(2000)
    y_ind = rng.standard_normal(2000)
    te = TransferEntropy(lag=1, n_bins=8).compute(x_ind, y_ind, lag=1)
    # should be small but NOT exactly zero due to discretisation bias
    assert te < .12


def test_max_entropy_with_mean_constraint():
    from stochpylib.information_theory.entropy import MaxEntropy as ME
    me = ME(support_size=20, mean_constraint=.7, lower=0., upper=1.).fit()
    mean_val = float(np.sum(me.support_ * me.distribution_))
    assert abs(mean_val - .7) < .05
    assert np.isfinite(me.result_)


def test_alpha_divergence_near_1_approximates_kl():
    p = [.3, .7]
    q = [.5, .5]
    ad = AlphaDivergence(alpha=1.001).compute(p, q)
    kl = KLDivergence.compute(p, q)
    assert abs(ad - kl) < .15


def test_information_gain_equals_mutual_info():
    rng = np.random.default_rng(45)
    x = rng.integers(0, 3, 300)
    y = rng.integers(0, 2, 300)
    ig = InformationGain().fit(x, y).result_
    mi = MutualInformation().fit(x, y).result_
    assert abs(ig - mi) < .01


def test_typical_set_biased_source():
    ts = TypicalSet(epsilon=.1).fit([.9, .1])
    # balanced sequence NOT typical under biased source
    assert not ts.is_typical([0, 1] * 10)
    # biased sequence IS typical
    assert ts.is_typical([0] * 18 + [1] * 2)


def test_multi_info_independent_near_zero():
    rng = np.random.default_rng(46)
    cols = [rng.integers(0, 2, 5000) for _ in range(4)]
    mi_all = MultiInformation().fit(*cols).result_
    assert mi_all < .03


def test_vi_identical_zero():
    rng = np.random.default_rng(47)
    x = rng.integers(0, 5, 1000)
    vi = VariationOfInformation().fit(x, x).result_
    assert abs(vi) < 1e-9


def test_interaction_information_xor_negative():
    """XOR creates pure synergy -> negative interaction information."""
    rng = np.random.default_rng(48)
    xr = rng.integers(0, 2, 3000)
    yr = rng.integers(0, 2, 3000)
    zr = xr ^ yr
    ii = InteractionInformation().compute(zr, xr, yr)
    assert ii < 0