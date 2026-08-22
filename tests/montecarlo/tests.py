"""Tests for stochpylib.montecarlo.

Covers: quasi-random sequences (exact values, uniformity, balance, discrepancy vs
pseudo-random, scrambling), core estimators (known integrals within tolerance),
variance-reduction techniques (strict SE improvement + correctness), applications
(pi, option pricing vs Black-Scholes oracle, VaR/ES, reliability with library
distributions, sensitivity), determinism, and the export surface.

All randomness is seeded. Statistical tolerances are set at >= 3 standard errors so the
suite is deterministic in practice while remaining meaningful.
"""

import numpy as np
import pytest
from scipy import stats as spt
from scipy.stats import qmc as qmc_mod

import stochpylib
from stochpylib import montecarlo as M
from stochpylib.distributions import Beta, Exponential, LogNormal, Normal

# ---------------------------------------------------------------- export surface

SPEC_NAMES = {
    "simulate", "importance_sampling", "rejection_sampling", "stratified_sampling",
    "quasi_montecarlo", "crude_mc",
    "AntitheticVariates", "ControlVariates", "StratifiedSampling",
    "LatinHypercubeSampling", "OrthogonalSampling", "ConditionedMC", "RejectionControl",
    "SobolSequence", "HaltonSequence", "FaureSequence", "NiederreiterSequence",
    "DigitalNet", "LowDiscrepancy",
    "MonteCarloIntegration", "pi_estimation", "option_pricing_mc", "risk_analysis",
    "reliability_mc", "sensitivity_analysis",
}


def test_module_exports_match_spec():
    missing = SPEC_NAMES - set(M.__all__)
    assert not missing, f"missing exports: {missing}"
    assert stochpylib.montecarlo is M


# ---------------------------------------------------------------- sequences

VDC16 = np.array([sum(((i >> k) & 1) / 2 ** (k + 1) for k in range(20)) for i in range(1, 17)])


def test_sobol_dim1_is_van_der_corput_natural_order():
    pts = M.SobolSequence(1).generate(16)
    assert np.array_equal(pts[:, 0], VDC16)


def test_sobol_dim2_canonical_prefix():
    pts = M.SobolSequence(2).generate(3)
    assert np.allclose(pts, [[0.5, 0.5], [0.25, 0.75], [0.75, 0.25]])


@pytest.mark.parametrize("name", ["sobol", "halton", "faure", "niederreiter"])
def test_sequence_uniformity(name):
    pts = M.LowDiscrepancy(name, dim=6).generate(4096)
    assert pts.shape == (4096, 6)
    for d in range(6):
        p = spt.kstest(pts[:, d], "uniform").pvalue
        assert p > 1e-4, f"{name} dim {d}: KS p={p}"


def test_sobol_uses_standard_direction_number_table():
    s = M.SobolSequence(4)
    assert s.uses_standard_table and s.BITS == 30


def test_generate_block_exact_net_balance():
    # aligned blocks including the origin are exactly balanced in every dimension
    for m, dim in [(8, 4), (10, 6)]:
        p = M.SobolSequence(dim).generate_block(m)
        n = 1 << m
        for d in range(dim):
            assert int((p[:, d] < 0.5).sum()) == n // 2
            for k in range(8):
                cnt = int(((p[:, d] >= k / 8) & (p[:, d] < (k + 1) / 8)).sum())
                assert cnt == n // 8, f"m={m} dim={d} eighth {k}: {cnt}"


def test_block_matches_scipy_set_and_gray_order():
    m = 9
    ours = M.SobolSequence(4).generate_block(m)
    ref = qmc_mod.Sobol(d=4, scramble=False, bits=30).random(1 << m)
    assert np.array_equal(np.sort(ours, axis=0), np.sort(ref, axis=0))
    g = np.arange(1 << m) ^ (np.arange(1 << m) >> 1)
    assert np.array_equal(ours[g], ref)  # scipy enumerates along the Gray-code walk


@pytest.mark.parametrize("name", ["sobol", "halton", "faure", "niederreiter"])
def test_sequence_uniformity(name):
    pts = M.LowDiscrepancy(name, dim=6).generate(4096)
    assert pts.shape == (4096, 6)
    for d in range(6):
        p = spt.kstest(pts[:, d], "uniform").pvalue
        assert p > 1e-4, f"{name} dim {d}: KS p={p}"


def test_sequences_beat_pseudo_random_discrepancy():
    rand_disc = qmc_mod.discrepancy(np.random.default_rng(1).uniform(size=(512, 4)))
    worst = max(
        qmc_mod.discrepancy(cls(4).generate(512))
        for cls in (M.SobolSequence, M.HaltonSequence, M.NiederreiterSequence, M.FaureSequence)
    )
    assert worst < 0.95 * rand_disc


def test_faure_coordinates_distinct():
    pts = M.FaureSequence(4).generate(243)
    diffs = [np.max(np.abs(pts[:, i] - pts[:, j])) for i in range(4) for j in range(i + 1, 4)]
    assert min(diffs) > 0.01


def test_digital_shift_reproducible_and_effective():
    a = M.SobolSequence(3, random_state=42).generate(64)
    b = M.SobolSequence(3, random_state=42).generate(64)
    c = M.SobolSequence(3, random_state=43).generate(64)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    unshifted = M.SobolSequence(3).generate(64)
    assert not np.array_equal(a, unshifted)


def test_stream_continues_and_reset():
    stream = M.SobolSequence(2)
    p1 = stream.generate(4)
    p2 = stream.generate(4)
    assert np.allclose(np.vstack([p1, p2]), M.SobolSequence(2).generate(8))
    stream.reset()
    assert np.array_equal(stream.generate(4), p1)


def test_halton_exact_values():
    from stochpylib.montecarlo.quasi_random import radical_inverse

    pts = M.HaltonSequence(2).generate(2)
    assert np.isclose(pts[0, 0], 0.5) and np.isclose(pts[0, 1], 1 / 3)
    assert np.isclose(pts[1, 1], radical_inverse(2, 3))


# ---------------------------------------------------------------- estimators


def test_crude_mc_known_integral():
    res = M.crude_mc(lambda p: p[:, 0] ** 2, n=200_000, random_state=1)
    assert abs(res.estimate - 1 / 3) < 4 * res.std_error


def test_crude_mc_box_domain():
    exact = (1 - np.exp(-2)) * (1 - np.exp(-3))
    res = M.crude_mc(lambda p: np.exp(-p[:, 0] - p[:, 1]), dim=2,
                     bounds=[[0, 2], [0, 3]], n=200_000, random_state=2)
    assert abs(res.estimate - exact) < 4 * res.std_error


def test_quasi_montecarlo_more_accurate_than_crude():
    crude = M.crude_mc(lambda p: p[:, 0] ** 2, n=200_000, random_state=1)
    q = M.quasi_montecarlo(lambda p: p[:, 0] ** 2, n=4096)
    assert abs(q.estimate - 1 / 3) < abs(crude.estimate - 1 / 3)


def test_importance_sampling_beta_mean_and_ess():
    b = Beta(2.0, 4.0)
    res = M.importance_sampling(
        lambda p: p[:, 0],
        lambda p: b.pdf(p),
        lambda n_, rng_: rng_.uniform(size=(n_, 1)),
        lambda p: np.ones(p.shape[0]),
        n=200_000, random_state=3,
    )
    assert abs(res.estimate - 1 / 3) < 4 * res.std_error
    assert 0 < res.extras["ess"] <= 200_000


def test_rejection_sampling_matches_scipy_target():
    b = Beta(2.0, 4.0)
    samples, rate = M.rejection_sampling(
        b.pdf,
        lambda n_, rng_: rng_.uniform(size=(n_, 1)),
        lambda p: np.ones(p.shape[0]),
        n_samples=5000, k_bound=float(b.pdf(1 / 3)) * 1.05, random_state=5,
    )
    assert samples.shape == (5000, 1)
    assert 0 < rate <= 1
    assert spt.kstest(samples[:, 0], spt.beta(2, 4).cdf).pvalue > 1e-3


def test_stratified_sampling_improves_on_crude():
    integrand = lambda p: p[:, 0] ** 2
    crude = M.crude_mc(integrand, n=32_768, random_state=9)
    strat = M.stratified_sampling(integrand, bounds=(0, 1), n_strata=64,
                                  n_per_stratum=512, random_state=10)
    assert abs(strat.estimate - 1 / 3) < 4 * strat.std_error
    assert strat.std_error < crude.std_error


def test_simulate_driver_dice():
    res = M.simulate(statistic=lambda s: s, sampler=lambda rng: int(rng.integers(1, 7)),
                     n_simulations=50_000, random_state=11)
    assert abs(res.estimate - 3.5) < 4 * res.std_error


# ---------------------------------------------------------------- variance reduction

BS_CALL = None  # filled lazily via helper


def _bs_call(S, K, T, r, sigma):
    from stochpylib.montecarlo.variance_reduction import _black_scholes_price

    return _black_scholes_price(S, K, T, r, sigma, kind="call")


def test_antithetic_pricing_matches_black_scholes():
    bs = _bs_call(100, 100, 1.0, 0.05, 0.2)
    res = M.AntitheticVariates(n_simulations=200_000, random_state=12).price_european_call()
    assert abs(res.estimate - bs) < 3 * res.std_error
    assert float(res) == pytest.approx(res.estimate)


def test_antithetic_reduces_variance_vs_naive():
    rng = np.random.default_rng(13)
    z = rng.standard_normal(100_000)
    terminal = 100 * np.exp((0.05 - 0.02) + 0.2 * z)
    naive_se = float(np.exp(-0.05) * np.maximum(terminal - 100, 0).std(ddof=1) / np.sqrt(100_000))
    anti = M.AntitheticVariates(n_simulations=200_000, random_state=14).price_european_call()
    assert anti.std_error < naive_se


def test_control_variate_recovers_integral_with_reduction():
    cv = M.ControlVariates(n_simulations=100_000, random_state=15)
    res = cv.estimate(lambda p: p[:, 0] ** 2, lambda p: p[:, 0], control_mean=0.5)
    assert abs(res.estimate - 1 / 3) < 4 * res.std_error
    assert res.extras["variance_reduction"] > 1


def test_lhs_one_draw_per_stratum():
    lhs = M.LatinHypercubeSampling(dim=3, n=128, random_state=16).generate()
    for d in range(3):
        counts = np.histogram(lhs[:, d], bins=128, range=(0, 1))[0]
        assert set(counts) == {1}


def test_conditioned_mc_partial_normal_expectation():
    # E[(Y+Z)+] with Y,Z ~ N(0,1): m(y) = phi(y) + y*Phi(y); oracle 1/sqrt(pi)
    cm = M.ConditionedMC(n_simulations=50_000, random_state=17)
    res = cm.estimate(
        cond_expectation=lambda y: spt.norm.pdf(y) + y * spt.norm.cdf(y),
        y_sampler=lambda rng: rng.standard_normal(),
    )
    assert abs(res.estimate - 1 / np.sqrt(np.pi)) < 4 * res.std_error


def test_rejection_control_correct_and_reduces_variance():
    rc = M.RejectionControl(n_simulations=100_000, random_state=18)
    res = rc.estimate(
        lambda p: p[:, 0] ** 2,
        target_pdf=lambda p: spt.norm.pdf(p[:, 0]),
        proposal_sampler=lambda n_, rng_: rng_.normal(0.0, 2.0, size=(n_, 1)),
        proposal_pdf=lambda p: spt.norm.pdf(p[:, 0], scale=2.0),
    )
    assert abs(res.estimate - 1.0) < 4 * res.std_error
    assert res.extras["variance_reduction"] > 1


def test_orthogonal_sampling_shape_and_balance():
    pts = M.OrthogonalSampling(dim=2, n=256, random_state=19).generate()
    assert pts.shape == (256, 2)
    assert abs(int((pts[:, 0] < 0.5).sum()) - 128) <= 2


def test_stratified_grid_class_xy():
    sg = M.StratifiedSampling(n_strata=16, dim=2, n_per_stratum=32, random_state=20)
    res = sg.estimate(lambda p: p[:, 0] * p[:, 1])
    assert abs(res.estimate - 0.25) < 4 * res.std_error


# ---------------------------------------------------------------- applications


def test_monte_carlo_integration_class_qmc_and_crude():
    exact = 1 - np.cos(1.0)
    integ = M.MonteCarloIntegration(lambda p: np.sin(p[:, 0]), random_state=21)
    crude = integ.estimate(200_000)
    assert abs(crude.estimate - exact) < 4 * crude.std_error
    integ.method = "qmc"
    q = integ.estimate(4096)
    assert abs(q.estimate - exact) < abs(crude.estimate - exact)


def test_pi_estimation():
    res = M.pi_estimation(n=400_000, random_state=22)
    assert abs(res.estimate - np.pi) < 4 * res.std_error


def test_option_pricing_matches_black_scholes_both_kinds():
    from stochpylib.montecarlo.variance_reduction import _black_scholes_price

    for kind in ("call", "put"):
        bs = _black_scholes_price(100, 100, 1.0, 0.05, 0.2, kind=kind)
        res = M.option_pricing_mc(S=100, K=100, T=1, r=0.05, sigma=0.2, kind=kind,
                                  n=200_000, random_state=23)
        assert abs(res.estimate - bs) < 3 * res.std_error


def test_risk_analysis_var_es_normal():
    pnl = spt.norm.rvs(size=500_000, random_state=24)
    res = M.risk_analysis(pnl, alpha=0.95)
    assert isinstance(res.expected_shortfall, float)
    assert abs(res.estimate - spt.norm.ppf(0.95)) < 0.01
    assert res.expected_shortfall > res.estimate


def test_reliability_mc_single_component():
    true_p = 1 - np.exp(-1.5)
    res = M.reliability_mc(lambda X: X[:, 0], [Exponential(rate=0.5)],
                           threshold=3.0, n=300_000, random_state=25)
    assert abs(res.estimate - true_p) < 4 * res.std_error


def test_reliability_mc_series_system():
    a, b = np.exp(-1.0), np.exp(-1.4)
    true_p = (1 - a) + (1 - b) - (1 - a) * (1 - b)
    res = M.reliability_mc(lambda X: np.minimum(X[:, 0], X[:, 1]),
                           [Exponential(rate=0.5), Exponential(rate=0.7)],
                           threshold=2.0, n=300_000, random_state=26)
    assert abs(res.estimate - true_p) < 4 * res.std_error


def test_sensitivity_analysis_identifies_dominant_input():
    out = M.sensitivity_analysis(
        lambda X: 3.0 * X[:, 0] + X[:, 1],
        [LogNormal(0.0, 0.4), Normal(0.0, 1.0)],
        n=200_000, random_state=27,
    )
    assert out[0]["pearson"] > out[1]["pearson"]
    assert all(v["spearman"] >= v["pearson"] - 0.15 for v in out.values())


# ---------------------------------------------------------------- determinism


def test_determinism_same_seed_bitwise():
    f = lambda p: np.sin(3 * p[:, 0]) * np.exp(-p[:, 1])
    r1 = M.crude_mc(f, n=50_000, dim=2, random_state=99)
    r2 = M.crude_mc(f, n=50_000, dim=2, random_state=99)
    assert r1.estimate == r2.estimate and r1.std_error == r2.std_error


def test_result_object_contract():
    res = M.crude_mc(lambda p: p[:, 0], n=1000, random_state=31)
    assert isinstance(res, M.MCResult)
    assert float(res) == res.estimate
    lo, hi = res.confidence_interval(0.99)
    assert lo < res.estimate < hi
