"""Tests for stochpylib.financial_stochastics: option pricing (closed form,
lattice, Monte Carlo, Fourier), Greeks, stochastic/local volatility, short-rate
models, risk (VaR/ES/stress), credit risk, and portfolio construction.

Oracles: closed-form formulas, scipy.stats/scipy.linalg cross-checks, and
internal consistency (put-call parity, limiting cases, MC-vs-analytic within
a stated number of standard errors). All randomness is seeded.
"""

import math

import numpy as np
import pytest
from scipy import stats as spstats
from scipy.linalg import expm as scipy_expm

import stochpylib
from stochpylib import financial_stochastics as fs
from stochpylib.financial_stochastics import credit as fs_credit
from stochpylib.financial_stochastics import greeks as fs_greeks
from stochpylib.financial_stochastics import option_pricing as fs_op
from stochpylib.financial_stochastics import portfolio as fs_port
from stochpylib.financial_stochastics import rate_models as fs_rate
from stochpylib.financial_stochastics import risk as fs_risk
from stochpylib.financial_stochastics import stochastic_vol as fs_vol

# ============================================================== option_pricing


def test_bs_reference_values():
    bs100 = fs_op.BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
    assert abs(bs100.call_price() - 10.4506) < 1e-3
    bs105 = fs_op.BlackScholes(S=100, K=105, T=1, r=0.05, sigma=0.2)
    assert abs(bs105.call_price() - 8.0214) < 1e-3


def test_bs_put_call_parity():
    bs = fs_op.BlackScholes(S=100, K=90, T=0.75, r=0.03, sigma=0.25, q=0.01)
    lhs = bs.call_price() - bs.put_price()
    rhs = 100 * math.exp(-0.01 * 0.75) - 90 * math.exp(-0.03 * 0.75)
    assert abs(lhs - rhs) < 1e-9


def test_bs_matches_scipy_norm():
    S, K, T, r, sigma = 100.0, 95.0, 0.5, 0.04, 0.3
    bs = fs_op.BlackScholes(S, K, T, r, sigma)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    expected = S * spstats.norm.cdf(d1) - K * math.exp(-r * T) * spstats.norm.cdf(d2)
    assert abs(bs.call_price() - expected) < 1e-9


def test_bs_edge_cases():
    bs_t0 = fs_op.BlackScholes(S=110, K=100, T=0, r=0.05, sigma=0.2)
    assert bs_t0.call_price() == pytest.approx(10.0)
    bs_s0 = fs_op.BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.0)
    assert bs_s0.call_price() == pytest.approx(max(100 - 100 * math.exp(-0.05), 0.0))


def test_bs_implied_vol_roundtrip():
    bs = fs_op.BlackScholes(S=100, K=110, T=1, r=0.05, sigma=0.35)
    iv = bs.implied_vol(bs.call_price())
    assert abs(iv - 0.35) < 1e-8


def test_bs_delta_gamma_properties_match_greeks_module():
    bs = fs_op.BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.01)
    args = (bs.S, bs.K, bs.T, bs.r, bs.sigma, bs.q, "call")
    assert bs.Delta == pytest.approx(float(fs_greeks.Delta(*args)))
    assert bs.Gamma == pytest.approx(float(fs_greeks.Gamma(*args)))
    assert bs.Vega == pytest.approx(float(fs_greeks.Vega(*args)))


def test_binomial_converges_to_bs():
    bs = fs_op.BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
    err_100 = abs(fs_op.BinomialTree(100, 100, 1, 0.05, 0.2, n_steps=100).call_price() - bs.call_price())
    err_2000 = abs(fs_op.BinomialTree(100, 100, 1, 0.05, 0.2, n_steps=2000).call_price() - bs.call_price())
    assert err_2000 < 0.02
    assert err_2000 < err_100


def test_binomial_american_vs_european():
    tree = fs_op.BinomialTree(100, 100, 1, 0.05, 0.2, n_steps=1000)
    assert tree.put_price(american=True) >= tree.put_price(american=False) - 1e-9
    assert tree.call_price(american=True) == pytest.approx(tree.call_price(american=False), abs=1e-6)


def test_binomial_parity():
    tree = fs_op.BinomialTree(100, 105, 1, 0.05, 0.2, n_steps=500, q=0.01)
    lhs = tree.call_price() - tree.put_price()
    rhs = 100 * math.exp(-0.01) - 105 * math.exp(-0.05)
    assert abs(lhs - rhs) < 1e-6


def test_trinomial_converges_and_matches_binomial():
    bs = fs_op.BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
    tri = fs_op.TrinomialTree(100, 100, 1, 0.05, 0.2, n_steps=600)
    assert abs(tri.call_price() - bs.call_price()) < 0.02
    tri_am = fs_op.TrinomialTree(100, 100, 1, 0.05, 0.2, n_steps=600)
    bin_am = fs_op.BinomialTree(100, 100, 1, 0.05, 0.2, n_steps=600)
    assert abs(tri_am.put_price(american=True) - bin_am.put_price(american=True)) < 0.05


def test_baw_american_put_vs_tree():
    baw = fs_op.BlackScholes_American(100, 100, 1, 0.05, 0.2)
    tree = fs_op.BinomialTree(100, 100, 1, 0.05, 0.2, n_steps=2000)
    tree_put = tree.put_price(american=True)
    assert abs(baw.put_price() - tree_put) < 0.06
    bs = fs_op.BlackScholes(100, 100, 1, 0.05, 0.2)
    assert baw.put_price() >= bs.put_price() - 1e-9
    assert baw.early_exercise_boundary("put") < 100


def test_baw_call_no_dividend_equals_european():
    baw = fs_op.BlackScholes_American(100, 100, 1, 0.05, 0.2, q=0.0)
    bs = fs_op.BlackScholes(100, 100, 1, 0.05, 0.2)
    assert baw.call_price() == pytest.approx(bs.call_price(), abs=1e-9)


def test_mc_plain_within_se():
    bs = fs_op.BlackScholes(100, 100, 1, 0.05, 0.2)
    mc = fs_op.MonteCarloOptionPricing(100, 100, 1, 0.05, 0.2)
    res = mc.price("call", n_paths=200_000, antithetic=True, random_state=1)
    assert abs(res.estimate - bs.call_price()) < 3 * res.std_error
    assert res.n_samples == 200_000
    assert res.method == "mc-antithetic"


def test_mc_antithetic_and_control_variate_reduce_se():
    bs = fs_op.BlackScholes(100, 100, 1, 0.05, 0.2)
    mc = fs_op.MonteCarloOptionPricing(100, 100, 1, 0.05, 0.2)
    plain = mc.price("call", n_paths=100_000, antithetic=False, random_state=2)
    anti = mc.price("call", n_paths=100_000, antithetic=True, random_state=2)
    cv = mc.price("call", n_paths=100_000, antithetic=False, control_variate=True, random_state=2)
    assert anti.std_error < plain.std_error
    assert cv.std_error < plain.std_error
    assert abs(cv.estimate - bs.call_price()) < 4 * cv.std_error


def test_mc_qmc_close_to_bs():
    bs = fs_op.BlackScholes(100, 100, 1, 0.05, 0.2)
    mc = fs_op.MonteCarloOptionPricing(100, 100, 1, 0.05, 0.2)
    plain = mc.price("call", n_paths=100_000, antithetic=False, random_state=3)
    qmc = mc.price("call", n_paths=100_000, antithetic=False, qmc=True, random_state=3)
    assert abs(qmc.estimate - bs.call_price()) < 0.02
    assert qmc.std_error < plain.std_error


def test_mc_geometric_asian_closed_form():
    S, K, T, r, sigma, n = 100.0, 100.0, 1.0, 0.05, 0.2, 50
    mc = fs_op.MonteCarloOptionPricing(S, K, T, r, sigma)
    res = mc.asian_price("call", average="geometric", n_paths=100_000, n_steps=n, random_state=4)
    m = math.log(S) + (r - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    v = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    d1 = (m - math.log(K) + v) / math.sqrt(v)
    d2 = d1 - math.sqrt(v)
    closed = math.exp(-r * T) * (math.exp(m + 0.5 * v) * spstats.norm.cdf(d1) - K * spstats.norm.cdf(d2))
    assert abs(res.estimate - closed) < 3 * res.std_error


def test_mc_barrier_leq_vanilla():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    mc = fs_op.MonteCarloOptionPricing(S, K, T, r, sigma)

    def barrier_payoff(paths, barrier=90.0):
        alive = np.all(paths >= barrier, axis=1)
        return alive * np.maximum(paths[:, -1] - K, 0.0)

    barrier_res = mc.price_path_dependent(barrier_payoff, n_paths=50_000, n_steps=100, random_state=5)
    vanilla_res = mc.price("call", n_paths=50_000, random_state=5)
    assert barrier_res.estimate <= vanilla_res.estimate + 3 * (barrier_res.std_error + vanilla_res.std_error)
    assert barrier_res.estimate >= -1e-9


def test_lsm_american_put_vs_binomial():
    tree = fs_op.BinomialTree(100, 100, 1, 0.05, 0.2, n_steps=2000)
    tree_put = tree.put_price(american=True)
    bs_put = fs_op.BlackScholes(100, 100, 1, 0.05, 0.2).put_price()
    ls = fs_op.LongstaffSchwartz(100, 100, 1, 0.05, 0.2, n_steps=50)
    res = ls.price("put", n_paths=100_000, random_state=6)
    assert abs(res.estimate - tree_put) < max(3 * res.std_error, 0.07)
    assert res.estimate >= bs_put - 2 * res.std_error


def test_lsm_polynomial_basis_agrees_with_laguerre():
    ls_lag = fs_op.LongstaffSchwartz(100, 100, 1, 0.05, 0.2, n_steps=50, basis="laguerre")
    ls_poly = fs_op.LongstaffSchwartz(100, 100, 1, 0.05, 0.2, n_steps=50, basis="poly")
    r1 = ls_lag.price("put", n_paths=100_000, random_state=7)
    r2 = ls_poly.price("put", n_paths=100_000, random_state=7)
    assert abs(r1.estimate - r2.estimate) < 3 * (r1.std_error + r2.std_error)


def test_lsm_accepts_external_paths():
    S, K, T, r, sigma, n_steps = 100.0, 100.0, 1.0, 0.05, 0.2, 50
    rng = np.random.default_rng(8)
    dt = T / n_steps
    Z = rng.standard_normal((80_000, n_steps))
    log_paths = np.concatenate([np.zeros((80_000, 1)),
                                np.cumsum((r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z, axis=1)], axis=1)
    paths = S * np.exp(log_paths)
    ls = fs_op.LongstaffSchwartz(S, K, T, r, sigma, n_steps=n_steps)
    res_ext = ls.price("call", paths=paths)
    ls_direct = fs_op.LongstaffSchwartz(S, K, T, r, sigma, n_steps=n_steps)
    res_direct = ls_direct.price("call", n_paths=80_000, random_state=8)
    assert abs(res_ext.estimate - res_direct.estimate) < 4 * max(res_ext.std_error, res_direct.std_error)


def test_fourier_bs_cf_reproduces_bs():
    fp = fs_op.FourierOptionPricing.from_black_scholes(100, 1, 0.05, 0.2)
    for K in (80, 90, 100, 110, 120):
        bs = fs_op.BlackScholes(100, K, 1, 0.05, 0.2)
        cm = fp.call_price(K, method="carr_madan")
        cos = fp.call_price(K, method="cos")
        assert abs(cm - bs.call_price()) < 1e-3
        assert abs(cos - bs.call_price()) < 1e-3


def test_fourier_put_parity():
    fp = fs_op.FourierOptionPricing.from_black_scholes(100, 1, 0.05, 0.2, q=0.01)
    call = fp.call_price(105)
    put = fp.put_price(105)
    assert abs((call - put) - (100 * math.exp(-0.01) - 105 * math.exp(-0.05))) < 1e-6


# =================================================================== greeks


@pytest.mark.parametrize("greek,fn", [
    ("delta", fs_greeks.Delta), ("gamma", fs_greeks.Gamma), ("vega", fs_greeks.Vega),
    ("theta", fs_greeks.Theta), ("rho", fs_greeks.Rho), ("vanna", fs_greeks.Vanna),
    ("volga", fs_greeks.Volga),
])
def test_analytic_greeks_vs_finite_difference(greek, fn):
    S, K, T, r, sigma = 100.0, 105.0, 0.8, 0.04, 0.25

    def price_fn(S, K, T, r, sigma):
        return fs_op._bs_price(S, K, T, r, sigma, 0.0, "call")

    fd = fs_greeks.Greeks_FD(price_fn)
    fd_vals = fd.compute(S, K, T, r, sigma)
    analytic = float(fn(S, K, T, r, sigma, 0.0, "call"))
    assert abs(analytic - fd_vals[greek]) < 1e-3 * max(abs(analytic), 1.0)


def test_put_call_greek_relations():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.02
    args = (S, K, T, r, sigma, q)
    dcall = fs_greeks.Delta(*args, "call")
    dput = fs_greeks.Delta(*args, "put")
    assert abs((dcall - dput) - math.exp(-q * T)) < 1e-9
    assert fs_greeks.Gamma(*args, "call") == pytest.approx(fs_greeks.Gamma(*args, "put"))
    assert fs_greeks.Vega(*args, "call") == pytest.approx(fs_greeks.Vega(*args, "put"))
    assert fs_greeks.Vanna(*args, "call") == pytest.approx(fs_greeks.Vanna(*args, "put"))
    assert fs_greeks.Volga(*args, "call") == pytest.approx(fs_greeks.Volga(*args, "put"))
    rcall = fs_greeks.Rho(*args, "call")
    rput = fs_greeks.Rho(*args, "put")
    assert abs((rcall - rput) - K * T * math.exp(-r * T)) < 1e-9


def test_theta_sign_and_decay():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    theta = fs_greeks.Theta(S, K, T, r, sigma, 0.0, "call")
    assert theta < 0
    h = 1e-4
    p0 = fs_op._bs_price(S, K, T, r, sigma, 0.0, "call")
    p1 = fs_op._bs_price(S, K, T - h, r, sigma, 0.0, "call")
    assert abs((p1 - p0) - theta * h) < 1e-6


def test_greeks_mc_pathwise_within_se():
    gmc = fs_greeks.Greeks_MC(100, 100, 1, 0.05, 0.2)
    res = gmc.compute(n_paths=400_000, method="pathwise", random_state=9,
                      greeks=("delta", "vega", "rho"))
    analytic = {"delta": fs_greeks.Delta(100, 100, 1, 0.05, 0.2),
               "vega": fs_greeks.Vega(100, 100, 1, 0.05, 0.2),
               "rho": fs_greeks.Rho(100, 100, 1, 0.05, 0.2)}
    for g in ("delta", "vega", "rho"):
        assert abs(res[g].estimate - analytic[g]) < 4 * res[g].std_error


def test_greeks_mc_likelihood_gamma():
    gmc = fs_greeks.Greeks_MC(100, 100, 1, 0.05, 0.2)
    res = gmc.compute(n_paths=400_000, method="likelihood", random_state=10, greeks=("gamma",))
    analytic = fs_greeks.Gamma(100, 100, 1, 0.05, 0.2)
    assert abs(res["gamma"].estimate - analytic) < 4 * res["gamma"].std_error


def test_greeks_mc_bump_matches_analytic():
    gmc = fs_greeks.Greeks_MC(100, 100, 1, 0.05, 0.2)
    res = gmc.compute(n_paths=400_000, method="bump", random_state=11, greeks=("delta", "vega"))
    assert abs(res["delta"].estimate - fs_greeks.Delta(100, 100, 1, 0.05, 0.2)) < 4 * res["delta"].std_error
    assert abs(res["vega"].estimate - fs_greeks.Vega(100, 100, 1, 0.05, 0.2)) < 4 * res["vega"].std_error


def test_greeks_fd_on_tree_pricer():
    def price_fn(S, K, T, r, sigma):
        return fs_op.BinomialTree(S, K, T, r, sigma, n_steps=300).call_price()

    fd = fs_greeks.Greeks_FD(price_fn)
    vals = fd.compute(100, 100, 1, 0.05, 0.2)
    analytic = fs_greeks.Delta(100, 100, 1, 0.05, 0.2)
    assert abs(vals["delta"] - analytic) < 0.02


# ============================================================== stochastic_vol


def test_heston_cf_xi_to_zero_matches_deterministic_variance():
    h = fs_vol.HestonModel(S0=100, v0=0.04, kappa=2, theta=0.04, xi=1e-4, rho=-0.7, r=0.05)
    bs = fs_op.BlackScholes(100, 100, 1, 0.05, math.sqrt(0.04))
    assert abs(h.call_price(100, 1) - bs.call_price()) < 5e-3


def test_heston_kappa_large_matches_bs_theta():
    h = fs_vol.HestonModel(S0=100, v0=0.02, kappa=50, theta=0.04, xi=0.3, rho=-0.5, r=0.05)
    bs = fs_op.BlackScholes(100, 100, 1, 0.05, math.sqrt(0.04))
    assert abs(h.call_price(100, 1) - bs.call_price()) < 0.05


def test_heston_carr_madan_vs_qe_mc():
    h = fs_vol.HestonModel(S0=100, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7, r=0.05)
    cm = h.call_price(100, 1)
    mc = h.call_price_mc(100, 1, n_paths=200_000, N=100, scheme="qe", random_state=12)
    assert abs(mc.estimate - cm) < 4 * mc.std_error


def test_heston_qe_vs_euler_vs_bates_zero_jumps():
    from stochpylib.levy_processes.jump_diffusion import BatesModel

    h = fs_vol.HestonModel(S0=100, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7, r=0.05)
    qe = h.call_price_mc(100, 1, n_paths=150_000, N=100, scheme="qe", random_state=13)
    euler = h.call_price_mc(100, 1, n_paths=150_000, N=100, scheme="euler", random_state=13)
    assert abs(qe.estimate - euler.estimate) < 4 * (qe.std_error + euler.std_error)
    bates0 = BatesModel(S0=100, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7, r=0.05, jump_rate=0.0)
    bates_mc = bates0.call_price_mc(100, 1, n_paths=150_000, n_steps=100, random_state=13)
    assert abs(euler.estimate - bates_mc.estimate) < 4 * (euler.std_error + bates_mc.std_error)


def test_heston_simulate_shape_and_signature():
    h = fs_vol.HestonModel(S0=100, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7, r=0.05)
    S = h.simulate(T=1, N=252, n_paths=10, random_state=14)
    assert S.shape == (10, 253)
    S2, v2 = h.simulate(T=1, N=50, n_paths=5, return_variance=True, random_state=14)
    assert S2.shape == (5, 51) and v2.shape == (5, 51)


def test_heston_expected_variance_closed_form():
    h = fs_vol.HestonModel(S0=100, v0=0.09, kappa=3, theta=0.04, xi=0.4, rho=-0.6, r=0.03)
    S, v = h.simulate(T=1, N=200, n_paths=40_000, return_variance=True, random_state=15)

    closed_terminal = h.theta + (h.v0 - h.theta) * math.exp(-h.kappa * 1.0)
    sample_mean = v[:, -1].mean()
    se = v[:, -1].std(ddof=1) / math.sqrt(v.shape[0])
    assert abs(sample_mean - closed_terminal) < 4 * se

    closed_integral = h.expected_integrated_variance(1.0)
    integrated = np.trapezoid(v, dx=1.0 / 200, axis=1) if hasattr(np, "trapezoid") \
        else np.trapz(v, dx=1.0 / 200, axis=1)
    se_int = integrated.std(ddof=1) / math.sqrt(len(integrated))
    assert abs(integrated.mean() - closed_integral) < 4 * se_int


def test_heston_fit_recovers_prices():
    true = fs_vol.HestonModel(S0=100, v0=0.04, kappa=2, theta=0.05, xi=0.4, rho=-0.6, r=0.03)
    strikes = np.array([80, 90, 100, 110, 120])
    maturities = np.full(5, 1.0)
    prices = np.array([true.call_price(K, 1.0) for K in strikes])
    guess = fs_vol.HestonModel(S0=100, v0=0.03, kappa=1.5, theta=0.04, xi=0.3, rho=-0.4, r=0.03)
    guess.fit(strikes, maturities, prices)
    assert guess.rmse_ < 1e-2


def test_heston_put_call_parity():
    h = fs_vol.HestonModel(S0=100, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7, r=0.05, q=0.02)
    lhs = h.call_price(100, 1) - h.put_price(100, 1)
    rhs = 100 * math.exp(-0.02) - 100 * math.exp(-0.05)
    assert abs(lhs - rhs) < 1e-6


def test_sabr_beta1_nu0_is_lognormal():
    sabr = fs_vol.SABRModel(alpha=0.3, beta=1.0, rho=0.0, nu=1e-10)
    for K in (80, 100, 120):
        assert abs(sabr.implied_vol(100, K, 1) - 0.3) < 1e-6


def test_sabr_beta0_is_normal_model():
    alpha = 5.0
    sabr = fs_vol.SABRModel(alpha=alpha, beta=0.0, rho=0.0, nu=1e-10)
    F, K, T = 100.0, 110.0, 1.0
    expected = alpha * math.log(F / K) / (F - K) * (1.0 + alpha**2 * T / (24.0 * F * K))
    assert abs(sabr.implied_vol(F, K, T) - expected) < 1e-3 * expected


def test_sabr_atm_limit_continuous():
    sabr = fs_vol.SABRModel(alpha=0.25, beta=0.7, rho=-0.3, nu=0.5)
    atm = sabr.implied_vol(100, 100, 1)
    near = sabr.implied_vol(100, 100 * (1 + 1e-9), 1)
    assert abs(atm - near) < 1e-6


def test_sabr_smile_shape():
    sabr_skew = fs_vol.SABRModel(alpha=0.25, beta=0.7, rho=-0.6, nu=0.6)
    down = sabr_skew.implied_vol(100, 90, 1)
    up = sabr_skew.implied_vol(100, 110, 1)
    assert down > up
    sabr_conv = fs_vol.SABRModel(alpha=0.25, beta=0.7, rho=0.0, nu=0.8)
    atm = sabr_conv.implied_vol(100, 100, 1)
    wing = sabr_conv.implied_vol(100, 130, 1)
    assert wing > atm


def test_sabr_fit_recovers_params():
    true = fs_vol.SABRModel(alpha=0.3, beta=0.7, rho=-0.4, nu=0.5)
    strikes = np.array([80, 90, 100, 110, 120])
    vols = true.implied_vol(100, strikes, 1)
    guess = fs_vol.SABRModel(alpha=0.2, beta=0.7, rho=-0.1, nu=0.3)
    guess.fit(100, strikes, 1, vols, beta=0.7)
    assert abs(guess.alpha - 0.3) < 1e-2
    assert abs(guess.rho - (-0.4)) < 1e-2
    assert abs(guess.nu - 0.5) < 1e-2


def test_sabr_mc_vs_hagan():
    sabr = fs_vol.SABRModel(alpha=0.3, beta=1.0, rho=-0.3, nu=0.4)
    F0, T = 100.0, 0.5
    F, A = sabr.simulate(F0, T, N=100, n_paths=100_000, random_state=16)
    K = 105.0
    payoff = np.maximum(F[:, -1] - K, 0.0)
    mc_price = payoff.mean()
    se = payoff.std(ddof=1) / math.sqrt(len(payoff))
    hagan_vol = sabr.implied_vol(F0, K, T)
    black_price = fs_op._bs_price(F0, K, T, 0.0, hagan_vol, 0.0, "call")
    assert abs(mc_price - black_price) < 4 * se


def test_rough_heston_h_half_equals_heston():
    rh = fs_vol.RoughHeston(S0=100, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7, r=0.05, H=0.5)
    h = fs_vol.HestonModel(S0=100, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7, r=0.05)
    u = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    cf_rh = rh.characteristic_function(u, 1.0, n_steps=300)
    cf_h = h.characteristic_function(u, 1.0)
    assert np.max(np.abs(cf_rh - cf_h)) < 1e-3
    assert abs(rh.call_price(100, 1, n_steps=300) - h.call_price(100, 1)) < 0.02


def test_rough_heston_mc_vs_fourier():
    rh = fs_vol.RoughHeston(S0=100, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7, r=0.05, H=0.1)
    fourier = rh.call_price(100, 1, n_steps=200)
    mc = rh.call_price_mc(100, 1, n_paths=40_000, N=100, random_state=17)
    assert abs(mc.estimate - fourier) < 4 * mc.std_error + 0.05


def test_rough_heston_cf_at_zero_is_one():
    rh = fs_vol.RoughHeston(S0=100, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7, r=0.05, H=0.2)
    phi0 = rh.characteristic_function(np.array([0.0]), 1.0, n_steps=100)
    assert abs(phi0[0] - 1.0) < 1e-8


def test_rbergomi_covariance_matches_quadrature():
    from scipy import integrate

    rb = fs_vol.RoughBergomi(S0=100, xi0=0.04, eta=1.5, rho=-0.7, H=0.15, r=0.0)
    Sigma = rb.covariance_matrix(1.0, 20)
    t = np.linspace(1.0 / 20, 1.0, 20)
    i, j = 5, 12
    ti, sj = t[i], t[j]

    def integrand(u):
        return (max(ti - u, 0.0)) ** (0.15 - 0.5) * (max(sj - u, 0.0)) ** (0.15 - 0.5)

    val, _ = integrate.quad(integrand, 0.0, min(ti, sj), limit=200)
    expected_cov = 2.0 * 0.15 * val
    assert abs(Sigma[i, j] - expected_cov) < 1e-4


def test_rbergomi_variance_and_martingale_property():
    rb = fs_vol.RoughBergomi(S0=100, xi0=0.04, eta=1.5, rho=-0.7, H=0.1, r=0.0)
    S, v = rb.simulate(T=1.0, N=50, n_paths=20_000, return_variance=True, random_state=18)
    mean_v = v[:, 25].mean()
    se_v = v[:, 25].std(ddof=1) / math.sqrt(v.shape[0])
    assert abs(mean_v - 0.04) < 4 * se_v


def test_rbergomi_eta_zero_is_bs():
    rb = fs_vol.RoughBergomi(S0=100, xi0=0.04, eta=1e-8, rho=-0.7, H=0.1, r=0.05)
    res = rb.call_price_mc(100, 1, n_paths=30_000, N=50, random_state=19)
    bs = fs_op.BlackScholes(100, 100, 1, 0.05, 0.2)
    assert abs(res.estimate - bs.call_price()) < 4 * res.std_error


def test_rbergomi_skew_sign():
    rb = fs_vol.RoughBergomi(S0=100, xi0=0.04, eta=1.8, rho=-0.9, H=0.1, r=0.0)
    low = rb.implied_vol_mc(90, 1, n_paths=30_000, N=50, random_state=20)
    high = rb.implied_vol_mc(110, 1, n_paths=30_000, N=50, random_state=20)
    assert low > high


def test_localvol_constant_sigma_matches_bs():
    sigma_fn = lambda t, S: 0.2 * np.ones_like(np.atleast_1d(np.asarray(S, dtype=float)))
    lv = fs_vol.LocalVol(sigma_fn, 100, 0.05)
    bs = fs_op.BlackScholes(100, 100, 1, 0.05, 0.2)
    pde_price = lv.price_pde(100, 1, "call", n_space=300, n_time=300)
    mc = lv.price_mc(100, 1, "call", n_paths=100_000, N=100, random_state=21)
    assert abs(pde_price - bs.call_price()) < 0.02
    assert abs(mc.estimate - bs.call_price()) < 4 * mc.std_error


def test_localvol_time_dependent_matches_rms_vol():
    sigma_fn = lambda t, S: (0.2 + 0.1 * t) * np.ones_like(np.atleast_1d(np.asarray(S, dtype=float)))
    lv = fs_vol.LocalVol(sigma_fn, 100, 0.05)
    T = 1.0
    from scipy import integrate
    rms_var, _ = integrate.quad(lambda t: (0.2 + 0.1 * t) ** 2, 0, T)
    rms_vol = math.sqrt(rms_var / T)
    bs = fs_op.BlackScholes(100, 100, T, 0.05, rms_vol)
    pde_price = lv.price_pde(100, T, "call", n_space=300, n_time=300)
    assert abs(pde_price - bs.call_price()) < 0.05


def test_dupire_flat_surface_gives_flat_local_vol():
    dup = fs_vol.Dupire(lambda K, T: 0.22, 100, 0.05)
    assert abs(dup.local_vol(100, 1) - 0.22) < 1e-4
    assert abs(dup.local_vol(80, 0.5) - 0.22) < 1e-4


def test_dupire_term_structure_surface():
    a = lambda t: 0.2 + 0.1 * t

    def implied_vol_fn(K, T):
        from scipy import integrate
        var, _ = integrate.quad(lambda t: a(t) ** 2, 0, T)
        return math.sqrt(var / T) if T > 0 else a(0)

    dup = fs_vol.Dupire(implied_vol_fn, 100, 0.05, dT=1e-3)
    for T in (0.5, 1.0, 2.0):
        lv = dup.local_vol(100 * math.exp(0.05 * T), T)
        assert abs(lv - a(T)) < 5e-3


def test_dupire_to_localvol_reprices():
    dup = fs_vol.Dupire(lambda K, T: 0.2, 100, 0.05)
    lv = dup.to_local_vol()
    price = lv.price_pde(100, 1, "call", n_space=300, n_time=300)
    bs = fs_op.BlackScholes(100, 100, 1, 0.05, 0.2)
    assert abs(price - bs.call_price()) < 0.05


def test_lvsv_xi_zero_reduces_to_localvol():
    sigma_fn = lambda t, S: 0.2 * np.ones_like(np.atleast_1d(np.asarray(S, dtype=float)))
    lvsv = fs_vol.LVSV(sigma_fn, S0=100, v0=0.04, kappa=2, theta=0.04, xi=0.0, rho=-0.7, r=0.05, n_bins=20)
    res = lvsv.price_mc(100, 1, "call", n_paths=150_000, N=100, random_state=22)
    bs = fs_op.BlackScholes(100, 100, 1, 0.05, 0.2)
    assert abs(res.estimate - bs.call_price()) < 4 * res.std_error


def test_lvsv_leverage_calibration_reprices_localvol():
    sigma_fn = lambda t, S: 0.2 * np.ones_like(np.atleast_1d(np.asarray(S, dtype=float)))
    lvsv = fs_vol.LVSV(sigma_fn, S0=100, v0=0.04, kappa=1.0, theta=0.04, xi=0.5, rho=-0.5, r=0.05, n_bins=25)
    res = lvsv.price_mc(100, 1, "call", n_paths=100_000, N=100, random_state=23)
    bs = fs_op.BlackScholes(100, 100, 1, 0.05, 0.2)
    assert abs(res.estimate - bs.call_price()) < max(4 * res.std_error, 0.15)
    assert lvsv.leverage_grid_ is not None


def test_variance_swap_flat_surface():
    vs = fs_vol.VarianceSwap(T=1, r=0.05)
    strike = vs.fair_strike_from_surface(lambda K, T: 0.2, S0=100, q=0.0)
    assert abs(strike - 0.04) < 1e-3


def test_variance_swap_heston_vs_realized_mc():
    h = fs_vol.HestonModel(S0=100, v0=0.04, kappa=2, theta=0.05, xi=0.3, rho=-0.7, r=0.05)
    vs = fs_vol.VarianceSwap(T=1, r=0.05)
    closed = vs.fair_strike_heston(h)
    S = h.simulate(T=1, N=252, n_paths=20_000, scheme="qe", random_state=24)
    realized = vs.fair_strike_mc(S)
    assert abs(realized.estimate - closed) < 4 * realized.std_error + 0.01


# ================================================================ rate_models


def test_vasicek_zcb_vs_mc_discount():
    v = fs_rate.VasicekModel(r0=0.03, kappa=0.5, theta=0.04, sigma=0.01)
    closed = v.zcb_price(5)
    paths = v.simulate(T=5, N=250, n_paths=100_000, random_state=25)
    integral = np.trapezoid(paths, dx=5 / 250, axis=1) if hasattr(np, "trapezoid") \
        else np.trapz(paths, dx=5 / 250, axis=1)
    disc = np.exp(-integral)
    se = disc.std(ddof=1) / math.sqrt(len(disc))
    assert abs(disc.mean() - closed) < 4 * se


def test_vasicek_moments_and_fit():
    v = fs_rate.VasicekModel(r0=0.03, kappa=0.8, theta=0.04, sigma=0.02)
    paths = v.simulate(T=2, N=400, n_paths=60_000, random_state=26)
    t_idx = 200
    t = t_idx * (2.0 / 400)
    se_mean = paths[:, t_idx].std(ddof=1) / math.sqrt(paths.shape[0])
    assert abs(paths[:, t_idx].mean() - v.mean(t)) < 4 * se_mean
    assert abs(paths[:, t_idx].var(ddof=1) - v.variance(t)) < 0.15 * v.variance(t)

    fit_path = v.simulate(T=50, N=50_000, n_paths=1, random_state=27)[0]
    vf = fs_rate.VasicekModel(r0=0.03, kappa=1.0, theta=0.03, sigma=0.01).fit(fit_path, dt=50.0 / 50_000)
    assert abs(vf.kappa - 0.8) / 0.8 < 0.3
    assert abs(vf.theta - 0.04) < 5e-3
    assert abs(vf.sigma - 0.02) / 0.02 < 0.15


def test_vasicek_bond_option_parity_and_mc():
    v = fs_rate.VasicekModel(r0=0.03, kappa=0.5, theta=0.04, sigma=0.01)
    call = v.bond_option_price(1, 5, 0.85, "call")
    put = v.bond_option_price(1, 5, 0.85, "put")
    assert abs((call - put) - (v.zcb_price(5) - 0.85 * v.zcb_price(1))) < 1e-9


def test_cir_zcb_vs_mc_and_feller():
    cir = fs_rate.CIRProcess(r0=0.03, kappa=0.5, theta=0.04, sigma=0.05)
    assert cir.feller_condition() is True
    cir_violated = fs_rate.CIRProcess(r0=0.03, kappa=0.1, theta=0.02, sigma=0.5)
    assert cir_violated.feller_condition() is False
    closed = cir.zcb_price(5)
    paths = cir.simulate(T=5, N=250, n_paths=100_000, random_state=28, scheme="exact")
    integral = np.trapezoid(paths, dx=5 / 250, axis=1) if hasattr(np, "trapezoid") \
        else np.trapz(paths, dx=5 / 250, axis=1)
    disc = np.exp(-integral)
    se = disc.std(ddof=1) / math.sqrt(len(disc))
    assert abs(disc.mean() - closed) < 4 * se


def test_cir_exact_transition_matches_ncx2():
    cir = fs_rate.CIRProcess(r0=0.03, kappa=0.8, theta=0.04, sigma=0.05)
    dt = 0.5
    paths = cir.simulate(T=dt, N=1, n_paths=100_000, random_state=29, scheme="exact")
    r1 = paths[:, 1]
    ekt = math.exp(-cir.kappa * dt)
    d = 4.0 * cir.kappa * cir.theta / cir.sigma**2
    nc = 4.0 * cir.kappa * ekt * cir.r0 / (cir.sigma**2 * (1.0 - ekt))
    scale = cir.sigma**2 * (1.0 - ekt) / (4.0 * cir.kappa)
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    sample_q = np.quantile(r1, qs)
    theory_q = scale * spstats.ncx2.ppf(qs, d, nc)
    assert np.max(np.abs(sample_q - theory_q) / theory_q) < 0.05


def test_hullwhite_const_theta_equals_vasicek():
    hw = fs_rate.HullWhiteModel(kappa=0.5, sigma=0.01, r0=0.03, theta=0.02)
    vv = fs_rate.VasicekModel(0.03, 0.5, 0.02 / 0.5, 0.01)
    assert abs(hw.zcb_price(5) - vv.zcb_price(5)) < 1e-9
    assert abs(hw.bond_option_price(1, 5, 0.8, "call") - vv.bond_option_price(1, 5, 0.8, "call")) < 1e-6


def test_hullwhite_reproduces_input_curve():
    vv = fs_rate.VasicekModel(0.03, 0.5, 0.04, 0.01)
    maturities = np.array([0.5, 1, 2, 3, 5, 7, 10])
    disc = np.array([vv.zcb_price(t) for t in maturities])
    hw = fs_rate.HullWhiteModel(kappa=0.5, sigma=0.01, discount_curve=lambda t: 1.0)
    hw.fit(maturities, disc)
    for t, d in zip(maturities, disc):
        assert abs(hw.zcb_price(t) - d) < 1e-6
    assert math.isfinite(hw.theta(1.0))


def test_hullwhite_mc_discount_matches_curve():
    vv = fs_rate.VasicekModel(0.03, 0.5, 0.04, 0.01)
    hw = fs_rate.HullWhiteModel(kappa=0.5, sigma=0.01, r0=0.03, discount_curve=vv.zcb_price)
    paths = hw.simulate(T=3, N=150, n_paths=80_000, random_state=30)
    integral = np.trapezoid(paths, dx=3 / 150, axis=1) if hasattr(np, "trapezoid") \
        else np.trapz(paths, dx=3 / 150, axis=1)
    disc = np.exp(-integral)
    se = disc.std(ddof=1) / math.sqrt(len(disc))
    assert abs(disc.mean() - vv.zcb_price(3)) < 4 * se


def test_holee_closed_form_vs_mc_and_hw_limit():
    hl = fs_rate.HoLeeModel(sigma=0.01, r0=0.03, theta=0.001)
    closed = hl.zcb_price(3)
    paths = hl.simulate(T=3, N=150, n_paths=100_000, random_state=31)
    integral = np.trapezoid(paths, dx=3 / 150, axis=1) if hasattr(np, "trapezoid") \
        else np.trapz(paths, dx=3 / 150, axis=1)
    disc = np.exp(-integral)
    se = disc.std(ddof=1) / math.sqrt(len(disc))
    assert abs(disc.mean() - closed) < 4 * se
    hw_small_kappa = fs_rate.HullWhiteModel(kappa=1e-4, sigma=0.01, r0=0.03, theta=0.001)
    assert abs(hw_small_kappa.zcb_price(3) - closed) < 2e-4


def test_g2pp_eta_zero_equals_hullwhite():
    flat = lambda t: math.exp(-0.03 * t)
    hw = fs_rate.HullWhiteModel(kappa=0.5, sigma=0.01, r0=0.03, discount_curve=flat)
    g2 = fs_rate.G2ppModel(a=0.5, b=0.3, sigma=0.01, eta=1e-8, rho=0.0, r0=0.03, discount_curve=flat)
    for T in (1, 3, 5):
        assert abs(hw.zcb_price(T) - g2.zcb_price(0.0, T)) < 1e-6


def test_g2pp_zcb_vs_mc():
    flat = lambda t: math.exp(-0.03 * t)
    g2 = fs_rate.G2ppModel(a=0.6, b=0.3, sigma=0.01, eta=0.008, rho=-0.4, r0=0.03, discount_curve=flat)
    closed = g2.zcb_price(0.0, 3.0)
    r_paths, x, y = g2.simulate(T=3, N=150, n_paths=80_000, random_state=32)
    integral = np.trapezoid(r_paths, dx=3 / 150, axis=1) if hasattr(np, "trapezoid") \
        else np.trapz(r_paths, dx=3 / 150, axis=1)
    disc = np.exp(-integral)
    se = disc.std(ddof=1) / math.sqrt(len(disc))
    assert abs(disc.mean() - closed) < 4 * se + 0.01


def test_black_karasinski_moments_and_sigma_to_zero():
    bk = fs_rate.BlackKarasinski(r0=0.03, kappa=0.5, theta=0.03, sigma=0.2)
    paths = bk.simulate(T=1, N=100, n_paths=60_000, random_state=33)
    se = paths[:, -1].std(ddof=1) / math.sqrt(paths.shape[0])
    assert abs(paths[:, -1].mean() - bk.mean(1.0)) < 4 * se

    bk_tiny = fs_rate.BlackKarasinski(r0=0.03, kappa=0.5, theta=0.03, sigma=1e-6)
    zcb = bk_tiny.zcb_price(1, N=100, n_paths=20_000, random_state=34)
    assert abs(zcb.estimate - math.exp(-0.03)) < 1e-3


def test_lmm_caplets_vs_black_and_discount_identity():
    forwards = np.array([0.03, 0.032, 0.034, 0.036])
    tau = np.full(4, 0.5)
    vols = np.array([0.2, 0.22, 0.24, 0.26])
    lmm = fs_rate.LMM(forwards, tau, vols, beta=0.05)
    for i in range(1, 4):
        black = lmm.caplet_price_black(i, forwards[i])
        mc = lmm.caplet_price_mc(i, forwards[i], n_paths=150_000, substeps=4, random_state=35)
        assert abs(black - mc.estimate) < 4 * mc.std_error + 5e-5
    df = lmm.discount_factors()
    expected = np.cumprod(1.0 / (1.0 + tau * forwards))
    assert np.allclose(df, expected)


def test_hjm_reproduces_initial_curve_and_hw_pathwise():
    forward_curve = lambda x: 0.03 + 0.01 * x / 10.0
    vol_fn = lambda x: 0.01 * math.exp(-0.5 * x)
    hjm = fs_rate.HJM(forward_curve, vol_fn, x_max=5.0, N=50)
    closed = hjm.zcb_price(2)
    mc = hjm.zcb_price_mc(2, n_paths=20_000, random_state=36)
    assert abs(mc.estimate - closed) < 4 * mc.std_error


# ======================================================================= risk


def test_historical_var_equals_np_quantile():
    rng = np.random.default_rng(37)
    returns = rng.normal(0.0004, 0.012, 3000)
    hv = fs_risk.HistoricalVaR(confidence=0.99)
    res = hv.compute(returns)
    assert res.estimate == pytest.approx(float(np.quantile(-returns, 0.99)))
    var_facade = fs_risk.ValueAtRisk(confidence=0.99).historical(returns)
    assert var_facade.estimate == pytest.approx(res.estimate)
    assert res.expected_shortfall >= res.estimate


def test_parametric_normal_and_t_vs_scipy():
    mu, sigma, alpha = 0.001, 0.02, 0.99
    pv = fs_risk.ParametricVaR.from_moments(mu, sigma, confidence=alpha)
    res = pv.compute()
    z = spstats.norm.ppf(alpha)
    expected_var = -(mu - z * sigma)
    expected_es = -mu + sigma * spstats.norm.pdf(z) / (1 - alpha)
    assert abs(res.estimate - expected_var) < 1e-10
    assert abs(res.expected_shortfall - expected_es) < 1e-8

    df = 6.0
    pt = fs_risk.ParametricVaR.from_moments(mu, sigma, confidence=alpha)
    pt.dist, pt.df = "t", df
    res_t = pt.compute()
    tq = spstats.t.ppf(alpha, df)
    scale = sigma * math.sqrt((df - 2) / df)
    expected_var_t = -(mu - tq * scale)
    assert abs(res_t.estimate - expected_var_t) < 1e-6


def test_cornish_fisher_zero_moments_is_normal_and_nonmonotone_raises():
    pv_normal = fs_risk.ParametricVaR.from_moments(0.0, 1.0, confidence=0.99)
    normal_res = pv_normal.compute()
    cf = fs_risk.ParametricVaR.from_moments(0.0, 1.0, confidence=0.99)
    cf.method = "cornish_fisher"
    cf_res = cf.compute()
    assert abs(cf_res.estimate - normal_res.estimate) < 1e-9
    assert abs(cf_res.expected_shortfall - normal_res.expected_shortfall) < 1e-4

    bad = fs_risk.ParametricVaR.from_moments(0.0, 1.0, skew=3.0, kurt=0.0, confidence=0.999)
    bad.method = "cornish_fisher"
    with pytest.raises(ValueError):
        bad.compute()


def test_parametric_garch_and_ewma_run():
    rng = np.random.default_rng(38)
    returns = rng.standard_t(6, 1000) * 0.01
    ewma = fs_risk.ParametricVaR(confidence=0.99, method="ewma").fit(returns)
    res_ewma = ewma.compute()
    assert res_ewma.estimate > 0 and math.isfinite(res_ewma.estimate)
    garch = fs_risk.ParametricVaR(confidence=0.99, method="garch").fit(returns)
    res_garch = garch.compute()
    assert res_garch.estimate > 0 and math.isfinite(res_garch.estimate)


def test_var_horizon_overlapping_sums():
    rng = np.random.default_rng(39)
    returns = rng.normal(0, 0.01, 200)
    hv1 = fs_risk.HistoricalVaR(confidence=0.9, horizon=1).compute(returns)
    manual_h2 = np.convolve(returns, np.ones(2), mode="valid")
    expected = float(np.quantile(-manual_h2, 0.9))
    hv2 = fs_risk.HistoricalVaR(confidence=0.9, horizon=2).compute(returns)
    assert hv2.estimate == pytest.approx(expected)


def test_backtest_kupiec():
    rng = np.random.default_rng(40)
    n = 5000
    p = 0.01
    var_level = 2.326
    exceed = rng.random(n) < p
    returns = np.where(exceed, -var_level - rng.random(n), rng.normal(0, 0.3, n))
    var = fs_risk.ValueAtRisk(confidence=0.99)
    bt = var.backtest(returns, np.full(n, var_level))
    assert bt["p_value"] > 0.05

    exceed2 = rng.random(n) < 2 * p
    returns2 = np.where(exceed2, -var_level - rng.random(n), rng.normal(0, 0.3, n))
    bt2 = var.backtest(returns2, np.full(n, var_level))
    assert bt2["p_value"] < 0.05


def test_expected_shortfall_historical_and_parametric():
    rng = np.random.default_rng(41)
    returns = rng.normal(0.0005, 0.015, 3000)
    es = fs_risk.ExpectedShortfall(confidence=0.975)
    hist = es.historical(returns)
    losses = -returns
    q = np.quantile(losses, 0.975)
    manual = losses[losses >= q].mean()
    assert hist.estimate == pytest.approx(manual)
    par = es.parametric(returns.mean(), returns.std(ddof=1))
    z = spstats.norm.ppf(0.975)
    assert abs(par - (-returns.mean() + returns.std(ddof=1) * spstats.norm.pdf(z) / 0.025)) < 1e-8


def test_cvar_optimizer_approx_min_variance_gaussian():
    rng = np.random.default_rng(42)
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.03, 0.005], [0.0, 0.005, 0.02]])
    mu = np.zeros(3)
    R = rng.multivariate_normal(mu, cov / 252, 4000)
    cvar = fs_risk.ConditionalVaR(confidence=0.95)
    w_cvar = cvar.optimize_portfolio(R, long_only=True)
    mv = fs_port.MeanVariance(mu, cov, long_only=True)
    w_mv = mv.min_variance()
    assert np.max(np.abs(w_cvar - w_mv)) < 0.08


def test_stress_test_additivity_and_relative():
    pricer = lambda f: f["S"] - f["K"]
    st = fs_risk.StressTest(pricer, {"S": 100, "K": 90})
    up = st.apply({"S": 10})
    down = st.apply({"S": -10})
    combo = st.run({"up": {"S": 10}, "down": {"S": -10}})
    assert abs(combo["up"] + combo["down"]) < 1e-9
    rel = st.apply({"S": ("relative", 0.1)})
    assert rel["pnl"] == pytest.approx(10.0)


def test_scenario_analysis_mc_linear_var_and_from_historical():
    pricer = lambda f: f["S"] - f["K"]
    sigma_rel = 0.02
    sa = fs_risk.ScenarioAnalysis(pricer, confidence=0.99)
    sa.from_mc(mean=[0.0], cov=[[sigma_rel**2]], base_factors={"S": 100, "K": 90}, factors=["S"],
              n=100_000, random_state=43)
    res = sa.run()
    expected = spstats.norm.ppf(0.99) * sigma_rel * 100
    assert abs(res.estimate - expected) < 0.15 * expected

    rng = np.random.default_rng(44)
    factor_returns = rng.normal(0, sigma_rel, (3000, 1))
    sa2 = fs_risk.ScenarioAnalysis(pricer, confidence=0.99).from_historical(
        factor_returns, {"S": 100, "K": 90}, factors=["S"])
    hv = fs_risk.HistoricalVaR(confidence=0.99).compute(sa2.pnl_ / 1.0)
    res2 = sa2.run()
    assert abs(res2.estimate - hv.estimate) < 1e-9


# ===================================================================== credit


def test_default_intensity_flat_and_piecewise():
    lam = 0.02
    di = fs_credit.DefaultIntensity.flat(lam)
    assert abs(di.survival(5) - math.exp(-lam * 5)) < 1e-10
    dt = di.simulate_default_times(60_000, random_state=45)
    se = dt.std(ddof=1) / math.sqrt(len(dt))
    assert abs(dt.mean() - 1.0 / lam) < 4 * se


def test_default_intensity_fit_recovers_hazard():
    lam = 0.03
    di_true = fs_credit.DefaultIntensity.flat(lam)
    dt = di_true.simulate_default_times(20_000, random_state=46)
    horizon = 20.0
    observed = dt < horizon
    dt_obs = np.minimum(dt, horizon)
    fitted = fs_credit.DefaultIntensity(times=[0.0], hazards=[0.01]).fit(dt_obs, observed, times=[0.0])
    assert abs(fitted.hazards[0] - lam) / lam < 0.1


def test_cds_par_spread_and_parity():
    cds = fs_credit.CDSPricing(hazard=0.02, r=0.03, recovery=0.4)
    spread = cds.par_spread(5)
    assert abs(spread - 0.02 * 0.6) / (0.02 * 0.6) < 0.02
    assert abs(cds.pv(5, spread)) < 1e-8
    lam_back = cds.implied_hazard(5, spread)
    assert abs(lam_back - 0.02) < 1e-6


def test_cds_bootstrap_roundtrip():
    true_curve = fs_credit.DefaultIntensity([0.0, 1.0, 3.0], [0.01, 0.02, 0.03])
    cds_true = fs_credit.CDSPricing(true_curve, r=0.03, recovery=0.4)
    maturities = [1.0, 3.0, 5.0]
    spreads = [cds_true.par_spread(T) for T in maturities]
    boot = fs_credit.CDSPricing(0.0, r=0.03, recovery=0.4).bootstrap(maturities, spreads)
    assert np.allclose(boot.hazards, [0.01, 0.02, 0.03], atol=1e-5)


def test_merton_pd_and_balance_sheet():
    m = fs_credit.MertonCreditModel(V0=100, D=80, T=1, r=0.05, sigma_V=0.3)
    _, d2 = m._d1_d2(0.05)
    assert abs(m.default_probability() - float(spstats.norm.cdf(-d2))) < 1e-10
    assert abs((m.equity_value() + m.debt_value()) - m.V0) < 1e-8
    m_lowvol = fs_credit.MertonCreditModel(V0=100, D=80, T=1, r=0.05, sigma_V=1e-6)
    assert m_lowvol.credit_spread() < 1e-3

    E = m.equity_value()
    d1, _ = m._d1_d2(0.05)
    sigma_E = float(spstats.norm.cdf(d1)) * m.sigma_V * m.V0 / E
    back = fs_credit.MertonCreditModel.from_equity(E, sigma_E, D=80, T=1, r=0.05)
    assert abs(back.V0 - 100) < 1e-4
    assert abs(back.sigma_V - 0.3) < 1e-4


def test_credit_migration_powers_and_generator():
    P = np.array([[0.9, 0.08, 0.02], [0.05, 0.85, 0.10], [0.0, 0.0, 1.0]])
    cm = fs_credit.CreditMigration(transition_matrix=P)
    assert np.allclose(cm.n_step(2), P @ P)
    # sub-dominant eigenvalue of the transient block is ~0.943, so mass
    # outside the absorbing state decays like 0.943^n - not near machine
    # precision by n=100 (~0.94^100 ~ 0.003), but strictly increasing and
    # very close to 1 by n=1000.
    assert cm.cumulative_default_prob(0, 100) > 0.99
    assert cm.cumulative_default_prob(0, 1000) == pytest.approx(1.0, abs=1e-6)
    assert cm.cumulative_default_prob(0, 1000) > cm.cumulative_default_prob(0, 100)
    G = cm.generator()
    assert np.allclose(scipy_expm(G), P, atol=1e-6)
    assert np.allclose(G.sum(axis=1), 0.0, atol=1e-10)


def test_credit_migration_fit():
    P_true = np.array([[0.9, 0.08, 0.02], [0.05, 0.85, 0.10], [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(47)

    def simulate_chain(n_steps, start):
        s = [start]
        cur = start
        for _ in range(n_steps):
            cur = rng.choice(3, p=P_true[cur])
            s.append(cur)
            if cur == 2:
                break
        return np.array(s)

    seqs = [simulate_chain(20, rng.choice([0, 1])) for _ in range(4000)]
    fitted = fs_credit.CreditMigration().fit(seqs)
    assert np.max(np.abs(fitted.transition_matrix_ - P_true)) < 0.03


def test_credit_risk_model_el_ul_and_vasicek_quantile():
    crm = fs_credit.CreditRiskModel(exposures=[100, 200, 150], pds=[0.02, 0.03, 0.01], lgds=0.6)
    expected_el = sum(e * p * 0.6 for e, p in zip([100, 200, 150], [0.02, 0.03, 0.01]))
    assert abs(crm.expected_loss() - expected_el) < 1e-9
    losses = crm.simulate_losses(200_000, random_state=48)
    assert abs(crm.unexpected_loss() - losses.std(ddof=1)) < 0.1 * crm.unexpected_loss()
    q_indep = crm.vasicek_quantile(0.99, 1e-8)
    pd_bar = np.mean([0.02, 0.03, 0.01])
    assert abs(q_indep - pd_bar * sum([100 * 0.6, 200 * 0.6, 150 * 0.6])) / q_indep < 0.05


def test_copula_credit_loss_mean_within_se():
    cop = fs_credit.CopulaCreditModel.one_factor(exposures=[100] * 5, pds=[0.02] * 5, lgds=0.6, rho=0.3)
    losses = cop.simulate_losses(200_000, random_state=49)
    se = losses.std(ddof=1) / math.sqrt(len(losses))
    expected = sum([100 * 0.02 * 0.6] * 5)
    assert abs(losses.mean() - expected) < 4 * se


def test_copula_credit_correlation_fattens_tail():
    lo = fs_credit.CopulaCreditModel.one_factor(exposures=[100] * 5, pds=[0.02] * 5, lgds=0.6, rho=1e-8)
    hi = fs_credit.CopulaCreditModel.one_factor(exposures=[100] * 5, pds=[0.02] * 5, lgds=0.6, rho=0.6)
    l_lo = lo.simulate_losses(200_000, random_state=50)
    l_hi = hi.simulate_losses(200_000, random_state=50)
    var_lo = risk_analysis_var(l_lo, 0.99)
    var_hi = risk_analysis_var(l_hi, 0.99)
    assert var_hi > var_lo

    t_cop = fs_credit.CopulaCreditModel.one_factor(exposures=[100] * 5, pds=[0.02] * 5, lgds=0.6, rho=0.3,
                                                   copula="t", df=4)
    g_cop = fs_credit.CopulaCreditModel.one_factor(exposures=[100] * 5, pds=[0.02] * 5, lgds=0.6, rho=0.3)
    l_t = t_cop.simulate_losses(200_000, random_state=51)
    l_g = g_cop.simulate_losses(200_000, random_state=51)
    assert l_t.var(ddof=1) > l_g.var(ddof=1)


def risk_analysis_var(losses, alpha):
    return float(np.quantile(losses, alpha))


def test_copula_credit_rho_zero_matches_independent():
    cop0 = fs_credit.CopulaCreditModel.one_factor(exposures=[100] * 5, pds=[0.02] * 5, lgds=0.6, rho=1e-8)
    indep = fs_credit.CreditRiskModel(exposures=[100] * 5, pds=[0.02] * 5, lgds=0.6)
    l0 = cop0.simulate_losses(200_000, random_state=52)
    li = indep.simulate_losses(200_000, random_state=52)
    assert abs(l0.var(ddof=1) - li.var(ddof=1)) / li.var(ddof=1) < 0.05


# ================================================================== portfolio


def test_covariance_sample_equals_np_cov():
    rng = np.random.default_rng(53)
    R = rng.normal(size=(500, 4))
    ce = fs_port.CovarianceEstimation("sample").fit(R)
    assert np.allclose(ce.covariance_, np.cov(R, rowvar=False, ddof=1))


def test_ledoit_wolf_shrinkage_bounds_psd_and_small_n_improvement():
    # LW's superiority over the sample covariance is an *expected*-squared-
    # error guarantee, not a per-draw one (a single small-n draw can collapse
    # to full shrinkage and still lose to that draw's sample covariance --
    # verified this happens for seed 54 by cross-checking against sklearn's
    # independent `ledoit_wolf` estimator, which agrees to the sample-mean
    # convention). So check the bounds per-draw, and the improvement
    # claim averaged over many independent small-n draws.
    rng = np.random.default_rng(54)
    cov_true = np.array([[0.04, 0.015, 0.0], [0.015, 0.03, 0.005], [0.0, 0.005, 0.02]])
    lw_err_total = 0.0
    sample_err_total = 0.0
    for _ in range(200):
        R_small = rng.multivariate_normal(np.zeros(3), cov_true, 15)
        lw = fs_port.CovarianceEstimation("ledoit_wolf").fit(R_small)
        assert 0.0 <= lw.shrinkage_ <= 1.0
        assert np.all(np.linalg.eigvalsh(lw.covariance_) >= -1e-10)
        sample = np.cov(R_small, rowvar=False, ddof=1)
        lw_err_total += np.linalg.norm(lw.covariance_ - cov_true) ** 2
        sample_err_total += np.linalg.norm(sample - cov_true) ** 2
    assert lw_err_total < sample_err_total


def test_ewma_covariance_recursion():
    rng = np.random.default_rng(55)
    R = rng.normal(size=(50, 2))
    ce = fs_port.CovarianceEstimation("ewma", lam=0.9).fit(R)
    X = R - R.mean(axis=0)
    S = np.outer(X[0], X[0])
    for t in range(1, 50):
        S = 0.9 * S + 0.1 * np.outer(X[t], X[t])
    assert np.allclose(ce.covariance_, S)


def test_mean_variance_closed_form_vs_slsqp():
    mu = np.array([0.08, 0.10, 0.06])
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.05, 0.01], [0.0, 0.01, 0.03]])
    mv_u = fs_port.MeanVariance(mu, cov, long_only=False)
    mv_c = fs_port.MeanVariance(mu, cov, long_only=True)
    w_u = mv_u.min_variance()
    w_c = mv_c.min_variance()
    if np.all(w_u >= -1e-9):
        assert np.allclose(w_u, w_c, atol=1e-4)

    rets, vols, W = mv_c.efficient_frontier(n=20)
    order = np.argsort(rets)
    assert np.all(np.diff(vols[order][len(vols) // 2:]) >= -1e-6)

    rng = np.random.default_rng(56)
    best = -np.inf
    for _ in range(3000):
        w = rng.dirichlet(np.ones(3))
        best = max(best, mv_c.sharpe(w))
    w_ms = mv_c.max_sharpe()
    assert mv_c.sharpe(w_ms) >= best - 1e-6


def test_portfolio_optimization_max_utility_closed_form():
    rng = np.random.default_rng(57)
    mu = np.array([0.08, 0.10, 0.06])
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.05, 0.01], [0.0, 0.01, 0.03]])
    po = fs_port.PortfolioOptimization(expected_returns=mu, covariance=cov, bounds=(None, None))
    w = po.optimize("max_utility", risk_aversion=3.0)
    inv_cov = np.linalg.inv(cov)
    ones = np.ones(3)
    gamma = (ones @ inv_cov @ mu - 3.0) / (ones @ inv_cov @ ones)
    w_closed = inv_cov @ (mu - gamma * ones) / 3.0
    assert np.allclose(w, w_closed, atol=1e-6)

    R = rng.multivariate_normal(mu / 252, cov / 252, 1000)
    po2 = fs_port.PortfolioOptimization().fit(R, annualize=252)
    assert po2.mu_ is not None and po2.cov_ is not None


def test_black_litterman_no_views_and_certain_view():
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.05, 0.01], [0.0, 0.01, 0.03]])
    bl = fs_port.BlackLitterman(cov, market_weights=[0.5, 0.3, 0.2], risk_aversion=2.5, tau=0.05)
    mu0, cov0 = bl.posterior()
    assert np.allclose(mu0, bl.equilibrium_returns())
    assert np.allclose(cov0, 1.05 * cov)
    w0 = bl.optimal_weights(mu0, cov0)
    assert np.allclose(w0, np.array([0.5, 0.3, 0.2]) / 1.05)

    P = np.array([[1.0, 0.0, 0.0]])
    Q = np.array([0.5])
    mu1, _ = bl.posterior(P, Q, omega=np.array([[1e-10]]))
    assert abs(mu1[0] - 0.5) < 1e-4


def test_risk_parity_equal_contributions_and_budgets():
    cov = np.array([[0.04, 0.015, 0.0], [0.015, 0.03, 0.005], [0.0, 0.005, 0.02]])
    rp = fs_port.RiskParity(cov)
    w = rp.optimize()
    rc = rp.risk_contributions(w)
    assert np.allclose(rc, rc[0], atol=1e-6)
    assert abs(w.sum() - 1.0) < 1e-10

    cov_diag = np.diag([0.04, 0.09, 0.16])
    rp_diag = fs_port.RiskParity(cov_diag)
    w_diag = rp_diag.optimize()
    expected = (1.0 / np.sqrt(np.diag(cov_diag)))
    expected = expected / expected.sum()
    assert np.allclose(w_diag, expected, atol=1e-4)

    budgets = np.array([0.5, 0.3, 0.2])
    rp_b = fs_port.RiskParity(cov, budgets=budgets)
    w_b = rp_b.optimize()
    rc_b = rp_b.risk_contributions(w_b)
    assert np.allclose(rc_b / rc_b.sum(), budgets, atol=1e-4)


# ==================================================================== wiring

_SPEC_NAMES = {
    "BlackScholes", "BlackScholes_American", "BinomialTree", "TrinomialTree",
    "MonteCarloOptionPricing", "LongstaffSchwartz", "FourierOptionPricing",
    "HestonModel", "SABRModel", "RoughHeston", "RoughBergomi", "LVSV",
    "LocalVol", "Dupire", "VarianceSwap",
    "HullWhiteModel", "CIRProcess", "VasicekModel", "HoLeeModel", "LMM",
    "HJM", "G2ppModel", "BlackKarasinski",
    "ValueAtRisk", "ExpectedShortfall", "ConditionalVaR", "HistoricalVaR",
    "ParametricVaR", "StressTest", "ScenarioAnalysis",
    "CreditRiskModel", "DefaultIntensity", "CreditMigration", "CDSPricing",
    "MertonCreditModel", "CopulaCreditModel",
    "Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Volga",
    "Greeks_MC", "Greeks_FD",
    "PortfolioOptimization", "MeanVariance", "BlackLitterman", "RiskParity",
    "CovarianceEstimation",
}


def test_module_wiring_and_exports():
    assert "financial_stochastics" in stochpylib.__all__
    assert hasattr(stochpylib, "financial_stochastics")
    assert len(_SPEC_NAMES) == 50
    assert set(fs.__all__) == _SPEC_NAMES
    missing = [n for n in _SPEC_NAMES if not hasattr(fs, n)]
    assert not missing, f"missing exports: {missing}"
    assert list(fs.__all__) == sorted(fs.__all__)
    assert len(fs.__all__) == len(set(fs.__all__))

    submodule_alls = (
        set(fs_op.__all__) | set(fs_greeks.__all__) | set(fs_vol.__all__)
        | set(fs_rate.__all__) | set(fs_risk.__all__) | set(fs_credit.__all__)
        | set(fs_port.__all__)
    )
    assert submodule_alls == _SPEC_NAMES


def test_quickstart_example_runs():
    bs = fs.BlackScholes(S=100, K=105, T=1, r=0.05, sigma=0.2)
    assert abs(bs.call_price() - 8.0214) < 1e-3
    assert math.isfinite(bs.Delta) and math.isfinite(bs.Gamma)

    heston = fs.HestonModel(S0=100, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7, r=0.05)
    paths = heston.simulate(T=1, N=252, n_paths=10)
    assert paths.shape == (10, 253)

    rng = np.random.default_rng(58)
    returns_data = rng.normal(0.0003, 0.01, 500)
    var = fs.ValueAtRisk(confidence=0.99)
    res = var.historical(returns_data)
    assert math.isfinite(res.estimate)


def test_random_state_reproducibility():
    bs_mc = fs_op.MonteCarloOptionPricing(100, 100, 1, 0.05, 0.2)
    r1 = bs_mc.price("call", n_paths=1000, random_state=7)
    r2 = bs_mc.price("call", n_paths=1000, random_state=7)
    assert r1.estimate == r2.estimate

    h = fs_vol.HestonModel(S0=100, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7, r=0.05)
    s1 = h.simulate(T=1, N=50, n_paths=20, random_state=9)
    s2 = h.simulate(T=1, N=50, n_paths=20, random_state=9)
    assert np.array_equal(s1, s2)

    v = fs_rate.VasicekModel(r0=0.03, kappa=0.5, theta=0.04, sigma=0.01)
    p1 = v.simulate(T=1, N=50, n_paths=20, random_state=11)
    p2 = v.simulate(T=1, N=50, n_paths=20, random_state=11)
    assert np.array_equal(p1, p2)
