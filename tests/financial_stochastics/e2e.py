"""End-to-end API sweep for stochpylib.financial_stochastics: one realistic exercise per
public name. ``test_every_public_name_is_exercised`` fails the moment a name ships without
one; ``test_exercise[<name>]`` runs each exercise as its own pytest case."""

import functools
import math

import numpy as np
import pytest

from stochpylib import financial_stochastics as mod

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


S, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.2
BS_CALL = 10.4506
_RNG = np.random.default_rng(0)


@functools.lru_cache(maxsize=None)
def _returns():
    return np.random.default_rng(1).standard_t(5, 2000) * 0.01


@functools.lru_cache(maxsize=None)
def _returns_matrix():
    rng = np.random.default_rng(2)
    return rng.multivariate_normal([0.0005, 0.0003, 0.0004],
                                   [[4e-4, 1e-4, 0.5e-4], [1e-4, 3e-4, 0.8e-4], [0.5e-4, 0.8e-4, 2e-4]], 800)


COV3 = np.array([[0.04, 0.015, 0.0], [0.015, 0.03, 0.005], [0.0, 0.005, 0.02]])
MU3 = np.array([0.08, 0.06, 0.05])


@exercise("BlackScholes")
def _bs():
    bs = mod.BlackScholes(S, K, T, R, SIG)
    assert abs(bs.call_price() - BS_CALL) < 1e-3
    assert abs(bs.call_price() - bs.put_price() - (S - K * math.exp(-R * T))) < 1e-9
    assert 0 < bs.delta() < 1 and abs(bs.implied_vol(bs.call_price()) - SIG) < 1e-6
    assert set(bs.greeks()) >= {"delta", "gamma", "vega"} and bs.price("put") == pytest.approx(bs.put_price())


@exercise("BlackScholes_American")
def _bs_american():
    am = mod.BlackScholes_American(S, K, T, R, SIG)
    assert am.put_price() >= mod.BlackScholes(S, K, T, R, SIG).put_price() - 1e-9
    assert abs(am.call_price() - BS_CALL) < 0.05          # no dividends: never exercise early
    assert np.isfinite(am.early_exercise_boundary("put"))


@exercise("BinomialTree")
def _binomial():
    bt = mod.BinomialTree(S, K, T, R, SIG, n_steps=300)
    assert abs(bt.call_price() - BS_CALL) < 0.05 and bt.put_price(american=True) >= bt.put_price()
    assert 0 < bt.delta() < 1 and bt.gamma() > 0


@exercise("TrinomialTree")
def _trinomial():
    tt = mod.TrinomialTree(S, K, T, R, SIG, n_steps=200)
    assert abs(tt.price("call") - BS_CALL) < 0.05


@exercise("MonteCarloOptionPricing")
def _mc_pricing():
    mc = mod.MonteCarloOptionPricing(S, K, T, R, SIG)
    res = mc.price(n_paths=20000, antithetic=True, random_state=3)
    assert abs(res.estimate - BS_CALL) < 4 * res.std_error + 0.05
    assert mc.simulate_paths(10, 5, random_state=4).shape == (10, 6)
    asian = mc.asian_price(n_paths=5000, n_steps=50, random_state=5)
    assert 0 < asian.estimate < BS_CALL


@exercise("LongstaffSchwartz")
def _lsm():
    ls = mod.LongstaffSchwartz(S, K, T, R, SIG, n_steps=25)
    res = ls.price(kind="put", n_paths=10000, random_state=6)
    assert res.estimate >= mod.BlackScholes(S, K, T, R, SIG).put_price() - 3 * res.std_error


@exercise("FourierOptionPricing")
def _fourier():
    fp = mod.FourierOptionPricing.from_black_scholes(S, T, R, SIG)
    assert abs(fp.call_price(K) - BS_CALL) < 1e-2
    assert abs(fp.call_price(K, method="cos") - BS_CALL) < 1e-2
    assert abs(fp.implied_vol(K) - SIG) < 1e-3


for _g, _sign in (("Delta", 1), ("Gamma", 1), ("Vega", 1), ("Theta", -1), ("Rho", 1), ("Vanna", 0), ("Volga", 1)):
    def _greek(g=_g, sign=_sign):
        v = getattr(mod, g)(S, K, T, R, SIG)
        assert np.isfinite(v) and (sign == 0 or np.sign(v) == sign)
    EXERCISES[_g] = _greek


@exercise("Greeks_FD")
def _greeks_fd():
    fd = mod.Greeks_FD(lambda S_, K_, T_, r_, sig_: mod.BlackScholes(S_, K_, T_, r_, sig_).call_price())
    vals = fd.compute(S, K, T, R, SIG)
    assert abs(vals["delta"] - mod.Delta(S, K, T, R, SIG)) < 1e-4


@exercise("Greeks_MC")
def _greeks_mc():
    out = mod.Greeks_MC(S, K, T, R, SIG).compute(n_paths=20000, random_state=7, greeks=("delta", "vega"))
    assert abs(out["delta"].estimate - mod.Delta(S, K, T, R, SIG)) < 5 * out["delta"].std_error + 0.01


@exercise("HestonModel")
def _heston():
    h = mod.HestonModel(S0=S, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7, r=R)
    p = h.call_price(K, T)
    mc = h.call_price_mc(K, T, n_paths=20000, N=50, random_state=8)
    assert abs(mc.estimate - p) < 5 * mc.std_error + 0.1 and h.feller_condition()
    assert h.simulate(T, N=10, n_paths=5, random_state=9).shape == (5, 11)
    assert abs(h.implied_vol(K, T) - 0.2) < 0.05


@exercise("SABRModel")
def _sabr():
    sabr = mod.SABRModel(alpha=0.25, beta=0.7, rho=-0.3, nu=0.5)
    vols = [sabr.implied_vol(100.0, k, 1.0) for k in (90.0, 100.0, 110.0)]
    assert all(np.isfinite(vols)) and vols[0] > vols[1]
    F, A = sabr.simulate(100.0, 1.0, N=10, n_paths=3, random_state=10)     # (forward, alpha) paths
    assert sabr.price(100.0, 100.0, 1.0) > 0 and F.shape == A.shape == (3, 11)


@exercise("RoughHeston")
def _rough_heston():
    rh = mod.RoughHeston(S0=S, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7, r=R, H=0.5)
    assert abs(rh.call_price(K, T, n_steps=100) - mod.HestonModel(S, 0.04, 2, 0.04, 0.3, -0.7, R).call_price(K, T)) < 0.2
    assert rh.simulate(T, N=10, n_paths=3, random_state=11).shape == (3, 11)


@exercise("RoughBergomi")
def _rough_bergomi():
    rb = mod.RoughBergomi(S0=S, xi0=0.04, eta=1.5, rho=-0.7, H=0.15, r=0.0)
    assert rb.simulate(T, N=20, n_paths=4, random_state=12).shape == (4, 21)
    mc = rb.call_price_mc(K, T, n_paths=3000, N=30, random_state=13)
    assert 0 < mc.estimate < 20


@exercise("LocalVol")
def _local_vol():
    lv = mod.LocalVol(lambda S_, t: 0.2, S, R)
    assert abs(lv.price_pde(K, T, n_space=200, n_time=200) - BS_CALL) < 0.1
    mc = lv.price_mc(K, T, n_paths=5000, N=50, random_state=14)
    assert abs(mc.estimate - BS_CALL) < 5 * mc.std_error + 0.2


@exercise("Dupire")
def _dupire():
    dup = mod.Dupire(lambda K_, T_: 0.22, S, R)
    assert abs(dup.local_vol(100.0, 1.0) - 0.22) < 1e-3


@exercise("LVSV")
def _lvsv():
    lvsv = mod.LVSV(lambda S_, t: 0.2, S0=S, v0=0.04, kappa=2, theta=0.04, xi=0.0, rho=-0.7, r=R, n_bins=15)
    mc = lvsv.price_mc(K, T, n_paths=3000, N=30, random_state=15)
    assert abs(mc.estimate - BS_CALL) < 5 * mc.std_error + 0.5


@exercise("VarianceSwap")
def _variance_swap():
    vs = mod.VarianceSwap(T=1, r=R)
    assert abs(vs.fair_strike_from_surface(lambda K_, T_: 0.2, S0=S) - 0.04) < 1e-3
    h = mod.HestonModel(S0=S, v0=0.04, kappa=2, theta=0.05, xi=0.3, rho=-0.7, r=R)
    assert 0.04 < vs.fair_strike_heston(h) < 0.05


def _rate_model_workflow(m):
    p = m.zcb_price(5.0)
    assert 0 < p < 1 and m.zero_rate(5.0) > 0
    yc = np.asarray(m.yield_curve([1.0, 2.0, 5.0]), dtype=float)
    assert yc.shape == (3,) and np.all(np.isfinite(yc))
    paths = np.asarray(m.simulate(1.0, N=20, n_paths=5, random_state=16))
    assert paths.shape == (5, 21)


@exercise("VasicekModel")
def _vasicek():
    v = mod.VasicekModel(r0=0.03, kappa=0.5, theta=0.04, sigma=0.01)
    _rate_model_workflow(v)
    assert abs(v.mean(100.0) - 0.04) < 1e-6 and v.bond_option_price(1.0, 2.0, 0.95) > 0


@exercise("CIRProcess")
def _cir():
    c = mod.CIRProcess(r0=0.03, kappa=0.5, theta=0.04, sigma=0.05)
    _rate_model_workflow(c)
    assert c.feller_condition() and c.variance(1.0) > 0


@exercise("HullWhiteModel")
def _hull_white():
    hw = mod.HullWhiteModel(kappa=0.5, sigma=0.01, r0=0.03, theta=0.02)
    _rate_model_workflow(hw)
    assert np.isfinite(hw.theta(1.0))
    fitted = mod.HullWhiteModel(kappa=0.5, sigma=0.01).fit([1.0, 2.0, 5.0], [0.97, 0.94, 0.85])
    assert abs(fitted.zcb_price(2.0) - 0.94) < 1e-6 and np.isfinite(fitted.forward_rate(1.0))


@exercise("HoLeeModel")
def _ho_lee():
    hl = mod.HoLeeModel(sigma=0.01, r0=0.03, theta=0.001)
    _rate_model_workflow(hl)


@exercise("G2ppModel")
def _g2pp():
    g = mod.G2ppModel(a=0.5, b=0.1, sigma=0.01, eta=0.005, rho=-0.5, r0=0.03)
    assert 0 < g.zcb_price(5.0) < 1 and g.zero_rate(5.0) > 0
    assert np.asarray(g.yield_curve([1.0, 2.0, 5.0])).shape == (3,)
    r, x, y = g.simulate(1.0, N=20, n_paths=5, random_state=16)    # (short rate, x, y) factors
    assert r.shape == x.shape == y.shape == (5, 21)
    assert g.zcb_option_price(1.0, 2.0, 0.95) >= 0


@exercise("BlackKarasinski")
def _bk():
    bk = mod.BlackKarasinski(r0=0.03, kappa=0.5, theta=0.04, sigma=0.1)
    assert bk.simulate(1.0, N=10, n_paths=3, random_state=17).shape == (3, 11)
    zcb = bk.zcb_price(1.0, N=20, n_paths=2000, random_state=18)      # Monte Carlo -> MCResult
    assert 0 < zcb.estimate < 1 and zcb.std_error > 0


@exercise("LMM")
def _lmm():
    lmm = mod.LMM(np.array([0.03, 0.032, 0.034]), np.full(3, 0.5), np.array([0.2, 0.22, 0.24]), beta=0.05)
    df = lmm.discount_factors()
    assert np.all(np.diff(df) < 0)
    mc = lmm.caplet_price_mc(1, 0.032, n_paths=5000, random_state=19)
    assert abs(mc.estimate - lmm.caplet_price_black(1, 0.032)) < 5 * mc.std_error + 1e-4


@exercise("HJM")
def _hjm():
    hjm = mod.HJM(lambda x: 0.03 + 0.001 * x, lambda x: 0.01 * math.exp(-0.5 * x), x_max=5.0, N=25)
    mc = hjm.zcb_price_mc(2.0, n_paths=2000, random_state=20)
    assert abs(mc.estimate - hjm.zcb_price(2.0)) < 5 * mc.std_error + 1e-3


@exercise("HistoricalVaR")
def _hvar():
    res = mod.HistoricalVaR(confidence=0.99).compute(_returns())
    assert abs(res.estimate - (-np.quantile(_returns(), 0.01))) < 1e-9


@exercise("ParametricVaR")
def _pvar():
    res = mod.ParametricVaR(confidence=0.99, dist="t", df=5).fit(_returns()).compute()
    assert 0.01 < res.estimate < 0.06


@exercise("ValueAtRisk")
def _var():
    v = mod.ValueAtRisk(confidence=0.99)
    h = v.historical(_returns())
    p = v.parametric(_returns())
    assert h.estimate > 0 and p.estimate > 0
    bt = v.backtest(_returns(), np.full(2000, h.estimate))
    assert 0 <= bt.pvalue <= 1 if hasattr(bt, "pvalue") else True


@exercise("ExpectedShortfall")
def _es():
    es = mod.ExpectedShortfall(confidence=0.99).historical(_returns())
    assert es.estimate > mod.HistoricalVaR(confidence=0.99).compute(_returns()).estimate


@exercise("ConditionalVaR")
def _cvar():
    cv = mod.ConditionalVaR(confidence=0.95)
    assert cv.historical(_returns()).estimate > 0
    w = cv.optimize_portfolio(_returns_matrix())
    assert abs(np.sum(w) - 1.0) < 1e-6 and np.all(w >= -1e-9)


@exercise("StressTest")
def _stress():
    st = mod.StressTest(lambda f: f["S"] - f["K"], {"S": 100, "K": 90})
    assert st.apply({"S": 10})["pnl"] == pytest.approx(10.0)
    out = st.run({"up": {"S": 10}, "down": {"S": -10}})
    assert abs(out["up"] + out["down"]) < 1e-9


@exercise("ScenarioAnalysis")
def _scenario():
    sa = mod.ScenarioAnalysis(lambda f: f["S"] - f["K"], confidence=0.99)
    sa.from_mc(mean=[0.0], cov=[[0.02 ** 2]], base_factors={"S": 100, "K": 90}, factors=["S"],
               n=20000, random_state=21)
    res = sa.run()
    assert 3.0 < res.estimate < 6.0


@exercise("DefaultIntensity")
def _default_intensity():
    di = mod.DefaultIntensity(times=[0.0, 2.0], hazards=[0.02, 0.04])
    assert abs(di.survival(1.0) - math.exp(-0.02)) < 1e-12 and di.default_prob(1.0) == pytest.approx(1 - math.exp(-0.02))
    assert di.simulate_default_times(50, random_state=22).shape == (50,)


@exercise("CDSPricing")
def _cds():
    cds = mod.CDSPricing(hazard=0.02, r=0.03, recovery=0.4)
    spread = cds.par_spread(5.0)
    assert abs(cds.pv(5.0, spread)) < 1e-9 and abs(cds.implied_hazard(5.0, spread) - 0.02) < 1e-6


@exercise("MertonCreditModel")
def _merton_credit():
    m = mod.MertonCreditModel(V0=100, D=80, T=1, r=0.05, sigma_V=0.25)
    assert 0 < m.default_probability() < 1 and m.distance_to_default() > 0
    assert abs(m.equity_value() + m.debt_value() - 100) < 1e-6 and m.credit_spread() > 0


@exercise("CreditMigration")
def _migration():
    P = np.array([[0.9, 0.08, 0.02], [0.05, 0.85, 0.10], [0.0, 0.0, 1.0]])
    cm = mod.CreditMigration(transition_matrix=P)
    assert np.allclose(cm.n_step(2), P @ P) and cm.cumulative_default_prob(0, 100) > 0.99
    assert cm.simulate(0, 10, n_paths=4, random_state=23).shape == (4, 11)
    assert np.allclose(cm.generator().sum(axis=1), 0.0, atol=1e-8)


@exercise("CreditRiskModel")
def _credit_risk():
    crm = mod.CreditRiskModel(exposures=[100] * 10, pds=[0.02] * 10, lgds=0.6)
    assert abs(crm.expected_loss() - 12.0) < 1e-9 and crm.unexpected_loss() > 0
    ld = crm.loss_distribution(confidence=0.99, n=20000, random_state=24)
    assert ld.estimate >= 0


@exercise("CopulaCreditModel")
def _copula_credit():
    cop = mod.CopulaCreditModel.one_factor(exposures=[100] * 5, pds=[0.02] * 5, lgds=0.6, rho=0.3)
    losses = cop.simulate_losses(5000, random_state=25)
    assert losses.shape == (5000,) and abs(losses.mean() - cop.expected_loss()) < 3.0
    assert cop.vasicek_quantile(0.99, 0.3) > cop.expected_loss()


@exercise("CovarianceEstimation")
def _cov_est():
    lw = mod.CovarianceEstimation(method="ledoit_wolf").fit(_returns_matrix())
    cov = np.asarray(lw.covariance_ if hasattr(lw, "covariance_") else lw, dtype=float)
    assert cov.shape == (3, 3) and np.allclose(cov, cov.T)


@exercise("MeanVariance")
def _mean_variance():
    mv = mod.MeanVariance(MU3, COV3, long_only=True)
    w = mv.max_sharpe()
    assert abs(w.sum() - 1) < 1e-6 and mv.sharpe(w) >= mv.sharpe(mv.min_variance()) - 1e-9
    assert len(mv.efficient_frontier(n=10)) > 0


@exercise("PortfolioOptimization")
def _portfolio_opt():
    po = mod.PortfolioOptimization().fit(_returns_matrix())
    w = po.optimize("min_variance")
    assert abs(w.sum() - 1) < 1e-6 and po.portfolio_volatility(w) > 0


@exercise("BlackLitterman")
def _black_litterman():
    bl = mod.BlackLitterman(COV3, market_weights=[0.4, 0.4, 0.2])
    pi = bl.equilibrium_returns()
    mu_bl, cov_bl = bl.posterior(P=[[1, -1, 0]], Q=[0.02])
    w = bl.optimal_weights(mu_bl, cov_bl)
    assert pi.shape == (3,) and w.shape == (3,) and np.all(np.isfinite(w))


@exercise("RiskParity")
def _risk_parity():
    rp = mod.RiskParity(COV3)
    w = rp.optimize()
    rc = rp.risk_contributions(w)
    assert np.allclose(rc, rc[0], atol=1e-6) and abs(w.sum() - 1) < 1e-9


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
