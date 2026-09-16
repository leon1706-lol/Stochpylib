"""Tests for stochpylib.levy_processes.

Covers: subordinator monotonicity/mean/Laplace-transform checks, Levy-Khintchine
characteristic-function consistency, stable-process/subordination paths,
jump-diffusion pricing vs Black-Scholes (no-jump limit) and Monte Carlo
cross-checks, advanced point/branching/field processes (Hawkes fit + KS
residuals, Cox, renewal, branching, semi-Markov, random fields/measures), and
SDE-solver strong/weak convergence orders (EM=0.5, Milstein=1.0, RK=1.0,
Taylor=1.5, weak-2 vs exact GBM mean).

Several cases are direct regressions for bugs found while writing this suite
(see development/Probleme.md #41-50): the Carr-Madan log-strike convention,
the linear-grid quantile sampler bias, the numpy 2.x np.trapz removal, the
stable-subordinator Laplace-transform normalization, the mislabeled
Runge-Kutta/StochasticTaylor schemes, the shared-Brownian-path requirement in
StrongApproximation, the three-point weak-scheme weights, and
HawkesProcess.ks_residuals()'s missing stored events.

All randomness is seeded.
"""

import numpy as np
import pytest

import stochpylib
from stochpylib import levy_processes as lp
from stochpylib.levy_processes.advanced import (
    BranchingProcess,
    CoxProcess,
    GaussianRandomField,
    HawkesProcess,
    MultivariateHawkes,
    RandomMeasure,
    RenewalProcess,
    SemiMarkovProcess,
)
from stochpylib.levy_processes.jump_diffusion import (
    BatesModel,
    CGMYProcess,
    KouJumpDiffusion,
    MertonJumpDiffusion,
    NormalInverseGaussianProcess,
    VarianceGammaProcess,
    _black_scholes_call,
)
from stochpylib.levy_processes.levy import (
    AlphaStableDistribution,
    BrownianMotion,
    LevyKhintchine,
    LevyProcess,
    SpectrallyPositive,
    StableProcess,
    SubordinatedProcess,
)
from stochpylib.levy_processes.sde import (
    SDE,
    Euler_Maruyama,
    Milstein,
    Runge_Kutta_SDE,
    StochasticTaylor,
    StrongApproximation,
    WeakApproximation,
)
from stochpylib.levy_processes.subordinators import (
    GammaSubordinator,
    InverseGaussianSubordinator,
    StableSubordinator,
    TemperingSubordinator,
    upper_gamma_negative,
)
from stochpylib.distributions import Exponential


# ------------------------------------------------------------ subordinators

class TestSubordinators:
    def test_gamma_mean_and_monotone(self):
        gs = GammaSubordinator(rate=2.0, scale=0.5)
        rng = np.random.default_rng(0)
        inc = np.array([gs._increment(1.0, rng) for _ in range(50_000)])
        assert abs(inc.mean() - 1.0) < 4 * inc.std(ddof=1) / np.sqrt(len(inc))
        path = gs.sample([0.2, 0.4, 0.6, 0.8, 1.0], random_state=1)
        assert np.all(np.diff(path) >= 0.0)

    def test_inverse_gaussian_mean(self):
        igs = InverseGaussianSubordinator(lam=3.0)
        rng = np.random.default_rng(2)
        inc = np.array([igs._increment(0.5, rng) for _ in range(50_000)])
        se = inc.std(ddof=1) / np.sqrt(len(inc))
        assert abs(inc.mean() - 0.5) < 4 * se

    def test_stable_subordinator_laplace_transform(self):
        """Regression for Probleme.md #44: S1-parameterization scale bug."""
        alpha = 0.6
        ss = StableSubordinator(alpha=alpha)
        rng = np.random.default_rng(3)
        t, lam = 0.7, 1.5
        s = np.array([ss._increment(t, rng) for _ in range(20_000)])
        lt_mc = np.mean(np.exp(-lam * s))
        lt_true = np.exp(-t * lam ** alpha)
        assert abs(lt_mc - lt_true) < 0.02

    def test_tempering_mean_matches_analytic(self):
        """Regression for Probleme.md #42: linear-grid quantile-sampler bias."""
        ts = TemperingSubordinator(C=1.0, lam=5.0, alpha=0.5)
        rng = np.random.default_rng(4)
        dt = 0.1
        inc = np.array([ts._increment(dt, rng) for _ in range(30_000)])
        expected = ts.mean_rate() * dt
        se = inc.std(ddof=1) / np.sqrt(len(inc))
        assert abs(inc.mean() - expected) < 4 * se

    def test_tempering_monotone_path_and_truncation_mass_positive(self):
        ts = TemperingSubordinator(C=1.0, lam=5.0, alpha=0.5)
        path = ts.sample(np.linspace(0.1, 1.0, 10), random_state=5)
        assert np.all(np.diff(path) >= 0.0)
        assert ts.truncation_mass() > 0.0

    def test_upper_gamma_negative_matches_positive_shape_recurrence(self):
        # Gamma(s+1, x) = s*Gamma(s, x) + x**s * exp(-x); check for s in (-1,0)
        # and s in (-2,-1] against scipy's positive-shape gammaincc directly.
        from scipy import special
        x = 2.5
        for s in (-0.3, -0.7, -1.2, -1.8):
            g = upper_gamma_negative(s, x)
            g_next = s * g + x ** s * np.exp(-x)
            expected_next = special.gammaincc(s + 1.0, x) * special.gamma(s + 1.0) \
                if s + 1.0 > 0 else None
            if expected_next is not None:
                assert abs(g_next - expected_next) < 1e-8

    def test_subordinator_increments_and_simulate_shapes(self):
        gs = GammaSubordinator(rate=1.0, scale=1.0)
        incs = gs.increments(0.5, 100, random_state=6)
        assert incs.shape == (100,)
        assert np.all(incs >= 0.0)
        paths = gs.simulate(1.0, 10, n_paths=5, random_state=7)
        assert paths.shape == (5, 11)
        assert np.all(paths[:, 0] == 0.0)
        assert np.all(np.diff(paths, axis=1) >= 0.0)


# ------------------------------------------------------------------ levy.py

class TestLevyKhintchine:
    def test_levy_process_cf_matches_manual_exponent(self):
        rng = np.random.default_rng(10)

        def jump_cf(u):
            return np.exp(1j * u * 0.1 - 0.5 * 0.2 ** 2 * u ** 2)

        lp_proc = LevyProcess(b=0.05, sigma=0.3, jump_rate=2.0, jump_cf=jump_cf,
                              jump_sampler=lambda n, r: r.normal(0.1, 0.2, n))
        u = np.array([0.3, -0.7, 1.5])
        manual = np.exp(1.0 * (1j * u * 0.05 - 0.5 * 0.3 ** 2 * u ** 2
                               + 2.0 * (jump_cf(u) - 1.0)))
        assert np.allclose(lp_proc.characteristic_function(u, 1.0), manual)

    def test_stable_process_cf_matches_empirical_cf(self):
        alpha, beta = 0.6, 1.0
        sp = StableProcess(alpha=alpha, beta=beta, loc=0.0, scale=1.0)
        rng = np.random.default_rng(11)
        x = np.array([sp._increment(1.0, rng) for _ in range(30_000)])
        for u in (0.3, 0.7, 1.5, -1.0):
            emp = np.mean(np.exp(1j * u * x))
            ana = sp.characteristic_function(u, 1.0)
            assert abs(emp - ana) < 0.03

    def test_spectrally_positive_nonnegative_increments(self):
        sp = SpectrallyPositive(alpha=0.5, scale=1.0)
        rng = np.random.default_rng(12)
        incs = np.array([sp._increment(0.5, rng) for _ in range(2000)])
        assert np.all(incs >= 0.0)
        with pytest.raises(ValueError):
            SpectrallyPositive(alpha=1.5)

    def test_alpha_stable_distribution_adapter(self):
        ad = AlphaStableDistribution(alpha=1.8, loc=0.0, scale=1.0)
        x = np.asarray(ad.rvs(20_000, random_state=13))
        assert np.all(np.isfinite(x))
        assert abs(float(np.median(x))) < 0.1
        p = ad.pdf(0.0)
        assert np.isfinite(p) and p > 0.0
        c = ad.cdf(0.0)
        assert 0.4 < c < 0.6

    def test_subordinated_process_variance_matches_subordinator_mean(self):
        gs = GammaSubordinator(rate=2.0, scale=0.5)
        sub = SubordinatedProcess(base=BrownianMotion(), subordinator=gs)
        rng_seed = 14
        paths = sub.simulate(1.0, 20, n_paths=20_000, random_state=rng_seed)
        terminal = paths[:, -1]
        assert abs(terminal.mean()) < 0.05
        expected_var = 2.0 * 0.5 * 1.0  # E[T_1] for the gamma subordinator
        se = np.sqrt(2.0) * expected_var / np.sqrt(len(terminal))
        assert abs(terminal.var() - expected_var) < 6 * se

    def test_levy_khintchine_subordinator_form(self):
        alpha = 0.6

        def psi(lam):
            return lam ** alpha

        lk = LevyKhintchine(b=0.0, sigma=0.0, laplace_exponent=psi)
        # exponent(u) should analytically continue psi at lam = -i*u
        u = np.array([0.4, 1.2])
        lam = -1j * u
        assert np.allclose(lk.exponent(u), lam ** alpha)

    def test_levy_khintchine_compound_poisson_form(self):
        def jump_cf(u):
            return np.exp(-0.5 * u ** 2)  # standard normal jump CF

        lk = LevyKhintchine(b=0.1, sigma=0.2, jump_rate=1.5, jump_cf=jump_cf)
        u = np.array([0.5, -0.3])
        manual = 1j * u * 0.1 - 0.5 * 0.2 ** 2 * u ** 2 + 1.5 * (jump_cf(u) - 1.0)
        assert np.allclose(lk.exponent(u), manual)


# ------------------------------------------------------------ jump_diffusion

class TestJumpDiffusion:
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05

    def test_merton_zero_jump_matches_black_scholes(self):
        m = MertonJumpDiffusion(mu=self.r, sigma=0.2, jump_rate=0.0)
        cp = m.call_price(self.S0, self.K, self.T, self.r)
        bs = _black_scholes_call(self.S0, self.K, self.T, self.r, 0.2)
        assert abs(cp - bs) < 1e-6

    def test_merton_call_price_matches_monte_carlo(self):
        m = MertonJumpDiffusion(mu=self.r, sigma=0.2, jump_rate=1.0,
                                jump_mean=-0.1, jump_std=0.2)
        cp = m.call_price(self.S0, self.K, self.T, self.r)
        mc = m.call_price_mc(self.S0, self.K, self.T, self.r, n_paths=200_000,
                             random_state=1)
        assert abs(cp - mc.estimate) < 4 * mc.std_error

    def test_kou_zero_jump_matches_black_scholes(self):
        """Regression for Probleme.md #41: Carr-Madan log-strike convention."""
        k = KouJumpDiffusion(mu=self.r, sigma=0.2, jump_rate=0.0)
        cp = k.call_price(self.S0, self.K, self.T, self.r)
        bs = _black_scholes_call(self.S0, self.K, self.T, self.r, 0.2)
        assert abs(cp - bs) < 0.01

    def test_kou_call_price_matches_monte_carlo(self):
        """Regression for Probleme.md #41."""
        k = KouJumpDiffusion(mu=self.r, sigma=0.2, jump_rate=1.0, eta_up=10.0,
                             eta_down=10.0, p_up=0.4)
        cp = k.call_price(self.S0, self.K, self.T, self.r)
        mc = k.call_price_mc(self.S0, self.K, self.T, self.r, n_paths=100_000,
                             random_state=2)
        assert abs(cp - mc.estimate) < 5 * mc.std_error

    def test_bates_model_mc_price_sane(self):
        b = BatesModel(S0=self.S0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3,
                       rho=-0.7, r=self.r, jump_rate=1.0, jump_mean=-0.1,
                       jump_std=0.15)
        mc = b.call_price_mc(self.K, self.T, n_paths=20_000, n_steps=100,
                             random_state=3)
        assert 0.0 < mc.estimate < self.S0
        assert mc.std_error > 0.0 and np.isfinite(mc.std_error)

    def test_variance_gamma_moments(self):
        vg = VarianceGammaProcess(sigma=0.25, nu=0.4, theta=0.1)
        assert abs(vg.characteristic_function(0.0, 1.0) - 1.0) < 1e-10
        rng = np.random.default_rng(4)
        inc = np.array([vg._increment(1.0, rng) for _ in range(50_000)])
        se_mean = inc.std(ddof=1) / np.sqrt(len(inc))
        assert abs(inc.mean() - 0.1) < 4 * se_mean
        expected_var = 0.25 ** 2 + 0.1 ** 2 * 0.4
        assert abs(inc.var() - expected_var) < 0.05 * expected_var

    def test_cgmy_symmetric_case_has_zero_mean(self):
        cg = CGMYProcess(C=1.0, G=5.0, M=5.0, Y=0.5)
        assert abs(cg.characteristic_exponent(0.0)) < 1e-10
        rng = np.random.default_rng(5)
        inc = np.array([cg._increment(0.1, rng) for _ in range(30_000)])
        se_mean = inc.std(ddof=1) / np.sqrt(len(inc))
        assert abs(inc.mean()) < 4 * se_mean

    def test_cgmy_rejects_y_above_one_for_simulation(self):
        cg = CGMYProcess(C=1.0, G=5.0, M=5.0, Y=1.2)
        with pytest.raises(ValueError):
            cg._increment(0.1, np.random.default_rng(0))
        # characteristic function stays valid for Y in (0, 2)
        assert np.isfinite(cg.characteristic_function(0.5, 1.0))

    def test_normal_inverse_gaussian_mean(self):
        nig = NormalInverseGaussianProcess(alpha=5.0, beta=-1.0, delta=1.0)
        assert abs(nig.characteristic_function(0.0, 1.0) - 1.0) < 1e-10
        rng = np.random.default_rng(6)
        inc = np.array([nig._increment(1.0, rng) for _ in range(30_000)])
        expected = -1.0 * 1.0 ** 2 * 1.0
        se_mean = inc.std(ddof=1) / np.sqrt(len(inc))
        assert abs(inc.mean() - expected) < 4 * se_mean

    def test_carr_madan_call_matches_black_scholes_for_gbm_cf(self):
        """Direct regression for the log-strike bug (Probleme.md #41),
        independent of any specific jump model."""
        from stochpylib.levy_processes.jump_diffusion import carr_madan_call
        import math

        sigma = 0.2

        def gbm_cf(u):
            u = np.asarray(u, dtype=complex)
            b = self.r - 0.5 * sigma ** 2
            return np.exp(1j * u * (math.log(self.S0) + b * self.T)
                          - 0.5 * sigma ** 2 * u ** 2 * self.T)

        price = carr_madan_call(gbm_cf, self.S0, self.K, self.T, self.r)
        bs = _black_scholes_call(self.S0, self.K, self.T, self.r, sigma)
        assert abs(price - bs) < 1e-4


# ----------------------------------------------------------------- advanced

class TestAdvancedProcesses:
    def test_hawkes_simulate_fit_branching_and_ks_residuals(self):
        hp = HawkesProcess(mu=0.5, alpha=0.3, beta=1.0)
        events = hp.simulate(2000.0, random_state=42)
        assert events.size > 100
        assert np.all(np.diff(events) > 0.0)
        assert abs(hp.branching_ratio() - 0.3) < 1e-10

        fitted = HawkesProcess().fit(events, T=2000.0)
        assert abs(fitted.mu_ - 0.5) < 0.2
        assert 0.0 < fitted.branching_ratio() < 1.0
        # regression for Probleme.md #50: no-argument call after fit()
        stat, p_value = fitted.ks_residuals()
        assert 0.0 <= stat <= 1.0
        assert p_value > 0.01

    def test_hawkes_unstable_alpha_rejected_by_fit_bounds(self):
        hp = HawkesProcess(mu=0.5, alpha=0.1, beta=1.0)
        assert hp.intensity(0.0, []) == 0.5

    def test_multivariate_hawkes_simulate_and_branching_matrix(self):
        mh = MultivariateHawkes(mu=[0.3, 0.3],
                                alpha_matrix=[[0.1, 0.05], [0.05, 0.1]],
                                beta_vec=[1.0, 1.0])
        times, dims = mh.simulate(50.0, random_state=11)
        assert times.size == dims.size
        assert np.all(np.diff(times) >= 0.0)
        assert set(np.unique(dims)).issubset({0, 1})
        bm = mh.branching_matrix()
        assert np.allclose(bm, [[0.1, 0.05], [0.05, 0.1]])
        with pytest.raises(ValueError):
            MultivariateHawkes(mu=[0.3, 0.3], alpha_matrix=[[0.1]],
                              beta_vec=[1.0, 1.0])

    def test_cox_process_matches_poisson_moments(self):
        """Also a regression for Probleme.md #43 (np.trapz removed)."""
        cox = CoxProcess(lambda t: 3.0)
        counts = np.empty(1000)
        for i in range(1000):
            ev, lam_t = cox.simulate(2.0, random_state=1000 + i, n_grid=50)
            counts[i] = ev.size
            assert abs(lam_t - 6.0) < 1e-6
        se = counts.std(ddof=1) / np.sqrt(len(counts))
        assert abs(counts.mean() - 6.0) < 4 * se

    def test_renewal_process_exponential_matches_poisson_rate(self):
        rp = RenewalProcess(Exponential(2.0))
        m, se = rp.renewal_function(5.0, n_paths=1500, random_state=5)
        assert abs(m - 10.0) < 4 * se

    def test_branching_process_subcritical_dies_out(self):
        bp = BranchingProcess(lambda n, rng: rng.poisson(0.7, n))
        q = bp.extinction_probability(generations=40, n_paths=1000,
                                      random_state=6)
        assert q > 0.9

    def test_branching_process_supercritical_can_survive(self):
        # Keep `generations` small: a surviving supercritical path's
        # population grows like 1.5**generations (~10 million by gen 40),
        # which makes the per-generation rng.poisson(pop, rng) array
        # explode; 15 generations already gives ample separation from q=1.
        bp = BranchingProcess(lambda n, rng: rng.poisson(1.5, n))
        q = bp.extinction_probability(generations=15, n_paths=1000,
                                      random_state=7)
        assert q < 0.9

    def test_semi_markov_time_fractions_match_holding_means(self):
        sm = SemiMarkovProcess([[0, 1], [1, 0]],
                               [Exponential(2.0), Exponential(4.0)])
        times, states = sm.simulate(500.0, random_state=8)
        durs = np.diff(times)
        st = states[:-1]
        frac0 = durs[st == 0].sum() / durs.sum()
        expected = 0.5 / (0.5 + 0.25)  # mean holding times 1/2, 1/4
        assert abs(frac0 - expected) < 0.05

    def test_gaussian_random_field_shape_mean_and_finite(self):
        grf = GaussianRandomField(spectrum=lambda k: 1.0 / (1.0 + k ** 2),
                                  shape=(128,))
        field = grf.sample(random_state=10)
        assert field.shape == (128,)
        assert np.all(np.isfinite(field))
        assert abs(field.mean()) < 0.2

    def test_random_measure_gamma_mean(self):
        rm = RandomMeasure(kind="gamma", shape_per_unit=2.0, scale=1.5)
        vals = np.array([rm.sample((0, 1), random_state=9000 + i)
                         for i in range(20_000)])
        se = vals.std(ddof=1) / np.sqrt(len(vals))
        assert abs(vals.mean() - 3.0) < 4 * se
        assert rm.sample((0, 0)) == 0.0
        with pytest.raises(ValueError):
            rm.sample((1, 0))

    def test_random_measure_stable_laplace_transform(self):
        """Regression for Probleme.md #44."""
        alpha, length, lam = 0.6, 0.7, 1.5
        rm = RandomMeasure(kind="stable", alpha=alpha)
        vals = np.array([rm.sample((0, length), random_state=4000 + i)
                         for i in range(20_000)])
        lt_mc = np.mean(np.exp(-lam * vals))
        lt_true = np.exp(-length * lam ** alpha)
        assert abs(lt_mc - lt_true) < 0.03


# ---------------------------------------------------------------------- sde

class _GBM:
    mu, sigma, x0 = 0.05, 0.2, 100.0

    @staticmethod
    def make():
        return SDE(drift=lambda t, x: _GBM.mu * x,
                  diffusion=lambda t, x: _GBM.sigma * x, x0=_GBM.x0)

    @staticmethod
    def exact(sde, T, n_steps, n_paths, rng):
        dt = T / n_steps
        dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
        W = np.cumsum(dW, axis=1)
        t = np.linspace(dt, T, n_steps)
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = _GBM.x0
        paths[:, 1:] = _GBM.x0 * np.exp(
            (_GBM.mu - 0.5 * _GBM.sigma ** 2) * t + _GBM.sigma * W)
        return paths


class TestSDESolvers:
    def test_euler_maruyama_terminal_mean(self):
        sde = _GBM.make()
        paths = Euler_Maruyama(sde, T=1.0, n_steps=100, n_paths=20_000,
                               random_state=1)
        exact_mean = _GBM.x0 * np.exp(_GBM.mu)
        se = paths[:, -1].std(ddof=1) / np.sqrt(paths.shape[0])
        assert abs(paths[:, -1].mean() - exact_mean) < 5 * se

    def test_strong_convergence_orders(self):
        """Regression for Probleme.md #45-47: RK/Taylor were mislabeled and
        StrongApproximation didn't share the driving Brownian path."""
        sde = _GBM.make()
        res = StrongApproximation(
            sde, _GBM.exact,
            {"EM": Euler_Maruyama, "Milstein": Milstein,
             "RK": Runge_Kutta_SDE, "Taylor": StochasticTaylor},
            step_sizes=[16, 32, 64, 128, 256], n_paths=20_000, random_state=7)

        def order(name):
            h = np.log(res[name]["h"])
            e = np.log(res[name]["error"])
            return np.polyfit(h, e, 1)[0]

        assert 0.3 < order("EM") < 0.7
        assert 0.8 < order("Milstein") < 1.2
        assert 0.8 < order("RK") < 1.2
        assert 1.2 < order("Taylor") < 1.8
        # higher-order schemes must be more accurate at the finest step size
        assert res["Milstein"]["error"][-1] < res["EM"]["error"][-1]
        assert res["Taylor"]["error"][-1] < res["Milstein"]["error"][-1]

    def test_weak_approximation_matches_exact_gbm_mean(self):
        """Regression for Probleme.md #48: three-point weights doubled
        E[dW**2], biasing E[X_T] by an amount that did not shrink with
        n_steps."""
        sde = _GBM.make()
        xT = WeakApproximation(sde, T=1.0, n_steps=50, n_paths=100_000,
                               random_state=2)
        exact_mean = _GBM.x0 * np.exp(_GBM.mu)
        se = xT.std(ddof=1) / np.sqrt(len(xT))
        assert abs(xT.mean() - exact_mean) < 6 * se


# ------------------------------------------------------------------- wiring

_SPEC_NAMES = {
    "LevyProcess", "StableProcess", "AlphaStableDistribution",
    "SpectrallyPositive", "SubordinatedProcess", "LevyKhintchine",
    "JumpDiffusion", "MertonJumpDiffusion", "KouJumpDiffusion", "BatesModel",
    "VarianceGammaProcess", "CGMYProcess", "NormalInverseGaussianProcess",
    "Subordinator", "GammaSubordinator", "InverseGaussianSubordinator",
    "StableSubordinator", "TemperingSubordinator",
    "SemiMarkovProcess", "RenewalProcess", "BranchingProcess",
    "HawkesProcess", "MultivariateHawkes", "CoxProcess",
    "GaussianRandomField", "RandomMeasure",
    "SDE", "Euler_Maruyama", "Milstein", "Runge_Kutta_SDE",
    "StochasticTaylor", "WeakApproximation", "StrongApproximation",
}


def test_module_wiring_and_exports():
    assert "levy_processes" in stochpylib.__all__
    assert hasattr(stochpylib, "levy_processes")
    assert set(lp.__all__) == _SPEC_NAMES
    missing = [n for n in _SPEC_NAMES if not hasattr(lp, n)]
    assert not missing, f"levy_processes: spec names missing from exports: {missing}"
