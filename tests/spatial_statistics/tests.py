"""Oracle suite for :mod:`stochpylib.spatial_statistics`.

Oracles: ``scipy.special.kv`` (general-nu Matern), ``scipy.stats.poisson``/``scipy.stats.norm``
(distributional checks only -- never wrapped in library code), brute-force double loops for
variogram/Ripley-K estimators, closed-form textbook identities (Clark-Evans on a hexagonal
lattice, Cliff & Ord moments), and cross-module identities against ``gaussian_processes``,
``experimental_design.KrigingSurrogate`` and ``levy_processes.GaussianRandomField``. All
randomness is seeded; statistical assertions are set at >= 3 standard errors.
"""

import numpy as np
import pytest
from scipy import integrate, special, stats

from stochpylib import spatial_statistics as ss
from stochpylib.spatial_statistics import (
    BrownianSheet, CARModel, CoKriging, DisjunctiveKriging, ExperimentalVariogram,
    FractionalBrownianSheet, GaussianRandomField, GearyC, IndicatorKriging,
    InhomogeneousPoisson, Kriging, LogGaussianCox, MaternCluster, MaternField, MoransI,
    NNDistanceTest, Nugget, OrdinaryKriging, OrnsteinUhlenbeckField, PairCorrelation,
    PoissonPointProcess, Range, RipleyK, SARModel, Semivariogram, Sill, SimpleKriging,
    SpatialAutocorrelation, SpatialCovariance, SpatialFunction, SpatialPointProcess,
    SpatialWeights, ThomasProcess, UniversalKriging, Variogram, VariogramFitting,
)
from stochpylib.spatial_statistics.variogram import _matern_corr
from stochpylib.gaussian_processes import GPRegression, MaternKernel, RBFKernel
from stochpylib.experimental_design import KrigingSurrogate
from stochpylib.statistics import TestResult as SpTestResult
from stochpylib.timeseries import ForecastResult

_RNG = np.random.default_rng(0)


def _grf_sample(coords, model="exponential", sill=1.0, rng=0, **kw):
    v = Semivariogram(model=model, nugget=0.0, sill=sill, **kw)
    return GaussianRandomField(covariance=v).sample(coords, random_state=rng), v


# ============================================================================ variogram

class TestSemivariogramModels:
    MODELS = ["spherical", "exponential", "gaussian", "matern", "cubic"]

    @pytest.mark.parametrize("model", MODELS)
    def test_gamma_zero_is_zero(self, model):
        v = Semivariogram(model=model, nugget=0.3, sill=1.0, range=2.0)
        assert v(0.0) == 0.0

    @pytest.mark.parametrize("model", MODELS)
    def test_gamma_reaches_sill_at_large_h(self, model):
        v = Semivariogram(model=model, nugget=0.2, sill=1.5, range=1.0)
        assert v(50.0) == pytest.approx(1.5, abs=1e-6)

    @pytest.mark.parametrize("model", ["spherical", "cubic"])
    def test_finite_range_models_reach_sill_exactly_at_range(self, model):
        v = Semivariogram(model=model, nugget=0.0, sill=1.0, range=2.0)
        assert v(2.0) == pytest.approx(1.0, abs=1e-12)
        assert v(2.5) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("model", ["exponential", "gaussian"])
    def test_practical_range_convention(self, model):
        # only exponential/gaussian are asymptotic: scaled so gamma(range) ~= 95% of sill
        v = Semivariogram(model=model, nugget=0.0, sill=1.0, range=3.0)
        frac = v(3.0) / v.sill
        assert frac == pytest.approx(0.95, abs=0.03)

    def test_covariance_plus_gamma_equals_sill(self):
        v = Semivariogram(model="exponential", nugget=0.1, sill=1.0, range=1.5)
        h = np.linspace(0, 5, 20)
        assert np.allclose(v.covariance(h) + v(h), v.sill)

    def test_covariance_raises_for_unbounded_model(self):
        v = Semivariogram(model="power", nugget=0.0, sill=1.0, exponent=1.0)
        assert not v.is_bounded
        with pytest.raises(ValueError):
            v.covariance(1.0)

    def test_matern_nu_half_equals_exponential(self):
        v_m = Semivariogram(model="matern", nugget=0.0, sill=1.0, range=1.0, nu=0.5)
        v_e = Semivariogram(model="exponential", nugget=0.0, sill=1.0, range=3.0)
        h = np.linspace(0.01, 5, 30)
        assert np.allclose(v_m.covariance(h), v_e.covariance(h), atol=1e-10)

    @pytest.mark.parametrize("nu,ls", [(0.5, 1.0), (1.5, 0.7), (2.5, 2.0)])
    def test_general_matern_matches_gp_kernel_at_closed_form_nu(self, nu, ls):
        h = np.linspace(0.01, 5, 25)
        mine = _matern_corr(h, nu, ls)
        kernel = MaternKernel(nu=nu, length_scale=ls, variance=1.0)
        theirs = kernel._shape_fn(h / ls)
        assert np.allclose(mine, theirs, atol=1e-10)

    @pytest.mark.parametrize("nu", [0.3, 1.0, 3.0])
    def test_general_matern_matches_scipy_kv_directly(self, nu):
        h = np.linspace(0.01, 5, 25)
        ell = 1.3
        mine = _matern_corr(h, nu, ell)
        r = np.sqrt(2 * nu) * h / ell
        theirs = (2.0 ** (1 - nu) / special.gamma(nu)) * r ** nu * special.kv(nu, r)
        assert np.allclose(mine, theirs, rtol=1e-8)

    def test_nested_sum_equals_sum_of_parts(self):
        v1 = Semivariogram(model="exponential", nugget=0.1, sill=0.6, range=1.0)
        v2 = Semivariogram(model="spherical", nugget=0.0, sill=0.9, range=3.0)
        nested = v1 + v2
        h = np.linspace(0, 5, 15)
        assert np.allclose(nested(h), v1(h) + v2(h))
        assert nested.nugget == pytest.approx(v1.nugget + v2.nugget)

    def test_power_and_linear_are_unbounded_with_slope(self):
        v = Semivariogram(model="linear", nugget=0.2, sill=0.5)
        h = np.array([1.0, 2.0, 10.0])
        assert np.allclose(v(h), 0.2 + 0.5 * h)
        assert not v.is_bounded

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            Semivariogram(model="bogus")

    def test_power_exponent_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Semivariogram(model="power", exponent=2.5)


class TestSpatialCovarianceKernelBridge:
    @pytest.mark.parametrize("model,kernel_cls", [("exponential", MaternKernel),
                                                    ("gaussian", RBFKernel)])
    def test_to_kernel_matrix_matches_gp_kernel(self, model, kernel_cls):
        v = Semivariogram(model=model, nugget=0.0, sill=2.0, range=2.5)
        cov = SpatialCovariance(v)
        kernel = cov.to_kernel()
        X = _RNG.uniform(0, 5, size=(15, 2))
        assert np.allclose(cov.matrix(X), kernel(X), atol=1e-10)

    def test_from_kernel_round_trip(self):
        kernel = MaternKernel(nu=1.5, length_scale=0.8, variance=1.3)
        cov = SpatialCovariance.from_kernel(kernel)
        h = np.linspace(0, 3, 10)
        assert np.allclose(cov(h), kernel.variance * kernel._shape_fn(h / kernel.length_scale),
                            atol=1e-10)

    def test_to_kernel_requires_nugget_free(self):
        v = Semivariogram(model="exponential", nugget=0.1, sill=1.0, range=1.0)
        with pytest.raises(ValueError):
            SpatialCovariance(v).to_kernel()


class TestExperimentalVariogram:
    def test_matheron_matches_brute_force(self):
        coords = _RNG.uniform(0, 10, size=(40, 2))
        values = _RNG.normal(size=40)
        ev = ExperimentalVariogram(bins=8, estimator="matheron").fit(coords, values)
        n = len(coords)
        h_all, dz_all = [], []
        for i in range(n):
            for j in range(i + 1, n):
                h_all.append(np.linalg.norm(coords[i] - coords[j]))
                dz_all.append((values[i] - values[j]) ** 2)
        h_all, dz_all = np.array(h_all), np.array(dz_all)
        edges = ev.bin_edges_
        for b in range(len(ev.lags_)):
            lo = edges[b] if b > 0 else -np.inf  # first bin includes h == 0
            mask = (h_all > lo) & (h_all <= edges[b + 1])
            assert ev.counts_[b] == np.sum(mask)
            assert ev.gamma_[b] == pytest.approx(0.5 * np.mean(dz_all[mask]))
        # the raw cloud has one (h, gamma) entry per pair
        h_cloud, g_cloud = ev.cloud()
        assert len(h_cloud) == n * (n - 1) // 2
        assert np.isclose(0.5 * (values[0] - values[1]) ** 2,
                           g_cloud[np.argmin(np.abs(h_cloud - np.linalg.norm(coords[0] - coords[1])))])

    def test_cressie_and_dowd_are_positive_and_finite(self):
        coords = _RNG.uniform(0, 10, size=(50, 2))
        values = _RNG.normal(size=50)
        for est in ("cressie", "dowd"):
            ev = ExperimentalVariogram(bins=8, estimator=est).fit(coords, values)
            assert np.all(ev.gamma_ >= 0) and np.all(np.isfinite(ev.gamma_))

    def test_directional_variogram_differs_by_direction_on_anisotropic_field(self):
        n = 200
        x = _RNG.uniform(0, 20, size=n)
        y = _RNG.uniform(0, 20, size=n)
        # strong correlation along x, none along y: a simple additive anisotropic field
        values = np.sin(x / 2.0) + 0.05 * _RNG.normal(size=n)
        coords = np.column_stack([x, y])
        ev_x = ExperimentalVariogram(bins=10, direction=0, tolerance=15).fit(coords, values)
        ev_y = ExperimentalVariogram(bins=10, direction=90, tolerance=15).fit(coords, values)
        assert not np.allclose(ev_x.gamma_[:5], ev_y.gamma_[:5], rtol=0.05)

    def test_directional_requires_2d(self):
        coords = _RNG.uniform(0, 10, size=(20, 3))
        values = _RNG.normal(size=20)
        with pytest.raises(ValueError):
            ExperimentalVariogram(direction=0).fit(coords, values)


class TestVariogramFitting:
    def test_recovers_known_parameters_from_noiseless_curve(self):
        true = Semivariogram(model="spherical", nugget=0.2, sill=1.5, range=3.0)
        lags = np.linspace(0.1, 8, 25)
        ev = ExperimentalVariogram.__new__(ExperimentalVariogram)
        ev.lags_ = lags
        ev.gamma_ = true(lags)
        ev.counts_ = np.full(len(lags), 50)
        vf = VariogramFitting(model="spherical", method="ols").fit(ev)
        assert vf.variogram_.nugget == pytest.approx(0.2, abs=1e-4)
        assert vf.variogram_.sill == pytest.approx(1.5, abs=1e-4)
        assert vf.variogram_.range == pytest.approx(3.0, abs=1e-3)

    def test_selects_true_model_from_a_list(self):
        true = Semivariogram(model="gaussian", nugget=0.1, sill=1.0, range=2.0)
        lags = np.linspace(0.1, 6, 20)
        ev = ExperimentalVariogram.__new__(ExperimentalVariogram)
        ev.lags_ = lags
        ev.gamma_ = true(lags)
        ev.counts_ = np.full(len(lags), 50)
        vf = VariogramFitting(model=["spherical", "exponential", "gaussian"]).fit(ev)
        assert vf.variogram_.model == "gaussian"


class TestNuggetSillRange:
    def test_nugget_and_sill_on_synthetic_spherical_curve(self):
        true = Semivariogram(model="spherical", nugget=0.25, sill=1.2, range=4.0)
        lags = np.linspace(0.1, 10, 30)
        ev = ExperimentalVariogram.__new__(ExperimentalVariogram)
        ev.lags_ = lags
        ev.gamma_ = true(lags)
        assert float(Nugget().fit(ev)) == pytest.approx(0.25, abs=0.05)
        assert float(Sill().fit(ev)) == pytest.approx(1.2, abs=0.05)

    def test_range_on_synthetic_exponential_curve(self):
        # Range()'s default fraction=0.95 crossing coincides with the model's own
        # `range` parameter only for the asymptotic exponential/gaussian scaling (see
        # test_practical_range_convention) -- spherical/cubic reach 100% *before* h=range,
        # so their 95%-crossing point is smaller than `range` by construction, not a bug.
        true = Semivariogram(model="exponential", nugget=0.1, sill=1.0, range=4.0)
        lags = np.linspace(0.1, 12, 30)
        ev = ExperimentalVariogram.__new__(ExperimentalVariogram)
        ev.lags_ = lags
        ev.gamma_ = true(lags)
        assert float(Range().fit(ev)) == pytest.approx(4.0, abs=0.5)

    def test_sill_variance_method(self):
        values = _RNG.normal(loc=0, scale=2.0, size=1000)
        ev = ExperimentalVariogram.__new__(ExperimentalVariogram)
        s = Sill(method="variance").fit(ev, values=values)
        assert float(s) == pytest.approx(4.0, rel=0.15)


# ============================================================================== kriging

class TestKrigingCore:
    def _setup(self, n=50):
        coords = _RNG.uniform(0, 10, size=(n, 2))
        values, v = _grf_sample(coords, model="exponential", sill=1.0, range=2.0, rng=1)
        return coords, values, v

    def test_ordinary_kriging_exact_interpolation(self):
        coords, values, v = self._setup()
        ok = OrdinaryKriging(variogram=v).fit(coords, values)
        mean, std = ok.predict(coords[:5], return_std=True)
        assert np.allclose(mean, values[:5], atol=1e-6)
        assert np.all(std < 1e-5)

    def test_ordinary_kriging_weights_sum_to_one(self):
        coords, values, v = self._setup()
        ok = OrdinaryKriging(variogram=v).fit(coords, values)
        w, lam = ok.weights(coords.mean(axis=0))
        assert np.sum(w) == pytest.approx(1.0, abs=1e-8)

    def test_universal_kriging_reproduces_linear_trend_exactly(self):
        coords = _RNG.uniform(0, 10, size=(40, 2))
        values = 2.0 + 1.5 * coords[:, 0] - 0.7 * coords[:, 1]
        v = Semivariogram(model="exponential", nugget=0.0, sill=1e-6, range=1.0)
        uk = UniversalKriging(drift="linear", variogram=v).fit(coords, values)
        pred = uk.predict(coords)
        assert np.allclose(pred, values, atol=1e-3)

    def test_simple_kriging_matches_gp_regression(self):
        coords, values, v = self._setup()
        sk = SimpleKriging(variogram=v, mean=0.0).fit(coords, values)
        test_pts = _RNG.uniform(0, 10, size=(20, 2))
        m1, s1 = sk.predict(test_pts, return_std=True)
        gp = GPRegression(MaternKernel(nu=0.5, length_scale=2.0 / 3.0, variance=1.0),
                           noise=1e-10).fit(coords, values)
        m2, s2 = gp.predict(test_pts, return_std=True)
        assert np.allclose(m1, m2, atol=1e-6)
        assert np.allclose(s1, s2, atol=1e-6)

    def test_ordinary_kriging_matches_kriging_surrogate_mean(self):
        coords, values, v = self._setup()
        ok = OrdinaryKriging(variogram=v).fit(coords, values)
        surrogate = KrigingSurrogate(kernel=SpatialCovariance(v).to_kernel(), trend="constant",
                                      normalize=False, optimize=False, noise=1e-10)
        surrogate.fit(coords, values)
        test_pts = _RNG.uniform(0, 10, size=(15, 2))
        m1 = ok.predict(test_pts)
        m2 = surrogate.predict(test_pts)
        assert np.allclose(m1, m2, atol=1e-4)

    def test_power_variogram_gives_exact_interpolant(self):
        coords = _RNG.uniform(0, 10, size=(25, 2))
        values = _RNG.normal(size=25)
        v = Semivariogram(model="power", nugget=0.0, sill=0.3, exponent=1.2)
        ok = OrdinaryKriging(variogram=v).fit(coords, values)
        assert np.allclose(ok.predict(coords), values, atol=1e-6)

    def test_local_kriging_with_all_neighbors_equals_global(self):
        coords, values, v = self._setup(n=25)
        test_pts = _RNG.uniform(0, 10, size=(10, 2))
        global_ok = OrdinaryKriging(variogram=v).fit(coords, values)
        local_ok = OrdinaryKriging(variogram=v, n_neighbors=25).fit(coords, values)
        assert np.allclose(global_ok.predict(test_pts), local_ok.predict(test_pts), atol=1e-6)

    def test_cross_validate_matches_brute_force_loo(self):
        coords, values, v = self._setup(n=20)
        ok = OrdinaryKriging(variogram=v).fit(coords, values)
        cv = ok.cross_validate()
        errors = np.empty(20)
        for i in range(20):
            mask = np.ones(20, dtype=bool)
            mask[i] = False
            m = OrdinaryKriging(variogram=v).fit(coords[mask], values[mask])
            errors[i] = values[i] - m.predict(coords[i:i + 1])[0]
        assert np.allclose(cv["errors"], errors, atol=1e-8)

    def test_predict_result_returns_forecast_result(self):
        coords, values, v = self._setup(n=20)
        ok = OrdinaryKriging(variogram=v).fit(coords, values)
        r = ok.predict_result(coords[:5])
        assert isinstance(r, ForecastResult)
        lo, hi = r.confidence_interval(0.95)
        assert np.all(lo <= r.mean) and np.all(hi >= r.mean)

    def test_kriging_base_defaults_to_ordinary_behavior(self):
        coords, values, v = self._setup(n=15)
        base = Kriging(variogram=v).fit(coords, values)
        ok = OrdinaryKriging(variogram=v).fit(coords, values)
        assert np.allclose(base.predict(coords[:5]), ok.predict(coords[:5]))


class TestCoKriging:
    def test_secondary_weights_sum_to_zero_primary_to_one(self):
        coords1 = _RNG.uniform(0, 10, size=(30, 2))
        coords2 = _RNG.uniform(0, 10, size=(45, 2))
        z1, v = _grf_sample(coords1, rng=2)
        z2 = GaussianRandomField(covariance=v).sample(coords2, random_state=3)
        ck = CoKriging(variogram=v).fit(coords1, z1, coords2, z2)
        X0 = _RNG.uniform(0, 10, size=(5, 2))
        c1_0 = ck.variogram_.covariance(np.sqrt(((coords1[:, None, :] - X0[None, :, :]) ** 2).sum(-1)))
        c2_0 = ck._cross_scale * ck._corr1(
            np.sqrt(((coords2[:, None, :] - X0[None, :, :]) ** 2).sum(-1)))
        RHS = np.vstack([c1_0, c2_0, np.ones((1, 5)), np.zeros((1, 5))])
        sol = ck._A_inv @ RHS
        w1 = sol[:ck._n1]
        w2 = sol[ck._n1:ck._n1 + ck._n2]
        assert np.allclose(w1.sum(axis=0), 1.0, atol=1e-6)
        assert np.allclose(w2.sum(axis=0), 0.0, atol=1e-6)

    def test_uncorrelated_secondary_is_close_to_ordinary_kriging(self):
        coords1 = _RNG.uniform(0, 10, size=(30, 2))
        coords2 = _RNG.uniform(0, 10, size=(200, 2))
        z1, v = _grf_sample(coords1, rng=4)
        z2 = _RNG.normal(size=200)  # independent of z1
        ck = CoKriging(variogram=v).fit(coords1, z1, coords2, z2)
        ok = OrdinaryKriging(variogram=v).fit(coords1, z1)
        test_pts = _RNG.uniform(0, 10, size=(20, 2))
        assert np.mean(np.abs(ck.predict(test_pts) - ok.predict(test_pts))) < 0.15


class TestIndicatorDisjunctiveKriging:
    def test_indicator_kriging_ccdf_monotone_and_bounded(self):
        coords = _RNG.uniform(0, 10, size=(60, 2))
        values, v = _grf_sample(coords, rng=5)
        thresholds = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
        ik = IndicatorKriging(thresholds=thresholds, variogram=v).fit(coords, values)
        cdf = ik.predict_cdf(_RNG.uniform(0, 10, size=(20, 2)))
        assert np.all(cdf >= -1e-9) and np.all(cdf <= 1 + 1e-9)
        assert np.all(np.diff(cdf, axis=1) >= -1e-9)

    def test_disjunctive_kriging_converges_to_simple_kriging(self):
        coords = _RNG.uniform(0, 20, size=(400, 2))  # large domain: high effective sample size
        values, v = _grf_sample(coords, model="exponential", sill=1.0, range=1.0, rng=6)
        sk = SimpleKriging(variogram=v, mean=0.0).fit(coords, values)
        dk = DisjunctiveKriging(n_hermite=20, variogram=v).fit(coords, values)
        test_pts = _RNG.uniform(0, 20, size=(30, 2))
        m1, _ = sk.predict(test_pts, return_std=True)
        m2, _ = dk.predict(test_pts, return_std=True)
        assert np.corrcoef(m1, m2)[0, 1] > 0.95

    def test_disjunctive_predict_proba_in_unit_interval(self):
        coords = _RNG.uniform(0, 10, size=(60, 2))
        values, v = _grf_sample(coords, rng=7)
        dk = DisjunctiveKriging(n_hermite=12, variogram=v).fit(coords, values)
        p = dk.predict_proba(_RNG.uniform(0, 10, size=(10, 2)), threshold=float(np.median(values)))
        assert np.all((p >= 0) & (p <= 1))


# ------------------------------------------------------------------------ random fields

class TestRandomFields:
    def test_cholesky_sample_covariance_matches_theory(self):
        v = Semivariogram(model="exponential", nugget=0.0, sill=1.0, range=1.5)
        grf = GaussianRandomField(covariance=v)
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        samples = grf.sample(coords, n_samples=8000, random_state=1)
        emp_cov = np.cov(samples.T)
        theory = v.covariance(np.array([[0.0, 1.0], [1.0, 0.0]]))
        se = np.sqrt(2.0 / 8000)
        assert abs(emp_cov[0, 1] - theory[0, 1]) < 5 * se

    def test_circulant_matches_cholesky_variance(self):
        v = Semivariogram(model="exponential", nugget=0.0, sill=1.0, range=1.0)
        grf_c = GaussianRandomField(covariance=v, method="circulant")
        grf_l = GaussianRandomField(covariance=v, method="cholesky")
        gc = grf_c.sample_grid((16, 16), spacing=0.3, n_samples=400, random_state=2)
        gl = grf_l.sample_grid((16, 16), spacing=0.3, n_samples=400, random_state=3)
        assert abs(gc.std() - gl.std()) < 0.1
        assert abs(gc.std() - 1.0) < 0.15

    def test_matern_field_marginals_pass_ks_normal_test(self):
        mf = MaternField(nu=1.0, length_scale=1.0, variance=1.0)
        coords = np.zeros((1, 2))
        s = mf.sample(coords, n_samples=500, random_state=4).ravel()
        stat, p = stats.kstest(s, "norm")
        assert p > 0.001

    def test_general_matern_matches_kv_covariance(self):
        mf = MaternField(nu=0.8, length_scale=1.2, variance=2.0)
        h = np.linspace(0.01, 4, 10)
        r = np.sqrt(2 * 0.8) * h / 1.2
        theory = 2.0 * (2.0 ** 0.2 / special.gamma(0.8)) * r ** 0.8 * special.kv(0.8, r)
        assert np.allclose(mf._cov_fn(h), theory, rtol=1e-8)

    def test_ou_field_grid_matches_ar1_covariance(self):
        ou = OrnsteinUhlenbeckField(length_scale=1.0, variance=1.0, separable=True)
        samples = ou.sample_grid((30,), spacing=0.2, n_samples=3000, random_state=5)
        lag1 = np.mean(samples[:, 0] * samples[:, 1])
        theory = np.exp(-0.2 / 1.0)
        assert abs(lag1 - theory) < 0.05

    def test_brownian_sheet_variance_and_zero_boundary(self):
        bs = BrownianSheet(extent=(1.0, 1.0))
        n = 2000
        vals = np.array([bs.sample_grid((6, 6), random_state=s)[3, 4] for s in range(n)])
        s_coord, t_coord = 3 / 5, 4 / 5
        theory_var = s_coord * t_coord
        se = theory_var * np.sqrt(2.0 / n)
        assert abs(vals.var() - theory_var) < 5 * se
        g = bs.sample_grid((5, 5), random_state=0)
        assert np.all(g[0, :] == 0.0) and np.all(g[:, 0] == 0.0)

    def test_fbm_sheet_with_hurst_half_matches_brownian_sheet_variance(self):
        fbs = FractionalBrownianSheet(hurst=(0.5, 0.5), extent=(1.0, 1.0))
        n = 1500
        vals = np.array([fbs.sample_grid((6, 6), random_state=s)[3, 4] for s in range(n)])
        s_coord, t_coord = 3 / 5, 4 / 5
        theory_var = s_coord * t_coord
        se = theory_var * np.sqrt(2.0 / n)
        assert abs(vals.var() - theory_var) < 6 * se

    def test_condition_honours_data_exactly(self):
        v = Semivariogram(model="exponential", nugget=0.0, sill=1.0, range=1.0)
        grf = GaussianRandomField(covariance=v)
        coords = _RNG.uniform(0, 5, size=(10, 2))
        values = grf.sample(coords, random_state=6)
        cond = grf.condition(coords, values)
        resampled = cond.sample(coords, n_samples=1, random_state=7)
        assert np.allclose(resampled, values, atol=1e-6)

    def test_spectral_method_matches_levy_grf_for_same_seed(self):
        from stochpylib.levy_processes import GaussianRandomField as LevyGRF

        spectrum = lambda k: 1.0 / (1.0 + k ** 2)
        mine = GaussianRandomField(method="spectral", spectrum=spectrum)
        theirs = LevyGRF(spectrum=spectrum, shape=(32,), length=1.0)
        a = mine.sample_grid((32,), spacing=1.0, random_state=42)
        b = theirs.sample(random_state=42)
        assert np.allclose(a, b)

    def test_circulant_raises_on_impossible_embedding(self):
        def bad_cov(h):
            # oscillatory, not a valid covariance -> negative circulant eigenvalues
            return np.cos(h * 50.0)

        grf = GaussianRandomField(covariance=bad_cov, method="circulant")
        with pytest.raises(np.linalg.LinAlgError):
            grf.sample_grid((8, 8), spacing=1.0, random_state=0)


# --------------------------------------------------------------------- point processes

class TestPoissonProcesses:
    def test_csr_count_matches_scipy_poisson(self):
        W = ((0.0, 10.0), (0.0, 10.0))
        pp = PoissonPointProcess(intensity=0.5, window=W)
        n_sims = 2000
        counts = np.array([len(pp.sample(random_state=s)) for s in range(n_sims)])
        assert abs(counts.mean() - 50.0) < 3 * np.sqrt(50.0 / n_sims)
        assert abs(counts.var() - 50.0) < 3 * 50.0 * np.sqrt(2.0 / n_sims)
        # a proper Poisson(50) reference from scipy for the same mean
        ref = stats.poisson(50.0)
        assert abs(counts.mean() - ref.mean()) < 3 * np.sqrt(ref.var() / n_sims)

    def test_theoretical_K_is_ball_volume(self):
        pp = PoissonPointProcess(intensity=1.0, window=((0, 1), (0, 1)))
        r = np.array([0.1, 0.3, 0.5])
        assert np.allclose(pp.K(r), np.pi * r ** 2)

    def test_inhomogeneous_expected_count_matches_quadrature(self):
        W = ((0.0, 10.0), (0.0, 5.0))
        f = lambda X: 0.2 + 0.03 * X[:, 0]
        ip = InhomogeneousPoisson(f, window=W)
        expected = ip.expected_count()
        xs = np.linspace(0, 10, 200)
        ys = np.linspace(0, 5, 200)
        grid = np.array([[x, y] for x in xs for y in ys])
        vals = f(grid).reshape(200, 200)
        ref = integrate.simpson(integrate.simpson(vals, ys, axis=1), xs)
        assert expected == pytest.approx(ref, rel=1e-3)

    def test_inhomogeneous_fit_recovers_beta_sign(self):
        W = ((0.0, 20.0), (0.0, 20.0))
        true_ip = InhomogeneousPoisson(lambda X: np.exp(-2.0 + 0.1 * X[:, 0]), window=W)
        pts = true_ip.sample(random_state=8)
        fit_ip = InhomogeneousPoisson(lambda X: np.zeros(len(X)), window=W).fit(pts, degree=1)
        assert fit_ip.beta_[1] > 0  # recovers the positive x-slope


class TestClusterProcesses:
    def test_thomas_mean_count_matches_kappa_mu(self):
        W = ((0.0, 10.0), (0.0, 10.0))
        tp = ThomasProcess(kappa=0.1, mu=6.0, sigma=0.3, window=W)
        n_sims = 300
        counts = np.array([len(tp.sample(random_state=s)) for s in range(n_sims)])
        expected = 0.1 * 6.0 * 100.0
        se = counts.std() / np.sqrt(n_sims)
        assert abs(counts.mean() - expected) < 4 * se

    def test_thomas_K_integrates_pcf_consistently(self):
        tp = ThomasProcess(kappa=0.2, mu=4.0, sigma=0.2)
        r = 0.6
        rr = np.linspace(1e-6, r, 400)
        ref = 2 * np.pi * integrate.simpson(rr * tp.pcf(rr), rr)
        assert tp.K(r) == pytest.approx(ref, rel=1e-3)

    def test_matern_cluster_pcf_matches_disc_overlap_formula(self):
        mc = MaternCluster(kappa=0.1, mu=5.0, radius=0.5)
        r = np.array([0.2, 0.6, 0.9, 1.2])
        R = mc.radius
        expected = np.ones_like(r)
        mask = r < 2 * R
        rr = r[mask] / R
        h = (2 / np.pi) * (np.arccos(rr / 2) - (rr / 2) * np.sqrt(1 - (rr / 2) ** 2))
        expected[mask] = 1.0 + h / (np.pi * R ** 2 * mc.kappa)
        assert np.allclose(mc.pcf(r), expected)

    def test_lgcp_mean_intensity_matches_lognormal_moment(self):
        v = Semivariogram(model="exponential", nugget=0.0, sill=0.4, range=1.0)
        lgcp = LogGaussianCox(mean=np.log(0.3), covariance=v, window=((0, 10), (0, 10)),
                               grid_shape=(48, 48))
        theory = np.exp(np.log(0.3) + 0.4 / 2.0)
        n_sims = 150
        counts = np.array([len(lgcp.sample(random_state=s)) for s in range(n_sims)])
        se = counts.std() / np.sqrt(n_sims)
        assert abs(counts.mean() / 100.0 - theory) < 4 * se / 100.0


class TestRipleyKAndPCF:
    def test_hand_example_matches_brute_force_no_correction(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        W = ((-2.0, 2.0), (-2.0, 2.0))
        k = RipleyK(pts, W, r=np.array([0.5, 1.5]), correction="none")
        lam = 3 / 16.0
        # brute force: sum over ordered pairs i != j with d(i,j) <= r
        D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        expected = np.array([np.sum(D <= r) for r in [0.5, 1.5]]) / (3 * lam)
        assert np.allclose(k.estimate, expected)

    def test_csr_mean_K_matches_theory(self):
        W = ((0.0, 20.0), (0.0, 20.0))
        pp = PoissonPointProcess(intensity=1.0, window=W)
        r = np.array([0.3, 0.6])
        n_sims = 200
        ests = np.array([RipleyK(pp.sample(random_state=s), W, r=r,
                                  correction="translation").estimate for s in range(n_sims)])
        theory = np.pi * r ** 2
        se = ests.std(axis=0) / np.sqrt(n_sims)
        assert np.all(np.abs(ests.mean(axis=0) - theory) < 4 * se)

    def test_envelope_pvalue_small_for_clustered_pattern(self):
        W = ((0.0, 10.0), (0.0, 10.0))
        tp = ThomasProcess(kappa=0.03, mu=15.0, sigma=0.15, window=W)
        pts = tp.sample(random_state=9)
        k = RipleyK(pts, W, n_simulations=99, random_state=10, correction="translation")
        assert k.pvalue < 0.1

    def test_pcf_near_one_on_csr(self):
        W = ((0.0, 20.0), (0.0, 20.0))
        pp = PoissonPointProcess(intensity=2.0, window=W)
        pts = pp.sample(random_state=11)
        r = np.array([0.3, 0.5, 0.7])
        g = PairCorrelation(pts, W, r=r)
        assert np.all(np.abs(g.estimate - 1.0) < 0.5)

    def test_spatial_function_L_transform(self):
        W = ((0.0, 10.0), (0.0, 10.0))
        pp = PoissonPointProcess(intensity=1.0, window=W)
        k = RipleyK(pp.sample(random_state=12), W)
        L = k.L()
        assert isinstance(L, SpatialFunction)
        assert np.allclose(L.theoretical, k.r, atol=1e-6)  # L(r) = r under CSR


# --------------------------------------------------------------- spatial autocorrelation

def _brute_moran(z, W):
    n = len(z)
    zc = z - z.mean()
    num = 0.0
    for i in range(n):
        for j in range(n):
            num += W[i, j] * zc[i] * zc[j]
    return (n / W.sum()) * (num / np.sum(zc ** 2))


def _brute_geary(z, W):
    n = len(z)
    zc = z - z.mean()
    num = 0.0
    for i in range(n):
        for j in range(n):
            num += W[i, j] * (zc[i] - zc[j]) ** 2
    return ((n - 1) * num) / (2.0 * W.sum() * np.sum(zc ** 2))


class TestSpatialAutocorrelation:
    def test_morans_i_matches_brute_force(self):
        W = SpatialWeights.lattice((5, 5), rule="rook")
        z = _RNG.normal(size=25)
        r = MoransI(z, W)
        assert r.statistic == pytest.approx(_brute_moran(z, W.W))

    def test_morans_i_expected_value(self):
        W = SpatialWeights.lattice((6, 6))
        z = _RNG.normal(size=36)
        r = MoransI(z, W)
        assert r.extras["expected"] == pytest.approx(-1.0 / 35.0)

    def test_morans_i_randomization_variance_matches_cliff_ord(self):
        W = SpatialWeights.lattice((5, 5)).row_standardize()
        z = _RNG.normal(size=25)
        r = MoransI(z, W)
        n = 25
        S0, S1, S2 = W.S0, W.S1, W.S2
        zc = z - z.mean()
        b2 = (np.sum(zc ** 4) / n) / (np.sum(zc ** 2) / n) ** 2
        expected = -1.0 / (n - 1)
        var_rand = ((n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
                     - b2 * (n * (n - 1) * S1 - 2 * n * S2 + 6 * S0 ** 2))
                    / ((n - 1) * (n - 2) * (n - 3) * S0 ** 2)) - expected ** 2
        assert r.extras["var_randomization"] == pytest.approx(var_rand)

    def test_permutation_pvalue_close_to_analytic_on_random_data(self):
        W = SpatialWeights.lattice((6, 6)).row_standardize()
        z = _RNG.normal(size=36)
        r_analytic = MoransI(z, W)
        r_perm = MoransI(z, W, permutations=999, random_state=0)
        assert abs(r_perm.extras["p_sim"] - r_analytic.pvalue) < 0.15

    def test_gearys_c_matches_brute_force_and_expectation(self):
        W = SpatialWeights.lattice((5, 5), rule="rook")
        z = _RNG.normal(size=25)
        r = GearyC(z, W)
        assert r.statistic == pytest.approx(_brute_geary(z, W.W))
        assert r.extras["expected"] == pytest.approx(1.0)

    def test_strong_positive_autocorrelation_detected(self):
        # a smooth spatial trend should give a strongly positive, significant Moran's I
        shape = (10, 10)
        W = SpatialWeights.lattice(shape, rule="queen")
        xs, ys = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
        z = (xs + ys).astype(float).ravel() + 0.1 * _RNG.normal(size=100)
        r = MoransI(z, W)
        assert r.statistic > 0.5 and r.pvalue < 0.01

    def test_getis_ord_general_g_on_nonnegative_data(self):
        W = SpatialWeights.knn(_RNG.uniform(0, 10, size=(30, 2)), k=5)
        z = _RNG.uniform(0.1, 2.0, size=30)  # General G assumes non-negative values
        r = SpatialAutocorrelation(z, W, statistic="getis_ord")
        assert r.statistic > 0
        assert r.extras["expected"] == pytest.approx(W.S0 / (30 * 29))

    def test_local_moran_shape_and_type(self):
        W = SpatialWeights.lattice((5, 5))
        z = _RNG.normal(size=25)
        r = SpatialAutocorrelation(z, W, statistic="moran", local=True)
        assert isinstance(r, SpTestResult)
        assert r.statistic.shape == (25,)

    def test_local_getis_ord_matches_weighted_mean(self):
        W = SpatialWeights.knn(_RNG.uniform(0, 10, size=(20, 2)), k=4)
        z = _RNG.uniform(0.5, 2.0, size=20)
        r = SpatialAutocorrelation(z, W, statistic="getis_ord", local=True)
        row_sum = W.W.sum(axis=1)
        expected = (W.W @ z) / row_sum
        assert np.allclose(r.statistic, expected)


class TestNearestNeighbourDistance:
    def test_hexagonal_lattice_clark_evans_near_2149(self):
        pts = []
        rows, cols = 20, 20
        dx = 1.0
        dy = np.sqrt(3) / 2.0
        for i in range(rows):
            for j in range(cols):
                x = j * dx + (0.5 * dx if i % 2 else 0.0)
                pts.append((x, i * dy))
        pts = np.array(pts)
        W = ((0.0, (cols - 1) * dx + 0.5 * dx), (0.0, (rows - 1) * dy))
        r = NNDistanceTest(pts, W, edge="donnelly")
        assert r.statistic == pytest.approx(2.149, abs=0.1)

    def test_csr_clark_evans_near_one(self):
        # R = r_obs / r_expected is the simple, universally-recognized ratio (not
        # edge-bias-corrected -- see NNDistanceTest's docstring), so a finite window gives
        # a small, well-documented positive bias; a loose sanity bound, not a tight SE one.
        W = ((0.0, 30.0), (0.0, 30.0))
        pp = PoissonPointProcess(intensity=0.3, window=W)
        ratios = []
        for s in range(200):
            pts = pp.sample(random_state=s)
            if len(pts) > 5:
                ratios.append(NNDistanceTest(pts, W).statistic)
        ratios = np.array(ratios)
        assert abs(ratios.mean() - 1.0) < 0.1

    def test_csr_pvalues_are_calibrated(self):
        # the Donnelly-corrected z-test (used for the p-value) should be well calibrated
        # under CSR even though R itself carries the edge-truncation bias checked above.
        W = ((0.0, 30.0), (0.0, 30.0))
        pp = PoissonPointProcess(intensity=0.3, window=W)
        pvals = []
        for s in range(300):
            pts = pp.sample(random_state=s)
            if len(pts) > 5:
                pvals.append(NNDistanceTest(pts, W).pvalue)
        pvals = np.array(pvals)
        assert abs(np.mean(pvals < 0.05) - 0.05) < 0.05

    def test_g_function_csr_envelope(self):
        W = ((0.0, 15.0), (0.0, 15.0))
        pp = PoissonPointProcess(intensity=0.5, window=W)
        pts = pp.sample(random_state=1)
        r = NNDistanceTest(pts, W, method="G", n_simulations=99, random_state=2)
        assert r.pvalue is not None and 0 <= r.pvalue <= 1


class TestSpatialWeights:
    def test_knn_matches_brute_force(self):
        coords = _RNG.uniform(0, 10, size=(20, 2))
        W = SpatialWeights.knn(coords, k=4)
        D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        for i in range(20):
            nn = np.argsort(D[i])[:4]
            assert set(np.where(W.W[i] > 0)[0]) == set(nn)

    def test_lattice_rook_and_queen_neighbor_counts(self):
        rook = SpatialWeights.lattice((5, 5), rule="rook")
        queen = SpatialWeights.lattice((5, 5), rule="queen")
        # interior cell (2,2) -> index 12
        assert rook.W[12].sum() == 4
        assert queen.W[12].sum() == 8
        # corner cell (0,0) -> index 0
        assert rook.W[0].sum() == 2
        assert queen.W[0].sum() == 3

    def test_distance_band_binary_matches_brute_force(self):
        coords = _RNG.uniform(0, 10, size=(15, 2))
        W = SpatialWeights.distance_band(coords, threshold=3.0)
        D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        expected = ((D > 0) & (D <= 3.0)).astype(float)
        assert np.array_equal(W.W, expected)

    def test_row_standardize_rows_sum_to_one(self):
        W = SpatialWeights.knn(_RNG.uniform(0, 10, size=(10, 2)), k=3).row_standardize()
        assert np.allclose(W.W.sum(axis=1), 1.0)


# ------------------------------------------------------------------------------- lattice

class TestLatticeModels:
    def test_sar_recovers_rho_and_beta(self):
        rng = np.random.default_rng(20)
        W = SpatialWeights.lattice((8, 8), rule="rook").row_standardize()
        n = W.n
        true_rho, beta_true = 0.4, np.array([1.0, 2.0])
        x = rng.normal(size=n)
        Xd = np.column_stack([np.ones(n), x])
        A = np.eye(n) - true_rho * W.W
        eps = rng.normal(scale=0.3, size=n)
        y = np.linalg.solve(A, Xd @ beta_true + eps)
        sar = SARModel(W).fit(y, x)
        assert sar.rho_ == pytest.approx(true_rho, abs=0.2)
        assert sar.beta_[0] == pytest.approx(1.0, abs=0.5)
        assert sar.beta_[1] == pytest.approx(2.0, abs=0.5)
        assert sar.std_errors_.shape == (3,)

    def test_sar_logdet_via_eigenvalues_matches_slogdet(self):
        W = SpatialWeights.lattice((6, 6)).row_standardize()
        eig = np.linalg.eigvals(W.W).real
        rho = 0.3
        via_eig = np.sum(np.log(np.abs(1 - rho * eig)))
        sign, direct = np.linalg.slogdet(np.eye(W.n) - rho * W.W)
        assert via_eig == pytest.approx(direct, rel=1e-6)

    def test_car_recovers_rho_and_matches_precision_covariance(self):
        rng = np.random.default_rng(21)
        W = SpatialWeights.lattice((7, 7), rule="rook")
        n = W.n
        true_rho, sigma2_true = 0.15, 0.5
        beta_true = np.array([1.0, -0.5])
        x = rng.normal(size=n)
        Xd = np.column_stack([np.ones(n), x])
        Q = (np.eye(n) - true_rho * W.W) / sigma2_true
        L = np.linalg.cholesky(np.linalg.inv(Q))
        y = Xd @ beta_true + L @ rng.standard_normal(n)
        car = CARModel(W).fit(y, x)
        assert car.rho_ == pytest.approx(true_rho, abs=0.15)
        assert car.sigma2_ == pytest.approx(sigma2_true, rel=0.5)

    def test_car_requires_symmetric_weights(self):
        W = SpatialWeights.knn(_RNG.uniform(0, 10, size=(10, 2)), k=3)  # generally asymmetric
        with pytest.raises(ValueError):
            CARModel(W).fit(_RNG.normal(size=10))

    def test_car_sample_covariance_matches_inverse_precision(self):
        W = SpatialWeights.lattice((5, 5))
        n = W.n
        car = CARModel(W)
        car.rho_, car.sigma2_ = 0.1, 1.0
        car.beta_ = np.array([0.0])
        car.X_ = np.ones((n, 1))
        car._W = W.W
        samples = np.array([car.sample(random_state=s) for s in range(3000)])
        emp_cov = np.cov(samples.T)
        Q = (np.eye(n) - 0.1 * W.W) / 1.0
        theory_cov = np.linalg.inv(Q)
        assert np.max(np.abs(emp_cov - theory_cov)) < 0.3


# -------------------------------------------------------------------------------- hygiene

class TestHygiene:
    def test_public_surface_is_unique(self):
        assert len(ss.__all__) == len(set(ss.__all__)) == 36

    def test_base_classes_and_subclass_relations(self):
        assert issubclass(OrdinaryKriging, Kriging)
        assert issubclass(UniversalKriging, Kriging)
        assert issubclass(Semivariogram, Variogram)
        for cls in (PoissonPointProcess, InhomogeneousPoisson, ThomasProcess,
                    MaternCluster, LogGaussianCox):
            assert issubclass(cls, SpatialPointProcess)

    def test_library_code_never_imports_scipy_stats_or_spatial(self):
        import ast
        import pathlib

        pkg_dir = pathlib.Path(ss.__file__).parent
        seen = 0
        for path in pkg_dir.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith("scipy.stats"), path
                    assert not node.module.startswith("scipy.spatial"), path
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("scipy.stats"), path
                        assert not alias.name.startswith("scipy.spatial"), path
            seen += 1
        assert seen >= 8

    def test_quickstart_example_runs(self):
        coords = np.random.default_rng(0).uniform(0, 10, size=(80, 2))
        values = np.random.default_rng(1).normal(size=80)

        ev = ExperimentalVariogram(bins=12).fit(coords, values)
        vf = VariogramFitting(model=["spherical", "exponential", "gaussian"]).fit(ev)

        ok = OrdinaryKriging(variogram=vf.variogram_).fit(coords, values)
        mean, std = ok.predict([[5.0, 5.0]], return_std=True)
        assert mean.shape == (1,) and std[0] >= 0

        k = RipleyK(coords, window=((0, 10), (0, 10)), n_simulations=99, random_state=2)
        assert 0 <= k.pvalue <= 1

        W = SpatialWeights.knn(coords, k=6)
        mi = MoransI(values, W)
        assert mi.pvalue is not None
