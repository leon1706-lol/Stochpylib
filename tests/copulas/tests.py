"""Tests for stochpylib.copulas.

Covers: pseudo-observations & dependence measures (vs scipy.stats oracle),
elliptical copulas (CDF vs independent quadrature oracle, sampling),
Archimedean families (textbook CDF formulas, generator round-trips,
density-vs-finite-difference, Genest-MacKay tau, sampler margins/tau),
empirical copulas (boundary identities, checkerboard mass, Bernstein
convergence), vines (fit/loglik, sequential-sampler Rosenblatt uniformity on
realization-diagonal edges, adjacent-margin taus, refit stability) and the
module-level methods (CopulaFit ranking, tail_dependence, conditional).

All randomness is seeded. No scipy.stats in library code — it is the
test oracle here only.
"""

import numpy as np
import pytest
from scipy import stats

import stochpylib
from stochpylib import copulas as cp
from stochpylib.copulas._utils import (
    kendall_tau_estimate, pseudo_obs, student_t_ppf,
)
from stochpylib.copulas.archimedean import (
    AliMikhailHaqCopula, BB1Copula, BB7Copula, ClaytonCopula, FrankCopula,
    GumbelCopula, JoeCopula, PlackettCopula,
)
from stochpylib.copulas.elliptical import GaussianCopula, StudentTCopula
from stochpylib.copulas.empirical import (
    BetaCopula, CheckerboardCopula, EmpiricalCopula,
)
from stochpylib.copulas.vine import CVine, DVine, RVine, VineStructureSelect

# ------------------------------------------------------------------ helpers

GRID = np.linspace(0.05, 0.95, 9)
_UU, _VV = np.meshgrid(GRID, GRID)
PTS = np.column_stack([_UU.ravel(), _VV.ravel()])
INT = np.linspace(0.15, 0.85, 7)
_UI, _VI = np.meshgrid(INT, INT)
PTS_INT = np.column_stack([_UI.ravel(), _VI.ravel()])


def _textbook_cdf(name):
    def c_clayton(u, v, t):
        return np.maximum(u ** -t + v ** -t - 1.0, 1e-300) ** (-1.0 / t)

    def c_gumbel(u, v, t):
        return np.exp(-((-np.log(u)) ** t + (-np.log(v)) ** t) ** (1.0 / t))

    def c_frank(u, v, t):
        return -np.log1p((np.exp(-t * u) - 1) * (np.exp(-t * v) - 1)
                         / np.expm1(-t)) / t

    def c_joe(u, v, t):
        return 1 - ((1 - u) ** t + (1 - v) ** t
                    - (1 - u) ** t * (1 - v) ** t) ** (1.0 / t)

    def c_amh(u, v, t):
        return u * v / (1 - t * (1 - u) * (1 - v))

    def c_bb1(u, v, t, d):
        return (1 + ((u ** -t - 1) ** d + (v ** -t - 1) ** d)
                ** (1.0 / d)) ** (-1.0 / t)

    def c_bb7(u, v, t, d):
        return 1 - (1 - ((1 - (1 - u) ** t) ** -d
                         + (1 - (1 - v) ** t) ** -d - 1) ** (-1.0 / d)) \
            ** (1.0 / t)

    def c_plackett(u, v, t):
        if abs(t - 1) < 1e-12:
            return u * v
        s = 1 + (t - 1) * (u + v)
        disc = np.maximum(s * s - 4 * t * (t - 1) * u * v, 0.0)
        return (s - np.sqrt(disc)) / (2 * (t - 1))

    return dict(clayton=c_clayton, gumbel=c_gumbel, frank=c_frank, joe=c_joe,
                amh=c_amh, bb1=c_bb1, bb7=c_bb7, plackett=c_plackett)[name]


# ------------------------------------------------------------------ utils

def test_pseudo_obs_in_open_cube_and_rank_preserving():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((100, 3))
    u = pseudo_obs(data)
    assert u.shape == (100, 3)
    assert np.all(u > 0) and np.all(u < 1)
    for j in range(3):
        assert (np.argsort(np.argsort(data[:, j])) ==
                np.argsort(np.argsort(u[:, j]))).all()


def test_kendall_tau_matches_scipy_with_and_without_ties():
    rng = np.random.default_rng(1)
    for trial in range(4):
        n = int(rng.integers(30, 400))
        x = rng.standard_normal(n)
        y = 0.6 * x + rng.standard_normal(n)
        if trial >= 2:
            x = np.round(x, 1)
            y = np.round(y, 1)
        assert abs(kendall_tau_estimate(x, y) -
                   stats.kendalltau(x, y)[0]) < 1e-10


def test_student_t_ppf_matches_scipy():
    q = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    for df in (2.5, 5, 30):
        assert np.max(np.abs(student_t_ppf(q, df) -
                             stats.t.ppf(q, df))) < 1e-8


# ------------------------------------------------------------------ elliptical

def test_gaussian_copula_fit_sample_and_cdf_vs_oracle():
    from scipy.integrate import quad
    rng = np.random.default_rng(3)
    data = rng.multivariate_normal([0, 0], [[1, .6], [.6, 1]], 3000)
    gp = GaussianCopula().fit(data)
    assert abs(gp.correlation_[0, 1] - 0.6) < 0.03
    s = gp.sample(50000, random_state=0)
    rho_emp = np.corrcoef(stats.norm.ppf(s.T))[0, 1]
    assert abs(rho_emp - 0.6) < 0.02

    rho = float(gp.correlation_[0, 1])

    def bvn(h, k):
        f = lambda x: stats.norm.cdf(
            (k - rho * x) / np.sqrt(1 - rho ** 2)) * stats.norm.pdf(x)
        return quad(f, -12, h)[0]

    for pt in [(.3, .4), (.7, .2), (.9, .9), (.05, .5)]:
        ours = float(gp.cdf([pt])[0])
        ref = bvn(stats.norm.ppf(pt[0]), stats.norm.ppf(pt[1]))
        assert abs(ours - ref) < 1e-6


def test_gaussian_copula_boundary_and_density_center():
    gp = GaussianCopula(dimension=2)
    gp.correlation_ = np.array([[1.0, .6], [.6, 1.0]])
    gp.dimension = 2
    assert abs(float(gp.cdf([[.3, 1.0]])[0]) - 0.3) < 1e-9
    assert abs(float(gp.density([[.5, .5]])[0]) -
               1 / np.sqrt(1 - .36)) < 1e-8


def test_student_t_copula_df_recovery_and_tail_dependence():
    rng1 = np.random.default_rng(5)
    base = rng1.standard_t(4, 3000)
    data = np.column_stack([base, .7 * base +
                            np.sqrt(.51) * rng1.standard_t(8, 3000)])
    tc = StudentTCopula().fit(data)
    assert 3.0 < tc.df_ < 9.0
    lam = tc.tail_dependence()["upper"]
    assert 0.02 < lam < 0.5


def test_student_t_h_function_matches_textbook_formula():
    tc = StudentTCopula(dimension=2, df=5)
    tc.correlation_ = np.array([[1.0, .5], [.5, 1.0]])
    tc.df_ = 5.0
    from stochpylib.copulas.elliptical import _t_cdf
    w = np.array([.3, .7])
    v = np.array([.4, .8])
    rho, nu = .5, 5.0
    a = student_t_ppf(w, nu)
    b = student_t_ppf(v, nu)
    scale = np.sqrt((1 - rho ** 2) * (nu + b ** 2) / (nu + 1))
    expected = _t_cdf((a - rho * b) / scale, nu + 1)
    assert np.allclose(tc._h_u(w, v), expected)


# ------------------------------------------------------------------ archimedean

_ARCH = [
    ("clayton", lambda: ClaytonCopula(theta=2.3), lambda u, v, p:
     _textbook_cdf("clayton")(u, v, p["theta"])),
    ("gumbel", lambda: GumbelCopula(theta=2.5), lambda u, v, p:
     _textbook_cdf("gumbel")(u, v, p["theta"])),
    ("frank", lambda: FrankCopula(theta=6.0), lambda u, v, p:
     _textbook_cdf("frank")(u, v, p["theta"])),
    ("frank_neg", lambda: FrankCopula(theta=-4.0), lambda u, v, p:
     _textbook_cdf("frank")(u, v, p["theta"])),
    ("joe", lambda: JoeCopula(theta=3.0), lambda u, v, p:
     _textbook_cdf("joe")(u, v, p["theta"])),
    ("amh", lambda: AliMikhailHaqCopula(theta=0.6), lambda u, v, p:
     _textbook_cdf("amh")(u, v, p["theta"])),
    ("bb1", lambda: BB1Copula(theta=1.4, delta=1.7), lambda u, v, p:
     _textbook_cdf("bb1")(u, v, p["theta"], p["delta"])),
    ("bb7", lambda: BB7Copula(theta=1.8, delta=1.1), lambda u, v, p:
     _textbook_cdf("bb7")(u, v, p["theta"], p["delta"])),
    ("plackett", lambda: PlackettCopula(theta=9.0), lambda u, v, p:
     _textbook_cdf("plackett")(u, v, p["theta"])),
]


@pytest.mark.parametrize("name,maker,ref", _ARCH,
                         ids=[a[0] for a in _ARCH])
def test_archimedean_cdf_matches_textbook(name, maker, ref):
    c = maker()
    c.dimension = 2
    kw = {}
    if hasattr(c, "theta_") and c.theta_ is not None:
        kw["theta"] = c.theta_
    if hasattr(c, "delta_") and c.delta_ is not None:
        kw["delta"] = c.delta_
    ours = np.asarray(c.cdf(PTS), dtype=float)
    theirs = ref(PTS[:, 0], PTS[:, 1], kw)
    assert np.max(np.abs(ours - theirs)) < 5e-10


@pytest.mark.parametrize("name,maker,_", _ARCH[::2],
                         ids=[a[0] for a in _ARCH][::2])
def test_archimedean_density_matches_finite_difference(name, maker, _):
    c = maker()
    c.dimension = 2
    h = 1e-4

    def sh(dx, dy):
        pts = PTS_INT.copy()
        pts[:, 0] += dx
        pts[:, 1] += dy
        return np.asarray(c.cdf(pts), dtype=float)

    fd = (sh(h, h) - sh(h, -h) - sh(-h, h) + sh(-h, -h)) / (4 * h * h)
    ours = np.asarray(c.density(PTS_INT), dtype=float)
    scale = max(float(np.median(np.abs(fd))), 1e-6)
    assert np.max(np.abs(ours - fd)) < 2e-3 * max(scale, 0.05)


@pytest.mark.parametrize("name,maker,_", _ARCH,
                         ids=[a[0] for a in _ARCH])
def test_archimedean_sampler_margins_and_tau(name, maker, _):
    c = maker()
    c.dimension = 2
    n = 12000
    s = c.sample(n, random_state=42)
    assert s.min() >= 0 and s.max() <= 1
    assert abs(s[:, 0].mean() - .5) < .02
    assert abs(s[:, 1].mean() - .5) < .02
    tau_th = (c.kendall_tau() if not isinstance(c, PlackettCopula)
              else c.kendall_tau())
    tau_emp = kendall_tau_estimate(s[:, 0], s[:, 1])
    se = 1.0 / np.sqrt(n - 1)
    assert abs(tau_emp - tau_th) < 5 * se


def test_clayton_analytic_tail_dependence():
    c = ClaytonCopula(theta=2.0)
    assert abs(c.tail_dependence()["lower"] - 2 ** -.5) < 1e-12
    g = GumbelCopula(theta=2.5)
    assert abs(g.tail_dependence()["upper"] - (2 - 2 ** .4)) < 1e-12


def test_generator_roundtrip_all_families():
    fams = [ClaytonCopula(theta=2.), GumbelCopula(theta=2.),
            FrankCopula(theta=4.), JoeCopula(theta=2.5),
            AliMikhailHaqCopula(theta=.6), BB1Copula(theta=1.4, delta=1.7),
            BB7Copula(theta=1.8, delta=1.1)]
    u = GRID
    for c in fams:
        rt = np.asarray(c._psi(c._psi_inv(u)), dtype=float)
        assert np.max(np.abs(rt - u)) < 1e-9, type(c).__name__


def test_fit_recovers_theta_from_samples():
    for cls, theta, attr in [(ClaytonCopula, 2.0, "theta_"),
                             (GumbelCopula, 3.0, "theta_"),
                             (FrankCopula, 5.0, "theta_")]:
        data = cls(theta=theta).sample(8000, random_state=7)
        fitted = cls().fit(data)
        # tau inversion: fitted parameter implies close tau to the truth
        tau_true = cls(theta=theta).kendall_tau()
        tau_fit = cls(theta=getattr(fitted, attr)).kendall_tau()
        assert abs(tau_true - tau_fit) < 0.03, cls.__name__


# ------------------------------------------------------------------ empirical

_dep_cache = {}


def _dep_data(n=2500):
    """Gaussian-copula pseudo-data with rho=0.6 (cached)."""
    if "data" not in _dep_cache:
        rng = np.random.default_rng(9)
        z = rng.multivariate_normal([0, 0], [[1, .6], [.6, 1]], n)
        _dep_cache["data"] = stats.norm.cdf(z)
    return _dep_cache["data"]


def test_empirical_copula_matches_bruteforce_and_frechet():
    ec = EmpiricalCopula().fit(_dep_data())
    obs = ec.u_obs_
    for pt in [(.25, .25), (.5, .75), (.8, .4)]:
        brute = np.mean(np.all(obs <= pt, axis=1))
        assert abs(float(ec.cdf([pt])[0]) - brute) < 1e-12
    assert float(ec.cdf([[.8, .4]])[0]) >= 0.2          # Frechet lower bound
    smp = ec.sample(500, random_state=2)
    assert smp.shape == (500, 2)


def test_checkerboard_mass_boundaries_and_density():
    cb = CheckerboardCopula(n_bins=20).fit(_dep_data())
    assert abs(cb.cell_mass_.sum() - 1.0) < 1e-12
    assert abs(float(cb.cdf([[1., 1.]])[0]) - 1.0) < 1e-9
    assert float(cb.cdf([[.7, 0.0]])[0]) == 0.0
    g = np.linspace(.001, .999, 99)
    uu, vv = np.meshgrid(g, g)
    d = cb.density(np.column_stack([uu.ravel(), vv.ravel()]))
    mass = np.trapezoid(np.trapezoid(d.reshape(99, 99), g, axis=1), g)
    assert 0.95 < mass < 1.05
    s = cb.sample(8000, random_state=3)
    assert np.all(np.abs(s.mean(axis=0) - 0.5) < 0.02)


def test_beta_copula_converges_to_checkerboard():
    bc20 = BetaCopula(n_bins=20).fit(_dep_data())
    bc60 = BetaCopula(n_bins=60).fit(_dep_data())
    cb = CheckerboardCopula(n_bins=60).fit(_dep_data())
    pt = (.5, .75)
    d60 = abs(float(bc60.cdf([pt])[0]) - float(cb.cdf([pt])[0]))
    d20 = abs(float(bc20.cdf([pt])[0]) - float(cb.cdf([pt])[0]))
    assert d60 < d20
    assert abs(float(bc60.cdf([[1., 1.]])[0]) - 1.0) < 1e-6


# ------------------------------------------------------------------ vines

_VINE_R = np.array([[1, .7, .4, .3, .2], [.7, 1, .5, .2, .1],
                    [.4, .5, 1, .6, .3], [.3, .2, .6, 1, .4],
                    [.2, .1, .3, .4, 1.]])
_rng_v = np.random.default_rng(2026)
_vz = _rng_v.standard_normal((1200, 5)) @ np.linalg.cholesky(_VINE_R).T
VINE_DATA = stats.norm.cdf(_vz)


@pytest.mark.parametrize("cls", [DVine, CVine, RVine])
def test_vine_fit_loglik_beats_independence(cls):
    v = cls(families=("gaussian",)).fit(VINE_DATA)
    assert np.isfinite(v.loglik_) and v.loglik_ > 50.0


@pytest.mark.parametrize("cls", [DVine, CVine])
def test_vine_sequential_sampler_is_rosenblatt_consistent(cls):
    """Realization-diagonal edges must produce uniform h-columns, and every
    tree-1 pair's sampled margin tau must equal its effective pair tau."""
    v = cls(families=("gaussian",)).fit(VINE_DATA)
    s = v.sample(20000, random_state=11)
    diag = {id(e) for e, _var in v._intro_plan_ if e is not None}
    for e in v._all_edges:
        if id(e) not in diag:
            continue
        x, y = v._columns(e, s)
        w = np.asarray(e.pair.h(np.clip(x, 1e-9, 1 - 1e-9),
                                np.clip(y, 1e-9, 1 - 1e-9)))
        assert stats.kstest(w, "uniform").statistic < 0.02
    prng = np.random.default_rng(777)
    for e in v.levels_[0]:
        a, b = sorted(e.heads)
        emp = kendall_tau_estimate(s[:, a], s[:, b])
        pv = prng.random(20000)
        bv = prng.random(20000)
        av = np.clip(e.pair.h_inv(pv, bv), 1e-9, 1 - 1e-9)
        mod = kendall_tau_estimate(av, bv)
        se = 1.0 / np.sqrt(20000 - 1)
        assert abs(emp - mod) < 5 * se, (a, b, emp, mod)


def test_dvine_refit_on_own_samples_is_stable():
    v1 = DVine(families=("gaussian",)).fit(VINE_DATA)
    s1 = v1.sample(3000, random_state=21)
    v2 = DVine(families=("gaussian",)).fit(s1)
    t1 = v1.kendall_tau()
    t2 = v2.kendall_tau()
    assert np.max(np.abs(t1 - t2)) < 0.12


def test_vine_structure_select_returns_finite_aic():
    best = VineStructureSelect(VINE_DATA, types=("CVine", "DVine"))
    assert type(best).__name__ in ("CVine", "DVine")
    assert np.isfinite(best.aic(VINE_DATA))


def test_vinecopula_facade():
    vc = cp.VineCopula(type="DVine").fit(VINE_DATA)
    assert vc.sample(200, random_state=0).shape == (200, 5)
    with pytest.raises(NotImplementedError):
        vc.cdf([[.5, .5]])


def test_pair_rotation_h_functions_match_fd_of_rotated_cdf():
    from stochpylib.copulas.pair import PairCopulaConstruction
    q = GumbelCopula(theta=3.0)
    q.dimension = 2
    us = np.array([.2, .5, .8])
    vs = np.array([.3, .55, .85])

    def crot(u, v, rot):
        if rot == 0:
            return float(q.cdf([[u, v]])[0])
        if rot == 90:
            return v - float(q.cdf([[1 - u, v]])[0])
        if rot == 180:
            return u + v - 1 + float(q.cdf([[1 - u, 1 - v]])[0])
        return u - float(q.cdf([[u, 1 - v]])[0])

    h = 1e-6
    for rot in (0, 90, 180, 270):
        pc = PairCopulaConstruction(q, rot)
        for uu, vv in zip(us, vs):
            fd = (crot(uu, vv + h, rot) - crot(uu, vv - h, rot)) / (2 * h)
            ana = float(pc.h(np.array([uu]), np.array([vv]))[0])
            assert abs(fd - ana) < 1e-4


# ------------------------------------------------------------------ methods

def test_copulafit_ranks_true_family_first():
    data = ClaytonCopula(theta=3.0).sample(3000, random_state=5)
    fit = cp.CopulaFit().fit(data)
    assert fit.best_name_ == "clayton"
    assert fit.table_[0][0] == "clayton"
    s = cp.CopulaSample(fit.best_).sample(4000, random_state=1)
    tau_s = kendall_tau_estimate(s[:, 0], s[:, 1])
    assert abs(tau_s - fit.best_.kendall_tau()) < 0.04


def test_methods_dependence_measures():
    rng = np.random.default_rng(12)
    x = rng.standard_normal(500)
    y = .5 * x + rng.standard_normal(500)
    assert abs(cp.kendall_tau(x, y) - stats.kendalltau(x, y)[0]) < 1e-10
    assert abs(cp.spearman_rho(x, y) - stats.spearmanr(x, y)[0]) < 1e-10


def test_tail_dependence_analytic_and_empirical():
    data = ClaytonCopula(theta=3.0).sample(4000, random_state=8)
    c = ClaytonCopula().fit(data)
    out = cp.tail_dependence(c, data=data)
    # fitted theta differs slightly from 3, so compare against its OWN lambda
    assert abs(out["lower"] - 2 ** (-1.0 / c.theta_)) < 1e-9
    assert "lower_emp" in out and "upper_emp" in out


def test_conditional_copula_boundaries():
    c = GumbelCopula(theta=2.5)
    c.dimension = 2
    out = cp.conditional_copula(c, [0.0, 1.0], [0.5, 0.5])
    assert np.allclose(out, [0.0, 1.0])


# ------------------------------------------------------------------ package

def test_module_wiring_and_exports():
    assert "copulas" in stochpylib.__all__
    assert hasattr(stochpylib, "copulas")
    for name in ("GaussianCopula", "ClaytonCopula", "BB1Copula",
                 "VineCopula", "CopulaFit"):
        assert name in cp.__all__


def test_doctests_pass():
    import doctest
    import stochpylib.copulas._utils as _utils
    import stochpylib.copulas.empirical as _emp
    assert doctest.testmod(_utils).failed == 0
    assert doctest.testmod(_emp).failed == 0
