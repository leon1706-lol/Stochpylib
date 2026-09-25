"""End-to-end API sweep for stochpylib.experimental_design: one realistic exercise per
public name. ``test_every_public_name_is_exercised`` fails the moment a name ships without
one; ``test_exercise[<name>]`` runs each exercise as its own pytest case."""

import numpy as np
import pytest

from stochpylib import experimental_design as mod

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


_RNG = np.random.default_rng(0)


def _yield(X):
    """A chemical-yield-like response: concave in temperature and time, with an interaction."""
    X = np.asarray(X, dtype=float)
    return (80 + 4 * X[:, 0] + 2.5 * X[:, 1] - 3 * X[:, 0] ** 2 - 2 * X[:, 1] ** 2
            + 1.2 * X[:, 0] * X[:, 1])


def _ishigami(X):
    return np.sin(X[:, 0]) + 7 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])


_MONTGOMERY_Y = np.array([45, 71, 48, 65, 68, 60, 80, 65, 43, 100, 45, 104, 75, 86, 70, 96],
                         dtype=float)


# ------------------------------------------------------------------ base and result

@exercise("Design")
def _design():
    d = mod.FullFactorial(2, 2, bounds=[(150, 200), (1.0, 3.0)]).generate()
    assert isinstance(d, mod.Design) and np.asarray(d).shape == (4, 2)
    nat = d.to_natural()
    assert np.allclose(nat[[0, 3]], [[150, 1.0], [200, 3.0]])
    assert np.allclose(d.to_coded(nat), d.points)
    assert d.with_center_points(3).n_runs == 7
    assert d.d_efficiency() == pytest.approx(1.0)


@exercise("DesignGenerator")
def _design_generator():
    gen = mod.LatinSquare(4, random_state=1)
    assert isinstance(gen, mod.DesignGenerator)
    d = gen.generate()
    assert gen.design_ is d and d.n_runs == 16


@exercise("OptimalDesign")
def _optimal_design():
    gen = mod.D_OptimalDesign(10, 2, model="quadratic", random_state=0)
    assert isinstance(gen, mod.OptimalDesign)
    d = gen.generate()
    assert 0 < d.properties["D_efficiency"] <= 1 and d.properties["n_candidates"] == 9


# ------------------------------------------------------------------------- classical

@exercise("FullFactorial")
def _full_factorial():
    d = mod.FullFactorial([2, 3]).generate()
    assert d.n_runs == 6 and d.properties["levels"] == [2, 3]


@exercise("FractionalFactorial")
def _fractional_factorial():
    d = mod.FractionalFactorial(6, p=2).generate()
    assert d.n_runs == 16 and d.properties["resolution"] == 4
    assert "I" in d.properties["defining_relation"]


@exercise("Plackett_Burman")
def _plackett_burman():
    d = mod.Plackett_Burman(11).generate()
    H = np.column_stack([np.ones(12), d.points])
    assert d.n_runs == 12 and np.allclose(H.T @ H, 12 * np.eye(12))


@exercise("CCD")
def _ccd():
    d = mod.CCD(3, center=6).generate()
    assert d.n_runs == 8 + 6 + 6 and d.properties["alpha"] == pytest.approx(8 ** 0.25)


@exercise("BoxBehnken")
def _box_behnken():
    d = mod.BoxBehnken(4, center=3).generate()
    assert d.n_runs == 27 and set(np.unique(d.points)) == {-1.0, 0.0, 1.0}


@exercise("LatinSquare")
def _latin_square():
    ls = mod.LatinSquare(4, random_state=2)
    d = ls.generate()
    y = 10 + d.points[:, 2] + _RNG.normal(0, 0.2, 16)   # a real treatment effect
    assert ls.analyze(y).pvalue < 0.01


@exercise("GraecoLatin")
def _graeco_latin():
    g = mod.GraecoLatin(5, random_state=3)
    g.generate()
    pairs = {(a, b) for a, b in zip(g.latin_.ravel(), g.greek_.ravel())}
    assert len(pairs) == 25


# --------------------------------------------------------------------------- optimal

@exercise("D_OptimalDesign")
def _d_optimal():
    d = mod.D_OptimalDesign(6, 1, model="quadratic", random_state=0).generate()
    assert sorted(np.round(d.points[:, 0], 9)) == [-1, -1, 0, 0, 1, 1]


@exercise("A_OptimalDesign")
def _a_optimal():
    d = mod.A_OptimalDesign(8, 1, model="quadratic", random_state=0).generate()
    assert int(np.sum(d.points[:, 0] == 0)) == 4


@exercise("G_OptimalDesign")
def _g_optimal():
    d = mod.G_OptimalDesign(6, 1, model="quadratic", random_state=0).generate()
    assert d.properties["G_efficiency"] == pytest.approx(1.0)


@exercise("I_OptimalDesign")
def _i_optimal():
    d = mod.I_OptimalDesign(12, 2, model="quadratic", random_state=0).generate()
    assert d.n_runs == 12 and np.isfinite(d.properties["criterion"])


@exercise("T_OptimalDesign")
def _t_optimal():
    d = mod.T_OptimalDesign(4, lambda X: X[:, 0] ** 2, n_factors=1, random_state=0).generate()
    assert sorted(d.points[:, 0]) == [-1.0, 0.0, 0.0, 1.0]


@exercise("BayesianDesign")
def _bayesian_design():
    eta = lambda X, th: th[0] * np.exp(-th[1] * X[:, 0])
    d = mod.BayesianDesign(2, model=eta, prior=np.array([[1.0, 0.5]]), bounds=(0, 10),
                           levels=51, random_state=0).generate()
    assert sorted(d.points[:, 0]) == pytest.approx([0.0, 2.0])


# --------------------------------------------------------------------- space filling

@exercise("LatinHypercubeDesign")
def _latin_hypercube():
    d = mod.LatinHypercubeDesign(10, 3, criterion="maximin", n_iter=20,
                                 random_state=0).generate()
    strata = np.floor(d.points * 10).astype(int)
    assert all(sorted(strata[:, j]) == list(range(10)) for j in range(3))


@exercise("MaximinLHD")
def _maximin_lhd():
    d = mod.MaximinLHD(10, 2, n_iter=500, random_state=0).generate()
    assert d.properties["phi_p"] <= d.properties["phi_p_initial"]


@exercise("MinimaxDesign")
def _minimax():
    d = mod.MinimaxDesign(4, 2, n_ref=1024, n_iter=20, random_state=0).generate()
    assert d.properties["fill_distance"] < 0.4


@exercise("UniformDesign")
def _uniform_design():
    d = mod.UniformDesign(13, 2, random_state=0).generate()
    assert d.properties["method"] == "glp"
    assert d.discrepancy() == pytest.approx(d.properties["discrepancy"])


@exercise("OrthogonalArrayDesign")
def _orthogonal_array():
    oa = mod.OrthogonalArrayDesign(3, 4, strength=2)
    oa.generate()
    for a in range(4):
        for b in range(a + 1, 4):
            assert len({tuple(r) for r in oa.array_[:, [a, b]]}) == 9


# ------------------------------------------------------------------ response surface

@exercise("ResponseSurface")
def _response_surface():
    bounds = [(150, 200), (1.0, 3.0)]
    d = mod.CCD(2, center=5).generate()
    y = _yield(d.points) + _RNG.normal(0, 0.1, d.n_runs)
    rs = mod.ResponseSurface(2, bounds=bounds).fit(d.to_natural(bounds), y)
    can = rs.canonical_analysis()
    assert can["nature"] == "maximum"
    assert np.all(can["stationary_point"] >= [150, 1.0])
    assert np.all(can["stationary_point"] <= [200, 3.0])


@exercise("RSM_ANOVA")
def _rsm_anova():
    d = mod.CCD(2, center=5).generate()
    y = _yield(d.points) + _RNG.normal(0, 0.1, d.n_runs)
    ra = mod.RSM_ANOVA().fit(d.points, y)
    assert ra.r2_ > 0.99 and ra.lack_of_fit_ is not None
    assert ra.lack_of_fit_.pvalue > 0.01


@exercise("MetaModel")
def _metamodel():
    d = mod.CCD(2, center=3).generate()
    mm = mod.MetaModel(cv=4, random_state=0).fit(d.points, _yield(d.points))
    assert mm.scores_[mm.best_name_] < 1e-6


@exercise("PolynomialChaos")
def _polynomial_chaos():
    pce = mod.PolynomialChaos(8, bounds=[(-np.pi, np.pi)] * 3).fit_function(_ishigami)
    assert pce.mean_ == pytest.approx(3.5, abs=1e-3)
    assert int(np.argmax(pce.sobol_first_)) == 1


@exercise("KrigingSurrogate")
def _kriging():
    X = mod.MaximinLHD(12, 1, n_iter=100, random_state=0).generate().points
    f = lambda Z: np.sin(6 * Z[:, 0]) + Z[:, 0]
    kr = mod.KrigingSurrogate().fit(X, f(X))
    xs = np.linspace(0.05, 0.95, 50)[:, None]
    mu, sd = kr.predict(xs, return_std=True)
    assert np.max(np.abs(mu - f(xs))) < 0.1 and np.all(sd >= 0)


# -------------------------------------------------------------------------- analysis

@exercise("ANOVA_DOE")
def _anova_doe():
    d = mod.FullFactorial(2, 4).generate()
    res = mod.ANOVA_DOE(model=["A", "C", "D", "AC", "AD"]).fit(d, _MONTGOMERY_Y)
    assert set(res.significant_) == {"A", "C", "D", "AC", "AD"}


@exercise("MainEffects")
def _main_effects():
    d = mod.FullFactorial(2, 4).generate()
    me = mod.MainEffects().fit(d, _MONTGOMERY_Y)
    assert me.effects_["A"] == pytest.approx(21.625) and me.ranking_[0] == "A"


@exercise("InteractionPlot")
def _interaction_plot():
    d = mod.FullFactorial(2, 4).generate()
    ip = mod.InteractionPlot(("A", "C")).fit(d, _MONTGOMERY_Y)
    assert ip.interaction_effect_ == pytest.approx(-18.125) and not ip.is_parallel()


@exercise("NormalPlot")
def _normal_plot():
    d = mod.FullFactorial(2, 4).generate()
    npl = mod.NormalPlot().fit(d, _MONTGOMERY_Y)
    assert set(npl.active_) == {"A", "C", "D", "AC", "AD"}
    assert npl.pse_ == pytest.approx(2.625)


@exercise("SensitivityIndex")
def _sensitivity_index():
    si = mod.SensitivityIndex("morris", n_trajectories=10, random_state=0).analyze(
        _ishigami, bounds=[(-np.pi, np.pi)] * 3)
    assert si.mu_star_[2] > 0 and si.n_evals_ == 40


@exercise("SobolIndex")
def _sobol_index():
    so = mod.SobolIndex(n_samples=2048, n_bootstrap=50, random_state=0).analyze(
        _ishigami, bounds=[(-np.pi, np.pi)] * 3)
    assert abs(so.S1_[1] - 0.4424) < 0.05 and abs(so.ST_[2] - 0.2437) < 0.05
    lo, hi = so.total_order_[0].confidence_interval()
    assert lo < so.ST_[0] < hi


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
