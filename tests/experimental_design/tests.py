"""Tests for stochpylib.experimental_design: classical, optimal and space-filling designs,
response surfaces and surrogates, and the analysis of designed experiments.

Oracles: statsmodels OLS / anova_lm (response-surface coefficients, prediction standard
errors, Type I/II ANOVA tables, Latin-square ANOVA), scipy.stats.qmc.discrepancy
(CD/WD/MD/L2-star), scipy.stats.sobol_indices and scipy.stats.norm, brute-force
enumeration of exact optimal designs, the published Plackett & Burman (1946) generator
rows, the known maximum resolutions of regular fractions (Montgomery, Table 8.14),
Montgomery's (Example 6.2) filtration-rate 2^4 data and its textbook effects, the
Ishigami function's analytic Sobol indices, Box & Lucas' (1959) optimal design for
exponential decay, and the library's own statistics / montecarlo / gaussian_processes
answers. All randomness is seeded; statistical assertions are set at >= 3 standard errors.
"""

import itertools

import numpy as np
import pytest
from scipy import stats
from scipy.stats import qmc

from stochpylib import experimental_design as ed
from stochpylib.distributions import Exponential, Normal, Uniform
from stochpylib.experimental_design import (
    ANOVA_DOE, A_OptimalDesign, BayesianDesign, BoxBehnken, CCD, D_OptimalDesign, Design,
    DesignGenerator, FractionalFactorial, FullFactorial, G_OptimalDesign, GraecoLatin,
    I_OptimalDesign, InteractionPlot, KrigingSurrogate, LatinHypercubeDesign, LatinSquare,
    MainEffects, MaximinLHD, MetaModel, MinimaxDesign, NormalPlot, OptimalDesign,
    OrthogonalArrayDesign, Plackett_Burman, PolynomialChaos, RSM_ANOVA, ResponseSurface,
    SensitivityIndex, SobolIndex, T_OptimalDesign, UniformDesign,
)
from stochpylib.experimental_design._common import (
    _are_orthogonal, _gf_tables, _is_latin, _is_lhs, _model_matrix, _ranks,
)

pd = pytest.importorskip("pandas")
smf = pytest.importorskip("statsmodels.formula.api")
from statsmodels.stats.anova import anova_lm  # noqa: E402

_RNG = np.random.default_rng(0)

# Montgomery, Design and Analysis of Experiments, Example 6.2 (pilot-plant filtration
# rate, 2^4 unreplicated), in standard order
MONTGOMERY_Y = np.array([45, 71, 48, 65, 68, 60, 80, 65, 43, 100, 45, 104, 75, 86, 70, 96],
                        dtype=float)
MONTGOMERY_EFFECTS = {"A": 21.625, "B": 3.125, "C": 9.875, "D": 14.625, "AB": 0.125,
                      "AC": -18.125, "AD": 16.625, "BC": 2.375, "BD": -0.375, "CD": -1.125,
                      "ABC": 1.875, "ABD": 4.125, "ACD": -1.625, "BCD": -2.625,
                      "ABCD": 1.375}

ISHIGAMI_S1 = np.array([0.313905191, 0.442411145, 0.0])
ISHIGAMI_ST = np.array([0.557588855, 0.442411145, 0.243683664])
ISHIGAMI_BOUNDS = [(-np.pi, np.pi)] * 3


def ishigami(X, a=7.0, b=0.1):
    return np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0])


def branin01(X):
    x1, x2 = 15 * X[:, 0] - 5, 15 * X[:, 1]
    return ((x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10)


def _counts(design):
    v = np.round(np.asarray(design)[:, 0], 9)
    return tuple(int(np.sum(v == x)) for x in (-1.0, 0.0, 1.0))


def _signs(row):
    return "".join("+" if v > 0 else "-" for v in row)


# ========================================================================== Design

def test_design_array_protocol_and_unit_conversions():
    d = FullFactorial(2, 2, bounds=[(10, 20), (0, 1)]).generate()
    assert isinstance(d, Design) and np.asarray(d).shape == (4, 2) and len(d) == 4
    assert d.n_runs == 4 and d.n_factors == 2 and d.shape == (4, 2)
    nat = d.to_natural()
    assert np.allclose(nat, [[10, 0], [20, 0], [10, 1], [20, 1]])
    assert np.allclose(d.to_coded(nat), d.points)
    assert np.allclose(d.unit_points(), (d.points + 1) / 2)
    u = LatinHypercubeDesign(6, 2, random_state=0).generate()
    assert np.allclose(u.to_natural([(0, 10), (5, 6)]), [0, 5] + u.points * [10, 1])
    with pytest.raises(ValueError):
        FullFactorial(2, 2).generate().to_natural()


def test_design_randomized_center_points_and_model_matrix():
    d = FullFactorial(2, 3).generate()
    r = d.randomized(random_state=1)
    assert sorted(map(tuple, r.points)) == sorted(map(tuple, d.points))
    assert np.array_equal(r.points, d.points[r.run_order])
    c = d.with_center_points(3)
    assert c.n_runs == 11 and np.all(c.points[-3:] == 0) and c.properties["n_center"] == 3
    F, names = d.model_matrix("interaction")
    assert names == ["1", "A", "B", "C", "A*B", "A*C", "B*C"]
    assert np.allclose(F.T @ F, 8 * np.eye(7))
    assert d.d_efficiency("linear") == pytest.approx(1.0)


def test_factor_names_skip_I():
    names = FullFactorial(2, 10).generate().factor_names
    assert names == list("ABCDEFGHJK")


def test_model_matrix_terms():
    X = np.array([[1.0, 2.0], [3.0, -1.0]])
    F, names = _model_matrix(X, "quadratic")
    assert names == ["1", "A", "B", "A*B", "A^2", "B^2"]
    assert np.allclose(F[0], [1, 1, 2, 2, 1, 4])
    F, names = _model_matrix(X, "cubic")
    assert F.shape[1] == 10
    F, names = _model_matrix(X, [(0, 0), (2, 1)])
    assert names == ["1", "A^2*B"] and np.allclose(F[:, 1], [2, -9])
    F, names = _model_matrix(X, lambda Z: np.column_stack([Z[:, 0], Z[:, 0] * 2]))
    assert names == ["f0", "f1"]
    with pytest.raises(ValueError):
        _model_matrix(X, "sextic")


# ================================================================ classical designs

def test_full_factorial_yates_order_and_product_rows():
    d = FullFactorial(2, 3).generate()
    assert np.array_equal(d.points, [[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
                                     [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]])
    assert np.allclose(d.points.T @ d.points, 8 * np.eye(3))
    m = FullFactorial([2, 3, 4]).generate()
    assert m.n_runs == 24
    expected = set(itertools.product(np.linspace(-1, 1, 2), np.linspace(-1, 1, 3),
                                     np.linspace(-1, 1, 4)))
    assert set(map(tuple, np.round(m.points, 12))) == {tuple(np.round(e, 12)) for e in expected}
    for j, lv in enumerate((2, 3, 4)):
        _, counts = np.unique(m.points[:, j], return_counts=True)
        assert len(counts) == lv and np.all(counts == 24 // lv)


def test_full_factorial_explicit_levels_are_natural_units():
    d = FullFactorial([[100, 150, 200], [1.5, 3.0]]).generate()
    assert d.space == "natural" and d.n_runs == 6
    assert set(d.points[:, 0]) == {100, 150, 200}
    assert d.properties["level_values"][1] == [1.5, 3.0]
    with pytest.raises(ValueError):
        FullFactorial(2).generate()
    with pytest.raises(ValueError):
        FullFactorial([1, 2]).generate()


@pytest.mark.parametrize("k,p,res", [(3, 1, 3), (4, 1, 4), (5, 1, 5), (5, 2, 3), (6, 1, 6),
                                     (6, 2, 4), (6, 3, 3), (7, 1, 7), (7, 2, 4), (7, 3, 4),
                                     (7, 4, 3), (8, 2, 5), (8, 3, 4), (8, 4, 4)])
def test_fractional_factorial_search_reaches_the_maximum_resolution(k, p, res):
    d = FractionalFactorial(k, p=p).generate()
    assert d.properties["resolution"] == res
    assert d.n_runs == 2 ** (k - p) and d.n_factors == k
    # every column balanced, every pair of columns orthogonal
    assert np.allclose(d.points.sum(0), 0)
    assert np.allclose(d.points.T @ d.points, d.n_runs * np.eye(k))
    assert len(d.properties["defining_relation"]) == 2 ** p


def test_fractional_factorial_generated_columns_are_generator_products():
    d = FractionalFactorial(generators="D=AB E=AC").generate()
    P = d.points
    assert d.factor_names == list("ABCDE") and d.n_runs == 8
    assert np.array_equal(P[:, 3], P[:, 0] * P[:, 1])
    assert np.array_equal(P[:, 4], P[:, 0] * P[:, 2])
    assert sorted(d.properties["defining_relation"]) == sorted(["I", "ABD", "ACE", "BCDE"])
    assert d.properties["resolution"] == 3
    assert d.properties["word_length_pattern"][:4] == (0, 0, 2, 1)
    six = FractionalFactorial(generators="E=ABC F=BCD").generate()
    assert six.properties["resolution"] == 4 and six.n_runs == 16
    assert np.array_equal(six.points[:, 5], np.prod(six.points[:, 1:4], axis=1))


def test_fractional_factorial_aliases_and_generator_formats():
    half = FractionalFactorial(generators=["ABC"]).generate()
    assert half.properties["defining_relation"] == ["I", "ABCD"]
    assert half.properties["aliases"]["A"] == ["BCD"]
    assert half.properties["aliases"]["AB"] == ["CD"]
    pydoe = FractionalFactorial(generators="a b c abc").generate()
    assert np.array_equal(pydoe.points, half.points)
    sat = FractionalFactorial(generators="D=AB E=AC F=BC G=ABC")
    d = sat.generate()
    assert d.properties["resolution"] == 3 and d.n_runs == 8 and d.n_factors == 7
    fold = sat.fold_over()
    assert fold.n_runs == 16 and fold.properties["resolution"] == 4
    assert np.allclose(fold.points.T @ fold.points, 16 * np.eye(7))


def test_fractional_factorial_resolution_target_and_errors():
    d = FractionalFactorial(7, resolution=4).generate()
    assert d.properties["fraction"] == "2^(7-3)" and d.properties["resolution"] >= 4
    with pytest.raises(ValueError):
        FractionalFactorial(4, resolution=6).generate()
    with pytest.raises(ValueError):
        FractionalFactorial(4, p=3).generate()
    with pytest.raises(ValueError):
        FractionalFactorial(generators="D=A").generate()
    with pytest.raises(ValueError):
        FractionalFactorial(5).generate()


@pytest.mark.parametrize("N", [4, 8, 12, 16, 20, 24, 28, 32, 40, 44, 48])
def test_plackett_burman_is_a_hadamard_design(N):
    d = Plackett_Burman(N - 1).generate()
    assert d.n_runs == N
    H = np.column_stack([np.ones(N), d.points])
    assert np.allclose(H.T @ H, N * np.eye(N))


def test_plackett_burman_reproduces_the_published_generator_rows():
    assert _signs(Plackett_Burman(11).generate().points[0]) == "++-+++---+-"
    assert _signs(Plackett_Burman(19).generate().points[0]) == "++--++++-+-+----++-"
    assert _signs(Plackett_Burman(23).generate().points[0]) == "+++++-+-++--++--+-+----"
    assert Plackett_Burman(7).generate().n_runs == 8
    assert Plackett_Burman(9).generate().n_runs == 12
    # N = 36 has no construction here and is skipped to the next constructible order
    assert Plackett_Burman(33).generate().n_runs == 40


def _quadratic_prediction_variance(X, points):
    F, _ = _model_matrix(X, "quadratic")
    Mi = np.linalg.inv(F.T @ F)
    Fp, _ = _model_matrix(points, "quadratic")
    return np.einsum("ip,pq,iq->i", Fp, Mi, Fp)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_ccd_rotatable_alpha_gives_spherical_prediction_variance(k):
    d = CCD(k, alpha="rotatable").generate()
    assert d.properties["alpha"] == pytest.approx((2 ** k) ** 0.25)
    assert d.n_runs == 2 ** k + 2 * k + 4
    u = np.random.default_rng(k).normal(size=(6, k))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    var = _quadratic_prediction_variance(d.points, u)
    assert np.ptp(var) < 1e-10


@pytest.mark.parametrize("k,n0,alpha", [(2, 5, 1.26711), (3, 6, 1.52465), (4, 4, 1.60717)])
def test_ccd_orthogonal_alpha_makes_squared_columns_orthogonal(k, n0, alpha):
    d = CCD(k, alpha="orthogonal", center=n0).generate()
    assert d.properties["alpha"] == pytest.approx(alpha, abs=1e-5)
    Q = d.points ** 2
    Q = Q - Q.mean(0)
    G = Q.T @ Q
    assert np.allclose(G - np.diag(np.diag(G)), 0, atol=1e-10)


def test_ccd_face_inscribed_fraction_and_errors():
    assert CCD(3, alpha="rotatable").generate().properties["alpha"] == pytest.approx(1.681793,
                                                                                     abs=1e-6)
    face = CCD(3, alpha="face").generate()
    assert set(np.unique(face.points)) == {-1.0, 0.0, 1.0}
    ins = CCD(3, alpha="inscribed").generate()
    assert np.max(np.abs(ins.points)) == pytest.approx(1.0)
    frac = CCD(5, fraction=1).generate()
    assert frac.properties["n_factorial"] == 16 and frac.n_runs == 16 + 10 + 4
    assert CCD(2, alpha=1.5).generate().properties["alpha"] == 1.5
    with pytest.raises(ValueError):
        CCD(1).generate()
    with pytest.raises(ValueError):
        CCD(2, alpha="bogus").generate()


@pytest.mark.parametrize("k,n_edge", [(3, 12), (4, 24), (5, 40), (6, 48), (7, 56)])
def test_box_behnken_runs_levels_and_quadratic_estimability(k, n_edge):
    d = BoxBehnken(k, center=3).generate()
    assert d.properties["n_edge"] == n_edge and d.n_runs == n_edge + 3
    assert set(np.unique(d.points)) == {-1.0, 0.0, 1.0}
    # no run at a corner of the cube
    assert np.all(np.sum(np.abs(d.points), axis=1) < k)
    F, _ = d.model_matrix("quadratic")
    assert np.linalg.matrix_rank(F) == F.shape[1]
    # every factor is varied equally often
    nonzero = (d.points != 0).sum(0)
    assert np.all(nonzero == nonzero[0])


def test_box_behnken_rejects_unsupported_sizes():
    for k in (2, 8):
        with pytest.raises(ValueError):
            BoxBehnken(k)


def test_latin_square_property_and_anova_matches_statsmodels():
    ls = LatinSquare(5, random_state=3)
    d = ls.generate()
    assert _is_latin(ls.square_)
    assert d.factor_names == ["row", "column", "treatment"] and d.n_runs == 25
    assert np.array_equal(d.points[:, 2].reshape(5, 5), ls.square_)
    rng = np.random.default_rng(4)
    y = 2 + 0.5 * d.points[:, 2] + rng.normal(0, 1, 25)
    res = ls.analyze(y)
    df = pd.DataFrame({"r": d.points[:, 0], "c": d.points[:, 1], "t": d.points[:, 2], "y": y})
    ref = anova_lm(smf.ols("y ~ C(r) + C(c) + C(t)", df).fit(), typ=1)
    got = {row["source"]: row for row in res.table}
    for src, key in (("Row", "C(r)"), ("Column", "C(c)"), ("Treatment", "C(t)")):
        assert got[src]["ss"] == pytest.approx(ref.loc[key, "sum_sq"], rel=1e-10)
        assert got[src]["F"] == pytest.approx(ref.loc[key, "F"], rel=1e-8)
        assert got[src]["p"] == pytest.approx(ref.loc[key, "PR(>F)"], rel=1e-6)
    assert got["Residual"]["df"] == 12
    assert res.pvalue == pytest.approx(ref.loc["C(t)", "PR(>F)"], rel=1e-6)
    unrandomized = LatinSquare(4, randomize=False).generate()
    assert np.array_equal(unrandomized.points[:, 2].reshape(4, 4),
                          (np.arange(4)[:, None] + np.arange(4)) % 4)


@pytest.mark.parametrize("n", [3, 4, 5, 7, 8, 9, 12])
def test_graeco_latin_squares_are_mutually_orthogonal(n):
    g = GraecoLatin(n, random_state=n)
    d = g.generate()
    assert _is_latin(g.latin_) and _is_latin(g.greek_)
    assert _are_orthogonal(g.latin_, g.greek_)
    assert d.n_runs == n * n and d.n_factors == 4


def test_graeco_latin_rejects_orders_2_mod_4_and_anova_matches_statsmodels():
    for n in (2, 6, 10):
        with pytest.raises(ValueError):
            GraecoLatin(n).generate()
    g = GraecoLatin(5, random_state=1)
    d = g.generate()
    rng = np.random.default_rng(2)
    y = d.points[:, 2] - 0.3 * d.points[:, 3] + rng.normal(0, 0.5, 25)
    res = g.analyze(y)
    df = pd.DataFrame({"r": d.points[:, 0], "c": d.points[:, 1], "l": d.points[:, 2],
                       "g": d.points[:, 3], "y": y})
    ref = anova_lm(smf.ols("y ~ C(r) + C(c) + C(l) + C(g)", df).fit(), typ=1)
    got = {row["source"]: row for row in res.table}
    for src, key in (("Latin", "C(l)"), ("Greek", "C(g)"), ("Row", "C(r)")):
        assert got[src]["ss"] == pytest.approx(ref.loc[key, "sum_sq"], rel=1e-10)
        assert got[src]["p"] == pytest.approx(ref.loc[key, "PR(>F)"], rel=1e-6)
    assert got["Residual"]["df"] == 8


def test_galois_field_tables_are_fields():
    for q in (4, 8, 9, 27):
        add, mul = _gf_tables(q)
        nz = range(1, q)
        # every nonzero element has a multiplicative inverse, addition is a group
        assert all(1 in mul[a, 1:] for a in nz)
        assert all(sorted(add[a]) == list(range(q)) for a in range(q))
        assert np.array_equal(mul, mul.T) and np.array_equal(add, add.T)
    with pytest.raises(ValueError):
        _gf_tables(6)


# ================================================================== optimal designs

def _brute_force(criterion, n, model="quadratic"):
    """Exact optimum over all multisets of {-1, 0, 1} of size n for a 1-factor model."""
    best, arg = np.inf, None
    for a in range(n + 1):
        for b in range(n + 1 - a):
            pts = np.array([-1.0] * a + [0.0] * b + [1.0] * (n - a - b))[:, None]
            F, _ = _model_matrix(pts, model)
            M = F.T @ F
            if np.linalg.matrix_rank(M) < F.shape[1]:
                continue
            val = criterion(M)
            if val < best - 1e-12:
                best, arg = val, (a, b, n - a - b)
    return arg, best


_GRID = np.linspace(-1, 1, 201)[:, None]
_FG, _ = _model_matrix(_GRID, "quadratic")
_MR = np.array([[1, 0, 1 / 3], [0, 1 / 3, 0], [1 / 3, 0, 1 / 5]])


@pytest.mark.parametrize("cls,crit,n,expected", [
    (D_OptimalDesign, lambda M: -np.linalg.slogdet(M)[1], 6, (2, 2, 2)),
    (D_OptimalDesign, lambda M: -np.linalg.slogdet(M)[1], 12, (4, 4, 4)),
    (A_OptimalDesign, lambda M: np.trace(np.linalg.inv(M)), 8, (2, 4, 2)),
    (A_OptimalDesign, lambda M: np.trace(np.linalg.inv(M)), 12, (3, 6, 3)),
    (I_OptimalDesign, lambda M: np.trace(_MR @ np.linalg.inv(M)), 8, (2, 4, 2)),
    (I_OptimalDesign, lambda M: np.trace(_MR @ np.linalg.inv(M)), 12, (3, 6, 3)),
    (G_OptimalDesign, None, 6, (2, 2, 2)),
    (G_OptimalDesign, None, 12, (4, 4, 4)),
])
def test_optimal_designs_match_brute_force_for_quadratic_regression(cls, crit, n, expected):
    d = cls(n, 1, model="quadratic", random_state=0).generate()
    assert _counts(d) == expected
    if crit is not None:
        arg, best = _brute_force(crit, n)
        assert arg == expected
        assert d.properties["criterion"] == pytest.approx(best, rel=1e-9)


def test_d_optimal_linear_designs_are_the_factorials():
    d = D_OptimalDesign(8, 3, random_state=0).generate()
    assert set(map(tuple, d.points)) == set(itertools.product([-1.0, 1.0], repeat=3))
    assert d.properties["D_efficiency"] == pytest.approx(1.0)
    assert d.properties["G_efficiency"] == pytest.approx(1.0)
    d2 = D_OptimalDesign(4, 2, random_state=1).generate()
    assert set(map(tuple, d2.points)) == set(itertools.product([-1.0, 1.0], repeat=2))


def test_g_efficiency_is_one_at_the_approximate_d_optimum():
    # Kiefer-Wolfowitz: the D-optimal design with equal weights on {-1, 0, 1} is G-optimal
    d = D_OptimalDesign(6, 1, model="quadratic", random_state=0).generate()
    assert d.properties["G_efficiency"] == pytest.approx(1.0)
    F, _ = d.model_matrix("quadratic")
    Mi = np.linalg.inv(F.T @ F)
    assert np.max(np.einsum("ip,pq,iq->i", _FG, Mi, _FG)) * 6 == pytest.approx(3.0)


def test_i_optimal_uses_the_exact_uniform_moment_matrix():
    d = I_OptimalDesign(8, 1, model="quadratic", random_state=0)
    d.generate()
    assert np.allclose(d._MR, _MR)
    mc = np.mean(np.einsum("ip,iq->ipq", _FG, _FG), axis=0)
    assert np.allclose(d._MR, mc, atol=1e-2)


def test_optimal_designs_on_a_two_factor_quadratic_and_custom_candidates():
    d = D_OptimalDesign(9, 2, model="quadratic", random_state=0).generate()
    # the D-optimal 9-run quadratic design on the 3^2 grid is the full 3^2 factorial
    assert set(map(tuple, d.points)) == set(itertools.product([-1.0, 0.0, 1.0], repeat=2))
    cand = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    c = D_OptimalDesign(2, candidates=cand, random_state=0).generate()
    assert sorted(c.points[:, 0]) == [-1.0, 1.0]
    assert c.properties["n_candidates"] == 4


def test_optimal_designs_reproducible_and_errors():
    a = D_OptimalDesign(10, 2, model="quadratic", random_state=5).generate().points
    b = D_OptimalDesign(10, 2, model="quadratic", random_state=5).generate().points
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        D_OptimalDesign(2, 1, model="quadratic").generate()
    with pytest.raises(ValueError):
        D_OptimalDesign(6).generate()


def test_t_optimal_design_discriminates_quadratic_from_linear():
    d = T_OptimalDesign(4, lambda X: X[:, 0] ** 2, n_factors=1, random_state=0).generate()
    assert sorted(d.points[:, 0]) == [-1.0, 0.0, 0.0, 1.0]
    assert d.properties["rss"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        T_OptimalDesign(4, "not callable", n_factors=1)


def test_bayesian_design_reduces_to_d_optimal_and_responds_to_the_prior():
    d = D_OptimalDesign(6, 1, model="quadratic", random_state=0).generate()
    b = BayesianDesign(6, model="quadratic", n_factors=1, random_state=0).generate()
    assert b.properties["criterion"] == pytest.approx(d.properties["criterion"])
    # a very precise prior on the curvature makes the centre point useless
    tight = BayesianDesign(6, model="quadratic", n_factors=1, prior_precision=[0, 0, 1e4],
                           random_state=0).generate()
    assert _counts(tight)[1] == 0


def test_bayesian_design_recovers_box_lucas_for_exponential_decay():
    eta = lambda X, th: th[0] * np.exp(-th[1] * X[:, 0])
    point = BayesianDesign(2, model=eta, prior=np.array([[1.0, 0.5]]), bounds=(0, 10),
                           levels=101, random_state=0).generate()
    assert point.space == "natural"
    assert sorted(point.points[:, 0]) == pytest.approx([0.0, 2.0])  # t = 0 and 1/theta2
    tight = BayesianDesign(2, model=eta, prior=[Normal(1.0, 0.01), Normal(0.5, 0.005)],
                           bounds=(0, 10), levels=101, random_state=0).generate()
    assert sorted(tight.points[:, 0]) == pytest.approx([0.0, 2.0])
    jac = lambda X, th: np.column_stack([np.exp(-th[1] * X[:, 0]),
                                         -th[0] * X[:, 0] * np.exp(-th[1] * X[:, 0])])
    analytic = BayesianDesign(2, model=eta, prior=np.array([[1.0, 0.25]]), jacobian=jac,
                              bounds=(0, 10), levels=101, random_state=0).generate()
    assert sorted(analytic.points[:, 0]) == pytest.approx([0.0, 4.0])
    with pytest.raises(ValueError):
        BayesianDesign(2, model=eta, n_factors=1).generate()


# ============================================================= space-filling designs

def test_latin_hypercube_design_delegates_to_montecarlo():
    from stochpylib.montecarlo import LatinHypercubeSampling

    d = LatinHypercubeDesign(12, 3, random_state=7).generate()
    ref = LatinHypercubeSampling(dim=3, n=12, random_state=np.random.default_rng(7)).generate()
    assert np.array_equal(d.points, ref)
    assert _is_lhs(d.points) and d.space == "unit"
    c = LatinHypercubeDesign(10, 2, centered=True, random_state=0).generate()
    assert np.allclose(np.sort(c.points, axis=0), ((np.arange(10) + 0.5) / 10)[:, None])


def test_latin_hypercube_criteria_improve_their_metric():
    plain = [LatinHypercubeDesign(10, 3, random_state=s).generate() for s in range(20)]
    mm = LatinHypercubeDesign(10, 3, criterion="maximin", n_iter=50, random_state=0).generate()
    cr = LatinHypercubeDesign(10, 3, criterion="correlation", n_iter=50,
                              random_state=0).generate()
    assert _is_lhs(mm.points) and _is_lhs(cr.points)
    assert mm.min_distance() >= np.median([p.min_distance() for p in plain])
    assert cr.max_abs_correlation() <= np.median([p.max_abs_correlation() for p in plain])
    with pytest.raises(ValueError):
        LatinHypercubeDesign(5, 2, criterion="bogus")


def test_maximin_lhd_beats_random_latin_hypercubes():
    d = MaximinLHD(20, 3, random_state=0).generate()
    assert _is_lhs(d.points)
    assert d.properties["phi_p"] <= d.properties["phi_p_initial"]
    rnd = [LatinHypercubeDesign(20, 3, random_state=s).generate().min_distance()
           for s in range(200)]
    assert d.min_distance() > np.percentile(rnd, 95)


def test_minimax_design_matches_the_known_one_and_two_dimensional_optima():
    one = MinimaxDesign(5, 1, random_state=0).generate()
    assert np.allclose(np.sort(one.points[:, 0]), (2 * np.arange(1, 6) - 1) / 10, atol=0.02)
    assert one.properties["fill_distance"] == pytest.approx(0.1, abs=0.005)
    two = MinimaxDesign(4, 2, random_state=0).generate()
    centres = {(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)}
    got = {tuple(np.round(p * 4) / 4) for p in two.points}
    assert got == centres
    assert np.allclose(np.sort(two.points, axis=0), [[0.25, 0.25], [0.25, 0.25],
                                                     [0.75, 0.75], [0.75, 0.75]], atol=0.04)
    from stochpylib.montecarlo import SobolSequence

    sob = Design(SobolSequence(dim=2).generate(4), space="unit")
    lhs = LatinHypercubeDesign(4, 2, random_state=0).generate()
    assert two.fill_distance() < sob.fill_distance()
    assert two.fill_distance() < lhs.fill_distance()


@pytest.mark.parametrize("method", ["CD", "WD", "MD", "L2-star"])
def test_discrepancy_matches_scipy_qmc(method):
    for X in (np.random.default_rng(1).random((25, 3)),
              LatinHypercubeDesign(16, 2, random_state=2).generate().points,
              np.array([[0.5, 0.5]]), np.random.default_rng(3).random((40, 5))):
        assert UniformDesign.discrepancy(X, method) == pytest.approx(
            qmc.discrepancy(X, method=method), rel=1e-12, abs=1e-15)
    d = LatinHypercubeDesign(9, 2, random_state=0).generate()
    assert d.discrepancy(method) == pytest.approx(qmc.discrepancy(d.points, method=method))


def test_uniform_design_glp_is_u_type_and_beats_random():
    d = UniformDesign(21, 3, random_state=0).generate()
    assert d.properties["method"] == "glp"
    assert np.allclose(np.sort(d.points, axis=0), ((np.arange(21) + 0.5) / 21)[:, None])
    rnd = min(LatinHypercubeDesign(21, 3, centered=True, random_state=s).generate()
              .discrepancy() for s in range(50))
    assert d.properties["discrepancy"] <= rnd
    assert d.properties["discrepancy"] == pytest.approx(qmc.discrepancy(d.points))
    s = UniformDesign(12, 3, method="lhs-search", n_iter=500, random_state=0).generate()
    assert _is_lhs(s.points) and s.properties["method"] == "lhs-search"
    assert s.properties["discrepancy"] <= np.median(
        [LatinHypercubeDesign(12, 3, centered=True, random_state=k).generate().discrepancy()
         for k in range(30)])
    for bad in (dict(criterion="XX"), dict(method="xx")):
        with pytest.raises(ValueError):
            UniformDesign(5, 2, **bad)


@pytest.mark.parametrize("s,t,k", [(5, 2, 6), (4, 2, 5), (3, 3, 4), (7, 2, 8), (8, 2, 4)])
def test_orthogonal_array_has_its_strength(s, t, k):
    oa = OrthogonalArrayDesign(s, k, strength=t)
    d = oa.generate()
    A = oa.array_
    assert A.shape == (s ** t, k)
    for cols in itertools.combinations(range(k), t):
        _, counts = np.unique(A[:, list(cols)], axis=0, return_counts=True)
        assert len(counts) == s ** t and np.all(counts == 1)
    assert d.properties["index"] == 1 and d.properties["strength"] == t
    assert np.allclose(np.unique(d.points), (np.arange(s) + 0.5) / s)


def test_orthogonal_array_lhs_keeps_projections_balanced():
    d = OrthogonalArrayDesign(5, 4, lhs=True, random_state=0).generate()
    assert _is_lhs(d.points) and d.properties["is_lhs"]
    coarse = np.floor(d.points * 5).astype(int)
    for a, b in itertools.combinations(range(4), 2):
        assert len(set(map(tuple, coarse[:, [a, b]]))) == 25
    for bad in (dict(levels=6, n_factors=3), dict(levels=5, n_factors=7),
                dict(levels=3, n_factors=2, strength=4)):
        with pytest.raises(ValueError):
            OrthogonalArrayDesign(**bad)


def test_space_filling_designs_are_reproducible():
    for cls, kw in ((MaximinLHD, dict(n=8, dim=2, n_iter=200)),
                    (UniformDesign, dict(n=10, dim=2, method="lhs-search", n_iter=200)),
                    (MinimaxDesign, dict(n=4, dim=2, n_ref=512, n_iter=10)),
                    (OrthogonalArrayDesign, dict(levels=3, n_factors=3, lhs=True))):
        a = cls(random_state=3, **kw).generate().points
        b = cls(random_state=3, **kw).generate().points
        assert np.array_equal(a, b), cls.__name__


# ================================================================ response surface

_CCD2 = CCD(2, center=5).generate().points


def _quad2(X):
    return (10 + 2 * X[:, 0] - 3 * X[:, 1] - 1.5 * X[:, 0] ** 2 - 2 * X[:, 1] ** 2
            + X[:, 0] * X[:, 1])


def test_response_surface_matches_statsmodels_ols():
    rng = np.random.default_rng(10)
    y = _quad2(_CCD2) + rng.normal(0, 0.2, len(_CCD2))
    rs = ResponseSurface(order=2).fit(_CCD2, y)
    df = pd.DataFrame({"a": _CCD2[:, 0], "b": _CCD2[:, 1], "y": y})
    ref = smf.ols("y ~ a + b + a:b + I(a**2) + I(b**2)", df).fit()
    assert np.allclose(rs.coef_, ref.params.values, atol=1e-10)
    assert np.allclose(rs.regression_.std_errors_, ref.bse.values, rtol=1e-8)
    Xn = np.array([[0.3, -0.2], [1.0, 1.0]])
    mu, se = rs.predict(Xn, return_std=True)
    pred = ref.get_prediction(pd.DataFrame({"a": Xn[:, 0], "b": Xn[:, 1]}))
    assert np.allclose(mu, pred.predicted_mean, atol=1e-10)
    assert np.allclose(se, pred.se_mean, rtol=1e-8)
    assert rs.terms_ == ["1", "A", "B", "A*B", "A^2", "B^2"]


def test_response_surface_canonical_analysis_on_exact_surfaces():
    rs = ResponseSurface(2).fit(_CCD2, _quad2(_CCD2))
    B = np.array([[-1.5, 0.5], [0.5, -2.0]])
    xs = -0.5 * np.linalg.solve(B, [2.0, -3.0])
    assert np.allclose(rs.stationary_point(), xs, atol=1e-10)
    can = rs.canonical_analysis()
    assert can["nature"] == "maximum"
    assert np.allclose(can["eigenvalues"], np.linalg.eigvalsh(B))
    assert can["predicted"] == pytest.approx(float(_quad2(xs[None])[0]))
    mini = ResponseSurface(2).fit(_CCD2, -_quad2(_CCD2))
    assert mini.canonical_analysis()["nature"] == "minimum"
    saddle = ResponseSurface(2).fit(_CCD2, _CCD2[:, 0] ** 2 - _CCD2[:, 1] ** 2)
    assert saddle.canonical_analysis()["nature"] == "saddle"
    ridge = ResponseSurface(2).fit(_CCD2, -_CCD2[:, 0] ** 2 + _CCD2[:, 1])
    assert ridge.canonical_analysis()["nature"] == "ridge"
    with pytest.raises(ValueError):
        ResponseSurface(1).fit(_CCD2, _quad2(_CCD2)).stationary_point()


def test_response_surface_natural_units_optimize_and_steepest_ascent():
    bounds = [(100.0, 200.0), (1.0, 3.0)]
    design = CCD(2, center=5).generate()
    Xn = design.to_natural(bounds)
    rs = ResponseSurface(2, bounds=bounds).fit(Xn, _quad2(design.points))
    xs_coded = -0.5 * np.linalg.solve([[-1.5, 0.5], [0.5, -2.0]], [2.0, -3.0])
    xs_nat = np.array(bounds)[:, 0] + (xs_coded + 1) / 2 * np.ptp(np.array(bounds), axis=1)
    assert np.allclose(rs.stationary_point(), xs_nat)
    best = rs.optimize(maximize=True, random_state=0)
    assert np.allclose(best["x"], xs_nat, atol=1e-6)
    # a boundary optimum: minimize the same (concave) surface over the box
    low = rs.optimize(maximize=False, random_state=0)
    g1, g2 = np.meshgrid(np.linspace(100, 200, 201), np.linspace(1, 3, 201))
    grid = np.column_stack([g1.ravel(), g2.ravel()])
    assert low["value"] == pytest.approx(float(np.min(rs.predict(grid))), abs=1e-3)
    path = rs.steepest_ascent(n_steps=3, step=0.25)  # 3 steps stay short of the maximum
    assert path["path"].shape == (4, 2)
    assert np.all(np.diff(path["predicted"]) > 0)


def test_response_surface_errors():
    with pytest.raises(ValueError):
        ResponseSurface(order=4)
    with pytest.raises(ValueError):
        ResponseSurface(2).fit(_CCD2[:5], _quad2(_CCD2[:5]))
    with pytest.raises(RuntimeError):
        ResponseSurface(2).predict(_CCD2)


def test_rsm_anova_matches_statsmodels_and_hand_lack_of_fit():
    rng = np.random.default_rng(11)
    X = _CCD2
    y = _quad2(X) + 0.3 * X[:, 0] ** 3 + rng.normal(0, 0.2, len(X))
    ra = RSM_ANOVA().fit(X, y)
    df = pd.DataFrame({"a": X[:, 0], "b": X[:, 1], "y": y})
    ref = anova_lm(smf.ols("y ~ a + b + a:b + I(a**2) + I(b**2)", df).fit(), typ=1)
    got = {r["source"]: r for r in ra.table_.table}
    assert got["linear"]["ss"] == pytest.approx(ref.loc[["a", "b"], "sum_sq"].sum())
    assert got["interaction"]["ss"] == pytest.approx(ref.loc["a:b", "sum_sq"])
    assert got["quadratic"]["ss"] == pytest.approx(
        ref.loc[["I(a ** 2)", "I(b ** 2)"], "sum_sq"].sum())
    assert got["residual"]["ss"] == pytest.approx(ref.loc["Residual", "sum_sq"])
    # lack of fit vs pure error from the 5 centre replicates, by hand
    centre = np.all(X == 0, axis=1)
    ss_pe = float(np.sum((y[centre] - y[centre].mean()) ** 2))
    assert got["pure error"]["ss"] == pytest.approx(ss_pe) and got["pure error"]["df"] == 4
    assert got["lack of fit"]["df"] == got["residual"]["df"] - 4
    ms_lof = got["lack of fit"]["ss"] / got["lack of fit"]["df"]
    assert ra.lack_of_fit_.statistic == pytest.approx(ms_lof / (ss_pe / 4))
    # predicted R^2 from PRESS equals brute-force leave-one-out refits
    F, _ = _model_matrix(X, "quadratic")
    press = 0.0
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        beta, *_ = np.linalg.lstsq(F[m], y[m], rcond=None)
        press += float((y[i] - F[i] @ beta) ** 2)
    assert ra.press_ == pytest.approx(press)
    sst = float(np.sum((y - y.mean()) ** 2))
    assert ra.pred_r2_ == pytest.approx(1 - press / sst)
    ref_r2 = 1 - ref.loc["Residual", "sum_sq"] / sst
    assert ra.r2_ == pytest.approx(ref_r2)
    assert ra.table_.pvalue < 1e-6 and ref_r2 > 0.9


def test_rsm_anova_without_replicates_and_first_order():
    X = FullFactorial(3, 2).generate().points
    y = 1 + X[:, 0] + X[:, 1] ** 2
    ra = RSM_ANOVA().fit(X, y + np.random.default_rng(0).normal(0, 0.01, 9))
    assert ra.lack_of_fit_ is None
    assert "pure error" not in [r["source"] for r in ra.table_.table]
    first = RSM_ANOVA(order=1).fit(X, 1 + X[:, 0] + 0.1 * X[:, 1] ** 2)
    assert [r["source"] for r in first.table_.table][0] == "linear"
    with pytest.raises(ValueError):
        RSM_ANOVA(order=3)


def test_polynomial_chaos_recovers_a_polynomial_exactly():
    rng = np.random.default_rng(12)
    X = rng.uniform(-1, 1, (60, 2))
    y = 1.0 + 2.0 * X[:, 0] + X[:, 0] * X[:, 1]
    pce = PolynomialChaos(3, bounds=[(-1, 1), (-1, 1)]).fit(X, y)
    # orthonormal Legendre on U(-1,1): x = P1/sqrt(3), so y = 1 + (2/sqrt3) P1(a) + (1/3) P1 P1
    assert pce.mean_ == pytest.approx(1.0)
    assert pce.var_ == pytest.approx(4 / 3 + 1 / 9)
    assert np.allclose(pce.predict(X), y, atol=1e-10)
    assert pce.sobol_first_ == pytest.approx([(4 / 3) / (4 / 3 + 1 / 9), 0.0], abs=1e-10)
    assert pce.sobol_total_[1] == pytest.approx((1 / 9) / (4 / 3 + 1 / 9))
    assert pce.loo_error_ < 1e-20


def test_polynomial_chaos_on_ishigami_matches_analytic_sobol_indices():
    from stochpylib.montecarlo import LatinHypercubeSampling

    U = LatinHypercubeSampling(dim=3, n=2000, random_state=0).generate()
    X = -np.pi + 2 * np.pi * U
    pce = PolynomialChaos(10, bounds=ISHIGAMI_BOUNDS).fit(X, ishigami(X))
    assert pce.mean_ == pytest.approx(3.5, abs=0.01)
    assert np.allclose(pce.sobol_first_, ISHIGAMI_S1, atol=0.02)
    assert np.allclose(pce.sobol_total_, ISHIGAMI_ST, atol=0.02)
    quad = PolynomialChaos(10, bounds=ISHIGAMI_BOUNDS).fit_function(ishigami)
    assert np.allclose(quad.sobol_first_, ISHIGAMI_S1, atol=0.01)
    assert np.allclose(quad.sobol_total_, ISHIGAMI_ST, atol=0.01)
    assert quad.mean_ == pytest.approx(3.5, abs=1e-6)


def test_polynomial_chaos_hermite_basis_and_transforms():
    ph = PolynomialChaos(2, distributions=[Normal(1.0, 2.0)]).fit_function(
        lambda X: X[:, 0] ** 2)
    assert ph.mean_ == pytest.approx(5.0)          # E[X^2] = mu^2 + sigma^2
    assert ph.var_ == pytest.approx(48.0)          # 2 sigma^4 + 4 mu^2 sigma^2
    pe = PolynomialChaos(4, distributions=[Exponential(1.0)]).fit_function(
        lambda X: X[:, 0], method="regression", n_samples=400, random_state=0)
    assert pe.mean_ == pytest.approx(1.0, abs=0.01) and pe.var_ == pytest.approx(1.0, abs=0.02)
    pu = PolynomialChaos(3, distributions=[Uniform(0.0, 2.0)] * 2).fit_function(
        lambda X: X[:, 0] + X[:, 1])
    assert pu.mean_ == pytest.approx(2.0) and pu.var_ == pytest.approx(2 / 3)
    hyp = PolynomialChaos(6, bounds=[(-1, 1)] * 3, truncation="hyperbolic", q=0.5)
    hyp._prepare(3)
    full = PolynomialChaos(6, bounds=[(-1, 1)] * 3)
    full._prepare(3)
    assert len(hyp.multi_indices_) < len(full.multi_indices_)


def test_polynomial_chaos_basis_is_orthonormal_by_quadrature():
    from stochpylib.numerical_methods import GaussHermite, GaussLegendre

    from stochpylib.experimental_design.response_surface import _hermite, _legendre

    g = GaussLegendre(12)
    P = _legendre(g.nodes_, 6)
    assert np.allclose((P * g.weights_ / 2) @ P.T, np.eye(7), atol=1e-12)
    h = GaussHermite(12)
    H = _hermite(h.nodes_, 6)
    assert np.allclose((H * h.weights_) @ H.T, np.eye(7), atol=1e-10)


def test_polynomial_chaos_errors():
    with pytest.raises(ValueError):
        PolynomialChaos(6, bounds=[(0, 1)] * 3).fit(np.random.default_rng(0).random((10, 3)),
                                                     np.ones(10))
    with pytest.raises(ValueError):
        PolynomialChaos(0)
    with pytest.raises(ValueError):
        PolynomialChaos(2).fit_function(lambda X: X[:, 0])
    pce = PolynomialChaos(2, bounds=[(0, 1)]).fit(np.linspace(0, 1, 10)[:, None],
                                                  np.linspace(0, 1, 10))
    with pytest.raises(ValueError):
        pce.predict([[0.5]], return_std=True)


_XK = MaximinLHD(20, 2, n_iter=500, random_state=0).generate().points
_YK = branin01(_XK)
_XT = np.random.default_rng(5).random((200, 2))


def test_kriging_interpolates_and_predicts_branin():
    kr = KrigingSurrogate().fit(_XK, _YK)
    mu, sd = kr.predict(_XK, return_std=True)
    assert np.max(np.abs(mu - _YK)) < 1e-4
    assert np.max(sd) < 1e-2 * np.std(_YK)
    assert kr.score(_XT, branin01(_XT)) > 0.9
    assert not np.allclose(kr.kernel_.length_scale, 0.5)  # hyperparameters were fitted
    x = np.linspace(0, 1, 8)[:, None]
    sine = KrigingSurrogate().fit(x, np.sin(2 * np.pi * x[:, 0]))
    xs = np.linspace(0, 1, 101)[:, None]
    assert np.max(np.abs(sine.predict(xs) - np.sin(2 * np.pi * xs[:, 0]))) < 0.05


def test_kriging_without_trend_equals_gp_regression():
    from stochpylib.gaussian_processes import GPRegression, MaternKernel

    k = KrigingSurrogate(kernel=MaternKernel(nu=2.5, length_scale=0.3), trend="none",
                         optimize=False, normalize=False, noise=1e-6).fit(_XK, _YK)
    g = GPRegression(MaternKernel(nu=2.5, length_scale=0.3), noise=1e-6).fit(_XK, _YK)
    a, b = k.predict(_XT, return_std=True), g.predict(_XT, return_std=True)
    assert np.allclose(a[0], b[0], atol=1e-10) and np.allclose(a[1], b[1], atol=1e-10)


@pytest.mark.parametrize("trend", ["constant", "linear", "quadratic"])
def test_universal_kriging_matches_an_independent_gls_computation(trend):
    from stochpylib.gaussian_processes import MaternKernel

    kern = MaternKernel(nu=2.5, length_scale=0.4, variance=2.0)
    noise = 1e-6
    kr = KrigingSurrogate(kernel=kern, trend=trend, optimize=False, normalize=False,
                          noise=noise).fit(_XK, _YK)
    K = kern(_XK) + noise * np.eye(len(_XK))
    Ki = np.linalg.inv(K)
    if trend == "constant":
        F, f = np.ones((len(_XK), 1)), np.ones((len(_XT), 1))
    else:
        F, _ = _model_matrix(_XK, trend)
        f, _ = _model_matrix(_XT, trend)
    A = F.T @ Ki @ F
    beta = np.linalg.solve(A, F.T @ Ki @ _YK)
    k = kern(_XT, _XK)
    mean = f @ beta + k @ Ki @ (_YK - F @ beta)
    u = f - k @ Ki @ F
    var = (kern.diag(_XT) - np.einsum("ij,jk,ik->i", k, Ki, k)
           + np.einsum("ip,pq,iq->i", u, np.linalg.inv(A), u))
    mu, sd = kr.predict(_XT, return_std=True)
    assert np.allclose(mu, mean, rtol=1e-6, atol=1e-6)
    assert np.allclose(sd, np.sqrt(np.maximum(var, 0)), rtol=1e-5, atol=1e-6)


def test_kriging_expected_improvement_and_sequential_design():
    kr = KrigingSurrogate().fit(_XK, _YK)
    ei = kr.expected_improvement(_XT)
    assert np.all(ei >= 0) and np.max(kr.expected_improvement(_XK)) < 1e-2
    s0 = kr.predict(_XT, return_std=True)[1].max()
    kr.sequential_design(branin01, [(0, 1), (0, 1)], 5, criterion="variance", random_state=0)
    assert kr.X_train_.shape == (25, 2)
    assert kr.predict(_XT, return_std=True)[1].max() < s0
    opt = KrigingSurrogate().fit(_XK, _YK)
    opt.sequential_design(branin01, [(0, 1), (0, 1)], 10, random_state=0)
    assert opt.y_train_.min() < _YK.min()
    with pytest.raises(ValueError):
        KrigingSurrogate(trend="cubic")
    with pytest.raises(ValueError):
        opt.sequential_design(branin01, [(0, 1), (0, 1)], 1, criterion="bogus")


def test_metamodel_selects_by_cross_validation():
    mm = MetaModel(random_state=0).fit(_CCD2, _quad2(_CCD2))
    assert mm.scores_[mm.best_name_] < 1e-8
    assert mm.best_name_ in ("response_surface_2", "polynomial_chaos_3")
    assert np.allclose(mm.predict(_CCD2), _quad2(_CCD2), atol=1e-8)
    smooth = lambda X: np.exp(-8 * ((X[:, 0] - 0.4) ** 2 + (X[:, 1] - 0.6) ** 2))
    X = MaximinLHD(40, 2, n_iter=500, random_state=1).generate().points
    mm2 = MetaModel(random_state=0).fit(X, smooth(X))
    assert mm2.best_name_ == "kriging"
    assert isinstance(mm2.best_, KrigingSurrogate)
    cv = ResponseSurface(2).cross_validate(_CCD2, _quad2(_CCD2), k=4, random_state=0)
    assert set(cv) == {"rmse", "q2", "fold_rmse"} and len(cv["fold_rmse"]) == 4
    assert cv["q2"] == pytest.approx(1.0)
    # cv=13 (leave-one-out): a 3-fold split can by chance remove all 4 factorial runs from
    # a training fold, leaving A*B inestimable on a 13-point CCD and letting the linear
    # model win that fold by luck (CI caught this at ~1/40 odds); LOO never drops more than
    # one point, so it can't break A*B's estimability, and random_state pins the outcome.
    custom = MetaModel(candidates=[ResponseSurface(1), ResponseSurface(2)], cv=13,
                       random_state=0).fit(_CCD2, _quad2(_CCD2))
    assert custom.best_name_ == "ResponseSurface_1"
    assert issubclass(ResponseSurface, MetaModel) and issubclass(KrigingSurrogate, MetaModel)
    assert issubclass(PolynomialChaos, MetaModel)


# ========================================================================= analysis

_M4 = FullFactorial(2, 4).generate()


def test_main_effects_reproduce_montgomery():
    me = MainEffects().fit(_M4, MONTGOMERY_Y)
    for f in "ABCD":
        assert me.effects_[f] == pytest.approx(MONTGOMERY_EFFECTS[f])
    assert me.ranking_ == ["A", "D", "C", "B"] and me.std_errors_ is None
    # an effect is twice the +/-1-coded OLS coefficient
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(16), _M4.points]), MONTGOMERY_Y,
                               rcond=None)
    assert np.allclose([me.effects_[f] for f in "ABCD"], 2 * beta[1:])
    assert me.level_means_["A"][1.0] - me.level_means_["A"][-1.0] == pytest.approx(21.625)


def test_main_effects_standard_errors_from_replicates():
    rng = np.random.default_rng(13)
    X = np.vstack([FullFactorial(2, 2).generate().points] * 4)
    y = 3 * X[:, 0] + rng.normal(0, 1, 16)
    me = MainEffects().fit(X, y)
    cells = [y[(X[:, 0] == a) & (X[:, 1] == b)] for a in (-1, 1) for b in (-1, 1)]
    s = np.sqrt(sum(float(np.sum((c - c.mean()) ** 2)) for c in cells) / 12)
    assert me.std_errors_["A"] == pytest.approx(2 * s / 4)
    assert abs(me.effects_["A"] - 6.0) < 3 * me.std_errors_["A"]


def test_anova_doe_contrasts_reproduce_montgomery():
    full = ANOVA_DOE().fit(_M4, MONTGOMERY_Y)
    for lab, eff in MONTGOMERY_EFFECTS.items():
        assert full.effects_[lab] == pytest.approx(eff)
    assert full.residual_df_ == 0
    red = ANOVA_DOE(model=["A", "C", "D", "AC", "AD"]).fit(_M4, MONTGOMERY_Y)
    rows = {r["source"]: r for r in red.table_.table}
    assert rows["A"]["ss"] == pytest.approx(1870.5625)
    assert rows["Residual"]["df"] == 10
    df = pd.DataFrame(_M4.points, columns=list("ABCD"))
    df["y"] = MONTGOMERY_Y
    ref = anova_lm(smf.ols("y ~ A + C + D + A:C + A:D", df).fit(), typ=2)
    for lab, key in (("A", "A"), ("C", "C"), ("AC", "A:C"), ("AD", "A:D")):
        assert rows[lab]["ss"] == pytest.approx(ref.loc[key, "sum_sq"])
        assert rows[lab]["F"] == pytest.approx(ref.loc[key, "F"])
        assert rows[lab]["p"] == pytest.approx(ref.loc[key, "PR(>F)"], rel=1e-6)
    assert set(red.significant_) == {"A", "C", "D", "AC", "AD"}
    main = ANOVA_DOE(model="main").fit(_M4, MONTGOMERY_Y)
    assert main.residual_df_ == 11


def test_anova_doe_replicated_and_multilevel_match_statsmodels_type_ii():
    rng = np.random.default_rng(14)
    X = np.vstack([FullFactorial(2, 3).generate().points] * 2)
    y = 2 * X[:, 0] - X[:, 1] + 1.5 * X[:, 0] * X[:, 2] + rng.normal(0, 1, 16)
    res = ANOVA_DOE().fit(X, y)
    df = pd.DataFrame(X, columns=["fa", "fb", "fc"])  # "C" would shadow patsy's C()
    df["y"] = y
    ref = anova_lm(smf.ols("y ~ C(fa) * C(fb) * C(fc)", df).fit(), typ=2)
    rows = {r["source"]: r for r in res.table_.table}
    for lab, key in (("A", "C(fa)"), ("AC", "C(fa):C(fc)"), ("ABC", "C(fa):C(fb):C(fc)")):
        assert rows[lab]["ss"] == pytest.approx(ref.loc[key, "sum_sq"])
        assert rows[lab]["p"] == pytest.approx(ref.loc[key, "PR(>F)"], rel=1e-6)
    ml = np.vstack([FullFactorial([3, 2]).generate().points] * 3)
    yy = 1 + ml[:, 0] + 0.5 * ml[:, 1] + 0.7 * ml[:, 0] * ml[:, 1] + rng.normal(0, 0.3, 18)
    res2 = ANOVA_DOE(model="interaction").fit(ml, yy)
    assert res2.table_.method == "DOE ANOVA (Type II SS)"
    d2 = pd.DataFrame({"A": ml[:, 0], "B": ml[:, 1], "y": yy})
    ref2 = anova_lm(smf.ols("y ~ C(A) * C(B)", d2).fit(), typ=2)
    rows2 = {r["source"]: r for r in res2.table_.table}
    for lab, key in (("A", "C(A)"), ("B", "C(B)"), ("AB", "C(A):C(B)")):
        assert rows2[lab]["ss"] == pytest.approx(ref2.loc[key, "sum_sq"])
        assert rows2[lab]["df"] == ref2.loc[key, "df"]
        assert rows2[lab]["F"] == pytest.approx(ref2.loc[key, "F"])


def test_anova_doe_agrees_with_statistics_two_way_anova():
    from stochpylib.statistics import ANOVA

    rng = np.random.default_rng(15)
    ml = np.vstack([FullFactorial([3, 2]).generate().points] * 4)
    y = ml[:, 0] + ml[:, 1] ** 2 + rng.normal(0, 1, 24)
    mine = {r["source"]: r for r in ANOVA_DOE(model="interaction").fit(ml, y).table_.table}
    ref = {r["source"]: r for r in ANOVA(y, factors=[ml[:, 0], ml[:, 1]]).table}
    for a, b in (("A", "A"), ("B", "B"), ("AB", "A:B"), ("Residual", "Residual")):
        assert mine[a]["ss"] == pytest.approx(ref[b]["ss"])


def test_anova_doe_term_parsing_and_errors():
    res = ANOVA_DOE(model=["A*C", (0,)]).fit(_M4, MONTGOMERY_Y)
    assert [r["source"] for r in res.table_.table][:2] == ["AC", "A"]
    with pytest.raises(ValueError):
        ANOVA_DOE(model=["Q"]).fit(_M4, MONTGOMERY_Y)
    with pytest.raises(ValueError):
        ANOVA_DOE(model="cubic").fit(_M4, MONTGOMERY_Y)
    with pytest.raises(ValueError):
        ANOVA_DOE().fit(_M4, MONTGOMERY_Y[:5])


def test_normal_plot_lenth_reproduces_montgomery():
    npl = NormalPlot().fit(_M4, MONTGOMERY_Y)
    assert npl.pse_ == pytest.approx(2.625)
    assert npl.me_ == pytest.approx(6.747777, abs=1e-6)
    assert npl.sme_ == pytest.approx(13.698960, abs=1e-6)
    assert set(npl.active_) == {"A", "C", "D", "AC", "AD"}
    m = 15
    i = np.arange(1, m + 1)
    assert np.allclose(npl.theoretical_quantiles_, stats.norm.ppf((i - 0.5) / m))
    assert [lab for lab, _ in npl.sorted_effects_][0] == "AC"
    assert npl.me_ == pytest.approx(stats.t.ppf(0.975, 5) * 2.625)
    half = NormalPlot(half=True).fit(_M4, MONTGOMERY_Y)
    assert np.allclose(half.theoretical_quantiles_, stats.halfnorm.ppf((i - 0.5) / m))
    assert [lab for lab, _ in half.sorted_effects_][-1] == "A"
    assert "PSE=2.6250" in npl.to_text()
    with pytest.raises(ValueError):
        NormalPlot().fit(FullFactorial([3, 2]).generate(), np.ones(6))


def test_interaction_plot_cell_means_and_parallelism():
    ip = InteractionPlot(("A", "C")).fit(_M4, MONTGOMERY_Y)
    A, C = _M4.points[:, 0], _M4.points[:, 2]
    for r, a in enumerate((-1, 1)):
        for c, cc in enumerate((-1, 1)):
            assert ip.cell_means_[r, c] == pytest.approx(MONTGOMERY_Y[(A == a) & (C == cc)].mean())
    assert ip.interaction_effect_ == pytest.approx(-18.125)
    assert not ip.is_parallel()
    assert set(ip.lines_) == {-1.0, 1.0}
    add = InteractionPlot((0, 1)).fit(_M4, 3 * _M4.points[:, 0] + _M4.points[:, 1])
    assert add.is_parallel() and add.interaction_effect_ == pytest.approx(0.0)
    assert "A \\ C" in ip.to_text()
    ml = InteractionPlot((0, 1)).fit(FullFactorial([3, 2]).generate(), np.arange(6.0))
    assert ml.cell_means_.shape == (3, 2) and ml.interaction_effect_ is None


def test_morris_screening_on_a_linear_model_is_exact():
    lin = lambda X: 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2]
    mo = SensitivityIndex("morris", n_trajectories=15, random_state=0).analyze(
        lin, bounds=[(0, 1), (0, 2), (-1, 1)])
    assert np.allclose(mo.mu_star_, [3.0, 4.0, 1.0])
    assert np.allclose(mo.mu_, [3.0, -4.0, 1.0])
    assert np.allclose(mo.sigma_, 0.0, atol=1e-12)
    assert mo.ranking_ == [1, 0, 2] and mo.n_evals_ == 15 * 4
    nonlin = SensitivityIndex("morris", random_state=0).analyze(
        lambda X: X[:, 0] * X[:, 1] + X[:, 2] ** 2, bounds=[(0, 1)] * 3)
    assert np.all(nonlin.sigma_ > 0)


def test_src_prcc_and_correlation_indices():
    lin = lambda X: 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2]
    src = SensitivityIndex("src", random_state=0).analyze(lin, bounds=[(0, 1), (0, 2), (-1, 1)])
    assert src.r2_ == pytest.approx(1.0)
    assert np.sum(src.src_ ** 2) == pytest.approx(1.0, abs=1e-2)
    sd = np.array([1, 2, 2]) / np.sqrt(12) * np.abs([3, 2, 0.5])
    assert np.allclose(np.abs(src.src_), sd / np.sqrt(np.sum(sd ** 2)), atol=0.02)
    prcc = SensitivityIndex("prcc", random_state=0).analyze(
        lambda X: np.exp(3 * X[:, 0]) + 0.1 * X[:, 1], bounds=[(0, 1)] * 3)
    assert prcc.prcc_[0] > 0.99 and abs(prcc.prcc_[2]) < 0.1
    rng = np.random.default_rng(16)
    X = rng.normal(size=(300, 3))
    y = X[:, 0] + 0.5 * X[:, 1] ** 3 + rng.normal(0, 0.1, 300)
    co = SensitivityIndex("correlation").fit(X, y)
    for i in range(3):
        assert co.pearson_[i] == pytest.approx(stats.pearsonr(X[:, i], y)[0])
        assert co.spearman_[i] == pytest.approx(stats.spearmanr(X[:, i], y)[0])
    assert np.allclose(_ranks([3, 1, 1, 2]), stats.rankdata([3, 1, 1, 2]))
    with pytest.raises(ValueError):
        SensitivityIndex("bogus")
    with pytest.raises(ValueError):
        SensitivityIndex("morris").fit(X, y)
    with pytest.raises(ValueError):
        SensitivityIndex("src").analyze(lambda X: X[:, 0])


def test_sobol_indices_of_ishigami_match_analytic_values():
    from stochpylib.montecarlo import MCResult

    so = SobolIndex(n_samples=2 ** 13, second_order=True, random_state=0).analyze(
        ishigami, bounds=ISHIGAMI_BOUNDS)
    for i in range(3):
        r1, rT = so.first_order_[i], so.total_order_[i]
        assert isinstance(r1, MCResult) and isinstance(rT, MCResult)
        assert abs(r1.estimate - ISHIGAMI_S1[i]) < max(3 * r1.std_error, 1e-3)
        assert abs(rT.estimate - ISHIGAMI_ST[i]) < max(3 * rT.std_error, 1e-3)
        assert abs(r1.estimate - ISHIGAMI_S1[i]) < 0.03
        assert abs(rT.estimate - ISHIGAMI_ST[i]) < 0.03
    assert so.S2_[0, 2] == pytest.approx(0.2437, abs=0.03)
    assert abs(so.S2_[0, 1]) < 0.03 and abs(so.S2_[1, 2]) < 0.03
    assert so.n_evals_ == 2 ** 13 * (3 + 2) + 2 ** 13 * 3
    lo, hi = so.first_order_[1].confidence_interval(0.999)
    assert lo < ISHIGAMI_S1[1] < hi


def test_sobol_indices_agree_with_scipy_and_the_g_function():
    so = SobolIndex(n_samples=2 ** 12, random_state=1).analyze(ishigami, bounds=ISHIGAMI_BOUNDS)
    ref = stats.sobol_indices(func=lambda x: ishigami(x.T), n=2 ** 12,
                              dists=[stats.uniform(-np.pi, 2 * np.pi)] * 3, random_state=1)
    assert np.allclose(so.S1_, ref.first_order, atol=0.05)
    assert np.allclose(so.ST_, ref.total_order, atol=0.05)
    a = np.array([0.0, 1.0, 4.5, 9.0])
    g = lambda X: np.prod((np.abs(4 * X - 2) + a) / (1 + a), axis=1)
    Vi = 1.0 / (3 * (1 + a) ** 2)
    V = np.prod(1 + Vi) - 1
    S1 = Vi / V
    ST = Vi * np.array([np.prod(1 + np.delete(Vi, i)) for i in range(4)]) / V
    sg = SobolIndex(n_samples=2 ** 13, random_state=2).analyze(g, bounds=[(0, 1)] * 4)
    assert np.allclose(sg.S1_, S1, atol=0.03) and np.allclose(sg.ST_, ST, atol=0.03)


def test_sobol_indices_with_distributions_random_sampler_and_reproducibility():
    sd = SobolIndex(n_samples=2 ** 12, random_state=3).analyze(
        ishigami, distributions=[Uniform(-np.pi, np.pi)] * 3)
    assert np.allclose(sd.S1_, ISHIGAMI_S1, atol=0.04)
    rnd = SobolIndex(n_samples=2 ** 13, sampler="random", random_state=4).analyze(
        ishigami, bounds=ISHIGAMI_BOUNDS)
    assert np.allclose(rnd.ST_, ISHIGAMI_ST, atol=0.05)
    a = SobolIndex(n_samples=1024, n_bootstrap=20, random_state=5).analyze(
        ishigami, bounds=ISHIGAMI_BOUNDS)
    b = SobolIndex(n_samples=1024, n_bootstrap=20, random_state=5).analyze(
        ishigami, bounds=ISHIGAMI_BOUNDS)
    assert np.array_equal(a.S1_, b.S1_)
    assert isinstance(a, SensitivityIndex) and a.S2_ is None
    with pytest.raises(ValueError):
        SobolIndex(sampler="lhs")
    with pytest.raises(ValueError):
        SobolIndex().fit(np.ones((3, 2)), np.ones(3))
    with pytest.raises(ValueError):
        SobolIndex(n_samples=64).analyze(lambda X: np.ones(len(X)), bounds=[(0, 1)])


# ========================================================================== hygiene

def test_base_classes_and_public_surface():
    assert len(ed.__all__) == len(set(ed.__all__)) == 32
    for cls in (FullFactorial, CCD, D_OptimalDesign, MaximinLHD, LatinSquare):
        assert issubclass(cls, DesignGenerator)
    for cls in (D_OptimalDesign, A_OptimalDesign, G_OptimalDesign, I_OptimalDesign,
                T_OptimalDesign, BayesianDesign):
        assert issubclass(cls, OptimalDesign)
    g = CCD(2)
    assert "unfitted" in repr(g)
    g.generate()
    assert "n_runs=12" in repr(g) and g.design_.kind == "central composite"
    with pytest.raises(NotImplementedError):
        DesignGenerator().generate()


def test_library_code_never_imports_scipy_stats_or_scipy_optimize():
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2] / "stochpylib" / "experimental_design"
    banned = re.compile(r"^\s*(?:from|import)\s+scipy\.(?:stats|optimize)", re.MULTILINE)
    seen = 0
    for path in sorted(root.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        assert not banned.search(src), path.name  # scipy is the oracle, never a backend
        assert not re.search(r"^\s*from\s+scipy\s+import\s+.*(stats|optimize)",
                             src, re.MULTILINE), path.name
        seen += 1
    assert seen >= 8


def test_quickstart_example_runs():
    from stochpylib.experimental_design import (CCD, FractionalFactorial, ResponseSurface,
                                                SobolIndex)

    # the exact snippet advertised in the module README and the vault quickstart
    ff = FractionalFactorial(5, p=1).generate()
    assert ff.properties["resolution"] == 5 and ff.properties["aliases"]["AB"] == ["CDE"]

    ccd = CCD(2, center=5).generate()
    X = ccd.to_natural([(150, 200), (1.0, 3.0)])
    y = _quad2(ccd.points)
    fit = ResponseSurface(order=2, bounds=[(150, 200), (1.0, 3.0)]).fit(X, y)
    assert fit.canonical_analysis()["nature"] == "maximum"
    assert np.all(fit.stationary_point() > [150, 1.0])

    S = SobolIndex(n_samples=4096, random_state=0).analyze(ishigami,
                                                           bounds=[(-3.14, 3.14)] * 3)
    lo, hi = S.first_order_[0].confidence_interval()
    assert lo < 0.33 and hi > 0.29
