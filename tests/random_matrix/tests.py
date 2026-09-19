"""Tests for stochpylib.random_matrix: ensembles, limiting spectral laws, Haar rotations,
and spectral statistics.

Oracles: closed forms (semicircle/Marchenko-Pastur moments, Catalan and Narayana numbers,
Edelman's hard-edge law, the Wigner surmises), scipy.stats (``kstest``, ``ortho_group``,
``unitary_group``, ``beta``), and the published Tracy-Widom moments / cdf values
(Bornemann 2010). All randomness is seeded; statistical assertions state their standard
error and use >= 3 SE (looser only where a finite-n bias is documented).
"""

import math

import numpy as np
import pytest
from scipy import integrate, special
from scipy import stats as sps

import stochpylib
from stochpylib import random_matrix as rm
from stochpylib.random_matrix import empirical_spectra as rm_es
from stochpylib.random_matrix import ensembles as rm_en
from stochpylib.random_matrix import random_rotations as rm_rr
from stochpylib.random_matrix import statistics as rm_st
from stochpylib.distributions import Exponential, Uniform
from stochpylib.distributions import InverseWishart as DistInverseWishart
from stochpylib.distributions import Wishart as DistWishart
from stochpylib.statistics import TestResult

# published Tracy-Widom moments (Bornemann 2010, Table 1 convention)
TW_REF = {1: (-1.2065336, 1.6077811), 2: (-1.7710868, 0.8131948), 4: (-2.3068848, 0.5177237)}


# ================================================================== ensembles

def test_goe_is_symmetric_with_documented_variances():
    H = rm.GOE(300).sample(random_state=0)
    assert H.dtype == float and np.allclose(H, H.T)
    off = H[np.triu_indices(300, 1)]
    assert abs(off.var() - 1.0) < 0.05          # SE ~ sqrt(2/44850) ~ 0.007
    assert abs(np.diag(H).var() - 2.0) < 0.35   # SE ~ 2*sqrt(2/300) ~ 0.16


def test_gue_is_hermitian_with_unit_offdiagonal_second_moment():
    H = rm.GUE(300).sample(random_state=0)
    assert np.iscomplexobj(H) and np.allclose(H, H.conj().T)
    off = H[np.triu_indices(300, 1)]
    assert abs(np.mean(np.abs(off) ** 2) - 1.0) < 0.03
    assert np.allclose(np.diag(H).imag, 0.0)


def test_gse_is_hermitian_and_kramers_degenerate():
    H = rm.GSE(40).sample(random_state=1)
    assert H.shape == (80, 80) and np.allclose(H, H.conj().T)
    full = np.linalg.eigvalsh(H)
    assert np.allclose(full[::2], full[1::2], atol=1e-10)
    assert rm.GSE(40).eigenvalues(random_state=1).shape == (40,)


@pytest.mark.parametrize("ensemble", [rm.GOE(1000), rm.GUE(1000), rm.GSE(500),
                                      rm.WignerMatrix(1000),
                                      rm.WignerMatrix(1000, Uniform(0.0, 1.0)),
                                      rm.WignerMatrix(1000, lambda s, r: r.exponential(1.0, s)),
                                      rm.BetaEnsemble(1000, 1.0), rm.BetaEnsemble(1000, 4.0)])
def test_normalized_spectrum_follows_the_semicircle(ensemble):
    e = ensemble.normalized_eigenvalues(random_state=3)
    law = ensemble.limit_law()
    assert isinstance(law, rm.WignerSemicircle) and law.radius == 2.0
    d = sps.kstest(e, law.cdf).statistic
    assert d < 0.03, d                           # finite-n edge effects only
    assert law.compare(e).pvalue > 0.05


def test_wigner_matrix_rejects_degenerate_entries():
    with pytest.raises(ValueError):
        rm.WignerMatrix(20, lambda s, r: np.ones(s)).sample(random_state=0)


def test_wishart_matches_library_distribution_and_marchenko_pastur():
    W = rm.WishartMatrix(p=200, n=1000)
    S = W.sample(random_state=0)
    assert S.shape == (200, 200) and np.allclose(S, S.T)
    # E[W] = n * Sigma, checked on the trace: Var(tr W)/n^2 small
    assert abs(np.trace(S) / 200 - 1000) < 30
    e = W.normalized_eigenvalues(random_state=0)
    mp = W.limit_law()
    assert isinstance(mp, rm.MarchenkoPastur) and mp.gamma == pytest.approx(0.2)
    assert sps.kstest(e, mp.cdf).statistic < 0.03
    assert mp.lam_minus - 0.05 < e.min() and e.max() < mp.lam_plus + 0.05


def test_wishart_with_covariance_reproduces_distribution_mean():
    sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
    W = rm.WishartMatrix(2, 50, sigma=sigma)
    mean = W.samples(2000, random_state=1).mean(axis=0)
    assert np.allclose(mean, DistWishart(50, sigma).mean(), rtol=0.05)


def test_inverse_wishart_reciprocal_spectrum_and_mean():
    IW = rm.InverseWishart(p=40, n=300)
    e = IW.normalized_eigenvalues(random_state=0)   # = n * eig(W^-1)
    assert np.all(e > 0)
    assert IW.limit_law().compare(1.0 / e).pvalue > 0.05
    mean_diag = np.mean([np.trace(IW.sample(rng)) / 40
                         for rng in [np.random.default_rng(s) for s in range(200)]])
    assert abs(mean_diag - np.trace(DistInverseWishart(300, np.eye(40)).mean()) / 40) < 2e-4
    with pytest.raises(ValueError):
        rm.InverseWishart(10, 10)


def test_cue_eigenvalues_lie_on_the_circle_with_uniform_angles():
    c = rm.CUE(400)
    ev = c.eigenvalues(random_state=0)
    assert np.allclose(np.abs(ev), 1.0, atol=1e-10)
    ang = c.eigenangles(random_state=0)
    assert sps.kstest(ang, sps.uniform(-np.pi, 2 * np.pi).cdf).pvalue > 0.01
    # angles scaled to unit mean spacing behave like GUE spacings
    r = rm.EigenvalueSpacing(c.normalized_eigenvalues(random_state=1)).mean_ratio()
    assert abs(r - 0.5996) < 0.03


@pytest.mark.parametrize("entries", ["gaussian", "complex", Exponential(1.0),
                                     lambda s, r: r.uniform(-1, 1, s)])
def test_muresan_ginibre_matrix_obeys_the_circular_law(entries):
    M = rm.MuresanMatrix(500, entries)
    z = M.normalized_eigenvalues(random_state=2)
    assert np.iscomplexobj(z) and z.shape == (500,)
    law = M.limit_law()
    assert isinstance(law, rm.CircularLaw)
    assert law.compare(z).pvalue > 0.05           # |z|^2 ~ Uniform(0, 1)
    assert np.abs(z).max() < 1.1
    assert abs(z.mean()) < 0.05


def test_circular_law_helper_is_self_consistent():
    law = rm.CircularLaw()
    z = law.rvs(4000, random_state=0)
    assert law.compare(z).pvalue > 0.05
    assert law.pdf(0.3 + 0.2j) == pytest.approx(1 / np.pi) and law.pdf(2.0) == 0.0
    assert law.radial_cdf(0.5) == pytest.approx(0.25)


def test_matrix_ensemble_batch_and_repr():
    S = rm.GOE(6).samples(4, random_state=0)
    assert S.shape == (4, 6, 6)
    assert repr(rm.GOE(6)) == "GOE(n=6)" and "WishartMatrix" in repr(rm.WishartMatrix(2, 5))
    with pytest.raises(ValueError):
        rm.GOE(0)


# ========================================================= empirical_spectra

def test_semicircle_closed_forms_match_quadrature():
    w = rm.WignerSemicircle(2.0)
    assert integrate.quad(w.pdf, -2, 2)[0] == pytest.approx(1.0, abs=1e-9)
    x = np.linspace(-2, 2, 9)
    assert np.allclose(w.ppf(w.cdf(x)), x, atol=1e-8)
    for k in (2, 4, 6, 8):
        catalan = math.comb(k, k // 2) // (k // 2 + 1)
        assert w.moment(k) == pytest.approx(catalan)
        assert integrate.quad(lambda t: t ** k * w.pdf(t), -2, 2)[0] == pytest.approx(catalan, rel=1e-6)
    assert w.mgf(0.7) == pytest.approx(integrate.quad(lambda t: np.exp(0.7 * t) * w.pdf(t), -2, 2)[0])
    assert w.cf(0.7).real == pytest.approx(integrate.quad(lambda t: np.cos(0.7 * t) * w.pdf(t), -2, 2)[0])
    assert w.entropy() == pytest.approx(
        integrate.quad(lambda t: -w.pdf(t) * np.log(w.pdf(t)) if w.pdf(t) > 0 else 0.0, -2, 2)[0], rel=1e-6)
    assert w.var() == pytest.approx(1.0) and w.kurtosis() == pytest.approx(-1.0)


def test_semicircle_sampler_and_fit():
    w = rm.WignerSemicircle(3.0)
    r = np.asarray(w.rvs(20000, random_state=5))
    # semicircle(R) = R (2 Beta(3/2, 3/2) - 1): compare against scipy's beta as oracle
    assert sps.kstest((r / 3.0 + 1) / 2, sps.beta(1.5, 1.5).cdf).pvalue > 0.01
    assert abs(rm.WignerSemicircle.fit(r).radius - 3.0) < 0.05
    stat, p = w.ks_test(r)
    assert p > 0.01


@pytest.mark.parametrize("gamma", [0.25, 1.0, 2.0])
def test_marchenko_pastur_mass_moments_and_inverse(gamma):
    mp = rm.MarchenkoPastur(gamma, sigma=1.5)
    cont = integrate.quad(mp.pdf, mp.lam_minus, mp.lam_plus, limit=200)[0]
    assert cont == pytest.approx(min(1.0, 1.0 / gamma), abs=1e-6)
    assert mp.atom == pytest.approx(max(0.0, 1.0 - 1.0 / gamma))
    assert mp.cdf(mp.lam_plus + 1.0) == pytest.approx(1.0, abs=1e-9)
    assert mp.mean() == pytest.approx(1.5 ** 2) and mp.var() == pytest.approx(gamma * 1.5 ** 4)
    # Narayana-number raw moments agree with quadrature (+ atom contributes nothing)
    for k in (1, 2, 3):
        quad_k = integrate.quad(lambda t: t ** k * mp.pdf(t), mp.lam_minus, mp.lam_plus, limit=200)[0]
        assert mp.moment(k) == pytest.approx(quad_k, rel=1e-5)
    q = np.array([0.1, 0.5, 0.9])
    x = mp.ppf(q)
    # generalized inverse: equality above the atom, cdf(ppf(q)) >= q (= atom) inside it
    above = q > mp.atom
    assert np.allclose(mp.cdf(x[above]), q[above], atol=2e-4)
    assert np.all(mp.cdf(x) >= q - 2e-4) and np.all(x[~above] == 0.0)


def test_marchenko_pastur_sampler_fit_and_tie_aware_ks():
    mp = rm.MarchenkoPastur(2.0)
    r = np.asarray(mp.rvs(20000, random_state=1))
    assert abs(np.mean(r == 0.0) - 0.5) < 0.02          # atom mass
    assert abs(r.mean() - 1.0) < 0.04 and abs(r.var() - 2.0) < 0.15
    f = rm.MarchenkoPastur.fit(r)
    assert abs(f.gamma - 2.0) < 0.1 and abs(f.sigma - 1.0) < 0.03
    stat, p = mp.ks_test(r)
    assert stat < 0.02 and p > 0.01


@pytest.mark.parametrize("beta", [1, 2, 4])
def test_tracy_widom_moments_match_published_values(beta):
    tw = rm.TracyWidomDistribution(beta)
    mean, var = TW_REF[beta]
    assert tw.mean() == pytest.approx(mean, abs=2e-4)
    assert tw.var() == pytest.approx(var, abs=2e-4)
    assert integrate.quad(tw.pdf, -10, 8, limit=200)[0] == pytest.approx(1.0, abs=1e-5)
    q = np.array([0.01, 0.5, 0.99])
    assert np.allclose(tw.cdf(tw.ppf(q)), q, atol=1e-8)
    r = np.asarray(tw.rvs(5000, random_state=beta))
    assert abs(r.mean() - mean) < 4 * math.sqrt(var / 5000)
    assert tw.ks_test(r)[1] > 0.01


def test_tracy_widom_2_cdf_reference_points_and_contract():
    tw = rm.TracyWidomDistribution(2)
    assert tw.cdf(-3.0) == pytest.approx(0.0803, abs=2e-4)   # Tracy-Widom (1994) table
    assert tw.cdf(0.0) == pytest.approx(0.9694, abs=2e-4)
    assert tw.skewness() == pytest.approx(0.224, abs=0.01)   # Bornemann 2010
    assert tw.kurtosis() == pytest.approx(0.093, abs=0.01)
    assert math.isfinite(tw.entropy()) and tw.mgf(0.0) == pytest.approx(1.0, abs=1e-4)
    assert abs(tw.cf(0.0) - 1.0) < 1e-4   # trapezoid over the cached grid
    assert rm.TracyWidomDistribution.fit([1.0, 2.0], beta=1).beta == 1
    with pytest.raises(ValueError):
        rm.TracyWidomDistribution(3)


@pytest.mark.parametrize("beta", [1, 2, 4])
def test_beta_ensemble_reproduces_gaussian_ensemble_spacing_statistics(beta):
    e = rm.BetaEnsemble(3000, beta).eigenvalues(random_state=7)
    r = rm.EigenvalueSpacing(e).mean_ratio()
    # SE of <r> ~ 0.28 / sqrt(3000) ~ 0.005
    assert abs(r - rm.EigenvalueSpacing.mean_ratio_reference(beta)) < 0.02
    assert rm.BetaEnsemble(10, beta).mean_ratio_reference() == pytest.approx(
        rm.EigenvalueSpacing.mean_ratio_reference(beta))


def test_beta_laguerre_follows_marchenko_pastur_and_validates():
    be = rm.BetaEnsemble(300, 1.0, kind="laguerre", a=1200)
    e = be.normalized_eigenvalues(random_state=0)
    law = be.limit_law()
    assert isinstance(law, rm.MarchenkoPastur) and law.gamma == pytest.approx(0.25)
    assert law.compare(e).pvalue > 0.05
    assert rm.BetaEnsemble(5, 2.5).mean_ratio_reference() is None
    with pytest.raises(ValueError):
        rm.BetaEnsemble(10, 1.0, kind="laguerre", a=5)
    with pytest.raises(ValueError):
        rm.BetaEnsemble(10, 1.0, kind="cubic")


@pytest.mark.parametrize("beta", [1, 2])
def test_jacobi_ensemble_support_mean_and_wachter_law(beta):
    j = rm.JacobiEnsemble(200, 800, 1200, beta=beta)
    e = j.eigenvalues(random_state=0)
    assert np.all(e >= 0) and np.all(e <= 1)
    assert abs(e.mean() - 800 / 2000) < 0.02
    grid = np.linspace(0, 1, 4001)
    pdf = j.limit_density(grid)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    assert cdf[-1] == pytest.approx(1.0, abs=1e-3)
    assert sps.kstest(e, lambda x: np.interp(x, grid, cdf)).pvalue > 0.05
    with pytest.raises(ValueError):
        rm.JacobiEnsemble(10, 5, 20)


# ========================================================== random_rotations

@pytest.mark.parametrize("group,n,dim", [("O", 7, 7), ("U", 7, 7), ("Sp", 4, 8)])
def test_haar_samples_are_in_the_group(group, n, dim):
    Q = rm.HaarMeasure(group, n).sample(random_state=0)
    assert Q.shape == (dim, dim)
    assert np.allclose(Q.conj().T @ Q, np.eye(dim), atol=1e-12)
    if group == "Sp":
        J = rm.HaarMeasure.symplectic_form(n)
        assert np.allclose(Q.T @ J @ Q, J, atol=1e-12)
    if group == "O":
        assert Q.dtype == float


@pytest.mark.parametrize("group", ["O", "U", "Sp"])
def test_haar_second_moments_and_invariance(group):
    h = rm.HaarMeasure(group, 5)
    S = h.samples(3000, random_state=1)
    m = S.shape[-1]
    # E|Q_ij|^2 = 1/m exactly under Haar; SE ~ sqrt(Var)/sqrt(3000) with Var < 1/m^2
    assert abs(np.mean(np.abs(S[:, 0, 0]) ** 2) - 1.0 / m) < 3 / (m * math.sqrt(3000)) + 0.005
    assert abs(S[:, 1, 2].mean()) < 0.03


def test_haar_orthogonal_matches_scipy_ortho_group_marginal():
    ours = rm.HaarMeasure("O", 6).samples(3000, random_state=2)[:, 0, 0]
    ref = sps.ortho_group.rvs(6, size=3000, random_state=3)[:, 0, 0]
    assert sps.ks_2samp(ours, ref).pvalue > 0.01


def test_haar_unitary_matches_scipy_unitary_group_marginal():
    ours = rm.HaarMeasure("U", 6).samples(3000, random_state=2)[:, 0, 0]
    ref = sps.unitary_group.rvs(6, size=3000, random_state=3)[:, 0, 0]
    assert sps.ks_2samp(np.abs(ours) ** 2, np.abs(ref) ** 2).pvalue > 0.01


def test_unitary_eigenangles_uniform_but_orthogonal_have_atoms():
    ang_u = np.concatenate([rm.RandomUnitaryMatrix(8).eigenangles(s) for s in range(200)])
    assert sps.kstest(ang_u, sps.uniform(-np.pi, 2 * np.pi).cdf).pvalue > 0.01
    ang_o = np.concatenate([rm.RandomOrthogonalMatrix(7).eigenangles(s) for s in range(200)])
    # odd n: every draw has a real eigenvalue +-1 -> an atom at angle 0 or pi
    assert np.mean(np.isclose(np.abs(ang_o) % np.pi, 0.0, atol=1e-8)) > 0.1


def test_special_orthogonal_and_facades():
    so = rm.RandomOrthogonalMatrix(5, det=1)
    assert all(np.linalg.det(so.sample(s)) == pytest.approx(1.0) for s in range(5))
    om = rm.RandomOrthogonalMatrix(5, det=-1)
    assert np.linalg.det(om.sample(0)) == pytest.approx(-1.0)
    assert rm.RandomSymplectic(3).sample(0).shape == (6, 6)
    assert rm.RandomSymplectic(3).dim == 6 and rm.RandomUnitaryMatrix(3).dim == 3
    with pytest.raises(ValueError):
        rm.HaarMeasure("SU", 3)
    with pytest.raises(ValueError):
        rm.RandomOrthogonalMatrix(3, det=2)


# ================================================================ statistics

@pytest.mark.parametrize("ensemble,beta", [(rm.GOE(2000), 1), (rm.GUE(2000), 2),
                                           (rm.GSE(1000), 4)])
def test_mean_gap_ratio_matches_large_n_reference(ensemble, beta):
    sp = rm.EigenvalueSpacing(ensemble.eigenvalues(random_state=11))
    ref = rm.EigenvalueSpacing.mean_ratio_reference(beta)
    assert abs(sp.mean_ratio() - ref) < 0.02          # SE ~ 0.28/sqrt(n) <= 0.009
    assert sp.classify() == {1: "GOE", 2: "GUE", 4: "GSE"}[beta]
    res = sp.compare(beta)
    assert isinstance(res, TestResult) and res.pvalue > 0.01
    assert res.extras["mean_ratio"] == pytest.approx(sp.mean_ratio())


def test_poisson_spectrum_has_uncorrelated_spacings():
    rng = np.random.default_rng(0)
    e = np.sort(rng.uniform(0.0, 1.0, 3000))
    sp = rm.EigenvalueSpacing(e)
    assert abs(sp.mean_ratio() - (2 * np.log(2) - 1)) < 0.02
    assert sp.classify() == "poisson" and sp.compare("poisson").pvalue > 0.01
    assert sp.compare(1).pvalue < 1e-6                 # GOE surmise rejected
    assert rm.EigenvalueSpacing(e, unfold="none").spacings().mean() == pytest.approx(1.0)


def test_wigner_surmises_are_unit_mean_densities():
    for beta in (1, 2, 4, "poisson"):
        f = lambda s: rm.EigenvalueSpacing.surmise(s, beta)
        assert integrate.quad(f, 0, 20)[0] == pytest.approx(1.0, abs=1e-8)
        assert integrate.quad(lambda s: s * f(s), 0, 20)[0] == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError):
        rm.EigenvalueSpacing.surmise(1.0, 3)
    with pytest.raises(ValueError):
        rm.EigenvalueSpacing([1.0, 2.0])


@pytest.mark.parametrize("source,target", [(rm.GOE(2000), 1.0), (rm.GUE(2000), 2.0),
                                           (rm.GSE(1000), 4.0), (rm.BetaEnsemble(2000, 4.0), 4.0)])
def test_level_repulsion_exponent_recovers_dyson_index(source, target):
    e = source.eigenvalues(random_state=0)
    lr = rm.LevelRepulsion(e)
    assert abs(lr.exponent_ - target) < 0.45
    assert lr.classify() == {1.0: "GOE", 2.0: "GUE", 4.0: "GSE"}[target]
    assert integrate.quad(lr.surmise, 0, 10)[0] == pytest.approx(1.0, abs=1e-6)
    assert integrate.quad(lambda s: s * lr.surmise(s), 0, 10)[0] == pytest.approx(1.0, abs=1e-6)
    small = rm.LevelRepulsion(e, method="smallgap")
    assert small.exponent_ > 0.3 and small.n_used_ >= 5


def test_level_repulsion_poisson_is_zero():
    e = np.sort(np.random.default_rng(1).uniform(size=3000))
    assert rm.LevelRepulsion(e).exponent_ < 0.2
    assert rm.LevelRepulsion(e).classify() == "poisson"
    assert abs(rm.LevelRepulsion(e, method="smallgap").exponent_) < 0.3
    with pytest.raises(ValueError):
        rm.LevelRepulsion(e, method="bogus")


def test_eigenvalue_distribution_cdf_kde_moments_and_compare():
    e = rm.GOE(800).normalized_eigenvalues(random_state=4)
    ed = rm.EigenvalueDistribution(e)
    assert ed.n == 800 and ed.cdf(0.0) == pytest.approx(0.5, abs=0.05)
    assert abs(ed.pdf(0.0) - rm.WignerSemicircle().pdf(0.0)) < 0.03
    assert np.allclose(ed.moments([2, 4]), [1.0, 2.0], atol=0.1)
    assert ed.moments(1) == pytest.approx(ed.mean_)
    assert ed.compare(rm.WignerSemicircle(2.0)).pvalue > 0.05
    assert ed.compare(rm.WignerSemicircle(1.5)).pvalue < 1e-4
    assert ed.compare(sps.norm(0, 2)).pvalue < 1e-4      # any object with .cdf works
    lo, hi = ed.support()
    assert lo == e.min() and hi == e.max()
    with pytest.raises(ValueError):
        rm.EigenvalueDistribution(rm.CUE(10).eigenvalues(0))


@pytest.mark.parametrize("ensemble,beta", [(rm.GUE(300), 2), (rm.GOE(300), 1), (rm.GSE(150), 4),
                                           (rm.BetaEnsemble(400, 2.0), 2)])
def test_largest_eigenvalue_is_tracy_widom_distributed(ensemble, beta):
    le = rm.LargestEigenvalue(ensemble)
    assert le.beta == beta and le.kind_ == "hermite"
    res = le.compare(n_samples=200, random_state=5)
    tw_mean, tw_var = TW_REF[beta]
    se = math.sqrt(tw_var / 200)
    # finite-n edge bias is O(n^-2/3) ~ 0.02-0.04 at these sizes; 4 SE + 0.05 covers it
    assert abs(res.extras["sample_mean"] - tw_mean) < 4 * se + 0.05
    assert res.statistic < 0.12 and res.pvalue > 0.005


def test_wishart_largest_eigenvalue_johnstone_scaling():
    W = rm.WishartMatrix(100, 400)
    le = rm.LargestEigenvalue(W)
    assert le.kind_ == "wishart" and le.beta == 1
    a, b = 400 - 0.5, 100 - 0.5
    assert le.center == pytest.approx((math.sqrt(a) + math.sqrt(b)) ** 2)
    res = le.compare(n_samples=200, random_state=6)
    assert abs(res.extras["sample_mean"] - TW_REF[1][0]) < 4 * math.sqrt(TW_REF[1][1] / 200) + 0.1
    assert res.pvalue > 0.005
    lag = rm.LargestEigenvalue(rm.BetaEnsemble(100, 1.0, "laguerre", a=400))
    assert lag.kind_ == "wishart" and lag.center == pytest.approx(le.center)
    with pytest.raises(ValueError):
        rm.LargestEigenvalue(rm.WishartMatrix(2, 10, sigma=[[2.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        rm.LargestEigenvalue(rm.CUE(10))
    custom = rm.LargestEigenvalue(rm.CUE(10), beta=2, center=0.0, scale=1.0)
    assert custom.kind_ == "custom" and np.allclose(custom.scaled([1.0, 2.0]), [1.0, 2.0])


def test_bulk_spectrum_density_moments_and_number_variance():
    b = rm.BulkSpectrum(rm.GOE(1500), random_state=0)
    assert b.compare().pvalue > 0.05                  # normalized against the ensemble's law
    assert abs(b.moments(2) / 1500 - 1.0) < 0.05     # raw eigenvalues: E[lambda^2] ~ n
    assert b.density(0.0) > 0 and b.spacing().mean_ratio() > 0.5
    for L in (1.0, 3.0):
        s2 = b.number_variance(L)
        assert abs(s2 - rm.BulkSpectrum.number_variance_reference(L, 1)) < 0.1
    pois = rm.BulkSpectrum(np.sort(np.random.default_rng(2).uniform(size=3000)))
    assert abs(pois.number_variance(3.0) - 3.0) < 0.5
    with pytest.raises(ValueError):
        pois.compare()
    with pytest.raises(ValueError):
        b.number_variance(1e6)
    with pytest.raises(ValueError):
        rm.BulkSpectrum.number_variance_reference(1.0, 3)


def test_spectral_edge_extremes_soft_edge_and_edelman_hard_edge():
    se = rm.SpectralEdge(rm.WishartMatrix(60, 60), random_state=0)
    lo, hi = se.edges()
    assert lo == se.smallest() and hi == se.largest() and lo < hi
    # Edelman's density integrates to one and its cdf is the stated closed form
    assert integrate.quad(rm.SpectralEdge.hard_edge_pdf, 0, np.inf)[0] == pytest.approx(1.0, abs=1e-8)
    assert rm.SpectralEdge.hard_edge_cdf(5.0) == pytest.approx(
        integrate.quad(rm.SpectralEdge.hard_edge_pdf, 0, 5)[0], abs=1e-8)
    res = se.hard_edge_compare(n_samples=300, random_state=1)
    assert res.pvalue > 0.01
    soft = rm.SpectralEdge(rm.GUE(200)).soft_edge(n_samples=120, random_state=2)
    assert soft.pvalue > 0.005
    with pytest.raises(ValueError):
        rm.SpectralEdge(rm.WishartMatrix(10, 20)).hard_edge_scaled(5)
    with pytest.raises(ValueError):
        rm.SpectralEdge(np.arange(10.0)).soft_edge()


# ==================================================================== wiring

_SPEC_NAMES = {
    "GOE", "GUE", "GSE", "WishartMatrix", "InverseWishart", "WignerMatrix", "CUE",
    "MuresanMatrix", "MarchenkoPastur", "WignerSemicircle", "TracyWidomDistribution",
    "BetaEnsemble", "JacobiEnsemble", "RandomOrthogonalMatrix", "RandomUnitaryMatrix",
    "HaarMeasure", "RandomSymplectic", "EigenvalueSpacing", "LevelRepulsion",
    "EigenvalueDistribution", "LargestEigenvalue", "BulkSpectrum", "SpectralEdge",
}


def test_module_wiring_and_exports():
    assert "random_matrix" in stochpylib.__all__
    assert stochpylib.random_matrix is rm
    assert len(_SPEC_NAMES) == 23 and _SPEC_NAMES <= set(rm.__all__)
    assert set(rm.__all__) - _SPEC_NAMES == {"MatrixEnsemble", "CircularLaw"}
    assert list(rm.__all__) == sorted(rm.__all__) and len(set(rm.__all__)) == len(rm.__all__)
    assert (set(rm_en.__all__) | set(rm_es.__all__) | set(rm_rr.__all__)
            | set(rm_st.__all__)) == set(rm.__all__)
    import statistics as stdlib_statistics
    assert stdlib_statistics.mean([1, 2, 3]) == 2  # random_matrix/statistics.py does not shadow


def test_limit_laws_satisfy_the_distribution_contract():
    from stochpylib.distributions import Distribution

    for law in (rm.WignerSemicircle(), rm.MarchenkoPastur(0.5), rm.TracyWidomDistribution(2)):
        assert isinstance(law, Distribution)
        for meth in ("pdf", "cdf", "ppf", "rvs", "mean", "var", "skewness", "kurtosis",
                     "entropy", "mgf", "cf", "fit", "ks_test", "compare", "histogram_vs_density"):
            assert callable(getattr(law, meth)), meth
        x = law.ppf(0.3)
        assert law.cdf(x) == pytest.approx(0.3, abs=1e-6)
        c, emp, theo = law.histogram_vs_density(law.rvs(2000, random_state=0), bins=10)
        assert c.shape == emp.shape == theo.shape == (10,)


def test_quickstart_example_runs():
    goe = rm.GOE(n=500)
    eigenvalues = goe.eigenvalues(random_state=0)
    res = goe.limit_law().compare(goe.normalize(eigenvalues))
    assert not res.reject(0.01)
    W = rm.WishartMatrix(p=200, n=1000)
    eigs = W.eigenvalues(random_state=0)
    mp = rm.MarchenkoPastur(gamma=0.2)
    assert mp.compare(eigs / 1000).pvalue > 0.05
    assert rm.EigenvalueSpacing(eigenvalues).classify() == "GOE"


def test_reproducibility_with_random_state():
    assert np.array_equal(rm.GOE(30).eigenvalues(3), rm.GOE(30).eigenvalues(3))
    assert not np.array_equal(rm.GOE(30).eigenvalues(3), rm.GOE(30).eigenvalues(4))
    rng = np.random.default_rng(9)
    a = rm.HaarMeasure("U", 4).sample(rng)
    b = rm.HaarMeasure("U", 4).sample(rng)
    assert not np.allclose(a, b)                      # a live Generator streams
    assert np.array_equal(rm.TracyWidomDistribution(1).rvs(5, random_state=0),
                          rm.TracyWidomDistribution(1).rvs(5, random_state=0))


def test_no_scipy_stats_in_library_code():
    import pathlib
    import re

    pattern = re.compile(r"^\s*(from scipy import .*stats|from scipy\.stats|import scipy\.stats)")
    pkg = pathlib.Path(rm.__file__).parent
    for f in pkg.glob("*.py"):
        for line in f.read_text(encoding="utf-8").splitlines():
            assert not pattern.search(line), f"{f.name}: {line.strip()}"
