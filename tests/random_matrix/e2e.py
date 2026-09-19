"""End-to-end API sweep for stochpylib.random_matrix: one realistic exercise per public
name. ``test_every_public_name_is_exercised`` fails the moment a name ships without one;
``test_exercise[<name>]`` runs each exercise as its own pytest case."""

import numpy as np
import pytest

from stochpylib import random_matrix as mod
from stochpylib.statistics import TestResult

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


def _finite(a):
    return bool(np.all(np.isfinite(np.asarray(a))))


@exercise("GOE")
def _goe():
    e = mod.GOE(200).normalized_eigenvalues(random_state=0)
    assert e.shape == (200,) and abs(e.max()) < 2.3 and _finite(e)


@exercise("GUE")
def _gue():
    H = mod.GUE(60).sample(random_state=0)
    assert np.allclose(H, H.conj().T) and mod.GUE(60).beta == 2


@exercise("GSE")
def _gse():
    e = mod.GSE(40).eigenvalues(random_state=0)
    assert e.shape == (40,) and np.all(np.diff(e) >= 0)


@exercise("WignerMatrix")
def _wigner():
    from stochpylib.distributions import Uniform
    e = mod.WignerMatrix(200, Uniform(0, 1)).normalized_eigenvalues(random_state=0)
    assert abs(e.max()) < 2.4 and abs(e.mean()) < 0.2


@exercise("WishartMatrix")
def _wishart():
    W = mod.WishartMatrix(p=30, n=120)
    e = W.normalized_eigenvalues(random_state=0)
    assert np.all(e > 0) and isinstance(W.limit_law(), mod.MarchenkoPastur)


@exercise("InverseWishart")
def _inverse_wishart():
    e = mod.InverseWishart(p=10, n=60).eigenvalues(random_state=0)
    assert np.all(e > 0) and e.shape == (10,)


@exercise("CUE")
def _cue():
    ang = mod.CUE(50).eigenangles(random_state=0)
    assert ang.shape == (50,) and np.all(np.abs(ang) <= np.pi)


@exercise("MuresanMatrix")
def _muresan():
    z = mod.MuresanMatrix(100, "complex").normalized_eigenvalues(random_state=0)
    assert np.iscomplexobj(z) and np.abs(z).max() < 1.3


@exercise("CircularLaw")
def _circular_law():
    law = mod.CircularLaw()
    z = law.rvs(500, random_state=0)
    assert law.compare(z).pvalue > 0.001 and law.radial_cdf(1.0) == 1.0


@exercise("MatrixEnsemble")
def _matrix_ensemble():
    assert issubclass(mod.GOE, mod.MatrixEnsemble)
    assert mod.GOE(5).samples(3, random_state=0).shape == (3, 5, 5)


@exercise("WignerSemicircle")
def _semicircle():
    w = mod.WignerSemicircle(2.0)
    x = w.rvs(300, random_state=0)
    assert abs(w.cdf(w.ppf(0.25)) - 0.25) < 1e-8 and w.compare(x).pvalue > 0.001
    assert w.moment(2) == 1.0 and _finite(w.mgf(0.3)) and _finite(w.cf(0.3).real)


@exercise("MarchenkoPastur")
def _mp():
    mp = mod.MarchenkoPastur(0.5)
    x = mp.rvs(300, random_state=0)
    assert abs(x.mean() - 1.0) < 0.15 and mod.MarchenkoPastur.fit(x).gamma > 0


@exercise("TracyWidomDistribution")
def _tw():
    tw = mod.TracyWidomDistribution(2)
    assert abs(tw.mean() + 1.771) < 0.01 and abs(tw.cdf(tw.ppf(0.9)) - 0.9) < 1e-6


@exercise("BetaEnsemble")
def _beta_ensemble():
    e = mod.BetaEnsemble(300, beta=3.0).normalized_eigenvalues(random_state=0)
    assert abs(e.max()) < 2.3
    lag = mod.BetaEnsemble(50, 1.0, kind="laguerre", a=200).normalized_eigenvalues(random_state=0)
    assert np.all(lag > 0)


@exercise("JacobiEnsemble")
def _jacobi():
    j = mod.JacobiEnsemble(20, 80, 120)
    e = j.eigenvalues(random_state=0)
    assert np.all((e >= 0) & (e <= 1)) and j.limit_density(0.4) > 0


@exercise("HaarMeasure")
def _haar():
    Q = mod.HaarMeasure("Sp", 3).sample(random_state=0)
    J = mod.HaarMeasure.symplectic_form(3)
    assert np.allclose(Q.T @ J @ Q, J, atol=1e-10)


@exercise("RandomOrthogonalMatrix")
def _rom():
    Q = mod.RandomOrthogonalMatrix(6, det=1).sample(random_state=0)
    assert np.allclose(Q.T @ Q, np.eye(6), atol=1e-10) and np.linalg.det(Q) > 0


@exercise("RandomUnitaryMatrix")
def _rum():
    U = mod.RandomUnitaryMatrix(6).sample(random_state=0)
    assert np.allclose(U.conj().T @ U, np.eye(6), atol=1e-10)


@exercise("RandomSymplectic")
def _rsp():
    assert mod.RandomSymplectic(2).eigenangles(random_state=0).shape == (4,)


@exercise("EigenvalueSpacing")
def _spacing():
    sp = mod.EigenvalueSpacing(mod.GOE(500).eigenvalues(random_state=0))
    assert 0.45 < sp.mean_ratio() < 0.6 and sp.classify() == "GOE"
    assert isinstance(sp.compare(1), TestResult) and sp.spacings().mean() == pytest.approx(1.0)


@exercise("LevelRepulsion")
def _repulsion():
    lr = mod.LevelRepulsion(mod.GUE(500).eigenvalues(random_state=0))
    assert 1.0 < lr.exponent_ < 3.5 and lr.classify() == "GUE"


@exercise("EigenvalueDistribution")
def _esd():
    ed = mod.EigenvalueDistribution(mod.GOE(300).normalized_eigenvalues(random_state=0))
    assert 0.3 < ed.cdf(0.0) < 0.7 and ed.pdf(0.0) > 0.2 and _finite(ed.moments([1, 2]))
    assert ed.compare(mod.WignerSemicircle()).pvalue > 0.001


@exercise("LargestEigenvalue")
def _largest():
    le = mod.LargestEigenvalue(mod.GUE(100))
    z = le.scaled(le.sample(20, random_state=0))
    assert z.shape == (20,) and -6 < z.mean() < 2 and le.tracy_widom().beta == 2


@exercise("BulkSpectrum")
def _bulk():
    b = mod.BulkSpectrum(mod.GOE(400), random_state=0)
    assert b.compare().pvalue > 0.001 and 0.1 < b.number_variance(2.0) < 2.0


@exercise("SpectralEdge")
def _edge():
    se = mod.SpectralEdge(mod.WishartMatrix(20, 20), random_state=0)
    lo, hi = se.edges()
    assert lo < hi and se.hard_edge_compare(40, random_state=1).pvalue > 1e-4
    assert mod.SpectralEdge.hard_edge_cdf(1e9) == pytest.approx(1.0)


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
