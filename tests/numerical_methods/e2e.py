"""End-to-end API sweep for stochpylib.numerical_methods: one realistic exercise per
public name. ``test_every_public_name_is_exercised`` fails the moment a name ships
without one; ``test_exercise[<name>]`` runs each exercise as its own pytest case."""

import numpy as np
import pytest

from stochpylib import numerical_methods as mod

EXERCISES = {}


def exercise(name):
    def deco(fn):
        EXERCISES[name] = fn
        return fn
    return deco


def _finite(a):
    return bool(np.all(np.isfinite(np.asarray(a))))


# ----------------------------------------------------------------- integration

@exercise("GaussLegendre")
def _gl():
    gl = mod.GaussLegendre(10)
    r = gl.integrate(lambda x: x ** 4, 0, 1)
    assert abs(r.value - 0.2) < 1e-10


@exercise("GaussHermite")
def _gh():
    gh = mod.GaussHermite(20, kind="probabilists")
    r = gh.expectation(lambda z: z ** 2)
    assert abs(r.value - 1.0) < 1e-8


@exercise("GaussChebyshev")
def _gc():
    gc = mod.GaussChebyshev(30, kind=1)
    r = gc.integrate_weighted(lambda x: 1.0)
    assert abs(r.value - np.pi) < 1e-8


@exercise("AdaptiveQuadrature")
def _aq():
    r = mod.AdaptiveQuadrature(np.sin, 0, np.pi).integrate()
    assert abs(r.value - 2.0) < 1e-6 and r.converged


@exercise("NumericalIntegration")
def _ni():
    r = mod.NumericalIntegration(np.sin, 0, np.pi, method="romberg").integrate()
    assert abs(r.value - 2.0) < 1e-8


@exercise("MonteCarloIntegration")
def _mci():
    r = mod.MonteCarloIntegration(lambda x: x ** 2, bounds=[(0, 1)], n=50_000,
                                  random_state=0).integrate()
    assert abs(r.value - 1 / 3) < 4 * r.error


@exercise("CubatureRule")
def _cr():
    cr = mod.CubatureRule(dim=2, rule="gauss_legendre", n=6, method="tensor")
    r = cr.integrate(lambda pt: pt[0] * pt[1], bounds=[(0, 1), (0, 1)])
    assert abs(r.value - 0.25) < 1e-10


# ----------------------------------------------------------------- ode_sde

@exercise("EulerMethod")
def _euler():
    sol = mod.EulerMethod().solve(lambda t, y: -y, (0, 1), [1.0], n_steps=1000)
    assert abs(sol.y[-1, 0] - np.exp(-1.0)) < 1e-2


@exercise("RungeKutta4")
def _rk4():
    sol = mod.RungeKutta4().solve(lambda t, y: -y, (0, 1), [1.0], n_steps=50)
    assert abs(sol.y[-1, 0] - np.exp(-1.0)) < 1e-6


@exercise("DormandPrince")
def _dp():
    sol = mod.DormandPrince(rtol=1e-8).solve(lambda t, y: np.array([y[1], -y[0]]),
                                             (0, 2 * np.pi), [1.0, 0.0])
    assert abs(sol.y[-1, 0] - 1.0) < 1e-4 and sol.success


@exercise("Adams_Bashforth")
def _ab():
    sol = mod.Adams_Bashforth(order=4).solve(lambda t, y: -y, (0, 1), [1.0], n_steps=80)
    assert abs(sol.y[-1, 0] - np.exp(-1.0)) < 1e-6


@exercise("BDF")
def _bdf():
    sol = mod.BDF(order=2).solve(lambda t, y: np.array([-1000 * y[0]]), (0, 1), [1.0], h=0.01)
    assert _finite(sol.y) and sol.success


@exercise("Euler_Maruyama_SDE")
def _em_sde():
    sol = mod.Euler_Maruyama_SDE(lambda t, x: 0.05 * x, lambda t, x: 0.2 * x).solve(
        (0, 1.0), [100.0], n_steps=100, n_paths=200, random_state=0)
    assert sol.terminal.shape == (200, 1) and _finite(sol.terminal)


@exercise("Milstein_SDE")
def _mil_sde():
    sol = mod.Milstein_SDE(lambda t, x: 0.05 * x, lambda t, x: 0.2 * x,
                           diffusion_dx=lambda t, x: 0.2 * np.ones_like(x)).solve(
        (0, 1.0), [100.0], n_steps=100, n_paths=200, random_state=0)
    assert _finite(sol.terminal)


# ------------------------------------------------------------ linear_algebra

@exercise("MatrixExponential")
def _me():
    A = np.array([[0.0, -1.0], [1.0, 0.0]])
    R = mod.MatrixExponential(A).at(np.pi / 2)
    assert np.allclose(R, [[0, -1], [1, 0]], atol=1e-8)


@exercise("MatrixLogarithm")
def _ml():
    A = np.diag([2.0, 3.0])
    lg = mod.MatrixLogarithm(A).compute()
    assert np.allclose(lg.logm_, np.diag([np.log(2.0), np.log(3.0)]), atol=1e-8)


@exercise("CholeskyDecomp")
def _chol():
    A = np.array([[4.0, 2.0], [2.0, 3.0]])
    ch = mod.CholeskyDecomp(A)
    assert np.allclose(ch.reconstruct(), A, atol=1e-10)


@exercise("EigenDecomp")
def _eig():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    ed = mod.EigenDecomp(A).compute()
    assert np.allclose(sorted(ed.eigenvalues_), [1.0, 3.0], atol=1e-8)


@exercise("SVD")
def _svd():
    A = np.array([[1.0, 0.0], [0.0, 2.0]])
    sv = mod.SVD(A).compute()
    assert np.allclose(sorted(sv.s_), [1.0, 2.0], atol=1e-8)


@exercise("QRDecomp")
def _qr():
    A = np.random.default_rng(0).standard_normal((5, 3))
    qr = mod.QRDecomp(A).compute()
    assert np.max(np.abs(qr.reconstruct() - A)) < 1e-8


@exercise("Schur")
def _schur():
    A = np.random.default_rng(1).standard_normal((5, 5))
    s = mod.Schur(A).compute("real")
    assert np.max(np.abs(s.reconstruct() - A)) < 1e-6


# ------------------------------------------------------------------ root_solve

@exercise("Bisection")
def _bis():
    r = mod.Bisection(lambda x: x ** 2 - 2, 0, 2)
    assert abs(r.root - np.sqrt(2)) < 1e-8


@exercise("Brent")
def _brent():
    r = mod.Brent(lambda x: x ** 2 - 2, 0, 2)
    assert abs(r.root - np.sqrt(2)) < 1e-10


@exercise("Secant")
def _secant():
    r = mod.Secant(lambda x: x ** 2 - 2, 1.0, 2.0)
    assert abs(r.root - np.sqrt(2)) < 1e-8


@exercise("NewtonRaphson")
def _newton():
    r = mod.NewtonRaphson(lambda x: x ** 2 - 2, 1.0, fprime=lambda x: 2 * x)
    assert abs(r.root - np.sqrt(2)) < 1e-10


@exercise("FixedPoint")
def _fixed():
    r = mod.FixedPoint(lambda x: 0.5 * (x + 2 / x), 1.0)
    assert abs(r.root - np.sqrt(2)) < 1e-8


@exercise("RootFinding")
def _rf():
    rf = mod.RootFinding(lambda x: x ** 2 - 2, bracket=(0, 2))
    r = rf.solve()
    assert abs(r.root - np.sqrt(2)) < 1e-8


# --------------------------------------------------------------- interpolation

@exercise("SplineInterpolation")
def _spline():
    x = np.linspace(0, 2 * np.pi, 20)
    sp = mod.SplineInterpolation(x, np.sin(x))
    assert abs(sp(1.0) - np.sin(1.0)) < 1e-3


@exercise("CubicHermite")
def _hermite():
    x = np.linspace(0, 2 * np.pi, 20)
    ch = mod.CubicHermite(x, np.sin(x))
    assert abs(ch(1.0) - np.sin(1.0)) < 1e-3


@exercise("BarycentricLagrange")
def _baryc():
    x = np.linspace(0, 2 * np.pi, 15)
    bl = mod.BarycentricLagrange(x, np.sin(x))
    assert abs(bl(1.0) - np.sin(1.0)) < 1e-2


@exercise("Chebyshev")
def _cheb():
    cb = mod.Chebyshev.from_function(np.exp, 15, -1, 1)
    assert abs(cb(0.5) - np.exp(0.5)) < 1e-10


@exercise("NURBS")
def _nurbs():
    circ = mod.NURBS.circle()
    pts = circ.evaluate(np.linspace(0, 1, 20))
    assert np.allclose(np.linalg.norm(pts, axis=1), 1.0, atol=1e-8)


@exercise("Interpolation")
def _interp():
    x = np.linspace(0, 10, 15)
    interp = mod.Interpolation(x, np.sin(x), method="cubic")
    assert abs(interp(2.5) - np.sin(2.5)) < 1e-2


# ---------------------------------------------------------------------- pde

@exercise("Mesh")
def _mesh():
    mesh = mod.Mesh.interval(0, 1, 10)
    assert mesh.n_nodes == 11 and mesh.dim == 1


@exercise("FiniteDifference")
def _fd():
    fd = mod.FiniteDifference()
    x, u = fd.poisson_1d(lambda x: np.pi ** 2 * np.sin(np.pi * x), 0, 1, 40)
    assert np.max(np.abs(u - np.sin(np.pi * x))) < 1e-2


@exercise("FiniteElement")
def _fe():
    mesh = mod.Mesh.interval(0, 1, 20)
    fe = mod.FiniteElement(mesh, degree=1)
    fe.assemble(p=1.0, f=lambda x: np.pi ** 2 * np.sin(np.pi * x),
               dirichlet={"left": 0.0, "right": 0.0})
    fe.solve()
    assert fe.l2_error(lambda x: np.sin(np.pi * x)) < 1e-2


@exercise("FEniCS_Interface")
def _fenics():
    mesh = mod.Mesh.interval(0, 1, 20)
    fen = mod.FEniCS_Interface(mesh)
    fen.set_equation(source=lambda x: np.pi ** 2 * np.sin(np.pi * x))
    fen.dirichlet_bc(0.0)
    fen.solve()
    assert _finite(fen.u_)


@exercise("BoundaryElement")
def _bem():
    n_per_side = 20
    pts = []
    for i in range(n_per_side):
        pts.append([i / n_per_side, 0.0])
    for i in range(n_per_side):
        pts.append([1.0, i / n_per_side])
    for i in range(n_per_side):
        pts.append([1.0 - i / n_per_side, 1.0])
    for i in range(n_per_side):
        pts.append([0.0, 1.0 - i / n_per_side])
    pts = np.array(pts)
    bem = mod.BoundaryElement(pts)
    bem.solve(lambda mids: mids[:, 0] ** 2 - mids[:, 1] ** 2)
    u = bem.evaluate(np.array([[0.5, 0.2]]))
    assert abs(u[0] - (0.5 ** 2 - 0.2 ** 2)) < 1e-2


@exercise("SpectralMethod")
def _spectral():
    sm = mod.SpectralMethod()
    n = 64
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    du = sm.fourier_derivative(np.sin(x), 2 * np.pi)
    assert np.max(np.abs(du - np.cos(x))) < 1e-8


# -------------------------------------------------------------- result objects

@exercise("QuadratureResult")
def _qres():
    r = mod.QuadratureResult(value=1.0, error=0.1, n_evals=10, method="test")
    assert float(r) == 1.0


@exercise("RootResult")
def _rres():
    r = mod.RootResult(root=1.5, f_root=0.0, iterations=5, converged=True, method="test")
    assert float(r) == 1.5


@exercise("ODESolution")
def _odesol():
    sol = mod.RungeKutta4().solve(lambda t, y: -y, (0, 1), [1.0], n_steps=20)
    v = sol(0.5)
    assert np.isfinite(v) if np.ndim(v) == 0 else _finite(v)


@exercise("SDESolution")
def _sdesol():
    sol = mod.Euler_Maruyama_SDE(lambda t, x: 0.0 * x, lambda t, x: np.ones_like(x)).solve(
        (0, 1.0), [0.0], n_steps=50, n_paths=100, random_state=0)
    assert sol.mean_path().shape == (51, 1)


def test_every_public_name_is_exercised():
    assert set(EXERCISES) == set(mod.__all__), (
        set(mod.__all__) - set(EXERCISES), set(EXERCISES) - set(mod.__all__))


@pytest.mark.parametrize("name", sorted(mod.__all__))
def test_exercise(name):
    EXERCISES[name]()
