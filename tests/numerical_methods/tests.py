"""Tests for stochpylib.numerical_methods: quadrature, ODE/SDE solvers, linear algebra,
root finding, interpolation, and PDE tools.

Oracles: scipy.integrate (quad, solve_ivp, simpson), scipy.linalg (expm, logm, schur,
qr, svd, eig), scipy.optimize (brentq, fsolve), scipy.interpolate (CubicSpline,
PchipInterpolator, BarycentricInterpolator, BSpline), numpy.polynomial (leggauss,
hermgauss, chebgauss), closed forms, and library modules (financial_stochastics,
distributions, levy_processes) as cross-module oracles. All randomness is seeded;
statistical/Monte-Carlo assertions use >= 3 standard errors.
"""

import numpy as np
import pytest
from numpy.polynomial import chebyshev as np_chebmod
from numpy.polynomial import hermite_e as np_herme
from numpy.polynomial import legendre as np_leg
from scipy import integrate as sci_integrate
from scipy import optimize as sci_optimize
from scipy.interpolate import BarycentricInterpolator, BSpline, CubicSpline, PchipInterpolator
from scipy.linalg import expm as sp_expm
from scipy.linalg import logm as sp_logm
from scipy.linalg import schur as sp_schur

from stochpylib import numerical_methods as nm
from stochpylib.distributions import Normal
from stochpylib.financial_stochastics import BlackScholes
from stochpylib.financial_stochastics.option_pricing import BinomialTree
from stochpylib.levy_processes.sde import SDE as _LegacySDE
from stochpylib.levy_processes.sde import Euler_Maruyama as _legacy_em
from stochpylib.levy_processes.sde import _brownian_increments as _legacy_binc
from stochpylib.numerical_methods._common import _gauss_kronrod_15, _thomas


# ============================================================== integration

def test_gauss_legendre_matches_leggauss_and_is_exact_for_polynomials():
    gl = nm.GaussLegendre(10)
    xn, wn = np_leg.leggauss(10)
    assert np.allclose(np.sort(gl.nodes_), np.sort(xn))
    assert np.allclose(np.sort(gl.weights_), np.sort(wn))
    r = gl.integrate(lambda x: x ** 6, -1, 1)
    assert abs(r.value - 2 / 7) < 1e-12
    r2 = gl.integrate_composite(np.sin, 0, np.pi, n_panels=4)
    assert abs(r2.value - 2.0) < 1e-10


def test_gauss_hermite_matches_hermgauss_and_normal_moments():
    gh = nm.GaussHermite(20, kind="probabilists")
    xn, wn = np_herme.hermegauss(20)
    assert np.allclose(np.sort(gh.nodes_), np.sort(xn), atol=1e-9)
    assert abs(gh.expectation(lambda z: z ** 2).value - 1.0) < 1e-10
    assert abs(gh.expectation(lambda z: z ** 4).value - 3.0) < 1e-10
    assert abs(gh.expectation(lambda z: np.exp(z)).value - np.exp(0.5)) < 1e-8


def test_gauss_hermite_expectation_matches_black_scholes_call():
    # The call payoff has a kink at S=K, so Gauss-Hermite (which assumes a smooth/
    # analytic integrand for its usual spectral accuracy) converges only slowly and
    # non-monotonically here -- 300 nodes is needed for a 1e-2 match, unlike the smooth
    # moment checks elsewhere in this file which converge to 1e-8 by n=20.
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)
    gh = nm.GaussHermite(300, kind="probabilists")

    def payoff(z):
        ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
        return np.maximum(ST - K, 0.0) * np.exp(-r * T)

    price = gh.expectation(payoff, vectorized=True).value
    assert abs(price - bs.call_price()) < 1e-2


def test_gauss_chebyshev_weighted_integral_and_oracle_nodes():
    gc = nm.GaussChebyshev(20, kind=1)
    r = gc.integrate_weighted(lambda x: x ** 2)
    assert abs(r.value - np.pi / 2) < 1e-10
    gc2 = nm.GaussChebyshev(30, kind=1)
    assert abs(np.sum(gc2.weights_) - np.pi) < 1e-10


def test_adaptive_quadrature_matches_scipy_quad_including_infinite_and_singular():
    r = nm.AdaptiveQuadrature(np.sin, 0, np.pi).integrate()
    exact, _ = sci_integrate.quad(np.sin, 0, np.pi)
    assert abs(r.value - exact) < 1e-8
    assert r.error >= abs(r.value - 2.0) * 0.01 or r.error < 1e-6

    r2 = nm.AdaptiveQuadrature(lambda x: np.exp(-x ** 2), -np.inf, np.inf).integrate()
    assert abs(r2.value - np.sqrt(np.pi)) < 1e-6

    r3 = nm.AdaptiveQuadrature(lambda x: x ** -0.5, 0, 1).integrate()
    assert abs(r3.value - 2.0) < 1e-6

    r4 = nm.AdaptiveQuadrature(np.log, 0, 1).integrate()
    assert abs(r4.value - (-1.0)) < 1e-6

    d = Normal(1.0, 2.0)
    r5 = nm.AdaptiveQuadrature(lambda x: np.asarray(d.pdf(x), dtype=float), -np.inf, np.inf).integrate()
    assert abs(r5.value - 1.0) < 1e-6


def test_adaptive_simpson_matches_quad():
    r = nm.AdaptiveQuadrature(lambda x: np.sin(50 * x), 0, 1, rule="simpson").integrate()
    exact, _ = sci_integrate.quad(lambda x: np.sin(50 * x), 0, 1)
    assert abs(r.value - exact) < 1e-4


def test_composite_simpson_exact_for_cubics():
    f = lambda x: 2 * x ** 3 - x + 1
    r = nm.NumericalIntegration(f, 0, 2, method="simpson", n=10).integrate()
    exact = (0.5 * 2 ** 4 - 0.5 * 2 ** 2 + 2)
    assert abs(r.value - exact) < 1e-10


def test_romberg_converges_tightly():
    r = nm.NumericalIntegration(np.sin, 0, np.pi, method="romberg").integrate()
    assert abs(r.value - 2.0) < 1e-12


def test_trapezoid_error_halves_quadratically():
    f = lambda x: np.sin(3 * x)
    exact, _ = sci_integrate.quad(f, 0, 2)
    errs = []
    for n in (200, 400):
        r = nm.NumericalIntegration(f, 0, 2, method="trapezoid", n=n).integrate()
        errs.append(abs(r.value - exact))
    assert errs[0] / errs[1] > 3.5


def test_monte_carlo_integration_matches_crude_mc_and_error_band():
    from stochpylib.montecarlo import crude_mc

    f = lambda x: x ** 2
    mci = nm.MonteCarloIntegration(f, bounds=[(0, 1)], n=200_000, random_state=0)
    r = mci.integrate()
    ref = crude_mc(lambda pts: pts[:, 0] ** 2, n=200_000, dim=1, bounds=[(0, 1)], random_state=0)
    assert abs(r.value - ref.estimate) < 1e-9
    assert abs(r.value - 1 / 3) < 4 * r.error


def test_cubature_tensor_grid_exact_for_product_polynomial():
    cr = nm.CubatureRule(dim=3, rule="gauss_legendre", n=5, method="tensor")
    r = cr.integrate(lambda pt: pt[0] * pt[1] * pt[2], bounds=[(0, 1)] * 3)
    assert abs(r.value - 0.125) < 1e-10


def test_cubature_smolyak_vs_dblquad():
    exact, _ = sci_integrate.dblquad(lambda y, x: np.cos(x + y), -1, 1, -1, 1)
    cr = nm.CubatureRule(dim=2, rule="clenshaw_curtis", method="smolyak", level=6)
    r = cr.integrate(lambda pt: np.cos(pt[0] + pt[1]), bounds=[(-1, 1), (-1, 1)])
    assert abs(r.value - exact) < 1e-4


def test_cubature_gauss_hermite_expectation_3d():
    a = np.array([1.0, 0.5, -0.3])
    cr = nm.CubatureRule(dim=3, rule="gauss_hermite", n=10, method="tensor")
    r = cr.integrate(lambda pt: np.exp(a @ pt), bounds=None)
    assert abs(r.value - np.exp(0.5 * np.sum(a ** 2))) < 1e-6


def test_gauss_kronrod_error_bounds_true_error():
    k15, err, _ = _gauss_kronrod_15(lambda x: np.exp(x), 0, 1)
    assert abs(k15 - (np.e - 1)) <= max(err, 1e-13)


# =================================================================== ode_sde

def _order_ratios(solver, ns_list, **kw):
    exact = np.exp(-1.0)
    errs = [abs(solver.solve(lambda t, y: -y, (0, 1), [1.0], n_steps=n, **kw).y[-1, 0] - exact)
           for n in ns_list]
    return [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]


def test_euler_method_is_order_one():
    ratios = _order_ratios(nm.EulerMethod(), [50, 100, 200, 400])
    assert all(abs(r - 1.0) < 0.05 for r in ratios)


def test_rk4_is_order_four():
    ratios = _order_ratios(nm.RungeKutta4(), [10, 20, 40, 80])
    assert all(abs(r - 4.0) < 0.3 for r in ratios)


def test_adams_bashforth_order_four():
    ratios = _order_ratios(nm.Adams_Bashforth(order=4), [20, 40, 80, 160])
    assert all(abs(r - 4.0) < 0.3 for r in ratios)


def test_adams_bashforth_moulton_more_accurate_than_bashforth():
    exact = np.exp(-1.0)
    ab = nm.Adams_Bashforth(order=4, corrector=False).solve(lambda t, y: -y, (0, 1), [1.0], n_steps=40)
    abm = nm.Adams_Bashforth(order=4, corrector=True).solve(lambda t, y: -y, (0, 1), [1.0], n_steps=40)
    assert abs(abm.y[-1, 0] - exact) <= abs(ab.y[-1, 0] - exact) + 1e-14


def test_bdf_order_two():
    ratios = _order_ratios(nm.BDF(order=2), [20, 40, 80, 160])
    assert all(abs(r - 2.0) < 0.15 for r in ratios)


def test_dormand_prince_matches_scipy_rk45_dense_output():
    def f(t, y):
        return np.array([y[1], -y[0]])

    dp = nm.DormandPrince(rtol=1e-8, atol=1e-10)
    sol = dp.solve(f, (0, 10), [1.0, 0.0])
    exact = np.column_stack([np.cos(sol.t), -np.sin(sol.t)])
    assert np.max(np.abs(sol.y - exact)) < 1e-6

    t_eval = np.linspace(0, 10, 50)
    sol2 = dp.solve(f, (0, 10), [1.0, 0.0], t_eval=t_eval)
    exact2 = np.column_stack([np.cos(t_eval), -np.sin(t_eval)])
    assert np.max(np.abs(sol2.y - exact2)) < 1e-5

    r = sci_integrate.solve_ivp(f, (0, 10), [1.0, 0.0], method="RK45",
                                rtol=1e-8, atol=1e-10, dense_output=True)
    assert np.max(np.abs(r.sol(sol.t).T - exact)) < 1e-6


def test_bdf_stays_bounded_on_stiff_problem_where_rk4_blows_up():
    def stiff(t, y):
        return np.array([-1000 * (y[0] - np.cos(t))])

    sol_bdf = nm.BDF(order=3).solve(stiff, (0, 1), [0.0], h=0.05)
    assert np.all(np.isfinite(sol_bdf.y))
    sol_rk4 = nm.RungeKutta4().solve(stiff, (0, 1), [0.0], h=0.05)
    assert np.all(np.isfinite(sol_rk4.y)) and np.max(np.abs(sol_rk4.y)) > 1e10


def test_bdf_matches_radau_on_robertson_stiff_system():
    def robertson(t, y):
        y1, y2, y3 = y
        return np.array([-0.04 * y1 + 1e4 * y2 * y3,
                         0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2 ** 2,
                         3e7 * y2 ** 2])

    sol = nm.BDF(order=3, newton_tol=1e-10, max_newton=30).solve(
        robertson, (0, 1), [1.0, 0.0, 0.0], h=1e-4)
    r = sci_integrate.solve_ivp(robertson, (0, 1), [1.0, 0.0, 0.0], method="Radau",
                                rtol=1e-8, atol=1e-10)
    rel = np.abs(sol.y[-1] - r.y[:, -1]) / (np.abs(r.y[:, -1]) + 1e-10)
    assert np.max(rel) < 1e-3


def test_euler_maruyama_sde_strong_order_half_and_milstein_order_one():
    mu, sigma, x0, T = 0.05, 0.3, 100.0, 1.0

    def drift(t, x):
        return mu * x

    def diffusion(t, x):
        return sigma * x

    def diffusion_dx(t, x):
        return sigma * np.ones_like(x)

    n_paths = 20000
    ns_list = [8, 16, 32, 64, 128, 256]
    errs_em, errs_mil = [], []
    for n_steps in ns_list:
        rng = np.random.default_rng(0)
        dt = T / n_steps
        dW = rng.standard_normal((n_paths, n_steps, 1)) * np.sqrt(dt)
        em = nm.Euler_Maruyama_SDE(drift, diffusion)
        sol_em = em.solve((0, T), [x0], n_steps=n_steps, n_paths=n_paths, brownian=dW)
        WT = dW[:, :, 0].sum(axis=1)
        exact = x0 * np.exp((mu - 0.5 * sigma ** 2) * T + sigma * WT)
        errs_em.append(np.sqrt(np.mean((sol_em.terminal[:, 0] - exact) ** 2)))
        mil = nm.Milstein_SDE(drift, diffusion, diffusion_dx=diffusion_dx)
        sol_mil = mil.solve((0, T), [x0], n_steps=n_steps, n_paths=n_paths, brownian=dW)
        errs_mil.append(np.sqrt(np.mean((sol_mil.terminal[:, 0] - exact) ** 2)))
    logh = np.log(1.0 / np.array(ns_list))
    slope_em = np.polyfit(logh, np.log(errs_em), 1)[0]
    slope_mil = np.polyfit(logh, np.log(errs_mil), 1)[0]
    assert abs(slope_em - 0.5) < 0.15
    assert abs(slope_mil - 1.0) < 0.2


def test_ou_process_mean_and_variance():
    theta, mu, sigma, x0, T = 1.5, 2.0, 0.5, 0.0, 2.0
    em = nm.Euler_Maruyama_SDE(lambda t, x: theta * (mu - x), lambda t, x: sigma * np.ones_like(x))
    sol = em.solve((0, T), [x0], n_steps=500, n_paths=20000, random_state=1)
    terminal = sol.terminal[:, 0]
    exact_mean = mu + (x0 - mu) * np.exp(-theta * T)
    exact_var = sigma ** 2 / (2 * theta) * (1 - np.exp(-2 * theta * T))
    se = np.sqrt(exact_var / 20000)
    assert abs(terminal.mean() - exact_mean) < 3 * se


def test_general_noise_terminal_covariance():
    G = np.array([[1.0, 0.3], [0.0, 0.8]])

    def drift2(t, x):
        return np.zeros_like(x)

    def diffusion2(t, x):
        return np.tile(G, (x.shape[0], 1, 1))

    em2 = nm.Euler_Maruyama_SDE(drift2, diffusion2, noise="general", n_brownian=2)
    sol2 = em2.solve((0, 1.0), [0.0, 0.0], n_steps=200, n_paths=20000, random_state=2)
    cov_emp = np.cov(sol2.terminal.T)
    cov_exact = G @ G.T
    assert np.max(np.abs(cov_emp - cov_exact)) < 0.05


def test_euler_maruyama_sde_matches_legacy_levy_processes_solver():
    legacy_sde = _LegacySDE(drift=lambda t, x: 0.05 * x, diffusion=lambda t, x: 0.2 * x, x0=100.0)
    legacy_paths = _legacy_em(legacy_sde, T=1.0, n_steps=100, n_paths=1, random_state=42)
    rng = np.random.default_rng(42)
    dW = _legacy_binc(100, 1, 1.0 / 100, rng)[:, :, None]
    sol = nm.Euler_Maruyama_SDE(lambda t, x: 0.05 * x, lambda t, x: 0.2 * x).solve(
        (0, 1.0), [100.0], n_steps=100, n_paths=1, brownian=dW)
    assert np.allclose(legacy_paths[0], sol.paths[0, :, 0], atol=1e-10)


# ============================================================== linear_algebra

def test_matrix_exponential_matches_scipy_pade_eig_and_scaling():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    assert np.max(np.abs(nm.MatrixExponential(A).compute("pade").expm_ - sp_expm(A))) < 1e-10

    Asym = A + A.T
    assert np.max(np.abs(nm.MatrixExponential(Asym).compute("eig").expm_ - sp_expm(Asym))) < 1e-10

    nilpotent = np.triu(rng.standard_normal((5, 5)), 1)
    assert np.max(np.abs(nm.MatrixExponential(nilpotent).compute("pade").expm_
                        - sp_expm(nilpotent))) < 1e-9

    Abig = rng.standard_normal((5, 5)) * 30
    S = sp_expm(Abig)
    R = nm.MatrixExponential(Abig).compute("pade").expm_
    rel = np.abs(R - S) / (np.abs(S) + 1.0)
    assert np.max(rel) < 1e-9

    inv = nm.MatrixExponential(-A).compute().expm_
    fwd = nm.MatrixExponential(A).compute().expm_
    assert np.max(np.abs(fwd @ inv - np.eye(6))) < 1e-8


def test_matrix_exponential_ctmc_generator_gives_valid_transition_matrix():
    Q = np.array([[-2.0, 1.0, 1.0], [1.0, -3.0, 2.0], [0.5, 0.5, -1.0]])
    P = nm.MatrixExponential(Q).at(1.0)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(P >= -1e-8)


def test_matrix_logarithm_matches_scipy_and_inverts_expm():
    rng = np.random.default_rng(1)
    A = np.abs(rng.standard_normal((5, 5))) * 0.1 + np.eye(5) * 5
    lg = nm.MatrixLogarithm(A).compute("inverse_scaling")
    assert np.max(np.abs(lg.logm_ - sp_logm(A))) < 1e-8

    B = rng.standard_normal((5, 5)) * 0.1
    expB = nm.MatrixExponential(B).compute().expm_
    logexpB = nm.MatrixLogarithm(expB).compute().logm_
    assert np.max(np.abs(logexpB - B)) < 1e-8


def test_matrix_logarithm_raises_on_negative_real_eigenvalue():
    with pytest.raises(ValueError):
        nm.MatrixLogarithm(np.array([[-1.0, 0.0], [0.0, -2.0]])).compute()


def test_cholesky_matches_numpy_and_jitter_path():
    rng = np.random.default_rng(2)
    M = rng.standard_normal((6, 6))
    A = M @ M.T + 6 * np.eye(6)
    ch = nm.CholeskyDecomp(A)
    assert np.max(np.abs(ch.L_ - np.linalg.cholesky(A))) < 1e-10
    b = rng.standard_normal(6)
    assert np.max(np.abs(ch.solve(b) - np.linalg.solve(A, b))) < 1e-10
    assert abs(ch.logdet() - np.linalg.slogdet(A)[1]) < 1e-8
    assert np.max(np.abs(ch.inverse() - np.linalg.inv(A))) < 1e-8

    singular = np.outer(rng.standard_normal(4), np.ones(4))
    singular = singular @ singular.T
    ch2 = nm.CholeskyDecomp(singular, jitter=True)
    assert ch2.jitter_used_ > 0
    assert np.allclose(ch2.reconstruct(), singular, atol=1e-6)

    with pytest.raises(ValueError):
        nm.CholeskyDecomp(rng.standard_normal((4, 4)))


def test_eigendecomp_jacobi_matches_eigh_and_residual_is_small():
    rng = np.random.default_rng(3)
    n = 30
    M = rng.standard_normal((n, n))
    A = M @ M.T + n * np.eye(n)
    ed = nm.EigenDecomp(A).compute("jacobi")
    assert np.max(np.abs(np.sort(ed.eigenvalues_) - np.sort(np.linalg.eigvalsh(A)))) < 1e-8
    resid = A @ ed.eigenvectors_ - ed.eigenvectors_ * ed.eigenvalues_
    assert np.max(np.abs(resid)) < 1e-8


def test_eigendecomp_qr_on_companion_matrices():
    C = np.array([[0, 0, 6], [1, 0, -11], [0, 1, 6]], dtype=float)
    ed = nm.EigenDecomp(C, symmetric=False).compute("qr")
    assert np.allclose(sorted(np.real(ed.eigenvalues_)), [1.0, 2.0, 3.0], atol=1e-6)

    C2 = np.array([[0, -1], [1, 0]], dtype=float)
    ed2 = nm.EigenDecomp(C2, symmetric=False).compute("qr")
    assert np.allclose(sorted(ed2.eigenvalues_.imag), [-1.0, 1.0], atol=1e-8)


def test_svd_one_sided_jacobi_matches_numpy():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((8, 5))
    sv = nm.SVD(A).compute("jacobi")
    _, s, _ = np.linalg.svd(A, full_matrices=False)
    assert np.max(np.abs(np.sort(sv.s_)[::-1] - np.sort(s)[::-1])) < 1e-8
    assert np.max(np.abs(sv.reconstruct() - A)) < 1e-8
    assert np.max(np.abs(sv.pinv() - np.linalg.pinv(A))) < 1e-8
    k = 3
    assert abs(np.linalg.norm(A - sv.low_rank(k), 2) - s[k]) < 1e-6


def test_qr_decomp_three_methods_and_least_squares():
    rng = np.random.default_rng(5)
    A = rng.standard_normal((7, 4))
    for method in ("householder", "gram_schmidt", "givens"):
        qr = nm.QRDecomp(A).compute(method)
        assert np.max(np.abs(qr.Q_.T @ qr.Q_ - np.eye(qr.Q_.shape[1]))) < 1e-8
        assert np.max(np.abs(qr.reconstruct() - A)) < 1e-8
    b = rng.standard_normal(7)
    lsq = nm.QRDecomp(A).compute("householder").solve_least_squares(b)
    lsq_np = np.linalg.lstsq(A, b, rcond=None)[0]
    assert np.max(np.abs(lsq - lsq_np)) < 1e-8


def test_real_schur_matches_numpy_eigvals_and_structure():
    rng = np.random.default_rng(6)
    for _ in range(5):
        n = rng.integers(3, 12)
        A = rng.standard_normal((n, n))
        s = nm.Schur(A).compute("real")
        assert np.max(np.abs(s.Z_.T @ s.Z_ - np.eye(n))) < 1e-7
        assert np.max(np.abs(s.reconstruct() - A)) < 1e-5
        assert np.max(np.abs(np.tril(s.T_, -2))) < 1e-6
        eig1 = np.sort_complex(s.eigenvalues())
        eig2 = np.sort_complex(np.linalg.eigvals(A))
        assert np.max(np.abs(eig1 - eig2)) < 1e-5


def test_real_schur_matches_scipy_schur_eigenvalues():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((10, 10))
    s = nm.Schur(A).compute("real")
    T2, Z2 = sp_schur(A, output="real")
    assert np.max(np.abs(Z2 @ T2 @ Z2.T - A)) < 1e-8
    eig1 = np.sort_complex(s.eigenvalues())
    eig2 = np.sort_complex(np.linalg.eigvals(A))
    assert np.max(np.abs(eig1 - eig2)) < 1e-6


def test_complex_schur_is_strictly_triangular_and_reconstructs():
    rng = np.random.default_rng(8)
    for _ in range(5):
        n = rng.integers(3, 12)
        A = rng.standard_normal((n, n))
        s = nm.Schur(A).compute("complex")
        assert np.max(np.abs(np.tril(s.T_, -1))) < 1e-6
        assert np.max(np.abs(s.reconstruct() - A)) < 1e-5


# ================================================================== root_solve

_COS_ROOT = 0.7390851332151607


@pytest.mark.parametrize("call", [
    lambda f: nm.Bisection(f, 0, 1),
    lambda f: nm.Brent(f, 0, 1),
    lambda f: nm.Secant(f, 0.5, 1.0),
    lambda f: nm.NewtonRaphson(f, 0.5, fprime=lambda x: -np.sin(x) - 1),
])
def test_scalar_root_solvers_find_cos_x_equals_x(call):
    f = lambda x: np.cos(x) - x
    r = call(f)
    assert r.converged
    assert abs(float(r) - _COS_ROOT) < 1e-9


def test_brent_matches_scipy_brentq_iteration_count():
    f = lambda x: np.cos(x) - x
    r = nm.Brent(f, 0, 1)
    _, info = sci_optimize.brentq(f, 0, 1, full_output=True)
    assert r.iterations <= info.iterations + 5


def test_wallis_cubic_root():
    f = lambda x: x ** 3 - 2 * x - 5
    r = nm.Brent(f, 2, 3)
    assert abs(r.root - 2.0945514815423265) < 1e-10


def test_newton_raphson_vector_matches_fsolve():
    def F(v):
        x, y = v
        return np.array([x ** 2 + y ** 2 - 1, y - x ** 2])

    r = nm.NewtonRaphson(F, np.array([0.8, 0.6]))
    sol = sci_optimize.fsolve(F, [0.8, 0.6])
    assert np.max(np.abs(np.asarray(r.root) - sol)) < 1e-8


def test_fixed_point_acceleration_reduces_iterations():
    r_plain = nm.FixedPoint(np.cos, 0.5)
    r_aitken = nm.FixedPoint(np.cos, 0.5, acceleration="aitken")
    r_steff = nm.FixedPoint(np.cos, 0.5, acceleration="steffensen")
    for r in (r_plain, r_aitken, r_steff):
        assert r.converged and abs(r.root - _COS_ROOT) < 1e-8
    assert r_aitken.iterations < r_plain.iterations
    assert r_steff.iterations < r_plain.iterations


def test_root_finding_all_roots_finds_multiples_of_pi():
    roots = nm.RootFinding.all_roots(np.sin, 0.5, 10)
    vals = sorted(float(r) for r in roots)
    assert np.allclose(vals, [np.pi, 2 * np.pi, 3 * np.pi], atol=1e-6)


def test_root_finding_find_bracket_locates_sign_change():
    f = lambda x: np.cos(x) - x
    a, b = nm.RootFinding.find_bracket(f, 0.5)
    assert f(a) * f(b) < 0


def test_brent_implied_volatility_recovers_black_scholes_sigma():
    S, K, T, r, true_sigma = 100.0, 100.0, 1.0, 0.05, 0.25
    target = BlackScholes(S=S, K=K, T=T, r=r, sigma=true_sigma).call_price()

    def diff(sigma):
        return BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma).call_price() - target

    rr = nm.Brent(diff, 0.01, 2.0)
    assert abs(rr.root - true_sigma) < 1e-8


def test_non_bracketing_interval_raises():
    with pytest.raises(ValueError):
        nm.Brent(lambda x: np.cos(x) - x, 0.5, 0.6)


# ================================================================ interpolation

def test_spline_natural_clamped_notaknot_match_scipy():
    rng = np.random.default_rng(9)
    x = np.sort(rng.uniform(0, 10, 12))
    y = np.sin(x)
    xq = np.linspace(x[0], x[-1], 200)

    sp_nat = nm.SplineInterpolation(x, y, bc="natural")
    cs_nat = CubicSpline(x, y, bc_type="natural")
    assert np.max(np.abs(sp_nat(xq) - cs_nat(xq))) < 1e-9
    assert np.max(np.abs(sp_nat.derivative(xq) - cs_nat(xq, 1))) < 1e-9
    assert abs(sp_nat.integral(x[0], x[-1]) - cs_nat.integrate(x[0], x[-1])) < 1e-9

    sp_cl = nm.SplineInterpolation(x, y, bc="clamped", bc_values=(0.3, -0.2))
    cs_cl = CubicSpline(x, y, bc_type=((1, 0.3), (1, -0.2)))
    assert np.max(np.abs(sp_cl(xq) - cs_cl(xq))) < 1e-9

    sp_nak = nm.SplineInterpolation(x, y, bc="not-a-knot")
    cs_nak = CubicSpline(x, y, bc_type="not-a-knot")
    assert np.max(np.abs(sp_nak(xq) - cs_nak(xq))) < 1e-9


def test_spline_reproduces_cubic_exactly_under_not_a_knot():
    xc = np.linspace(0, 5, 7)
    yc = 2 * xc ** 3 - xc ** 2 + 3 * xc - 1
    sp = nm.SplineInterpolation(xc, yc, bc="not-a-knot")
    xq = np.linspace(0, 5, 50)
    exact = 2 * xq ** 3 - xq ** 2 + 3 * xq - 1
    assert np.max(np.abs(sp(xq) - exact)) < 1e-10


def test_spline_converges_at_fourth_order():
    def f(x):
        return np.sin(x)

    errs = []
    for n in (10, 20, 40):
        x = np.linspace(0, 2 * np.pi, n)
        sp = nm.SplineInterpolation(x, f(x), bc="not-a-knot")
        xq = np.linspace(0, 2 * np.pi, 300)
        errs.append(np.max(np.abs(sp(xq) - f(xq))))
    assert errs[0] / errs[1] > 8 and errs[1] / errs[2] > 8


def test_cubic_hermite_pchip_matches_scipy():
    rng = np.random.default_rng(10)
    x = np.sort(rng.uniform(0, 10, 12))
    y = np.sin(x)
    ch = nm.CubicHermite(x, y)
    pc = PchipInterpolator(x, y)
    assert np.max(np.abs(ch.slopes_ - pc.derivative()(x))) < 1e-10
    xq = np.linspace(x[0], x[-1], 200)
    assert np.max(np.abs(ch(xq) - pc(xq))) < 1e-9
    assert abs(ch.integral(x[0], x[-1]) - pc.integrate(x[0], x[-1])) < 1e-8


def test_cubic_hermite_given_slopes_matches_scipy_hermite_spline():
    from scipy.interpolate import CubicHermiteSpline
    rng = np.random.default_rng(11)
    x = np.sort(rng.uniform(0, 5, 8))
    y = rng.standard_normal(8)
    dydx = rng.standard_normal(8)
    ch = nm.CubicHermite(x, y, dydx=dydx)
    ref = CubicHermiteSpline(x, y, dydx)
    xq = np.linspace(x[0], x[-1], 100)
    assert np.max(np.abs(ch(xq) - ref(xq))) < 1e-9


def test_pchip_preserves_monotonicity():
    xm = np.arange(6.0)
    ym = np.array([1, 2, 2, 3, 5, 8.0])
    ch = nm.CubicHermite(xm, ym)
    xq = np.linspace(0, 5, 500)
    assert np.all(np.diff(ch(xq)) >= -1e-9)


def test_barycentric_lagrange_matches_scipy():
    rng = np.random.default_rng(12)
    x = np.sort(rng.uniform(0, 10, 12))
    y = np.sin(x)
    bl = nm.BarycentricLagrange(x, y)
    bi = BarycentricInterpolator(x, y)
    xq = np.linspace(x[0], x[-1], 200)
    assert np.max(np.abs(bl(xq) - bi(xq))) < 1e-8


def test_runge_phenomenon_chebyshev_vs_equispaced():
    runge = lambda t: 1 / (1 + 25 * t ** 2)
    blr = nm.BarycentricLagrange.from_function(runge, 40, -1, 1)
    xeq = np.linspace(-1, 1, 40)
    bleq = nm.BarycentricLagrange(xeq, runge(xeq))
    xtest = np.linspace(-0.99, 0.99, 300)
    err_cheb = np.max(np.abs(blr(xtest) - runge(xtest)))
    err_eq = np.max(np.abs(bleq(xtest) - runge(xtest)))
    assert err_cheb < 1e-2
    assert err_eq > 100


def test_chebyshev_series_exp_and_derivative_and_roots():
    cb = nm.Chebyshev.from_function(np.exp, 20, -1, 1)
    xt = np.linspace(-1, 1, 100)
    assert np.max(np.abs(cb(xt) - np.exp(xt))) < 1e-12
    cbd = cb.derivative()
    assert np.max(np.abs(cbd(xt) - np.exp(xt))) < 1e-10

    cbsin = nm.Chebyshev.from_function(np.sin, 20, -1, 1)
    assert np.max(np.abs(cbsin.derivative()(xt) - np.cos(xt))) < 1e-10

    r = nm.Chebyshev.from_function(lambda x: x ** 3 - x, 5, -1, 1).roots()
    assert np.allclose(sorted(r), [-1.0, 0.0, 1.0], atol=1e-8)

    cb2 = nm.Chebyshev.from_function(lambda x: x ** 2, 6, -1, 1)
    assert abs(cb2.definite_integral() - 2 / 3) < 1e-10

    cb3 = nm.Chebyshev(np.array([1.0, 0.5, 1e-16, 1e-17]))
    assert cb3.truncate().degree == 1


def test_chebyshev_interpolate_matches_numpy_polynomial():
    # numpy's Chebyshev.interpolate uses Chebyshev-Gauss (no-endpoint) nodes while this
    # class uses Chebyshev-Gauss-Lobatto (endpoint-including) nodes -- different node
    # sets agree only once both have resolved the function (n large enough); at low n
    # they can legitimately disagree by more than either's own error against f.
    n = 16
    f = lambda x: np.exp(x) * np.cos(2 * x)
    cb = nm.Chebyshev.from_function(f, n, -1, 1)
    ref = np_chebmod.Chebyshev.interpolate(f, n, domain=[-1, 1])
    xt = np.linspace(-1, 1, 50)
    assert np.max(np.abs(cb(xt) - ref(xt))) < 1e-10
    assert np.max(np.abs(cb(xt) - f(xt))) < 1e-10


def test_nurbs_with_unit_weights_matches_bspline():
    ctrl = np.array([[0, 0], [1, 2], [3, 3], [4, 0], [5, 1]], dtype=float)
    degree = 3
    nurbs = nm.NURBS(ctrl, degree)
    bs = BSpline(nurbs.knots, ctrl, degree)
    u = np.linspace(*nurbs.domain, 50)
    assert np.max(np.abs(nurbs.evaluate(u) - bs(u))) < 1e-9


def test_nurbs_circle_exact_radius():
    circ = nm.NURBS.circle(center=(1.0, 1.0), radius=2.0)
    u = np.linspace(0, 1, 200)
    pts = circ.evaluate(u)
    dist = np.linalg.norm(pts - np.array([1.0, 1.0]), axis=1)
    assert np.max(np.abs(dist - 2.0)) < 1e-10


def test_nurbs_interpolate_passes_through_points():
    pts_target = np.array([[0, 0], [1, 2], [2, 1], [4, 3], [5, 0]], dtype=float)
    ncurve = nm.NURBS.interpolate(pts_target, degree=3)
    d = np.sum(np.linalg.norm(np.diff(pts_target, axis=0), axis=1))
    ub = np.zeros(5)
    cum = 0.0
    for i in range(1, 5):
        cum += np.linalg.norm(pts_target[i] - pts_target[i - 1])
        ub[i] = cum / d
    evals = ncurve.evaluate(ub)
    assert np.max(np.abs(evals - pts_target)) < 1e-8


def test_nurbs_derivative_matches_finite_difference():
    ctrl = np.array([[0, 0], [1, 2], [3, 3], [4, 0], [5, 1]], dtype=float)
    nurbs = nm.NURBS(ctrl, degree=3)
    ua = np.array([0.3])
    eps = 1e-6
    fd = (nurbs.evaluate(ua + eps) - nurbs.evaluate(ua - eps)) / (2 * eps)
    assert np.max(np.abs(nurbs.derivative(ua) - fd)) < 1e-4


def test_interpolation_facade_dispatches_every_method():
    xi = np.linspace(0, 10, 15)
    yi = np.sin(xi)
    for m in ("linear", "nearest", "polynomial", "cubic", "pchip", "spline_natural"):
        interp = nm.Interpolation(xi, yi, method=m)
        v = interp(np.array([2.5, 7.3]))
        assert np.all(np.isfinite(v))
    with pytest.raises(ValueError):
        nm.Interpolation(xi, yi, method="bogus")


# ======================================================================= pde

def test_fornberg_stencil_matches_standard_coefficients():
    fd = nm.FiniteDifference()
    w2 = fd.stencil(2, [-1, 0, 1])
    assert np.allclose(w2, [1.0, -2.0, 1.0])
    w4 = fd.stencil(2, [-2, -1, 0, 1, 2])
    assert np.allclose(w4, [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])


def test_finite_difference_derivative_accuracy_scales_with_h():
    fd = nm.FiniteDifference()
    errs = [abs(fd.derivative(np.sin, 1.0, order=1, h=h, accuracy=2) - np.cos(1.0))
           for h in (0.1, 0.05, 0.025)]
    assert errs[0] / errs[1] > 3.5 and errs[1] / errs[2] > 3.5


def test_poisson_1d_is_second_order():
    fd = nm.FiniteDifference()
    f = lambda x: np.pi ** 2 * np.sin(np.pi * x)
    errs = []
    for n in (10, 20, 40, 80):
        x, u = fd.poisson_1d(f, 0, 1, n)
        errs.append(np.max(np.abs(u - np.sin(np.pi * x))))
    ratios = [errs[i] / errs[i + 1] for i in range(3)]
    assert all(abs(r - 4.0) < 0.2 for r in ratios)


def test_poisson_2d_cg_matches_dense_and_is_second_order():
    fd = nm.FiniteDifference()
    f2 = lambda x, y: 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    X1, Y1, U1 = fd.poisson_2d(f2, [(0, 1), (0, 1)], 15, 15, solver="cg", tol=1e-12)
    X2, Y2, U2 = fd.poisson_2d(f2, [(0, 1), (0, 1)], 15, 15, solver="dense")
    assert np.max(np.abs(U1 - U2)) < 1e-8
    errs = []
    for n in (10, 20, 40):
        X, Y, U = fd.poisson_2d(f2, [(0, 1), (0, 1)], n, n, tol=1e-12)
        errs.append(np.max(np.abs(U - np.sin(np.pi * X) * np.sin(np.pi * Y))))
    assert errs[0] / errs[1] > 3.5 and errs[1] / errs[2] > 3.5


def test_heat_1d_crank_nicolson_matches_decaying_mode():
    fd = nm.FiniteDifference()
    alpha = 0.5
    u0 = lambda x: np.sin(np.pi * x)
    x, t, U = fd.heat_1d(u0, alpha, (0, 1), (0, 0.5), 50, 200, scheme="crank_nicolson")
    exact_final = np.exp(-alpha * np.pi ** 2 * 0.5) * np.sin(np.pi * x)
    assert np.max(np.abs(U[-1] - exact_final)) < 1e-3


def test_heat_1d_explicit_raises_on_cfl_violation():
    fd = nm.FiniteDifference()
    with pytest.raises(ValueError):
        fd.heat_1d(lambda x: np.sin(np.pi * x), 0.5, (0, 1), (0, 0.5), 50, 50, scheme="explicit")


def test_black_scholes_pde_matches_closed_form_and_american_put_dominates():
    K, T, r, sigma, S0 = 100.0, 1.0, 0.05, 0.2, 100.0
    bs = BlackScholes(S=S0, K=K, T=T, r=r, sigma=sigma)
    fd = nm.FiniteDifference()
    pde_call = fd.black_scholes(K, T, r, sigma, kind="call").price(S0)
    assert abs(pde_call - bs.call_price()) < 1e-2

    fd2 = nm.FiniteDifference()
    pde_put = fd2.black_scholes(K, T, r, sigma, kind="put").price(S0)
    assert abs(pde_put - bs.put_price()) < 1e-2

    fd3 = nm.FiniteDifference()
    amer_put = fd3.black_scholes(K, T, r, sigma, kind="put", american=True,
                                 n_S=400, n_t=400).price(S0)
    assert amer_put >= pde_put
    bt = BinomialTree(S0, K, T, r, sigma, n_steps=2000)
    assert abs(amer_put - bt.put_price(american=True)) < 5e-2


def test_advection_upwind_conserves_mass_and_lax_wendroff_more_accurate():
    fd = nm.FiniteDifference()
    c = 1.0
    u0 = lambda x: np.sin(2 * np.pi * x)
    x, t, U_up = fd.advection_1d(u0, c, (0, 1), (0, 1.0), 200, 400, scheme="upwind")
    exact_final = np.sin(2 * np.pi * (x - c * 1.0))
    assert abs(U_up[0].sum() - U_up[-1].sum()) < 1e-8
    err_up = np.max(np.abs(U_up[-1] - exact_final))
    _, _, U_lw = fd.advection_1d(u0, c, (0, 1), (0, 1.0), 200, 400, scheme="lax_wendroff")
    err_lw = np.max(np.abs(U_lw[-1] - exact_final))
    assert err_lw < err_up


def test_fem_p1_and_p2_convergence_rates():
    f = lambda x: np.pi ** 2 * np.sin(np.pi * x)
    errs1 = []
    for n in (8, 16, 32, 64):
        mesh = nm.Mesh.interval(0, 1, n)
        fe = nm.FiniteElement(mesh, degree=1)
        fe.assemble(p=1.0, f=f, dirichlet={"left": 0.0, "right": 0.0})
        fe.solve()
        errs1.append(fe.l2_error(lambda x: np.sin(np.pi * x)))
    rates1 = [np.log(errs1[i] / errs1[i + 1]) / np.log(2) for i in range(3)]
    assert all(abs(r - 2.0) < 0.15 for r in rates1)

    errs2 = []
    for n in (8, 16, 32):
        mesh = nm.Mesh.interval(0, 1, n)
        fe = nm.FiniteElement(mesh, degree=2)
        fe.assemble(p=1.0, f=f, dirichlet={"left": 0.0, "right": 0.0})
        fe.solve()
        errs2.append(fe.l2_error(lambda x: np.sin(np.pi * x)))
    rates2 = [np.log(errs2[i] / errs2[i + 1]) / np.log(2) for i in range(2)]
    assert all(r >= 2.8 for r in rates2)


def test_fem_neumann_bc_exact_at_nodes():
    mesh = nm.Mesh.interval(0, 1, 20)
    fe = nm.FiniteElement(mesh, degree=1)
    fe.assemble(p=1.0, f=1.0, dirichlet={"left": 0.0}, neumann={"right": 0.0})
    fe.solve()
    x = mesh.nodes
    exact = x - x ** 2 / 2
    assert np.max(np.abs(fe.u_ - exact)) < 1e-10


def test_fem_2d_second_order_and_matrix_properties():
    f2 = lambda pt: 2 * np.pi ** 2 * np.sin(np.pi * pt[0]) * np.sin(np.pi * pt[1])
    errs = []
    for n in (8, 16, 32):
        mesh = nm.Mesh.rectangle(0, 1, 0, 1, n, n)
        fe = nm.FiniteElement(mesh, degree=1)
        fe.assemble(f=f2, dirichlet=0.0)
        fe.solve()
        errs.append(fe.l2_error(lambda pt: np.sin(np.pi * pt[0]) * np.sin(np.pi * pt[1])))
    rates = [np.log(errs[i] / errs[i + 1]) / np.log(2) for i in range(2)]
    assert all(abs(r - 2.0) < 0.2 for r in rates)

    mesh = nm.Mesh.rectangle(0, 1, 0, 1, 10, 10)
    fe = nm.FiniteElement(mesh, degree=1)
    fe.assemble(f=0.0, dirichlet=0.0)
    interior = [i for i in range(mesh.n_nodes) if i not in mesh.boundary_nodes]
    assert np.max(np.abs(fe.stiffness_[interior].sum(axis=1))) < 1e-10
    assert abs(fe.mass_.sum() - 1.0) < 1e-10


def test_fenics_interface_matches_native_finite_element():
    mesh = nm.Mesh.interval(0, 1, 20)
    fen = nm.FEniCS_Interface(mesh, degree=1)
    fen.set_equation(source=lambda x: np.pi ** 2 * np.sin(np.pi * x))
    fen.dirichlet_bc(0.0)
    fen.solve()
    fe = nm.FiniteElement(mesh, degree=1)
    fe.assemble(p=1.0, f=lambda x: np.pi ** 2 * np.sin(np.pi * x),
               dirichlet={"left": 0.0, "right": 0.0})
    fe.solve()
    assert np.max(np.abs(fen.u_ - fe.u_)) < 1e-10

    mesh2 = nm.Mesh.rectangle(0, 1, 0, 1, 10, 10)
    src = lambda pt: 2 * np.pi ** 2 * np.sin(np.pi * pt[0]) * np.sin(np.pi * pt[1])
    fen2 = nm.FEniCS_Interface(mesh2)
    fen2.set_equation(source=src)
    fen2.dirichlet_bc(0.0)
    fen2.solve()
    fe2 = nm.FiniteElement(mesh2, degree=1)
    fe2.assemble(f=src, dirichlet=0.0)
    fe2.solve()
    assert np.max(np.abs(fen2.u_ - fe2.u_)) < 1e-10


def test_fenics_mesh_export_xml_round_trip():
    import os
    import tempfile

    mesh = nm.Mesh.interval(0, 1, 10)
    fen = nm.FEniCS_Interface(mesh)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mesh.xml")
        fen.export_mesh(path, format="dolfin_xml")
        m2 = nm.FEniCS_Interface.read_mesh(path)
        assert np.allclose(m2.nodes, mesh.nodes)
        assert np.array_equal(m2.cells, mesh.cells)
        assert list(m2.boundary_nodes) == list(mesh.boundary_nodes)

    mesh2 = nm.Mesh.rectangle(0, 1, 0, 1, 5, 5)
    fen2 = nm.FEniCS_Interface(mesh2)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mesh2.xml")
        fen2.export_mesh(path, format="dolfin_xml")
        m3 = nm.FEniCS_Interface.read_mesh(path)
        assert np.allclose(m3.nodes, mesh2.nodes)
        assert sorted(m3.boundary_nodes) == sorted(mesh2.boundary_nodes)

        xdmf_path = os.path.join(d, "mesh2.xdmf")
        fen2.export_mesh(xdmf_path, format="xdmf")
        content = open(xdmf_path, encoding="utf-8").read()
        assert f'NumberOfElements="{mesh2.n_cells}"' in content


def test_fenics_to_fenics_raises_import_error_when_absent():
    import builtins

    mesh = nm.Mesh.interval(0, 1, 5)
    fen = nm.FEniCS_Interface(mesh)
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name in ("dolfinx", "dolfin"):
            raise ImportError(f"no module {name}")
        return real_import(name, *a, **kw)

    builtins.__import__ = fake_import
    try:
        with pytest.raises(ImportError, match="FEniCS"):
            fen.to_fenics()
    finally:
        builtins.__import__ = real_import


def test_fenics_to_fenics_uses_stub_dolfinx_when_present():
    import sys
    import types

    calls = {}
    dolfinx = types.ModuleType("dolfinx")
    mesh_mod = types.ModuleType("dolfinx.mesh")

    def create_interval(comm, n, bounds):
        calls["interval"] = (n, bounds)
        return "SENTINEL"

    def create_rectangle(comm, corners, n):
        calls["rectangle"] = (corners, n)
        return "SENTINEL2"

    mesh_mod.create_interval = create_interval
    mesh_mod.create_rectangle = create_rectangle
    dolfinx.mesh = mesh_mod
    mpi4py = types.ModuleType("mpi4py")
    mpi_mod = types.ModuleType("mpi4py.MPI")
    mpi_mod.COMM_WORLD = "COMM"
    mpi4py.MPI = mpi_mod
    saved = {k: sys.modules.get(k) for k in ("dolfinx", "dolfinx.mesh", "mpi4py", "mpi4py.MPI")}
    sys.modules["dolfinx"] = dolfinx
    sys.modules["dolfinx.mesh"] = mesh_mod
    sys.modules["mpi4py"] = mpi4py
    sys.modules["mpi4py.MPI"] = mpi_mod
    try:
        mesh = nm.Mesh.interval(0, 1, 10)
        fen = nm.FEniCS_Interface(mesh)
        result = fen.to_fenics()
        assert result["backend"] == "dolfinx" and result["mesh"] == "SENTINEL"
        assert calls["interval"][0] == 10
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _square_boundary_points(n_per_side):
    pts = []
    for i in range(n_per_side):
        pts.append([i / n_per_side, 0.0])
    for i in range(n_per_side):
        pts.append([1.0, i / n_per_side])
    for i in range(n_per_side):
        pts.append([1.0 - i / n_per_side, 1.0])
    for i in range(n_per_side):
        pts.append([0.0, 1.0 - i / n_per_side])
    return np.array(pts)


def test_boundary_element_harmonic_functions_and_flux_balance():
    pts = _square_boundary_points(50)
    for u_exact, dudx, dudy in (
        (lambda p: p[0] ** 2 - p[1] ** 2, lambda p: 2 * p[0], lambda p: -2 * p[1]),
        (lambda p: np.exp(p[0]) * np.cos(p[1]),
         lambda p: np.exp(p[0]) * np.cos(p[1]), lambda p: -np.exp(p[0]) * np.sin(p[1])),
    ):
        bem = nm.BoundaryElement(pts)
        bem.solve(lambda mids: np.array([u_exact(m) for m in mids]))
        interior = np.array([[0.5, 0.5], [0.3, 0.7], [0.7, 0.2], [0.5, 0.1]])
        u_bem = bem.evaluate(interior)
        u_ex = np.array([u_exact(p) for p in interior])
        assert np.max(np.abs(u_bem - u_ex)) < 2e-3

        dudn_ex = np.array([dudx(m) * n[0] + dudy(m) * n[1]
                            for m, n in zip(bem.midpoints_, bem.normals_)])
        rel_err = np.linalg.norm(bem.flux_ - dudn_ex) / np.linalg.norm(dudn_ex)
        assert rel_err < 0.05
        # exact-Laplace-solution flux integrates to zero (Gauss); our numerically solved
        # flux only approximately does, at the discretization's own error level
        assert abs(np.sum(bem.flux_ * bem.lengths_)) < 1e-3


def test_spectral_fourier_derivative_and_periodic_poisson_heat():
    sm = nm.SpectralMethod()
    n = 64
    L = 2 * np.pi
    x = np.linspace(0, L, n, endpoint=False)
    du = sm.fourier_derivative(np.sin(3 * x), L, order=1)
    assert np.max(np.abs(du - 3 * np.cos(3 * x))) < 1e-10

    x2 = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    u_p = sm.poisson_periodic(np.sin(x2), 2 * np.pi)
    assert np.max(np.abs(u_p - np.sin(x2))) < 1e-10

    alpha, T = 0.3, 0.5
    u_heat = sm.heat_periodic(np.sin(2 * x2), alpha, 2 * np.pi, T)
    exact_heat = np.exp(-alpha * 4 * T) * np.sin(2 * x2)
    assert np.max(np.abs(u_heat - exact_heat)) < 1e-10


def test_spectral_chebyshev_bvp_matches_trefethen_p13():
    sm = nm.SpectralMethod()
    n = 16
    x, u = sm.solve_bvp(lambda x: np.exp(4 * x), -1, 1, n, bc=(0.0, 0.0))
    exact = (np.exp(4 * x) - np.sinh(4) * x - np.cosh(4)) / 16.0
    assert np.max(np.abs(u - exact)) < 1e-9


def test_spectral_beats_finite_difference_at_equal_resolution():
    n = 16
    sm = nm.SpectralMethod()
    _, u_spec = sm.solve_bvp(lambda x: np.exp(4 * x), -1, 1, n, bc=(0.0, 0.0))
    x_spec = sm.chebyshev_diff_matrix(n)[1]
    exact_spec = (np.exp(4 * x_spec) - np.sinh(4) * x_spec - np.cosh(4)) / 16.0
    err_spec = np.max(np.abs(u_spec - exact_spec))

    fd = nm.FiniteDifference()
    x_fd, u_fd = fd.poisson_1d(lambda x: -np.exp(4 * x), -1, 1, n)
    exact_fd = (np.exp(4 * x_fd) - np.sinh(4) * x_fd - np.cosh(4)) / 16.0
    err_fd = np.max(np.abs(u_fd - exact_fd))
    assert err_spec < err_fd / 1e4


# ======================================================================= misc

def test_thomas_solver_matches_numpy():
    lower = np.array([1.0, 1.0, 1.0, 1.0])
    diag = np.array([4.0, 4.0, 4.0, 4.0, 4.0])
    upper = np.array([1.0, 1.0, 1.0, 1.0])
    rhs = np.random.default_rng(0).standard_normal(5)
    A = np.diag(diag) + np.diag(lower, -1) + np.diag(upper, 1)
    assert np.allclose(_thomas(lower, diag, upper, rhs), np.linalg.solve(A, rhs))
