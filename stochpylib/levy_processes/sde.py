"""SDE definition and solvers: Euler-Maruyama, Milstein, a Runge-Kutta scheme,
the strong Taylor 1.5 expansion, weak approximation, and strong-error
convergence studies.

All solvers operate on :class:`SDE` definitions with callable drift
``a(t, x)`` and diffusion ``b(t, x)``; derivatives needed by the higher-order
schemes are obtained by central finite differences (``fd_eps``), keeping the
user API to two plain functions. Exact GBM and Ornstein-Uhlenbeck solutions
serve as the convergence-study oracles in the tests.
"""

import numpy as np

__all__ = [
    "SDE", "Euler_Maruyama", "Milstein", "Runge_Kutta_SDE",
    "StochasticTaylor", "WeakApproximation", "StrongApproximation",
]


class SDE:
    """Itô SDE definition: ``dX_t = a(t, X_t) dt + b(t, X_t) dW_t``.

    ``a`` and ``b`` are callables ``(t, x) -> array`` (vectorized over the
    leading path axis); ``x0`` is the initial value (scalar or array).
    """

    def __init__(self, drift, diffusion, x0=0.0):
        self.a = drift
        self.b = diffusion
        self.x0 = x0

    def _b_x(self, t, x, eps=1e-5):
        """Central-difference d sigma / d x for Milstein/Taylor schemes."""
        return (self.b(t, x + eps) - self.b(t, x - eps)) / (2.0 * eps)

    def _b_xx(self, t, x, eps=1e-4):
        return (self.b(t, x + eps) - 2.0 * self.b(t, x)
                + self.b(t, x - eps)) / eps ** 2

    def _a_x(self, t, x, eps=1e-5):
        return (self.a(t, x + eps) - self.a(t, x - eps)) / (2.0 * eps)


def _brownian_increments(n_steps, n_paths, dt, rng):
    return rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)


def _assemble(x0, incs):
    paths = np.empty((incs.shape[0], incs.shape[1] + 1))
    paths[:, 0] = x0
    np.cumsum(incs, axis=1, out=paths[:, 1:])
    return paths


def Euler_Maruyama(sde, T=1.0, n_steps=100, n_paths=1, random_state=None):
    """Strong order 0.5 Euler-Maruyama; returns ``(n_paths, n_steps + 1)``."""
    rng = np.random.default_rng(random_state)
    dt = T / n_steps
    dW = _brownian_increments(n_steps, n_paths, dt, rng)
    x = np.full(n_paths, sde.x0, dtype=float)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x
    t = 0.0
    for i in range(n_steps):
        x = x + sde.a(t, x) * dt + sde.b(t, x) * dW[:, i]
        paths[:, i + 1] = x
        t += dt
    return paths


def Milstein(sde, T=1.0, n_steps=100, n_paths=1, random_state=None):
    """Strong order 1.0 Milstein scheme (numeric derivative of sigma)."""
    rng = np.random.default_rng(random_state)
    dt = T / n_steps
    dW = _brownian_increments(n_steps, n_paths, dt, rng)
    x = np.full(n_paths, sde.x0, dtype=float)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x
    t = 0.0
    for i in range(n_steps):
        bx = sde.b(t, x)
        dsigma = sde._b_x(t, x)
        x = x + sde.a(t, x) * dt + bx * dW[:, i] \
            + 0.5 * bx * dsigma * (dW[:, i] ** 2 - dt)
        paths[:, i + 1] = x
        t += dt
    return paths


def Runge_Kutta_SDE(sde, T=1.0, n_steps=100, n_paths=1, random_state=None):
    """Strong order 1.0 derivative-free Milstein (Platen) scheme.

    Uses the supporting value ``H = x + a*dt + b*sqrt(dt)`` to approximate the
    Milstein correction ``0.5 * b * b' * (dW^2 - dt)`` by a finite difference
    of ``b`` alone, ``(b(H) - b(x)) / (2*sqrt(dt)) * (dW^2 - dt)`` — this
    removes the EM evaluation-point bias and reaches strong order 1.0 without
    an explicit derivative of ``b`` (Kloeden-Platen 1992, eq. 11.1.4). A plain
    trapezoidal (stochastic-Heun) average of endpoint drift/diffusion, as
    used previously here, lacks this Ito correction term and only achieves
    strong order 0.5 — no better than Euler-Maruyama (development/Probleme.md).
    """
    rng = np.random.default_rng(random_state)
    dt = T / n_steps
    sqdt = np.sqrt(dt)
    dW = _brownian_increments(n_steps, n_paths, dt, rng)
    x = np.full(n_paths, sde.x0, dtype=float)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x
    t = 0.0
    for i in range(n_steps):
        a0 = sde.a(t, x)
        b0 = sde.b(t, x)
        H = x + a0 * dt + b0 * sqdt
        b_h = sde.b(t, H)
        dw = dW[:, i]
        x = x + a0 * dt + b0 * dw \
            + (0.5 / sqdt) * (b_h - b0) * (dw ** 2 - dt)
        paths[:, i + 1] = x
        t += dt
    return paths


def StochasticTaylor(sde, T=1.0, n_steps=100, n_paths=1, random_state=None):
    """Strong order 1.5 stochastic Taylor scheme for scalar SDEs.

    Implements the Kloeden-Platen strong Taylor expansion (1992, eq. 10.4.3):
    with ``dZ`` the double Ito integral ``I_(1,0) = int_0^dt int_0^s dW(u) ds``
    (jointly Gaussian with ``dW``: mean 0, ``Var(dZ) = dt^3/3``,
    ``Cov(dW, dZ) = dt^2/2``, generated as
    ``0.5*dt*dW + 0.5/sqrt(3)*dt^1.5*z`` with an independent standard normal
    ``z``) and ``I_(1,1,1) = (dW^3 - 3*dt*dW)/6`` the triple same-integrand
    integral (a deterministic function of ``dW``/``dt`` alone — no extra
    randomness), the update is
    ``a dt + b dW + 0.5 b b' (dW^2-dt) + a' b dZ
    + 0.5(a a' + 0.5 b^2 a'') dt^2 + (a b' + 0.5 b^2 b'')(dW dt - dZ)
    + 0.5 b(b b'' + b'^2)((1/3) dW^2 - dt) dW``.
    All drift/diffusion derivatives are central finite differences, keeping
    the user API to two plain callables.
    """
    rng = np.random.default_rng(random_state)
    dt = T / n_steps
    dW = _brownian_increments(n_steps, n_paths, dt, rng)
    eps = 1e-5
    x = np.full(n_paths, sde.x0, dtype=float)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x
    t = 0.0
    for i in range(n_steps):
        dw = dW[:, i]
        z = rng.standard_normal(n_paths)
        a0 = sde.a(t, x)
        b0 = sde.b(t, x)
        ap = sde._a_x(t, x, eps=eps)                       # a'
        app = (sde.a(t, x + eps) - 2 * a0 + sde.a(t, x - eps)) / eps ** 2
        bp = sde._b_x(t, x, eps=eps)                       # b'
        bpp = sde._b_xx(t, x, eps=1e-4)                    # b''
        dZ = 0.5 * dt * dw + 0.5 / np.sqrt(3.0) * dt ** 1.5 * z
        x = (x
             + a0 * dt
             + b0 * dw
             + 0.5 * b0 * bp * (dw ** 2 - dt)
             + ap * b0 * dZ
             + 0.5 * (a0 * ap + 0.5 * b0 ** 2 * app) * dt ** 2
             + (a0 * bp + 0.5 * b0 ** 2 * bpp) * (dw * dt - dZ)
             + 0.5 * b0 * (b0 * bpp + bp ** 2)
             * ((1.0 / 3.0) * dw ** 2 - dt) * dw)
        paths[:, i + 1] = x
        t += dt
    return paths


def WeakApproximation(sde, T=1.0, n_steps=50, n_paths=100_000,
                      random_state=None):
    """Weak order 2 terminal values of X_T (Talay-Tubaro scheme).

    Uses the three-point distributed increment ``P(+-sqrt(3*dt)) = 1/6``,
    ``P(0) = 2/3`` (matching the first four moments of ``N(0, dt)`` exactly,
    which weak order 2 requires) and the second-order corrections
    ``0.5 * L0(a) * dt**2 + L0(b) * dW * dt`` with
    ``L0 = a*d/dx + 0.5*b^2*d^2/dx^2``. Feed the returned array through your
    functional and average; the estimator carries Monte-Carlo standard errors
    (see the tests for the E[GBM_T] check).
    """
    rng = np.random.default_rng(random_state)
    dt = T / n_steps
    sq3 = np.sqrt(3.0)
    eps = 1e-5
    x = np.full(n_paths, sde.x0, dtype=float)
    t = 0.0
    for _ in range(n_steps):
        # P(-sqrt(3dt)) = P(+sqrt(3dt)) = 1/6, P(0) = 2/3 -- NOT a uniform
        # three-way split (development/Probleme.md documents the incident:
        # rng.integers(0, 3, ...) gives 1/3 each, doubling E[dW**2] to 2*dt
        # and biasing every moment computed downstream by an O(1) amount
        # that never shrinks as n_steps grows).
        u = rng.random(n_paths)
        dW = np.where(u < 1.0 / 6.0, -sq3,
                     np.where(u < 5.0 / 6.0, 0.0, sq3)) * np.sqrt(dt)
        a0 = sde.a(t, x)
        b0 = sde.b(t, x)
        ap = sde._a_x(t, x, eps=eps)
        app = (sde.a(t, x + eps) - 2 * a0 + sde.a(t, x - eps)) / eps ** 2
        bp = sde._b_x(t, x, eps=eps)
        bpp = sde._b_xx(t, x, eps=1e-4)
        L0a = a0 * ap + 0.5 * b0 ** 2 * app
        L0b = a0 * bp + 0.5 * b0 ** 2 * bpp
        x = (x + a0 * dt + b0 * dW
             + 0.5 * b0 * bp * (dW ** 2 - dt)
             + 0.5 * L0a * dt ** 2
             + L0b * dW * dt)
        t += dt
    return x


def StrongApproximation(sde, exact_solver, scheme_solvers, step_sizes,
                        n_paths=20_000, random_state=None):
    """Strong-error convergence study.

    ``exact_solver(T, n_steps, n_paths, rng)`` produces exact paths;
    ``scheme_solvers`` maps names to solver callables with the same signature.
    For each scheme and step size, computes the root-mean-square strong error
    against the exact solution on the common grid. Returns a dict
    ``{name: {"h": [...], "error": [...]}}`` ready for log-log order fitting.

    ``random_state`` must be a reusable seed (int, array-like, or ``None`` —
    not a live ``Generator``): the exact and approximate paths are each drawn
    from a *fresh* generator reseeded from it, so both see the identical
    driving Brownian path at each step size — the strong-error comparison is
    meaningless otherwise (development/Probleme.md documents the incident:
    passing one already-advanced ``Generator`` to both calls made the "exact"
    and "approximate" paths independent, so the measured "error" was just
    Monte-Carlo noise between two unrelated paths and never shrank with h).
    """
    results = {}
    for name, solver in scheme_solvers.items():
        hs, errs = [], []
        for n_steps in step_sizes:
            exact = exact_solver(sde, 1.0, n_steps, n_paths,
                                 np.random.default_rng(random_state))
            approx = solver(sde, 1.0, n_steps, n_paths,
                            np.random.default_rng(random_state))
            err = np.sqrt(np.mean((approx[:, -1] - exact[:, -1]) ** 2))
            hs.append(1.0 / n_steps)
            errs.append(float(err))
        results[name] = {"h": hs, "error": errs}
    return results
