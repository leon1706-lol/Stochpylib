"""ODE initial-value-problem solvers (explicit Euler through embedded Runge-Kutta,
multistep Adams-Bashforth, implicit BDF) and SDE path solvers (Euler-Maruyama, Milstein).

Every ODE solver's ``.solve(f, t_span, y0, ...)`` returns an
:class:`~stochpylib.numerical_methods._result.ODESolution`; every SDE solver's
``.solve(t_span, x0, ...)`` returns an
:class:`~stochpylib.numerical_methods._result.SDESolution`.
"""

import numpy as np

from stochpylib.numerical_methods._common import _numeric_jacobian
from stochpylib.numerical_methods._result import ODESolution, SDESolution

__all__ = [
    "EulerMethod", "RungeKutta4", "DormandPrince", "Adams_Bashforth", "BDF",
    "Euler_Maruyama_SDE", "Milstein_SDE",
]


def _wrap_f(f, args):
    def g(t, y):
        return np.atleast_1d(np.asarray(f(t, y, *args), dtype=float))
    return g


def _y0_array(y0):
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    return y0


class EulerMethod:
    """Explicit (forward) Euler: strong/local order 1."""

    def solve(self, f, t_span, y0, h=None, n_steps=None, t_eval=None, args=()):
        t0, tf = float(t_span[0]), float(t_span[1])
        n = n_steps if n_steps is not None else (int(round((tf - t0) / h)) if h else 100)
        g = _wrap_f(f, args)
        y0a = _y0_array(y0)
        t = np.linspace(t0, tf, n + 1)
        y = np.empty((n + 1, len(y0a)))
        y[0] = y0a
        n_evals = 0
        for i in range(n):
            dt = t[i + 1] - t[i]
            y[i + 1] = y[i] + dt * g(t[i], y[i])
            n_evals += 1
        sol = ODESolution(t=t, y=y, method="euler", n_evals=n_evals, n_steps=n, success=True)
        return _apply_t_eval(sol, t_eval)

    def __repr__(self):
        return "EulerMethod()"


class RungeKutta4:
    """Classic 4th-order Runge-Kutta (fixed step)."""

    def __init__(self, tableau=None):
        self.tableau = tableau  # optional (A, b, c) explicit Butcher tableau

    def solve(self, f, t_span, y0, h=None, n_steps=None, t_eval=None, args=()):
        t0, tf = float(t_span[0]), float(t_span[1])
        n = n_steps if n_steps is not None else (int(round((tf - t0) / h)) if h else 100)
        g = _wrap_f(f, args)
        y0a = _y0_array(y0)
        t = np.linspace(t0, tf, n + 1)
        y = np.empty((n + 1, len(y0a)))
        y[0] = y0a
        n_evals = 0
        if self.tableau is None:
            for i in range(n):
                dt = t[i + 1] - t[i]
                ti, yi = t[i], y[i]
                k1 = g(ti, yi)
                k2 = g(ti + dt / 2, yi + dt / 2 * k1)
                k3 = g(ti + dt / 2, yi + dt / 2 * k2)
                k4 = g(ti + dt, yi + dt * k3)
                y[i + 1] = yi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                n_evals += 4
        else:
            A, b, c = self.tableau
            s = len(b)
            for i in range(n):
                dt = t[i + 1] - t[i]
                ti, yi = t[i], y[i]
                ks = []
                for j in range(s):
                    yj = yi + dt * sum(A[j][l] * ks[l] for l in range(j)) if j > 0 else yi
                    ks.append(g(ti + c[j] * dt, yj))
                    n_evals += 1
                y[i + 1] = yi + dt * sum(b[j] * ks[j] for j in range(s))
        sol = ODESolution(t=t, y=y, method="rk4", n_evals=n_evals, n_steps=n, success=True)
        return _apply_t_eval(sol, t_eval)

    def __repr__(self):
        return "RungeKutta4()"


class DormandPrince:
    """Embedded RK5(4)7FM (Dormand-Prince) with adaptive step control (FSAL) and cubic
    Hermite dense output (3rd/4th order accurate, not DP's own 5th-order interpolant).
    """

    _C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
    _A = [
        [],
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, -56 / 15, 32 / 9],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ]
    _B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
    _B4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200,
                    187 / 2100, 1 / 40])

    def __init__(self, rtol=1e-6, atol=1e-9, h0=None, h_min=1e-12, h_max=np.inf,
                max_steps=100_000):
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.h0 = h0
        self.h_min = float(h_min)
        self.h_max = float(h_max)
        self.max_steps = int(max_steps)

    def solve(self, f, t_span, y0, t_eval=None, args=()):
        t0, tf = float(t_span[0]), float(t_span[1])
        g = _wrap_f(f, args)
        y = _y0_array(y0)
        direction = 1.0 if tf >= t0 else -1.0
        f0 = g(t0, y)
        n_evals = 1
        h = self.h0 if self.h0 is not None else _initial_step(g, t0, y, f0, direction, self.rtol, self.atol)
        h = direction * abs(h)
        ts, ys, dys = [t0], [y.copy()], [f0.copy()]
        t, fk = t0, f0
        steps = 0
        success = True
        message = ""
        while (direction > 0 and t < tf - 1e-14) or (direction < 0 and t > tf + 1e-14):
            if steps >= self.max_steps:
                success, message = False, "max_steps exceeded"
                break
            if abs(h) < self.h_min:
                success, message = False, "step size underflow"
                break
            if (direction > 0 and t + h > tf) or (direction < 0 and t + h < tf):
                h = tf - t
            ks = [fk]
            for i in range(1, 7):
                yi = y + h * sum(self._A[i][j] * ks[j] for j in range(i))
                ks.append(g(t + self._C[i] * h, yi))
                n_evals += 1
            y5 = y + h * sum(self._B5[j] * ks[j] for j in range(7))
            y4 = y + h * sum(self._B4[j] * ks[j] for j in range(7))
            err = y5 - y4
            scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y5))
            err_norm = np.sqrt(np.mean((err / scale) ** 2))
            if err_norm <= 1.0 or abs(h) <= self.h_min:
                t = t + h
                y = y5
                fk = g(t, y)  # FSAL: reuse as k1 of next step (still counted)
                n_evals += 1
                ts.append(t)
                ys.append(y.copy())
                dys.append(fk.copy())
                steps += 1
                factor = 5.0 if err_norm < 1e-300 else min(5.0, max(0.2, 0.9 * err_norm ** (-0.2)))
            else:
                factor = max(0.2, 0.9 * err_norm ** (-0.2))
            h = h * factor
            if abs(h) > self.h_max:
                h = direction * self.h_max
        t_arr = np.array(ts)
        y_arr = np.array(ys)
        dy_arr = np.array(dys)
        sol = ODESolution(t=t_arr, y=y_arr, dy=dy_arr, method="dormand_prince",
                          n_evals=n_evals, n_steps=steps, success=success, message=message)
        return _apply_t_eval(sol, t_eval)

    def __repr__(self):
        return f"DormandPrince(rtol={self.rtol}, atol={self.atol})"


def _initial_step(g, t0, y0, f0, direction, rtol, atol):
    """Hairer-Norsett-Wanner initial step-size heuristic (Hairer & Wanner II.4)."""
    scale = atol + rtol * np.abs(y0)
    d0 = np.sqrt(np.mean((y0 / scale) ** 2)) if np.any(scale > 0) else 0.0
    d1 = np.sqrt(np.mean((f0 / scale) ** 2)) if np.any(scale > 0) else 0.0
    h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1
    y1 = y0 + direction * h0 * f0
    f1 = g(t0 + direction * h0, y1)
    d2 = np.sqrt(np.mean(((f1 - f0) / scale) ** 2)) / h0 if np.any(scale > 0) else 0.0
    h1 = (max(1e-6, h0 * 1e-3) if max(d1, d2) <= 1e-15
         else (0.01 / max(d1, d2)) ** (1.0 / 5))
    return min(100 * h0, h1)


def _apply_t_eval(sol, t_eval):
    """Return a new ODESolution whose grid is exactly ``t_eval`` (dense-output query),
    keeping the original solve's diagnostics. ``ODESolution.__post_init__`` normalizes
    ``y`` back to ``(n, d)`` whether ``sol(t_eval)`` came back 1-D or 2-D.
    """
    if t_eval is None:
        return sol
    t_eval = np.asarray(t_eval, dtype=float)
    y_at = sol(t_eval)
    return ODESolution(t=t_eval, y=y_at, method=sol.method, n_evals=sol.n_evals,
                       n_steps=sol.n_steps, success=sol.success, message=sol.message)


_AB_COEFFS = {
    1: [1.0],
    2: [3 / 2, -1 / 2],
    3: [23 / 12, -16 / 12, 5 / 12],
    4: [55 / 24, -59 / 24, 37 / 24, -9 / 24],
    5: [1901 / 720, -2774 / 720, 2616 / 720, -1274 / 720, 251 / 720],
}
_AM_COEFFS = {
    1: [1.0],
    2: [1 / 2, 1 / 2],
    3: [5 / 12, 8 / 12, -1 / 12],
    4: [9 / 24, 19 / 24, -5 / 24, 1 / 24],
    5: [251 / 720, 646 / 720, -264 / 720, 106 / 720, -19 / 720],
}


class Adams_Bashforth:
    """Explicit Adams-Bashforth (orders 1-5), RK4-started; optional PECE corrector
    (Adams-Bashforth-Moulton) of the same order.
    """

    def __init__(self, order=4, corrector=False):
        if order not in _AB_COEFFS:
            raise ValueError("order must be in 1..5")
        self.order = int(order)
        self.corrector = bool(corrector)

    def solve(self, f, t_span, y0, h=None, n_steps=None, t_eval=None, args=()):
        t0, tf = float(t_span[0]), float(t_span[1])
        n = n_steps if n_steps is not None else (int(round((tf - t0) / h)) if h else 100)
        g = _wrap_f(f, args)
        y0a = _y0_array(y0)
        t = np.linspace(t0, tf, n + 1)
        y = np.empty((n + 1, len(y0a)))
        y[0] = y0a
        p = self.order
        n_evals = 0
        f_hist = []
        rk4 = RungeKutta4()
        start = min(p - 1, n)
        for i in range(start):
            dt = t[i + 1] - t[i]
            ti, yi = t[i], y[i]
            k1 = g(ti, yi)
            k2 = g(ti + dt / 2, yi + dt / 2 * k1)
            k3 = g(ti + dt / 2, yi + dt / 2 * k2)
            k4 = g(ti + dt, yi + dt * k3)
            y[i + 1] = yi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            f_hist.append(k1)
            n_evals += 4
        if start < n:
            f_hist.append(g(t[start], y[start]))
            n_evals += 1
        coeffs = _AB_COEFFS[p]
        for i in range(start, n):
            dt = t[i + 1] - t[i]
            recent = f_hist[-p:][::-1]  # most recent first, matches coeffs order
            pred = y[i] + dt * sum(c * fk for c, fk in zip(coeffs, recent))
            if self.corrector:
                f_pred = g(t[i + 1], pred)
                n_evals += 1
                am = _AM_COEFFS[p]
                hist_for_corr = [f_pred] + recent[:p - 1]
                pred = y[i] + dt * sum(c * fk for c, fk in zip(am, hist_for_corr))
            y[i + 1] = pred
            f_hist.append(g(t[i + 1], y[i + 1]))
            n_evals += 1
        sol = ODESolution(t=t, y=y, method=f"adams_bashforth{p}{'_moulton' if self.corrector else ''}",
                          n_evals=n_evals, n_steps=n, success=True)
        return _apply_t_eval(sol, t_eval)

    def __repr__(self):
        return f"Adams_Bashforth(order={self.order}, corrector={self.corrector})"


_BDF_COEFFS = {
    # alpha coefficients for y_{n+1} - sum(alpha_j * y_{n+1-j}) = h*beta*f, most-recent-first
    1: {"alpha": [1.0], "beta": 1.0},
    2: {"alpha": [4 / 3, -1 / 3], "beta": 2 / 3},
    3: {"alpha": [18 / 11, -9 / 11, 2 / 11], "beta": 6 / 11},
    4: {"alpha": [48 / 25, -36 / 25, 16 / 25, -3 / 25], "beta": 12 / 25},
    5: {"alpha": [300 / 137, -300 / 137, 200 / 137, -75 / 137, 12 / 137], "beta": 60 / 137},
}


class BDF:
    """Implicit backward differentiation formula (orders 1-5, fixed step), Newton
    corrector per step. Order 1 is backward Euler. Ramps order during startup.
    """

    def __init__(self, order=2, newton_tol=1e-10, max_newton=20, jac=None):
        if order not in _BDF_COEFFS:
            raise ValueError("order must be in 1..5")
        self.order = int(order)
        self.newton_tol = float(newton_tol)
        self.max_newton = int(max_newton)
        self.jac = jac

    def solve(self, f, t_span, y0, h=None, n_steps=None, t_eval=None, args=()):
        t0, tf = float(t_span[0]), float(t_span[1])
        n = n_steps if n_steps is not None else (int(round((tf - t0) / h)) if h else 100)
        g = _wrap_f(f, args)
        y0a = _y0_array(y0)
        t = np.linspace(t0, tf, n + 1)
        d = len(y0a)
        y = np.empty((n + 1, d))
        y[0] = y0a
        n_evals = 0
        for i in range(n):
            dt = t[i + 1] - t[i]
            cur_order = min(self.order, i + 1)
            spec = _BDF_COEFFS[cur_order]
            hist = y[max(0, i - cur_order + 1):i + 1][::-1]  # most recent first
            rhs_const = sum(a * hk for a, hk in zip(spec["alpha"], hist))
            beta = spec["beta"]
            t_new = t[i + 1]
            y_guess = y[i].copy()

            def G(yv):
                return yv - beta * dt * g(t_new, yv) - rhs_const

            def J(yv):
                if self.jac is not None:
                    return np.asarray(self.jac(t_new, yv, *args), dtype=float) * (-beta * dt) + np.eye(d)
                return _numeric_jacobian(lambda z: G(z), yv)

            yv = y_guess
            for _ in range(self.max_newton):
                Gv = G(yv)
                n_evals += 1
                if np.linalg.norm(Gv) < self.newton_tol:
                    break
                Jv = J(yv)
                n_evals += d if self.jac is None else 0
                try:
                    step = np.linalg.solve(Jv, -Gv)
                except np.linalg.LinAlgError:
                    break
                yv = yv + step
            y[i + 1] = yv
        sol = ODESolution(t=t, y=y, method=f"bdf{self.order}", n_evals=n_evals,
                          n_steps=n, success=True)
        return _apply_t_eval(sol, t_eval)

    def __repr__(self):
        return f"BDF(order={self.order})"


def _brownian_increments(n_paths, n_steps, dt, m, rng, brownian=None):
    if brownian is not None:
        return brownian
    return rng.standard_normal((n_paths, n_steps, m)) * np.sqrt(dt)


class Euler_Maruyama_SDE:
    """Euler-Maruyama SDE path solver: strong order 0.5.

    ``drift(t, x)`` and ``diffusion(t, x)`` are called with ``x`` of shape
    ``(n_paths, d)`` and must broadcast over the path axis. ``noise='diagonal'``:
    ``diffusion`` returns ``(n_paths, d)``, multiplied elementwise by scalar-per-dimension
    Brownian increments (``m = d``). ``noise='general'``: ``diffusion`` returns
    ``(n_paths, d, m)`` and the update uses ``einsum('pdm,pm->pd')``.
    """

    def __init__(self, drift, diffusion, noise="diagonal", n_brownian=None):
        if noise not in ("diagonal", "general"):
            raise ValueError("noise must be 'diagonal' or 'general'")
        self.drift = drift
        self.diffusion = diffusion
        self.noise = noise
        self.n_brownian = n_brownian

    def solve(self, t_span, x0, n_steps=100, n_paths=1, random_state=None, brownian=None):
        rng = np.random.default_rng(random_state)
        t0, tf = float(t_span[0]), float(t_span[1])
        dt = (tf - t0) / n_steps
        x0a = np.atleast_1d(np.asarray(x0, dtype=float))
        d = len(x0a)
        m = self.n_brownian if self.n_brownian is not None else d
        x = np.tile(x0a, (n_paths, 1))
        paths = np.empty((n_paths, n_steps + 1, d))
        paths[:, 0, :] = x
        dW = _brownian_increments(n_paths, n_steps, dt, m, rng, brownian)
        t = t0
        for i in range(n_steps):
            a = np.atleast_2d(np.asarray(self.drift(t, x), dtype=float))
            b = np.asarray(self.diffusion(t, x), dtype=float)
            if self.noise == "diagonal":
                x = x + a * dt + b * dW[:, i, :]
            else:
                x = x + a * dt + np.einsum("pdm,pm->pd", b, dW[:, i, :])
            paths[:, i + 1, :] = x
            t += dt
        t_grid = np.linspace(t0, tf, n_steps + 1)
        return SDESolution(t=t_grid, paths=paths, method="euler_maruyama_sde", brownian=dW)

    def __repr__(self):
        return f"Euler_Maruyama_SDE(noise={self.noise!r})"


class Milstein_SDE:
    """Milstein SDE path solver (diagonal/commutative noise only): strong order 1.0."""

    def __init__(self, drift, diffusion, diffusion_dx=None, fd_eps=1e-6):
        self.drift = drift
        self.diffusion = diffusion
        self.diffusion_dx = diffusion_dx
        self.fd_eps = float(fd_eps)

    def _dg_dx(self, t, x):
        if self.diffusion_dx is not None:
            return np.asarray(self.diffusion_dx(t, x), dtype=float)
        eps = self.fd_eps
        gp = np.asarray(self.diffusion(t, x + eps), dtype=float)
        gm = np.asarray(self.diffusion(t, x - eps), dtype=float)
        return (gp - gm) / (2 * eps)

    def solve(self, t_span, x0, n_steps=100, n_paths=1, random_state=None, brownian=None):
        rng = np.random.default_rng(random_state)
        t0, tf = float(t_span[0]), float(t_span[1])
        dt = (tf - t0) / n_steps
        x0a = np.atleast_1d(np.asarray(x0, dtype=float))
        d = len(x0a)
        x = np.tile(x0a, (n_paths, 1))
        paths = np.empty((n_paths, n_steps + 1, d))
        paths[:, 0, :] = x
        dW = _brownian_increments(n_paths, n_steps, dt, d, rng, brownian)
        t = t0
        for i in range(n_steps):
            a = np.atleast_2d(np.asarray(self.drift(t, x), dtype=float))
            b = np.asarray(self.diffusion(t, x), dtype=float)
            bp = self._dg_dx(t, x)
            dw = dW[:, i, :]
            x = x + a * dt + b * dw + 0.5 * b * bp * (dw ** 2 - dt)
            paths[:, i + 1, :] = x
            t += dt
        t_grid = np.linspace(t0, tf, n_steps + 1)
        return SDESolution(t=t_grid, paths=paths, method="milstein_sde", brownian=dW)

    def __repr__(self):
        return "Milstein_SDE()"
