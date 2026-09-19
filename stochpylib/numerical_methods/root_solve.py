"""Root finding: bracketing (bisection, Brent), open (secant, Newton-Raphson, fixed
point) and a dispatcher/bracket-search/multi-root facade.

Every scalar solver returns a :class:`~stochpylib.numerical_methods._result.RootResult`
and never raises on non-convergence (``converged=False`` instead) -- the one exception
being an invalid bracket (no sign change), which is a usage error.
"""

import numpy as np

from stochpylib.numerical_methods._common import _numeric_jacobian
from stochpylib.numerical_methods._result import RootResult

__all__ = ["RootFinding", "Bisection", "NewtonRaphson", "Brent", "Secant", "FixedPoint"]


def Bisection(f, a, b, xtol=1e-12, max_iter=200, args=()):
    """Bisection method on a bracket ``[a, b]`` with ``f(a)`` and ``f(b)`` of opposite sign."""
    a, b = float(a), float(b)
    fa, fb = float(f(a, *args)), float(f(b, *args))
    if fa == 0.0:
        return RootResult(a, fa, 0, True, "bisection", bracket=(a, b))
    if fb == 0.0:
        return RootResult(b, fb, 0, True, "bisection", bracket=(a, b))
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    history = []
    converged = False
    c, fc = a, fa
    for i in range(1, int(max_iter) + 1):
        c = 0.5 * (a + b)
        fc = float(f(c, *args))
        history.append(c)
        if fc == 0.0 or (b - a) / 2.0 < xtol:
            converged = True
            break
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return RootResult(c, fc, len(history), converged, "bisection", bracket=(a, b), history=history)


def Brent(f, a, b, xtol=1e-12, ftol=0.0, max_iter=200):
    """Brent-Dekker method: inverse quadratic interpolation / secant, bisection fallback."""
    a, b = float(a), float(b)
    fa, fb = float(f(a)), float(f(b))
    if fa == 0.0:
        return RootResult(a, fa, 0, True, "brent", bracket=(a, b))
    if fb == 0.0:
        return RootResult(b, fb, 0, True, "brent", bracket=(a, b))
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    c, fc = a, fa
    mflag = True
    d = a
    history = [b]
    for i in range(1, int(max_iter) + 1):
        if fb == 0.0 or abs(b - a) < xtol or abs(fb) <= ftol:
            return RootResult(b, fb, i - 1, True, "brent", bracket=(a, b), history=history)
        if fa != fc and fb != fc:
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            s = b - fb * (b - a) / (fb - fa)
        cond = (
            not (min((3 * a + b) / 4, b) <= s <= max((3 * a + b) / 4, b))
            or (mflag and abs(s - b) >= abs(b - c) / 2)
            or (not mflag and abs(s - b) >= abs(c - d) / 2)
            or (mflag and abs(b - c) < xtol)
            or (not mflag and abs(c - d) < xtol)
        )
        if cond:
            s = 0.5 * (a + b)
            mflag = True
        else:
            mflag = False
        fs = float(f(s))
        d = c
        c, fc = b, fb
        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        history.append(b)
    return RootResult(b, fb, int(max_iter), False, "brent", bracket=(a, b), history=history)


def Secant(f, x0, x1=None, xtol=1e-12, ftol=0.0, max_iter=200):
    """Secant method: two-point derivative-free open iteration."""
    x0 = float(x0)
    x1 = x0 + 1e-4 * max(1.0, abs(x0)) if x1 is None else float(x1)
    f0, f1 = float(f(x0)), float(f(x1))
    history = [x0, x1]
    converged = False
    for i in range(1, int(max_iter) + 1):
        if f1 == f0:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        f2 = float(f(x2))
        history.append(x2)
        x0, f0, x1, f1 = x1, f1, x2, f2
        if abs(x1 - x0) < xtol or abs(f1) <= ftol:
            converged = True
            break
    return RootResult(x1, f1, len(history) - 2, converged, "secant", history=history)


def NewtonRaphson(f, x0, fprime=None, xtol=1e-12, ftol=0.0, max_iter=100, damping=True, args=()):
    """Newton-Raphson: scalar (``fprime`` a derivative) or vector (``fprime`` a Jacobian).

    Without ``fprime``, uses a central finite-difference derivative/Jacobian.
    """
    x0_arr = np.atleast_1d(np.asarray(x0, dtype=float))
    scalar = np.ndim(x0) == 0

    def F(x):
        return np.atleast_1d(np.asarray(f(x[0] if scalar else x, *args), dtype=float))

    def J(x):
        if fprime is not None:
            Jx = np.asarray(fprime(x[0] if scalar else x, *args), dtype=float)
            return np.atleast_2d(Jx) if Jx.ndim < 2 else Jx
        return _numeric_jacobian(F, x)

    x = x0_arr.copy()
    Fx = F(x)
    converged = False
    history = [x.copy()]
    it = 0
    for it in range(1, int(max_iter) + 1):
        if np.linalg.norm(Fx) <= ftol:
            converged = True
            break
        Jx = J(x)
        try:
            step = np.linalg.solve(Jx, -Fx) if Jx.shape[0] > 1 else np.array([-Fx[0] / Jx[0, 0]])
        except np.linalg.LinAlgError:
            break
        alpha = 1.0
        if damping:
            base_norm = np.linalg.norm(Fx)
            for _ in range(10):
                x_new = x + alpha * step
                Fx_new = F(x_new)
                if np.linalg.norm(Fx_new) <= base_norm or alpha < 1e-6:
                    break
                alpha *= 0.5
        else:
            x_new = x + step
            Fx_new = F(x_new)
        if np.linalg.norm(x_new - x) < xtol:
            x, Fx = x_new, Fx_new
            history.append(x.copy())
            converged = True
            break
        x, Fx = x_new, Fx_new
        history.append(x.copy())
    if np.linalg.norm(Fx) <= max(ftol, 1e-10) and not converged:
        converged = True
    root = float(x[0]) if scalar else x
    f_root = float(Fx[0]) if scalar else Fx
    return RootResult(root, f_root, it, converged, "newton", history=history)


def FixedPoint(g, x0, xtol=1e-12, max_iter=500, acceleration="none", args=()):
    """Fixed-point iteration ``x_{n+1} = g(x_n)``.

    ``acceleration='aitken'``: component-wise Aitken delta-squared every 3 iterates.
    ``acceleration='steffensen'``: scalar-only single-step Aitken acceleration.
    """
    if acceleration not in ("none", "aitken", "steffensen"):
        raise ValueError("acceleration must be 'none', 'aitken' or 'steffensen'")
    x0_arr = np.atleast_1d(np.asarray(x0, dtype=float))
    scalar = np.ndim(x0) == 0
    if acceleration == "steffensen" and not scalar:
        raise ValueError("steffensen acceleration requires a scalar x0")

    def G(x):
        return np.atleast_1d(np.asarray(g(x[0] if scalar else x, *args), dtype=float))

    x = x0_arr.copy()
    history = [x.copy()]
    converged = False
    it = 0
    if acceleration == "steffensen":
        for it in range(1, int(max_iter) + 1):
            x1 = G(x)
            x2 = G(x1)
            denom = (x2 - 2 * x1 + x)
            if np.abs(denom) < 1e-300:
                x_new = x2
            else:
                x_new = x - (x1 - x) ** 2 / denom
            history.append(x_new.copy())
            if np.linalg.norm(x_new - x) < xtol:
                x = x_new
                converged = True
                break
            x = x_new
    else:
        buf = [x.copy()]
        for it in range(1, int(max_iter) + 1):
            x_new = G(x)
            buf.append(x_new.copy())
            history.append(x_new.copy())
            if acceleration == "aitken" and len(buf) >= 3:
                x0_, x1_, x2_ = buf[-3], buf[-2], buf[-1]
                denom = (x2_ - 2 * x1_ + x0_)
                accel = np.where(np.abs(denom) < 1e-300, x2_,
                                 x0_ - (x1_ - x0_) ** 2 / np.where(np.abs(denom) < 1e-300, 1.0, denom))
                if np.linalg.norm(accel - x) < xtol:
                    x = accel
                    converged = True
                    break
                x = accel
                buf = [x.copy()]
                continue
            if np.linalg.norm(x_new - x) < xtol:
                x = x_new
                converged = True
                break
            x = x_new
    root = float(x[0]) if scalar else x
    return RootResult(root, None, it, converged, f"fixed_point_{acceleration}", history=history)


class RootFinding:
    """Dispatcher over the scalar/vector solvers, plus bracket search and multi-root scan."""

    def __init__(self, f, bracket=None, x0=None, fprime=None, method="auto", **kwargs):
        self.f = f
        self.bracket = bracket
        self.x0 = x0
        self.fprime = fprime
        self.kwargs = kwargs
        if method == "auto":
            if bracket is not None:
                method = "brent"
            elif fprime is not None:
                method = "newton"
            elif x0 is not None:
                method = "secant"
            else:
                raise ValueError("provide bracket, x0, or both")
        if method not in ("brent", "bisection", "newton", "secant", "fixed_point"):
            raise ValueError(f"unknown method {method!r}")
        self.method = method

    def solve(self):
        if self.method == "brent":
            return Brent(self.f, *self.bracket, **self.kwargs)
        if self.method == "bisection":
            return Bisection(self.f, *self.bracket, **self.kwargs)
        if self.method == "newton":
            return NewtonRaphson(self.f, self.x0, fprime=self.fprime, **self.kwargs)
        if self.method == "secant":
            return Secant(self.f, self.x0, **self.kwargs)
        if self.method == "fixed_point":
            return FixedPoint(self.f, self.x0, **self.kwargs)
        raise AssertionError

    @staticmethod
    def find_bracket(f, x0, factor=1.6, max_tries=50):
        """Expanding-interval search for a sign change, starting near ``x0``.

        Named ``find_bracket`` (not ``bracket``) to avoid colliding with the ``self.bracket``
        constructor attribute that stores an already-known ``(a, b)`` interval.
        """
        x0 = float(x0)
        a = x0
        b = x0 + (1.0 if x0 == 0 else abs(x0) * 0.01)
        fa, fb = float(f(a)), float(f(b))
        for _ in range(int(max_tries)):
            if fa * fb < 0:
                return (a, b) if a < b else (b, a)
            if abs(fa) < abs(fb):
                a = a + factor * (a - b)
                fa = float(f(a))
            else:
                b = b + factor * (b - a)
                fb = float(f(b))
        raise ValueError("find_bracket failed to find a sign change")

    @staticmethod
    def all_roots(f, a, b, n_grid=1000):
        """Scan ``[a, b]`` on a grid and Brent-refine every sign change found."""
        x = np.linspace(float(a), float(b), int(n_grid))
        fx = np.array([float(f(xi)) for xi in x])
        roots = []
        zero_idx = np.where(fx == 0.0)[0]
        for i in zero_idx:
            roots.append(RootResult(float(x[i]), 0.0, 0, True, "grid_exact"))
        sign_change = np.where(np.sign(fx[:-1]) * np.sign(fx[1:]) < 0)[0]
        for i in sign_change:
            roots.append(Brent(f, x[i], x[i + 1]))
        roots.sort(key=lambda r: float(r))
        return roots

    def __repr__(self):
        return f"RootFinding(method={self.method!r})"
