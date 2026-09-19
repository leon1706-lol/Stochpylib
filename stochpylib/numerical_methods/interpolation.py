"""Interpolation: cubic splines, PCHIP-style cubic Hermite, barycentric Lagrange,
Chebyshev series, NURBS/B-splines, and a facade dispatcher.
"""

import numpy as np

from stochpylib.numerical_methods._common import _thomas
from stochpylib.numerical_methods.integration import GaussLegendre

__all__ = [
    "Interpolation", "SplineInterpolation", "CubicHermite", "BarycentricLagrange",
    "Chebyshev", "NURBS",
]


def _validate_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("x must be 1-D and y must share its length")
    if len(x) < 2:
        raise ValueError("need at least 2 points")
    order = np.argsort(x)
    if not np.array_equal(order, np.arange(len(x))):
        x, y = x[order], y[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("x values must be strictly increasing (after sorting, no duplicates)")
    return x, y


class SplineInterpolation:
    """Natural, clamped, or not-a-knot cubic spline (second-derivative formulation)."""

    def __init__(self, x, y, bc="not-a-knot", bc_values=(0.0, 0.0)):
        if bc not in ("natural", "clamped", "not-a-knot"):
            raise ValueError("bc must be 'natural', 'clamped' or 'not-a-knot'")
        self.x_, self.y_ = _validate_xy(x, y)
        self.bc = bc
        self.bc_values = bc_values
        self._fit()

    def _fit(self):
        x, y = self.x_, self.y_
        n = len(x) - 1
        h = np.diff(x)
        if self.bc == "not-a-knot" and n >= 2:
            M = self._solve_not_a_knot(x, y, h)
        else:
            M = self._solve_tridiagonal(x, y, h)
        self.second_derivs_ = M
        coeffs = np.empty((n, 4))
        for i in range(n):
            coeffs[i, 0] = y[i]
            coeffs[i, 1] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * M[i] + M[i + 1]) / 6.0
            coeffs[i, 2] = M[i] / 2.0
            coeffs[i, 3] = (M[i + 1] - M[i]) / (6.0 * h[i])
        self.coefficients_ = coeffs

    def _solve_tridiagonal(self, x, y, h):
        n = len(x) - 1
        M = np.zeros(n + 1)
        if n == 1:
            return M
        lower = np.zeros(n - 1)
        diag = np.zeros(n - 1)
        upper = np.zeros(n - 1)
        rhs = np.zeros(n - 1)
        for i in range(1, n):
            diag[i - 1] = 2.0 * (h[i - 1] + h[i])
            if i - 2 >= 0:
                lower[i - 2] = h[i - 1]
            if i <= n - 2:
                upper[i - 1] = h[i]
            rhs[i - 1] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
        if self.bc == "clamped":
            fp0, fpn = self.bc_values
            # augment as full (n+1) system instead for clamped
            return self._solve_clamped(x, y, h, fp0, fpn)
        inner = _thomas(lower, diag, upper, rhs)
        M[1:n] = inner
        return M

    def _solve_clamped(self, x, y, h, fp0, fpn):
        n = len(x) - 1
        A = np.zeros((n + 1, n + 1))
        b = np.zeros(n + 1)
        A[0, 0] = 2 * h[0]
        A[0, 1] = h[0]
        b[0] = 6.0 * ((y[1] - y[0]) / h[0] - fp0)
        A[n, n - 1] = h[-1]
        A[n, n] = 2 * h[-1]
        b[n] = 6.0 * (fpn - (y[n] - y[n - 1]) / h[-1])
        for i in range(1, n):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            b[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
        return np.linalg.solve(A, b)

    def _solve_not_a_knot(self, x, y, h):
        n = len(x) - 1
        if n == 2:
            return self._solve_tridiagonal(x, y, h)
        A = np.zeros((n + 1, n + 1))
        b = np.zeros(n + 1)
        # not-a-knot: third derivative continuous at x[1] and x[n-1]
        A[0, 0] = h[1]
        A[0, 1] = -(h[0] + h[1])
        A[0, 2] = h[0]
        A[n, n - 2] = h[-1]
        A[n, n - 1] = -(h[-2] + h[-1])
        A[n, n] = h[-2]
        for i in range(1, n):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            b[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
        return np.linalg.solve(A, b)

    def _segment(self, xq):
        idx = np.searchsorted(self.x_, xq, side="right") - 1
        return np.clip(idx, 0, len(self.x_) - 2)

    def __call__(self, xq):
        xq_arr = np.atleast_1d(np.asarray(xq, dtype=float))
        idx = self._segment(xq_arr)
        dx = xq_arr - self.x_[idx]
        c = self.coefficients_[idx]
        out = c[:, 0] + c[:, 1] * dx + c[:, 2] * dx ** 2 + c[:, 3] * dx ** 3
        return out[0] if np.ndim(xq) == 0 else out

    def derivative(self, xq, order=1):
        if order not in (1, 2, 3):
            raise ValueError("order must be 1, 2 or 3")
        xq_arr = np.atleast_1d(np.asarray(xq, dtype=float))
        idx = self._segment(xq_arr)
        dx = xq_arr - self.x_[idx]
        c = self.coefficients_[idx]
        if order == 1:
            out = c[:, 1] + 2 * c[:, 2] * dx + 3 * c[:, 3] * dx ** 2
        elif order == 2:
            out = 2 * c[:, 2] + 6 * c[:, 3] * dx
        else:
            out = 6 * c[:, 3]
        return out[0] if np.ndim(xq) == 0 else out

    def integral(self, a, b):
        a, b = float(a), float(b)
        sign = 1.0
        if a > b:
            a, b = b, a
            sign = -1.0

        def antideriv_at(t):
            idx = int(self._segment(np.array([t]))[0])
            dx = t - self.x_[idx]
            c = self.coefficients_[idx]
            local = c[0] * dx + c[1] * dx ** 2 / 2 + c[2] * dx ** 3 / 3 + c[3] * dx ** 4 / 4
            # sum full segments before idx
            total = 0.0
            for j in range(idx):
                hj = self.x_[j + 1] - self.x_[j]
                cj = self.coefficients_[j]
                total += cj[0] * hj + cj[1] * hj ** 2 / 2 + cj[2] * hj ** 3 / 3 + cj[3] * hj ** 4 / 4
            return total + local

        return sign * (antideriv_at(b) - antideriv_at(a))

    def __repr__(self):
        return f"SplineInterpolation(n={len(self.x_)}, bc={self.bc!r})"


def _pchip_slopes(x, y):
    h = np.diff(x)
    delta = np.diff(y) / h
    n = len(x)
    d = np.zeros(n)
    for k in range(1, n - 1):
        if delta[k - 1] == 0 or delta[k] == 0 or np.sign(delta[k - 1]) != np.sign(delta[k]):
            d[k] = 0.0
        else:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    # shape-preserving one-sided endpoint estimates (Fritsch-Carlson / de Boor)
    def edge(h0, h1, d0, d1):
        d_ = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(d_) != np.sign(d0):
            d_ = 0.0
        elif np.sign(d0) != np.sign(d1) and abs(d_) > 3 * abs(d0):
            d_ = 3 * d0
        return d_

    if n == 2:
        d[0] = d[1] = delta[0]
    else:
        d[0] = edge(h[0], h[1], delta[0], delta[1])
        d[-1] = edge(h[-1], h[-2], delta[-1], delta[-2])
    return d


class CubicHermite:
    """Piecewise cubic Hermite interpolant; PCHIP (Fritsch-Carlson) slopes if none given."""

    def __init__(self, x, y, dydx=None):
        self.x_, self.y_ = _validate_xy(x, y)
        self.slopes_ = _pchip_slopes(self.x_, self.y_) if dydx is None else np.asarray(dydx, dtype=float)
        if len(self.slopes_) != len(self.x_):
            raise ValueError("dydx must have the same length as x")

    def _segment(self, xq):
        idx = np.searchsorted(self.x_, xq, side="right") - 1
        return np.clip(idx, 0, len(self.x_) - 2)

    def __call__(self, xq):
        xq_arr = np.atleast_1d(np.asarray(xq, dtype=float))
        idx = self._segment(xq_arr)
        x0, x1 = self.x_[idx], self.x_[idx + 1]
        y0, y1 = self.y_[idx], self.y_[idx + 1]
        m0, m1 = self.slopes_[idx], self.slopes_[idx + 1]
        h = x1 - x0
        s = (xq_arr - x0) / h
        h00 = 2 * s ** 3 - 3 * s ** 2 + 1
        h10 = s ** 3 - 2 * s ** 2 + s
        h01 = -2 * s ** 3 + 3 * s ** 2
        h11 = s ** 3 - s ** 2
        out = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
        return out[0] if np.ndim(xq) == 0 else out

    def derivative(self, xq):
        xq_arr = np.atleast_1d(np.asarray(xq, dtype=float))
        idx = self._segment(xq_arr)
        x0, x1 = self.x_[idx], self.x_[idx + 1]
        y0, y1 = self.y_[idx], self.y_[idx + 1]
        m0, m1 = self.slopes_[idx], self.slopes_[idx + 1]
        h = x1 - x0
        s = (xq_arr - x0) / h
        dh00 = 6 * s ** 2 - 6 * s
        dh10 = 3 * s ** 2 - 4 * s + 1
        dh01 = -6 * s ** 2 + 6 * s
        dh11 = 3 * s ** 2 - 2 * s
        out = (dh00 * y0 + dh10 * h * m0 + dh01 * y1 + dh11 * h * m1) / h
        return out[0] if np.ndim(xq) == 0 else out

    def integral(self, a, b):
        a, b = float(a), float(b)
        sign = 1.0
        if a > b:
            a, b = b, a
            sign = -1.0

        def F(t):
            idx = int(self._segment(np.array([t]))[0])
            x0, x1 = self.x_[idx], self.x_[idx + 1]
            y0, y1 = self.y_[idx], self.y_[idx + 1]
            m0, m1 = self.slopes_[idx], self.slopes_[idx + 1]
            h = x1 - x0
            s = (t - x0) / h
            H00 = s ** 4 / 2 - s ** 3 + s
            H10 = s ** 4 / 4 - 2 * s ** 3 / 3 + s ** 2 / 2
            H01 = -s ** 4 / 2 + s ** 3
            H11 = s ** 4 / 4 - s ** 3 / 3
            local = h * (H00 * y0 + H10 * h * m0 + H01 * y1 + H11 * h * m1)
            total = 0.0
            for j in range(idx):
                total += self._segment_integral(j)
            return total + local

        return sign * (F(b) - F(a))

    def _segment_integral(self, j):
        x0, x1 = self.x_[j], self.x_[j + 1]
        y0, y1 = self.y_[j], self.y_[j + 1]
        m0, m1 = self.slopes_[j], self.slopes_[j + 1]
        h = x1 - x0
        # integral of Hermite basis over s in [0,1] at s=1
        H00, H10, H01, H11 = 0.5, 1.0 / 12.0, 0.5, -1.0 / 12.0
        return h * (H00 * y0 + H10 * h * m0 + H01 * y1 + H11 * h * m1)

    def __repr__(self):
        return f"CubicHermite(n={len(self.x_)})"


class BarycentricLagrange:
    """Barycentric Lagrange interpolation (numerically stable, O(n) per evaluation)."""

    def __init__(self, x, y, weights=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(np.unique(x)) != len(x):
            raise ValueError("x values must be distinct")
        self.x_ = x
        self.y_ = y
        self.weights_ = self._default_weights(x) if weights is None else np.asarray(weights, dtype=float)

    @staticmethod
    def _default_weights(x):
        n = len(x)
        scale = 4.0 / (np.max(x) - np.min(x)) if np.max(x) != np.min(x) else 1.0
        w = np.ones(n)
        for j in range(n):
            diffs = scale * (x[j] - np.delete(x, j))
            w[j] = 1.0 / np.prod(diffs)
        return w

    def update_y(self, y):
        y = np.asarray(y, dtype=float)
        if len(y) != len(self.x_):
            raise ValueError("y must match the stored x length")
        self.y_ = y
        return self

    def __call__(self, xq):
        xq_arr = np.atleast_1d(np.asarray(xq, dtype=float))
        out = np.empty_like(xq_arr)
        for i, xv in enumerate(xq_arr):
            diff = xv - self.x_
            exact = np.where(diff == 0.0)[0]
            if len(exact):
                out[i] = self.y_[exact[0]]
                continue
            terms = self.weights_ / diff
            out[i] = np.sum(terms * self.y_) / np.sum(terms)
        return out[0] if np.ndim(xq) == 0 else out

    @classmethod
    def chebyshev_points(cls, n, a=-1.0, b=1.0, kind=2):
        n = int(n)
        if kind == 2:
            j = np.arange(n)
            x = np.cos(j * np.pi / (n - 1)) if n > 1 else np.array([0.0])
        else:
            j = np.arange(1, n + 1)
            x = np.cos((2 * j - 1) * np.pi / (2 * n))
        x = 0.5 * (a + b) + 0.5 * (b - a) * x
        return np.sort(x)

    @classmethod
    def from_function(cls, f, n, a=-1.0, b=1.0, kind=2):
        x = cls.chebyshev_points(n, a, b, kind)
        y = np.array([float(f(xi)) for xi in x])
        n_ = len(x)
        if kind == 2:
            w = np.array([(-1.0) ** j for j in range(n_)], dtype=float)
            w[0] *= 0.5
            w[-1] *= 0.5
        else:
            j = np.arange(n_)
            w = ((-1.0) ** j) * np.sin((2 * j + 1) * np.pi / (2 * n_))
        return cls(x, y, weights=w)

    def __repr__(self):
        return f"BarycentricLagrange(n={len(self.x_)})"


class Chebyshev:
    """Chebyshev polynomial series on an arbitrary domain via the Clenshaw recurrence."""

    def __init__(self, coefficients=None, domain=(-1.0, 1.0)):
        self.domain = (float(domain[0]), float(domain[1]))
        self.coefficients_ = np.array([0.0]) if coefficients is None else np.asarray(coefficients, dtype=float)

    @property
    def degree(self):
        return len(self.coefficients_) - 1

    def _to_unit(self, x):
        a, b = self.domain
        return (2.0 * x - (a + b)) / (b - a)

    @classmethod
    def from_function(cls, f, n, a=-1.0, b=1.0, vectorized=False):
        """Fit via Chebyshev-Gauss-Lobatto nodes and the DCT-I coefficient formula."""
        n = int(n)
        j = np.arange(n + 1)
        t = np.cos(j * np.pi / n) if n > 0 else np.array([0.0])
        x = 0.5 * (a + b) + 0.5 * (b - a) * t
        if vectorized:
            fv = np.asarray(f(x), dtype=float)
        else:
            fv = np.array([float(f(xi)) for xi in x])
        c = np.zeros(n + 1)
        for k in range(n + 1):
            terms = fv * np.cos(k * j * np.pi / n)
            terms[0] *= 0.5
            terms[-1] *= 0.5
            s = 2.0 / n * np.sum(terms) if n > 0 else fv[0]
            c[k] = s
        c[0] *= 0.5
        if n > 0:
            c[-1] *= 0.5
        return cls(c, domain=(a, b))

    def __call__(self, xq):
        xq_arr = np.atleast_1d(np.asarray(xq, dtype=float))
        t = self._to_unit(xq_arr)
        c = self.coefficients_
        n = len(c) - 1
        b_k1 = np.zeros_like(t)
        b_k2 = np.zeros_like(t)
        for k in range(n, 0, -1):
            b_k = c[k] + 2 * t * b_k1 - b_k2
            b_k2 = b_k1
            b_k1 = b_k
        out = c[0] + t * b_k1 - b_k2
        return out[0] if np.ndim(xq) == 0 else out

    def derivative(self):
        """Chebyshev coefficients of f' via the standard backward recurrence
        (Trefethen, *Spectral Methods in MATLAB*, eq. 8): with ``b_n = 0``,
        ``b_{k-1} = b_{k+1} + 2k*c_k`` for ``k = n..1``, then ``c'_0 = b_0/2``,
        ``c'_k = b_k`` for ``k >= 1``, scaled by ``2/(domain span)`` for the chain rule.
        """
        c = self.coefficients_
        n = len(c) - 1
        if n == 0:
            return Chebyshev(np.array([0.0]), domain=self.domain)
        b = np.zeros(n + 2)  # b[n+1] unused sentinel, b[n] = 0
        for k in range(n, 0, -1):
            b[k - 1] = b[k + 1] + 2 * k * c[k]
        cp = b[:n].copy()
        cp[0] *= 0.5
        scale = 2.0 / (self.domain[1] - self.domain[0])
        return Chebyshev(cp * scale, domain=self.domain)

    def integral(self):
        """Indefinite integral (antiderivative) with ``C(domain[0]) == 0``.

        ``C_k = scale*(c_{k-1} - c_{k+1})/(2k)`` for ``k >= 2`` (``c_{n+1}, c_{n+2} := 0``);
        ``C_1`` needs the special case ``scale*(c_0 - c_2/2)`` (the ``T_1`` term picks up an
        extra, un-halved contribution from ``integral(T_0) = T_1``, unlike every higher
        ``T_k`` whose antiderivative splits evenly between ``T_{k-1}`` and ``T_{k+1}``).
        ``C_0`` is fixed afterward so the antiderivative vanishes at the left endpoint.
        """
        c = self.coefficients_
        n = len(c) - 1
        scale = (self.domain[1] - self.domain[0]) / 2.0
        c_ext = np.concatenate([c, [0.0, 0.0]])
        C = np.zeros(n + 2)
        if n >= 0:
            C[1] = scale * (c_ext[0] - 0.5 * c_ext[2])
        for k in range(2, n + 2):
            C[k] = scale * (c_ext[k - 1] - c_ext[k + 1]) / (2 * k)
        result = Chebyshev(C, domain=self.domain)
        result.coefficients_[0] -= result(self.domain[0])
        return result

    def definite_integral(self, a=None, b=None):
        F = self.integral()
        lo = self.domain[0] if a is None else a
        hi = self.domain[1] if b is None else b
        return float(F(hi) - F(lo))

    def roots(self):
        """Real roots inside the domain, via the colleague (companion) matrix."""
        c = self.coefficients_
        n = self.degree
        if n < 1:
            return np.array([])
        c = c / c[-1] if c[-1] != 0 else c
        C = np.zeros((n, n))
        for i in range(1, n):
            C[i, i - 1] = 0.5
            C[i - 1, i] = 0.5
        C[0, 1] = 1.0
        for i in range(n):
            C[i, n - 1] -= 0.5 * c[i] if i < n - 1 else 0.0
        C[:, n - 1] -= 0.5 * c[:n]
        eig = np.linalg.eigvals(C)
        real_roots = eig[np.abs(eig.imag) < 1e-8].real
        real_roots = real_roots[(real_roots >= -1 - 1e-8) & (real_roots <= 1 + 1e-8)]
        a, b = self.domain
        x = 0.5 * (a + b) + 0.5 * (b - a) * real_roots
        return np.sort(x)

    def truncate(self, tol=1e-14):
        c = self.coefficients_
        mag = np.max(np.abs(c)) if len(c) else 0.0
        keep = len(c)
        for i in range(len(c) - 1, 0, -1):
            if abs(c[i]) > tol * max(mag, 1.0):
                keep = i + 1
                break
        else:
            keep = 1
        return Chebyshev(c[:keep].copy(), domain=self.domain)

    def __repr__(self):
        return f"Chebyshev(degree={self.degree}, domain={self.domain})"


class NURBS:
    """Non-uniform rational B-spline curve via the Cox-de Boor recursion."""

    def __init__(self, control_points, degree, knots=None, weights=None):
        P = np.asarray(control_points, dtype=float)
        if P.ndim != 2:
            raise ValueError("control_points must be (n, dim)")
        self.control_points = P
        self.degree = int(degree)
        n = P.shape[0]
        if knots is None:
            m = n + self.degree + 1
            interior = m - 2 * (self.degree + 1)
            knots = np.concatenate([
                np.zeros(self.degree + 1),
                np.linspace(0, 1, interior + 2)[1:-1] if interior > 0 else np.array([]),
                np.ones(self.degree + 1),
            ])
        self.knots = np.asarray(knots, dtype=float)
        self.weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        if len(self.knots) != n + self.degree + 1:
            raise ValueError("len(knots) must equal n_control_points + degree + 1")

    @property
    def domain(self):
        p = self.degree
        return float(self.knots[p]), float(self.knots[-p - 1])

    def _find_span(self, u):
        p, kv = self.degree, self.knots
        n = len(self.control_points) - 1
        if u >= kv[n + 1]:
            return n
        if u <= kv[p]:
            return p
        idx = np.searchsorted(kv, u, side="right") - 1
        return int(np.clip(idx, p, n))

    def _basis_at(self, u, span):
        p, kv = self.degree, self.knots
        N = np.zeros(p + 1)
        left = np.zeros(p + 1)
        right = np.zeros(p + 1)
        N[0] = 1.0
        for j in range(1, p + 1):
            left[j] = u - kv[span + 1 - j]
            right[j] = kv[span + j] - u
            saved = 0.0
            for r in range(j):
                denom = right[r + 1] + left[j - r]
                temp = N[r] / denom if denom != 0 else 0.0
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            N[j] = saved
        return N

    def basis(self, u):
        u_arr = np.atleast_1d(np.asarray(u, dtype=float))
        n = len(self.control_points)
        out = np.zeros((len(u_arr), n))
        for i, uv in enumerate(u_arr):
            span = self._find_span(uv)
            Nb = self._basis_at(uv, span)
            out[i, span - self.degree:span + 1] = Nb
        return out

    def evaluate(self, u):
        u_arr = np.atleast_1d(np.asarray(u, dtype=float))
        N = self.basis(u_arr)
        num = N @ (self.control_points * self.weights[:, None])
        den = N @ self.weights
        out = num / den[:, None]
        return out[0] if np.ndim(u) == 0 else out

    def derivative(self, u):
        u_arr = np.atleast_1d(np.asarray(u, dtype=float))
        p, kv = self.degree, self.knots
        n = len(self.control_points)
        # derivative of the numerator B-spline curve A(u) = sum N_i w_i P_i and
        # denominator w(u) = sum N_i w_i via the standard degree-lowering formula.
        Qw = self.control_points * self.weights[:, None]
        Q_deriv_ctrl = np.zeros((n - 1, Qw.shape[1]))
        w_deriv_ctrl = np.zeros(n - 1)
        for i in range(n - 1):
            denom = kv[i + p + 1] - kv[i + 1]
            factor = p / denom if denom != 0 else 0.0
            Q_deriv_ctrl[i] = factor * (Qw[i + 1] - Qw[i])
            w_deriv_ctrl[i] = factor * (self.weights[i + 1] - self.weights[i])
        deriv_knots = kv[1:-1]
        deriv_curve = NURBS(Q_deriv_ctrl, p - 1, knots=deriv_knots, weights=np.ones(n - 1))
        deriv_w_curve = NURBS(w_deriv_ctrl[:, None], p - 1, knots=deriv_knots, weights=np.ones(n - 1))
        A = self.basis(u_arr) @ Qw
        w = self.basis(u_arr) @ self.weights
        # A'(u), w'(u) via plain (non-rational) B-spline evaluation of the derivative ctrl pts
        Ap = _bspline_eval(Q_deriv_ctrl, p - 1, deriv_knots, u_arr) if p > 0 else np.zeros((len(u_arr), Qw.shape[1]))
        wp = _bspline_eval(w_deriv_ctrl[:, None], p - 1, deriv_knots, u_arr)[:, 0] if p > 0 else np.zeros(len(u_arr))
        Cu = A / w[:, None]
        out = (Ap - wp[:, None] * Cu) / w[:, None]
        return out[0] if np.ndim(u) == 0 else out

    @classmethod
    def circle(cls, center=(0.0, 0.0), radius=1.0):
        c = np.asarray(center, dtype=float)
        r = radius
        s = np.sqrt(2.0) / 2.0
        pts = np.array([
            [r, 0], [r, r], [0, r], [-r, r], [-r, 0],
            [-r, -r], [0, -r], [r, -r], [r, 0],
        ], dtype=float) + c
        weights = np.array([1, s, 1, s, 1, s, 1, s, 1], dtype=float)
        knots = np.array([0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1], dtype=float)
        return cls(pts, degree=2, knots=knots, weights=weights)

    @classmethod
    def interpolate(cls, points, degree=3):
        """Global interpolation through ``points`` (Piegl-Tiller chord-length parameterization)."""
        P = np.asarray(points, dtype=float)
        n = P.shape[0]
        p = min(int(degree), n - 1)
        d = np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))
        if d == 0:
            u_bar = np.linspace(0, 1, n)
        else:
            u_bar = np.zeros(n)
            cum = 0.0
            for i in range(1, n):
                cum += np.linalg.norm(P[i] - P[i - 1])
                u_bar[i] = cum / d
        knots = np.zeros(n + p + 1)
        knots[-(p + 1):] = 1.0
        for j in range(1, n - p):
            knots[j + p] = np.mean(u_bar[j:j + p])
        A = np.zeros((n, n))
        tmp = cls(P, p, knots=knots, weights=np.ones(n))
        for i, uv in enumerate(u_bar):
            span = tmp._find_span(uv)
            Nb = tmp._basis_at(uv, span)
            A[i, span - p:span + 1] = Nb
        ctrl = np.linalg.solve(A, P)
        return cls(ctrl, p, knots=knots, weights=np.ones(n))

    def __repr__(self):
        return f"NURBS(n_control={len(self.control_points)}, degree={self.degree})"


def _bspline_eval(control_points, degree, knots, u_arr):
    """Plain (non-rational) B-spline evaluation, used internally by NURBS.derivative."""
    n = len(control_points)
    out = np.zeros((len(u_arr), control_points.shape[1]))
    kv = knots
    for idx, uv in enumerate(u_arr):
        if uv >= kv[n]:
            span = n - 1
        elif uv <= kv[degree]:
            span = degree
        else:
            span = int(np.clip(np.searchsorted(kv, uv, side="right") - 1, degree, n - 1))
        N = np.zeros(degree + 1)
        left = np.zeros(degree + 1)
        right = np.zeros(degree + 1)
        N[0] = 1.0
        for j in range(1, degree + 1):
            left[j] = uv - kv[span + 1 - j]
            right[j] = kv[span + j] - uv
            saved = 0.0
            for r in range(j):
                denom = right[r + 1] + left[j - r]
                temp = N[r] / denom if denom != 0 else 0.0
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            N[j] = saved
        out[idx] = N @ control_points[span - degree:span + 1]
    return out


class Interpolation:
    """Facade over the interpolation methods with a common ``__call__``/derivative/integral."""

    _METHODS = ("linear", "nearest", "polynomial", "cubic", "pchip", "spline_natural")

    def __init__(self, x, y, method="cubic"):
        if method not in self._METHODS:
            raise ValueError(f"method must be one of {self._METHODS}")
        self.x_, self.y_ = _validate_xy(x, y)
        self.method = method
        if method == "cubic":
            self.interpolator_ = SplineInterpolation(self.x_, self.y_, bc="not-a-knot")
        elif method == "spline_natural":
            self.interpolator_ = SplineInterpolation(self.x_, self.y_, bc="natural")
        elif method == "pchip":
            self.interpolator_ = CubicHermite(self.x_, self.y_)
        elif method == "polynomial":
            self.interpolator_ = BarycentricLagrange(self.x_, self.y_)
        else:
            self.interpolator_ = None  # linear / nearest use np.interp directly

    def __call__(self, xq):
        if self.method == "linear":
            return np.interp(xq, self.x_, self.y_)
        if self.method == "nearest":
            xq_arr = np.atleast_1d(np.asarray(xq, dtype=float))
            idx = np.array([np.argmin(np.abs(self.x_ - v)) for v in xq_arr])
            out = self.y_[idx]
            return out[0] if np.ndim(xq) == 0 else out
        return self.interpolator_(xq)

    def derivative(self, xq, eps=1e-6):
        if self.method in ("cubic", "spline_natural", "pchip"):
            return self.interpolator_.derivative(xq)
        xq_arr = np.atleast_1d(np.asarray(xq, dtype=float))
        out = (np.asarray(self(xq_arr + eps)) - np.asarray(self(xq_arr - eps))) / (2 * eps)
        return out[0] if np.ndim(xq) == 0 else out

    def integral(self, a, b):
        if self.method in ("cubic", "spline_natural", "pchip"):
            return self.interpolator_.integral(a, b)
        n = 2000
        xs = np.linspace(a, b, n)
        ys = np.asarray(self(xs))
        return float(np.trapezoid(ys, xs))

    def __repr__(self):
        return f"Interpolation(n={len(self.x_)}, method={self.method!r})"
