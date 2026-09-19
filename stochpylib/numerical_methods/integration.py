"""Numerical integration: Gauss quadrature families, adaptive Gauss-Kronrod/Simpson,
Romberg, tensor/sparse cubature, and a Monte-Carlo facade over :mod:`stochpylib.montecarlo`.

All Gauss node/weight tables are computed natively via the Golub-Welsch eigenvalue
algorithm (symmetric tridiagonal Jacobi matrix -> ``numpy.linalg.eigh``), not looked up
from ``numpy.polynomial`` or ``scipy.special``.
"""

import heapq

import numpy as np

from stochpylib.numerical_methods._common import (
    _as_scalar_fn,
    _check_interval,
    _gauss_kronrod_15,
    _transform_infinite,
)
from stochpylib.numerical_methods._result import QuadratureResult

__all__ = [
    "GaussLegendre", "GaussHermite", "GaussChebyshev", "AdaptiveQuadrature",
    "NumericalIntegration", "MonteCarloIntegration", "CubatureRule",
]


class GaussLegendre:
    """n-point Gauss-Legendre quadrature on [-1, 1] via Golub-Welsch.

    Exact for polynomials up to degree ``2n - 1``.
    """

    def __init__(self, n):
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        k = np.arange(1, n)
        beta = k / np.sqrt(4.0 * k * k - 1.0)
        J = np.diag(beta, 1) + np.diag(beta, -1)
        vals, vecs = np.linalg.eigh(J)
        order = np.argsort(vals)
        self.nodes_ = vals[order]
        self.weights_ = 2.0 * vecs[0, order] ** 2

    def nodes_weights(self, a=-1.0, b=1.0):
        a, b = _check_interval(a, b)
        c, h = 0.5 * (a + b), 0.5 * (b - a)
        return c + h * self.nodes_, h * self.weights_

    def integrate(self, f, a=-1.0, b=1.0, vectorized=False):
        g = _as_scalar_fn(f, vectorized)
        x, w = self.nodes_weights(a, b)
        value = float(np.sum(w * g(x)))
        return QuadratureResult(value=value, n_evals=self.n, method="gauss_legendre")

    def integrate_composite(self, f, a, b, n_panels=1, vectorized=False):
        a, b = _check_interval(a, b)
        edges = np.linspace(a, b, int(n_panels) + 1)
        total = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            total += self.integrate(f, lo, hi, vectorized=vectorized).value
        return QuadratureResult(value=total, n_evals=self.n * int(n_panels),
                                method="gauss_legendre_composite",
                                extras={"n_panels": int(n_panels)})

    def __repr__(self):
        return f"GaussLegendre(n={self.n})"


class GaussHermite:
    """n-point Gauss-Hermite quadrature via Golub-Welsch.

    ``kind='physicists'``: weight ``exp(-x^2)``, nodes/weights integrate against that
    weight directly (``sum(w) == sqrt(pi)``). ``kind='probabilists'`` (default): weight is
    the standard-normal density (``sum(w) == 1``), suited to ``E[g(Z)]`` for ``Z ~ N(0,1)``.
    """

    def __init__(self, n, kind="probabilists"):
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1")
        if kind not in ("physicists", "probabilists"):
            raise ValueError("kind must be 'physicists' or 'probabilists'")
        self.n = n
        self.kind = kind
        k = np.arange(1, n)
        beta = np.sqrt(k / 2.0)
        J = np.diag(beta, 1) + np.diag(beta, -1)
        vals, vecs = np.linalg.eigh(J)
        order = np.argsort(vals)
        x = vals[order]
        w = np.sqrt(np.pi) * vecs[0, order] ** 2
        if kind == "probabilists":
            x = x * np.sqrt(2.0)
            w = w / np.sqrt(np.pi)
        self.nodes_ = x
        self.weights_ = w

    def integrate(self, f, vectorized=False):
        """Integral of ``f(x) * weight(x)`` over the real line (weight per ``kind``)."""
        g = _as_scalar_fn(f, vectorized)
        return QuadratureResult(value=float(np.sum(self.weights_ * g(self.nodes_))),
                                n_evals=self.n, method=f"gauss_hermite_{self.kind}")

    def expectation(self, f, mean=0.0, std=1.0, vectorized=False):
        """``E[f(mean + std*Z)]`` for ``Z ~ N(0,1)`` (probabilists' nodes only)."""
        if self.kind != "probabilists":
            raise ValueError("expectation() requires kind='probabilists'")
        g = _as_scalar_fn(f, vectorized)
        x = mean + std * self.nodes_
        return QuadratureResult(value=float(np.sum(self.weights_ * g(x))),
                                n_evals=self.n, method="gauss_hermite_expectation")

    def __repr__(self):
        return f"GaussHermite(n={self.n}, kind={self.kind!r})"


class GaussChebyshev:
    """n-point Gauss-Chebyshev quadrature (closed form nodes/weights).

    ``kind=1``: weight ``1/sqrt(1-x^2)``. ``kind=2``: weight ``sqrt(1-x^2)``.
    """

    def __init__(self, n, kind=1):
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1")
        if kind not in (1, 2):
            raise ValueError("kind must be 1 or 2")
        self.n = n
        self.kind = kind
        j = np.arange(1, n + 1)
        if kind == 1:
            self.nodes_ = np.cos((2 * j - 1) * np.pi / (2 * n))
            self.weights_ = np.full(n, np.pi / n)
        else:
            self.nodes_ = np.cos(j * np.pi / (n + 1))
            self.weights_ = (np.pi / (n + 1)) * np.sin(j * np.pi / (n + 1)) ** 2

    def integrate_weighted(self, f, vectorized=False):
        """Integral of ``f(x) * weight(x)`` over [-1, 1]."""
        g = _as_scalar_fn(f, vectorized)
        return QuadratureResult(value=float(np.sum(self.weights_ * g(self.nodes_))),
                                n_evals=self.n, method=f"gauss_chebyshev_{self.kind}")

    def integrate(self, f, a=-1.0, b=1.0, vectorized=False):
        """Plain integral of ``f`` over ``[a, b]`` (weight divided/multiplied back out)."""
        a, b = _check_interval(a, b)
        c, h = 0.5 * (a + b), 0.5 * (b - a)
        g = _as_scalar_fn(f, vectorized)
        x = c + h * self.nodes_
        core = np.sqrt(np.maximum(1.0 - self.nodes_ ** 2, 0.0))
        if self.kind == 1:
            fv = g(x) * core
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                fv = np.where(core > 1e-14, g(x) / np.where(core > 1e-14, core, 1.0), 0.0)
        value = h * float(np.sum(self.weights_ * fv))
        return QuadratureResult(value=value, n_evals=self.n,
                                method=f"gauss_chebyshev_{self.kind}_plain")

    def __repr__(self):
        return f"GaussChebyshev(n={self.n}, kind={self.kind})"


class AdaptiveQuadrature:
    """Globally adaptive quadrature (QUADPACK-style QAG / adaptive Simpson).

    Subdivides the interval with the worst local error estimate until the requested
    tolerance is met or ``max_subdivisions`` is exhausted (never raises; check
    ``.converged``). Infinite limits are handled via a change of variables.
    """

    def __init__(self, f, a, b, rtol=1e-10, atol=1e-12, rule="gauss_kronrod",
                max_subdivisions=500, vectorized=False):
        if rule not in ("gauss_kronrod", "simpson"):
            raise ValueError("rule must be 'gauss_kronrod' or 'simpson'")
        self.f_raw = f
        self.a_raw, self.b_raw = float(a), float(b)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.rule = rule
        self.max_subdivisions = int(max_subdivisions)
        self.vectorized = vectorized

    def _prepare(self):
        g = _as_scalar_fn(self.f_raw, self.vectorized)
        g, lo, hi = _transform_infinite(g, self.a_raw, self.b_raw)
        return g, lo, hi

    def integrate(self):
        g, a, b = self._prepare()
        if self.rule == "gauss_kronrod":
            return self._integrate_gk(g, a, b)
        return self._integrate_simpson(g, a, b)

    def _integrate_gk(self, g, a, b):
        k15, err, _ = _gauss_kronrod_15(g, a, b)
        heap = [(-err, a, b, k15, err)]
        total_val, total_err = k15, err
        n_evals = 15
        n_intervals = 1
        for _ in range(self.max_subdivisions):
            tol = max(self.atol, self.rtol * abs(total_val))
            if total_err <= tol:
                return QuadratureResult(value=total_val, error=total_err, n_evals=n_evals,
                                        method="adaptive_gauss_kronrod", converged=True,
                                        extras={"n_intervals": n_intervals})
            _, lo, hi, val, e = heapq.heappop(heap)
            mid = 0.5 * (lo + hi)
            k15_l, err_l, _ = _gauss_kronrod_15(g, lo, mid)
            k15_r, err_r, _ = _gauss_kronrod_15(g, mid, hi)
            n_evals += 30
            n_intervals += 1
            heapq.heappush(heap, (-err_l, lo, mid, k15_l, err_l))
            heapq.heappush(heap, (-err_r, mid, hi, k15_r, err_r))
            total_val += (k15_l + k15_r) - val
            total_err += (err_l + err_r) - e
        return QuadratureResult(value=total_val, error=total_err, n_evals=n_evals,
                                method="adaptive_gauss_kronrod", converged=False,
                                extras={"n_intervals": n_intervals})

    def _integrate_simpson(self, g, a, b):
        def simpson(lo, hi, flo, fmid, fhi):
            return (hi - lo) / 6.0 * (flo + 4.0 * fmid + fhi)

        def refine(lo, hi, flo, fmid, fhi, whole, tol, evals, depth):
            mid = 0.5 * (lo + hi)
            lmid = 0.5 * (lo + mid)
            rmid = 0.5 * (mid + hi)
            flmid = float(g(np.array([lmid]))[0])
            frmid = float(g(np.array([rmid]))[0])
            evals[0] += 2
            left = simpson(lo, mid, flo, flmid, fmid)
            right = simpson(mid, hi, fmid, frmid, fhi)
            err = abs(left + right - whole) / 15.0
            if err <= tol or depth <= 0:
                return left + right + (left + right - whole) / 15.0, err
            lv, le = refine(lo, mid, flo, flmid, fmid, left, tol / 2, evals, depth - 1)
            rv, re = refine(mid, hi, fmid, frmid, fhi, right, tol / 2, evals, depth - 1)
            return lv + rv, le + re

        mid = 0.5 * (a + b)
        fa = float(g(np.array([a]))[0])
        fm = float(g(np.array([mid]))[0])
        fb = float(g(np.array([b]))[0])
        whole = simpson(a, b, fa, fm, fb)
        evals = [3]
        tol = max(self.atol, self.rtol * abs(whole))
        value, err = refine(a, b, fa, fm, fb, whole, tol, evals,
                            depth=int(np.log2(self.max_subdivisions + 1)) + 1)
        converged = err <= max(self.atol, self.rtol * abs(value)) or evals[0] < self.max_subdivisions
        return QuadratureResult(value=value, error=err, n_evals=evals[0],
                                method="adaptive_simpson", converged=bool(converged))

    def __repr__(self):
        return f"AdaptiveQuadrature(a={self.a_raw}, b={self.b_raw}, rule={self.rule!r})"


class NumericalIntegration:
    """Facade over the integration methods with one ``.integrate()`` entry point."""

    _METHODS = ("adaptive", "trapezoid", "simpson", "romberg", "gauss_legendre",
                "gauss_kronrod", "monte_carlo")

    def __init__(self, f, a, b, method="adaptive", n=None, vectorized=False, **kwargs):
        if method not in self._METHODS:
            raise ValueError(f"method must be one of {self._METHODS}")
        self.f = f
        self.a, self.b = float(a), float(b)
        self.method = method
        self.n = n
        self.vectorized = vectorized
        self.kwargs = kwargs

    def integrate(self):
        f, a, b, n = self.f, self.a, self.b, self.n
        g = _as_scalar_fn(f, self.vectorized)
        if self.method == "adaptive":
            return AdaptiveQuadrature(f, a, b, vectorized=self.vectorized, **self.kwargs).integrate()
        if self.method == "trapezoid":
            n = n or 1000
            x = np.linspace(a, b, n + 1)
            y = g(x)
            value = float(np.trapezoid(y, x))
            coarse = float(np.trapezoid(y[::2], x[::2])) if n % 2 == 0 else value
            return QuadratureResult(value=value, error=abs(value - coarse) / 3.0,
                                    n_evals=n + 1, method="trapezoid")
        if self.method == "simpson":
            n = n or 1000
            if n % 2 == 1:
                n += 1
            x = np.linspace(a, b, n + 1)
            y = g(x)
            h = (b - a) / n
            value = h / 3.0 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]))
            return QuadratureResult(value=float(value), n_evals=n + 1, method="simpson")
        if self.method == "romberg":
            return self._romberg(g, a, b)
        if self.method == "gauss_legendre":
            gl = GaussLegendre(n or 20)
            return gl.integrate(f, a, b, vectorized=self.vectorized)
        if self.method == "gauss_kronrod":
            k15, err, _ = _gauss_kronrod_15(g, a, b)
            return QuadratureResult(value=k15, error=err, n_evals=15, method="gauss_kronrod")
        if self.method == "monte_carlo":
            return MonteCarloIntegration(f, bounds=[(a, b)], vectorized=self.vectorized,
                                         **self.kwargs).integrate(n=n or 100_000)
        raise AssertionError

    def _romberg(self, g, a, b, tol=1e-12, max_levels=20):
        table = [[0.5 * (b - a) * (float(g(np.array([a]))[0]) + float(g(np.array([b]))[0]))]]
        n_evals = 2
        h = b - a
        for k in range(1, max_levels + 1):
            h /= 2
            npts = 2 ** (k - 1)
            xs = a + h * (2 * np.arange(npts) + 1)
            trap = 0.5 * table[k - 1][0] + h * float(np.sum(g(xs)))
            n_evals += npts
            row = [trap]
            for j in range(1, k + 1):
                row.append(row[j - 1] + (row[j - 1] - table[k - 1][j - 1]) / (4 ** j - 1))
            table.append(row)
            if k >= 1 and abs(row[-1] - table[k - 1][-1]) < tol:
                return QuadratureResult(value=row[-1], error=abs(row[-1] - table[k - 1][-1]),
                                        n_evals=n_evals, method="romberg", converged=True,
                                        extras={"table": table, "levels": k})
        return QuadratureResult(value=table[-1][-1],
                                error=abs(table[-1][-1] - table[-2][-1]),
                                n_evals=n_evals, method="romberg", converged=False,
                                extras={"table": table, "levels": max_levels})

    @staticmethod
    def compare(f, a, b, exact=None, methods=("adaptive", "simpson", "romberg",
                                              "gauss_legendre", "monte_carlo")):
        out = {}
        for m in methods:
            res = NumericalIntegration(f, a, b, method=m).integrate()
            if exact is not None:
                res.extras["abs_error_vs_exact"] = abs(res.value - exact)
            out[m] = res
        return out

    def __repr__(self):
        return f"NumericalIntegration(a={self.a}, b={self.b}, method={self.method!r})"


class MonteCarloIntegration:
    """Integral over a box via crude or quasi-Monte-Carlo, delegating to
    :mod:`stochpylib.montecarlo` for the actual sampling.

    Distinct from :class:`stochpylib.montecarlo.MonteCarloIntegration` (which returns an
    ``MCResult``): this class lives in ``numerical_methods`` for interface parity with the
    module's other quadrature classes (``.integrate() -> QuadratureResult``).
    """

    def __init__(self, f, bounds, n=100_000, method="crude", sequence="sobol",
                vectorized=False, random_state=None):
        if method not in ("crude", "qmc"):
            raise ValueError("method must be 'crude' or 'qmc'")
        self.f = f
        self.bounds = list(bounds)
        self.n = int(n)
        self.method = method
        self.sequence = sequence
        self.vectorized = vectorized
        self.random_state = random_state

    def integrate(self, n=None):
        from stochpylib.montecarlo import crude_mc, quasi_montecarlo

        dim = len(self.bounds)
        g = _as_scalar_fn(self.f, self.vectorized)

        def integrand(pts):
            if dim == 1:
                return g(pts[:, 0])
            return _eval_multi(g, pts)

        n = int(n) if n is not None else self.n
        if self.method == "crude":
            mc = crude_mc(integrand, n=n, dim=dim, bounds=self.bounds, random_state=self.random_state)
        else:
            mc = quasi_montecarlo(integrand, dim=dim, n=n, sequence=self.sequence,
                                  bounds=self.bounds, random_state=self.random_state)
        return QuadratureResult(value=float(mc.estimate), error=float(mc.std_error), n_evals=n,
                                method=f"monte_carlo:{self.method}", extras={"mc_result": mc})

    def __repr__(self):
        return f"MonteCarloIntegration(dim={len(self.bounds)}, method={self.method!r})"


def _eval_multi(g, pts):
    """Evaluate a scalar-argument function g over each row of an (m, dim) array."""
    pts = np.asarray(pts, dtype=float)
    return np.asarray([float(np.asarray(g(row))) for row in pts], dtype=float)


class CubatureRule:
    """Multidimensional quadrature: tensor-product or Smolyak sparse grids."""

    def __init__(self, dim, rule="gauss_legendre", n=5, method="tensor", level=3):
        if rule not in ("gauss_legendre", "gauss_hermite", "clenshaw_curtis"):
            raise ValueError("rule must be 'gauss_legendre', 'gauss_hermite' or 'clenshaw_curtis'")
        if method not in ("tensor", "smolyak"):
            raise ValueError("method must be 'tensor' or 'smolyak'")
        self.dim = int(dim)
        self.rule = rule
        self.n = n
        self.method = method
        self.level = int(level)

    def _rule_1d(self, npts):
        npts = max(int(npts), 1)
        if self.rule == "gauss_legendre":
            gl = GaussLegendre(npts)
            return gl.nodes_.copy(), gl.weights_.copy()
        if self.rule == "gauss_hermite":
            gh = GaussHermite(npts, kind="probabilists")
            return gh.nodes_.copy(), gh.weights_.copy()
        # clenshaw_curtis on [-1, 1], npts points (npts=1 -> single node at 0)
        if npts == 1:
            return np.array([0.0]), np.array([2.0])
        j = np.arange(npts)
        x = np.cos(np.pi * j / (npts - 1))
        w = np.ones(npts)
        for i in range(npts):
            theta = np.pi * i / (npts - 1)
            s = 0.0
            for k in range(1, (npts - 1) // 2 + 1):
                bk = 1.0 if 2 * k != npts - 1 else 0.5
                s += bk * np.cos(2 * k * theta) / (4 * k * k - 1)
            c0 = 1.0 if (i == 0 or i == npts - 1) else 2.0
            w[i] = (c0 / (npts - 1)) * (1.0 - s * 2.0)
        w *= 1.0
        # normalize so weights sum to 2 (matches Gauss-Legendre's [-1,1] measure)
        w = w * (2.0 / np.sum(w))
        return x, w

    def nodes_weights(self, bounds=None):
        if self.rule == "gauss_hermite" and bounds is not None:
            raise ValueError("gauss_hermite cubature has no finite bounds")
        if self.rule != "gauss_hermite" and bounds is None:
            raise ValueError("bounds are required for gauss_legendre/clenshaw_curtis")
        if self.method == "tensor":
            return self._tensor_nodes_weights(bounds)
        return self._smolyak_nodes_weights(bounds)

    def _scale(self, x1d, w1d, lo, hi):
        if self.rule == "gauss_hermite":
            return x1d, w1d
        c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
        return c + h * x1d, h * w1d

    def _tensor_nodes_weights(self, bounds):
        n_each = self.n if isinstance(self.n, (list, tuple, np.ndarray)) else [self.n] * self.dim
        per_dim = []
        for d in range(self.dim):
            x1d, w1d = self._rule_1d(n_each[d])
            if bounds is not None:
                x1d, w1d = self._scale(x1d, w1d, bounds[d][0], bounds[d][1])
            per_dim.append((x1d, w1d))
        grids = np.meshgrid(*[p[0] for p in per_dim], indexing="ij")
        wgrids = np.meshgrid(*[p[1] for p in per_dim], indexing="ij")
        nodes = np.stack([g.ravel() for g in grids], axis=1)
        weights = np.ones(nodes.shape[0])
        for wg in wgrids:
            weights *= wg.ravel()
        return nodes, weights

    def _smolyak_nodes_weights(self, bounds):
        d = self.dim
        ell = self.level
        combined = {}

        def clenshaw_curtis_n(k):
            return 1 if k == 0 else 2 ** k + 1

        min_k = max(ell - d + 1, 0)
        for total in range(min_k, ell + 1):
            for k in _compositions(total, d):
                coeff_pow = ell - total
                sign = (-1) ** coeff_pow
                from math import comb
                coeff = sign * comb(d - 1, coeff_pow)
                if coeff == 0:
                    continue
                per_dim = []
                for ki in k:
                    npts = clenshaw_curtis_n(ki) if self.rule == "clenshaw_curtis" else ki + 1
                    x1d, w1d = self._rule_1d(npts)
                    if bounds is not None:
                        x1d, w1d = self._scale(x1d, w1d, bounds[len(per_dim)][0], bounds[len(per_dim)][1])
                    per_dim.append((x1d, w1d))
                grids = np.meshgrid(*[p[0] for p in per_dim], indexing="ij")
                wgrids = np.meshgrid(*[p[1] for p in per_dim], indexing="ij")
                nodes = np.stack([g.ravel() for g in grids], axis=1)
                weights = np.ones(nodes.shape[0])
                for wg in wgrids:
                    weights *= wg.ravel()
                for pt, w in zip(nodes, weights):
                    key = tuple(np.round(pt, 12))
                    combined[key] = combined.get(key, 0.0) + coeff * w
        pts = np.array(list(combined.keys()))
        wts = np.array(list(combined.values()))
        return pts, wts

    def integrate(self, f, bounds=None, vectorized=False):
        g = _as_scalar_fn(f, vectorized) if self.dim == 1 else None
        nodes, weights = self.nodes_weights(bounds)
        if self.dim == 1:
            value = float(np.sum(weights * g(nodes[:, 0])))
        else:
            fv = np.array([float(np.asarray(f(row))) for row in nodes])
            value = float(np.sum(weights * fv))
        return QuadratureResult(value=value, n_evals=len(nodes),
                                method=f"cubature_{self.method}_{self.rule}",
                                extras={"n_nodes": len(nodes)})

    def __repr__(self):
        return f"CubatureRule(dim={self.dim}, rule={self.rule!r}, method={self.method!r})"


def _compositions(total, d):
    """All d-tuples of non-negative integers summing to total."""
    if d == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, d - 1):
            yield (first,) + rest
