"""Variogram models, experimental (empirical) variograms and fitting.

``Variogram`` is the abstract base (subclasses implement ``_gamma(h)``); ``Semivariogram``
is the concrete parametric family (spherical/exponential/gaussian/matern/cubic/
linear/power/nugget). ``ExperimentalVariogram`` bins pairwise squared differences and
``VariogramFitting`` least-squares-fits a ``Semivariogram`` (or the best of several) to it.
``Nugget``/``Sill``/``Range`` are simple curve-based initial-value estimators, reused both
standalone and as ``VariogramFitting``'s starting point. Native numpy/scipy.special/optimize
only -- scipy.stats is the test suite's oracle only.
"""

import numpy as np
from scipy import optimize, special

from stochpylib.spatial_statistics._common import _as_coords, _as_values, _pairwise

__all__ = [
    "Variogram", "Semivariogram", "SpatialCovariance", "ExperimentalVariogram",
    "VariogramFitting", "Nugget", "Sill", "Range",
]

_FIT_MODELS = ("spherical", "exponential", "gaussian", "matern", "cubic", "linear", "power")


def _matern_corr(h, nu, length_scale):
    """General-nu Matern correlation function via scipy.special.kv.

    Equals the closed-form ``gaussian_processes.MaternKernel`` shape at nu in
    {0.5, 1.5, 2.5} exactly (verified in tests); for other nu this is the only source
    (the GP kernel only has closed forms for those three).
    """
    h = np.asarray(h, dtype=float)
    ell = float(length_scale)
    nu = float(nu)
    out = np.ones_like(h)
    pos = h > 0
    if np.any(pos):
        r = np.sqrt(2.0 * nu) * h[pos] / ell
        log_coef = (1.0 - nu) * np.log(2.0) - special.gammaln(nu)
        out[pos] = np.exp(log_coef) * r ** nu * special.kv(nu, r)
    return out


class Variogram:
    """Abstract variogram model: subclasses implement ``_gamma(h)`` for h > 0."""

    def _gamma(self, h):
        raise NotImplementedError

    def __call__(self, h):
        h = np.asarray(h, dtype=float)
        out = self._gamma(np.abs(h))
        return np.where(h == 0, 0.0, out)

    @property
    def nugget(self):
        raise NotImplementedError

    @property
    def sill(self):
        raise NotImplementedError

    @property
    def partial_sill(self):
        return self.sill - self.nugget

    @property
    def range(self):
        raise NotImplementedError

    @property
    def is_bounded(self):
        raise NotImplementedError

    def covariance(self, h):
        if not self.is_bounded:
            raise ValueError("covariance() needs a bounded (finite-sill) variogram")
        return self.sill - self(h)

    def __add__(self, other):
        parts = (self.components if isinstance(self, _NestedVariogram) else [self])
        parts = parts + (other.components if isinstance(other, _NestedVariogram) else [other])
        return _NestedVariogram(parts)

    __radd__ = __add__

    def __repr__(self):
        return f"{type(self).__name__}(nugget={self.nugget:.4g})"


class _NestedVariogram(Variogram):
    """Sum of independent nested structures (each a plain ``Variogram``)."""

    def __init__(self, components):
        self.components = list(components)

    def _gamma(self, h):
        return sum(c._gamma(h) for c in self.components)

    @property
    def nugget(self):
        return float(sum(c.nugget for c in self.components))

    @property
    def sill(self):
        return float(sum(c.sill for c in self.components))

    @property
    def range(self):
        return float(max(c.range for c in self.components))

    @property
    def is_bounded(self):
        return all(c.is_bounded for c in self.components)


def _resolve(v):
    return float(v.value_) if hasattr(v, "value_") else float(v)


class Semivariogram(Variogram):
    """A parametric semivariogram model.

    ``model``: ``"spherical"``/``"exponential"``/``"gaussian"`` use the practical-range
    convention (gamma reaches ~95% of the partial sill at ``h = range``); ``"matern"`` uses
    ``range`` directly as the correlation length scale (not rescaled to a practical range) and
    ``nu`` as the smoothness; ``"cubic"`` is bounded with a practical range at ``h = range``;
    ``"linear"``/``"power"`` are unbounded -- ``sill`` is reinterpreted as the slope
    coefficient and ``exponent`` (in (0, 2), fixed at 1 for ``"linear"``) is the power of h;
    ``"nugget"`` is a pure nugget effect (no partial sill). ``nugget``/``sill``/``range``
    accept a float or a fitted ``Nugget``/``Sill``/``Range`` (its ``value_`` is used).
    """

    def __init__(self, model="spherical", nugget=0.0, sill=1.0, range=1.0, nu=0.5,
                 exponent=1.0):
        if model not in _FIT_MODELS + ("nugget",):
            raise ValueError(f"unknown model {model!r}")
        self.model = model
        self._nugget_in = nugget
        self._sill_in = sill
        self._range_in = range
        self.nu = float(nu)
        self.exponent = float(exponent)
        if model in ("linear", "power") and not (0.0 < self.exponent < 2.0):
            raise ValueError("exponent must be in (0, 2)")

    @property
    def nugget(self):
        return _resolve(self._nugget_in)

    @property
    def sill(self):
        if self.model in ("linear", "power"):
            raise ValueError("the power/linear variogram is unbounded and has no sill")
        if self.model == "nugget":
            return self.nugget
        return _resolve(self._sill_in)

    @property
    def range(self):
        if self.model == "nugget":
            return 0.0
        return _resolve(self._range_in)

    @property
    def is_bounded(self):
        return self.model not in ("linear", "power")

    def _slope(self):
        """The slope coefficient for linear/power models (the ``sill`` argument reused)."""
        return _resolve(self._sill_in)

    def _gamma(self, h):
        h = np.asarray(h, dtype=float)
        c0 = self.nugget
        if self.model == "nugget":
            return np.full_like(h, c0)
        if self.model in ("linear", "power"):
            return c0 + self._slope() * h ** self.exponent
        a = self.range
        c1 = _resolve(self._sill_in) - c0
        if self.model == "spherical":
            hn = np.clip(h / a, 0.0, None)
            shape = np.where(hn < 1.0, 1.5 * hn - 0.5 * hn ** 3, 1.0)
        elif self.model == "exponential":
            shape = 1.0 - np.exp(-3.0 * h / a)
        elif self.model == "gaussian":
            shape = 1.0 - np.exp(-3.0 * (h / a) ** 2)
        elif self.model == "cubic":
            hn = np.clip(h / a, 0.0, None)
            shape = np.where(
                hn < 1.0,
                7.0 * hn ** 2 - 8.75 * hn ** 3 + 3.5 * hn ** 5 - 0.75 * hn ** 7,
                1.0,
            )
        elif self.model == "matern":
            shape = 1.0 - _matern_corr(h, self.nu, a)
        else:
            raise ValueError(f"unknown model {self.model!r}")
        return c0 + c1 * shape


class SpatialCovariance:
    """The covariance function of a bounded ``Variogram``: ``C(h) = sill - gamma(h)``."""

    def __init__(self, variogram):
        if not isinstance(variogram, Variogram):
            raise TypeError("variogram must be a Variogram instance")
        if not variogram.is_bounded:
            raise ValueError("SpatialCovariance needs a bounded (finite-sill) variogram")
        self.variogram = variogram

    def __call__(self, h):
        return self.variogram.sill - self.variogram(h)

    def matrix(self, X, Y=None):
        X = _as_coords(X)
        Y = X if Y is None else _as_coords(Y)
        return self(_pairwise(X, Y))

    def to_kernel(self):
        v = self.variogram
        model = getattr(v, "model", None)
        if model not in ("exponential", "gaussian", "matern"):
            raise ValueError("to_kernel supports exponential, gaussian and matern models only")
        if v.nugget != 0.0:
            raise ValueError("to_kernel requires a nugget-free variogram")
        from stochpylib.gaussian_processes import MaternKernel, RBFKernel

        if model == "exponential":
            return MaternKernel(nu=0.5, length_scale=v.range / 3.0, variance=v.partial_sill)
        if model == "gaussian":
            return RBFKernel(length_scale=v.range / np.sqrt(6.0), variance=v.partial_sill)
        if v.nu not in (0.5, 1.5, 2.5):
            raise ValueError("to_kernel only supports matern nu in {0.5, 1.5, 2.5}")
        return MaternKernel(nu=v.nu, length_scale=v.range, variance=v.partial_sill)

    @classmethod
    def from_kernel(cls, kernel):
        from stochpylib.gaussian_processes.kernels.kernels import MaternKernel, RBFKernel

        ls = float(np.atleast_1d(kernel.length_scale)[0])
        if isinstance(kernel, MaternKernel):
            if kernel.nu == 0.5:
                v = Semivariogram(model="exponential", nugget=0.0, sill=kernel.variance,
                                   range=3.0 * ls)
            else:
                v = Semivariogram(model="matern", nugget=0.0, sill=kernel.variance,
                                   range=ls, nu=kernel.nu)
        elif isinstance(kernel, RBFKernel):
            v = Semivariogram(model="gaussian", nugget=0.0, sill=kernel.variance,
                               range=ls * np.sqrt(6.0))
        else:
            raise TypeError("from_kernel supports RBFKernel and MaternKernel only")
        return cls(v)

    def __repr__(self):
        return f"SpatialCovariance({self.variogram!r})"


class ExperimentalVariogram:
    """Binned empirical semivariogram: Matheron, Cressie-Hawkins or Dowd estimators."""

    def __init__(self, bins=15, max_dist=None, estimator="matheron", direction=None,
                 tolerance=22.5, bandwidth=None):
        if estimator not in ("matheron", "cressie", "dowd"):
            raise ValueError("estimator must be 'matheron', 'cressie' or 'dowd'")
        self.bins = int(bins)
        self.max_dist = max_dist
        self.estimator = estimator
        self.direction = direction
        self.tolerance = float(tolerance)
        self.bandwidth = bandwidth

    def fit(self, coords, values):
        X = _as_coords(coords)
        z = _as_values(values, len(X))
        n = len(X)
        if n < 2:
            raise ValueError("need at least 2 points")
        H = _pairwise(X)
        iu = np.triu_indices(n, 1)
        hij = H[iu]
        dz = z[iu[0]] - z[iu[1]]

        if self.direction is not None:
            if X.shape[1] != 2:
                raise ValueError("directional variograms are supported for 2-D coords only")
            dx = X[iu[0], 0] - X[iu[1], 0]
            dy = X[iu[0], 1] - X[iu[1], 1]
            ang = np.degrees(np.arctan2(dy, dx)) % 180.0
            target = float(self.direction) % 180.0
            diff = np.abs(((ang - target + 90.0) % 180.0) - 90.0)
            mask = diff <= self.tolerance
            hij, dz = hij[mask], dz[mask]
            if hij.size == 0:
                raise ValueError("no pairs fall within the directional tolerance")

        max_dist = self.max_dist if self.max_dist is not None else 0.5 * float(hij.max())
        edges = np.linspace(0.0, max_dist, self.bins + 1)
        idx = np.clip(np.digitize(hij, edges) - 1, 0, self.bins - 1)
        idx[hij > max_dist] = -1

        lags, gamma, counts = [], [], []
        for b in range(self.bins):
            m = idx == b
            cnt = int(m.sum())
            if cnt == 0:
                continue
            hb, dzb = hij[m], dz[m]
            if self.estimator == "matheron":
                g = 0.5 * float(np.mean(dzb ** 2))
            elif self.estimator == "cressie":
                g = 0.5 * float(np.mean(np.abs(dzb) ** 0.5) ** 4) / (0.457 + 0.494 / cnt)
            else:  # dowd
                g = 0.5 * (1.4826 * float(np.median(np.abs(dzb)))) ** 2
            lags.append(float(hb.mean()))
            gamma.append(g)
            counts.append(cnt)
        if not lags:
            raise ValueError("no non-empty lag bins; increase max_dist or reduce bins")
        self.lags_ = np.asarray(lags)
        self.gamma_ = np.asarray(gamma)
        self.counts_ = np.asarray(counts, dtype=int)
        self.bin_edges_ = edges
        self.n_pairs_ = int(hij.size)
        self._h_all, self._dz_all = hij, dz
        return self

    def cloud(self):
        """The raw (unbinned) ``(h, gamma)`` cloud: ``h_ij, 0.5*(z_i - z_j)**2``."""
        if not hasattr(self, "_h_all"):
            raise RuntimeError("fit() must be called first")
        return self._h_all, 0.5 * self._dz_all ** 2

    def __repr__(self):
        n = len(self.lags_) if hasattr(self, "lags_") else 0
        return f"ExperimentalVariogram(estimator={self.estimator!r}, n_bins={n})"


class Nugget:
    """Nugget estimate: the WLS line through the first 3 lags, extrapolated to h=0."""

    def fit(self, experimental, values=None):
        lags, gamma = experimental.lags_, experimental.gamma_
        k = min(3, len(lags))
        if k < 2:
            self.value_ = float(gamma[0]) if len(gamma) else 0.0
            return self
        A = np.column_stack([np.ones(k), lags[:k]])
        coef, *_ = np.linalg.lstsq(A, gamma[:k], rcond=None)
        self.value_ = float(max(coef[0], 0.0))
        return self

    def __float__(self):
        return float(self.value_)

    def __repr__(self):
        return f"Nugget(value_={getattr(self, 'value_', None)!r})"


class Sill:
    """Sill estimate: the mean gamma over the last third of lags, or the sample variance."""

    def __init__(self, method="plateau"):
        if method not in ("plateau", "variance"):
            raise ValueError("method must be 'plateau' or 'variance'")
        self.method = method

    def fit(self, experimental, values=None):
        if self.method == "variance":
            if values is None:
                raise ValueError("Sill(method='variance') needs values=")
            self.value_ = float(np.var(np.asarray(values, dtype=float), ddof=1))
        else:
            gamma = experimental.gamma_
            k = max(1, len(gamma) // 3)
            self.value_ = float(np.mean(gamma[-k:]))
        return self

    def __float__(self):
        return float(self.value_)

    def __repr__(self):
        return f"Sill(value_={getattr(self, 'value_', None)!r})"


class Range:
    """Range estimate: first lag crossing ``fraction`` of the sill, linearly interpolated."""

    def __init__(self, fraction=0.95):
        self.fraction = float(fraction)

    def fit(self, experimental, values=None):
        lags, gamma = experimental.lags_, experimental.gamma_
        if len(lags) < 2:
            self.value_ = float(lags[-1]) if len(lags) else 1.0
            return self
        nugget = float(Nugget().fit(experimental).value_)
        sill = float(Sill().fit(experimental).value_)
        target = nugget + self.fraction * max(sill - nugget, 0.0)
        idx = np.where(gamma >= target)[0]
        if len(idx) == 0:
            self.value_ = float(lags[-1])
            return self
        i = int(idx[0])
        if i == 0:
            self.value_ = float(lags[0])
        else:
            g0, g1 = gamma[i - 1], gamma[i]
            h0, h1 = lags[i - 1], lags[i]
            t = 0.0 if g1 == g0 else (target - g0) / (g1 - g0)
            self.value_ = float(h0 + t * (h1 - h0))
        return self

    def __float__(self):
        return float(self.value_)

    def __repr__(self):
        return f"Range(value_={getattr(self, 'value_', None)!r})"


class VariogramFitting:
    """Least-squares fit of a ``Semivariogram`` (or the best of several) to an experimental one."""

    def __init__(self, model="spherical", method="wls", fit_nugget=True):
        if method not in ("ols", "wls", "cressie"):
            raise ValueError("method must be 'ols', 'wls' or 'cressie'")
        self.model = model
        self.method = method
        self.fit_nugget = bool(fit_nugget)

    def _weights(self, counts):
        if self.method == "ols":
            return np.ones(len(counts))
        return np.sqrt(counts.astype(float))

    def _fit_single(self, exp, model_name):
        lags, gamma, counts = exp.lags_, exp.gamma_, exp.counts_
        w0 = self._weights(counts)
        nug0 = float(Nugget().fit(exp).value_)
        sill0 = max(float(Sill().fit(exp).value_), nug0 + 1e-6)
        rng0 = max(float(Range().fit(exp).value_), 1e-6)
        # weakly-structured (near-nugget) data leaves a flat ridge in (partial_sill, range)
        # space -- an unbounded optimizer can report an arbitrarily large "range" that is
        # not actually resolved by the data. Cap the search at generous but physically
        # interpretable multiples of what the experimental variogram can actually see.
        max_lag = float(lags[-1])
        max_level = float(max(gamma.max(), nug0, 1e-6))

        names, x0, lo, hi = [], [], [], []
        if self.fit_nugget:
            names.append("nugget"); x0.append(nug0); lo.append(0.0); hi.append(5.0 * max_level)
        if model_name in ("linear", "power"):
            names.append("slope")
            x0.append(max(sill0 / rng0, 1e-6)); lo.append(1e-12)
            hi.append(5.0 * max_level / max(max_lag, 1e-6))
            if model_name == "power":
                names.append("exponent"); x0.append(1.0); lo.append(1e-3); hi.append(1.999)
        else:
            # fit the partial sill (>= 0) rather than the total sill directly, so the
            # optimizer can never drive sill below nugget (a nonphysical decreasing gamma)
            names += ["partial_sill", "range"]
            x0 += [max(sill0 - nug0, 1e-6), min(rng0, 3.0 * max_lag)]
            lo += [1e-9, 1e-9]
            hi += [5.0 * max_level, 3.0 * max_lag]
            if model_name == "matern":
                names.append("nu"); x0.append(1.0); lo.append(0.05); hi.append(10.0)

        def build(theta):
            p = dict(zip(names, theta))
            c0 = p.get("nugget", 0.0)
            if model_name in ("linear", "power"):
                return Semivariogram(model=model_name, nugget=c0, sill=p["slope"],
                                      range=1.0, exponent=p.get("exponent", 1.0))
            return Semivariogram(model=model_name, nugget=c0, sill=c0 + p["partial_sill"],
                                  range=p["range"], nu=p.get("nu", 0.5))

        def resid(theta):
            v = build(theta)
            pred = v(lags)
            if self.method == "cressie":
                w = np.sqrt(counts.astype(float)) / np.maximum(pred, 1e-8)
            else:
                w = w0
            return w * (pred - gamma)

        res = optimize.least_squares(resid, x0, bounds=(lo, hi), method="trf")
        v = build(res.x)
        rss = float(np.sum((v(lags) - gamma) ** 2))
        n = len(lags)
        k = len(x0)
        aic = n * np.log(max(rss, 1e-300) / n) + 2 * k if n > 0 else np.inf
        return v, dict(zip(names, res.x)), rss, aic

    def fit(self, experimental):
        models = list(self.model) if isinstance(self.model, (list, tuple)) else [self.model]
        for m in models:
            if m not in _FIT_MODELS:
                raise ValueError(f"unknown model {m!r}")
        rows = []
        best = None
        for m in models:
            v, params, rss, aic = self._fit_single(experimental, m)
            rows.append({"model": m, "params": params, "rss": rss, "aic": aic})
            if best is None or aic < best[3]:
                best = (v, params, rss, aic)
        self.variogram_, self.params_, self.rss_, self.aic_ = best
        if len(models) > 1:
            self.table_ = rows
        return self

    def __repr__(self):
        return f"VariogramFitting(model={self.model!r}, method={self.method!r})"
