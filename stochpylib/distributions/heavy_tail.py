"""Heavy-tailed distributions, implemented from scratch.

General 4-parameter stable distributions have a closed-form characteristic function but no
closed-form density in general (only 3 special cases do: Normal, Cauchy, Levy). ``pdf``/``cdf``
for the general case are computed via numerical inversion of the characteristic function
(Gil-Pelaez theorem) using ``scipy.integrate.quad`` — the standard "from scratch" approach when
no closed density exists. The two exactly-solvable special cases are delegated to their closed
forms for speed and accuracy: ``alpha=2`` is always Gaussian (any ``beta``), and
``alpha=1, beta=0`` is Cauchy.

Sampling (``rvs``) uses the Chambers–Mallows–Leckie (CML) algorithm for ``alpha != 1``
(any ``beta``), whose constants were validated empirically against this class's own
closed-form characteristic function (Monte-Carlo noise level across symmetric and skewed
cases). For the ``alpha == 1`` corner — where no published closed-form sampler matches our
parametrization — ``rvs`` uses a cached **numerical quantile function**: the exact
Gil-Pelaez CDF is inverted once onto a monotone knot grid (PCHIP-refined), after which
draws are a vectorized table lookup with quantile error <= ~1e-3 of the scale (asserted in
tests). Both routes are fast; nothing here falls back to per-sample root finding.

``SubGaussian``/``SubExponential`` are tail-behavior *classes*, not single canonical
distributions — implemented here as specific, documented parametric families representing each
tail class (generalized normal with shape >= 2, and a shape-constrained Weibull respectively),
not as "the" sub-Gaussian/sub-exponential distribution (there isn't one).
"""

import numpy as np
from scipy import integrate, special
from scipy.interpolate import PchipInterpolator

from stochpylib.distributions._base import Distribution
from stochpylib.distributions.continuous import Cauchy as _Cauchy
from stochpylib.distributions.continuous import Normal as _Normal
from stochpylib.distributions.continuous import Weibull as _Weibull

import warnings

# class-level cache of numerical quantile tables for the alpha=1 corner:
# (alpha, beta, loc, scale) -> (q_grid, x_grid)
_QUANTILE_LUT_CACHE = {}


class StableDistribution(Distribution):
    """General 4-parameter stable distribution (alpha, beta, loc, scale)."""

    def __init__(self, alpha=2.0, beta=0.0, loc=0.0, scale=1.0):
        if not (0 < alpha <= 2):
            raise ValueError("alpha must be in (0, 2]")
        if not (-1 <= beta <= 1):
            raise ValueError("beta must be in [-1, 1]")
        self.alpha, self.beta, self.loc, self.scale = float(alpha), float(beta), float(loc), float(scale)
        # Exact special cases: alpha=2 is always Gaussian; alpha=1,beta=0 is Cauchy.
        self._gauss = _Normal(self.loc, self.scale * np.sqrt(2)) if self.alpha == 2 else None
        self._cauchy = _Cauchy(self.loc, self.scale) if (self.alpha == 1 and self.beta == 0) else None

    def cf(self, t):
        if self._gauss is not None:
            return self._gauss.cf(t)
        if self._cauchy is not None:
            return self._cauchy.cf(t)
        a, b, mu, c = self.alpha, self.beta, self.loc, self.scale
        if t == 0:
            return complex(1.0)
        if a == 1.0:
            omega = -(2.0 / np.pi) * np.log(abs(t))
        else:
            omega = np.tan(np.pi * a / 2.0)
        log_phi = 1j * mu * t - abs(c * t) ** a * (1 - 1j * b * np.sign(t) * omega)
        return complex(np.exp(log_phi))

    def pdf(self, x):
        if self._gauss is not None:
            return self._gauss.pdf(x)
        if self._cauchy is not None:
            return self._cauchy.pdf(x)
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        out = np.array([self._pdf_scalar(xi) for xi in x])
        return out.item() if scalar else out

    def _pdf_scalar(self, x):
        integrand = lambda t: (self.cf(t) * np.exp(-1j * t * x)).real
        val, _ = integrate.quad(integrand, 0, np.inf, limit=400)
        return max(val / np.pi, 0.0)

    def cdf(self, x):
        if self._gauss is not None:
            return self._gauss.cdf(x)
        if self._cauchy is not None:
            return self._cauchy.cdf(x)
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        out = np.array([self._cdf_scalar(xi) for xi in x])
        return out.item() if scalar else out

    def _cdf_scalar(self, x):
        def integrand(t):
            if t == 0:
                return 0.0
            return (self.cf(t) * np.exp(-1j * t * x)).imag / t

        val, _ = integrate.quad(integrand, 0, np.inf, limit=400)
        return float(np.clip(0.5 - val / np.pi, 0.0, 1.0))

    def ppf(self, q):
        if self._gauss is not None:
            return self._gauss.ppf(q)
        if self._cauchy is not None:
            return self._cauchy.ppf(q)
        return super().ppf(q)

    def mean(self):
        if self._gauss is not None:
            return self._gauss.mean()
        if self._cauchy is not None:
            return self._cauchy.mean()
        return self.loc if self.alpha > 1 else np.nan

    def var(self):
        if self._gauss is not None:
            return self._gauss.var()
        if self._cauchy is not None:
            return self._cauchy.var()
        return np.inf if self.alpha < 2 else self.scale**2 * 2

    def rvs(self, size=1, random_state=None):
        """Fast exact-or-validated sampling.

        - ``alpha == 2`` (any beta): Gaussian closed form.
        - ``alpha == 1, beta == 0``: Cauchy closed form.
        - ``alpha != 1``: Chambers–Mallows–Leckie (validated against the closed-form cf).
        - ``alpha == 1, beta != 0``: cached numerical quantile table built by inverting
          the exact Gil-Pelaez CDF once (quantile error asserted <= ~1e-3 of scale).
        """
        if self._gauss is not None:
            return self._gauss.rvs(size, random_state=random_state)
        if self._cauchy is not None:
            return self._cauchy.rvs(size, random_state=random_state)
        rng = np.random.default_rng(random_state)
        n = size if isinstance(size, int) else int(np.prod(size))
        a, b = self.alpha, self.beta
        if a != 1.0:
            v = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
            w = rng.exponential(1.0, size=n)
            zeta = b * np.tan(np.pi * a / 2.0)
            theta = np.arctan(zeta) / a
            x_std = (
                (1.0 + zeta**2) ** (1.0 / (2.0 * a))
                * np.sin(a * (v + theta)) / np.cos(v) ** (1.0 / a)
                * (np.cos(v - a * (v + theta)) / w) ** ((1.0 - a) / a)
            )
            out = self.loc + self.scale * x_std
            return out[0] if size == 1 else out
        # alpha == 1, beta != 0: numerical quantile lookup
        q_grid, x_grid = self._quantile_table()
        u = rng.uniform(size=n)
        out = np.interp(u, q_grid, x_grid)
        return out[0] if size == 1 else out

    def _quantile_table(self):
        """Monotone (q -> x) interpolation table for the alpha=1 corner.

        Three-part construction:

        1. **Central region** (|x - loc| <= 25*scale): exact Gil-Pelaez CDF evaluated on a
           grid, refined through monotone PCHIP — this is where the quadrature is reliable.
        2. **Tails**: alpha=1 stables have exact power-law asymptotics,
           ``1-F(x) ~ c(1+beta)/(pi*x)`` and ``F(x) ~ c(1-beta)/(pi*|x|)`` (c = scale),
           so quantiles beyond the central window are computed analytically from those
           formulas down to q = 1e-9 / up to 1 - 1e-9.

        The pieces are stitched into one strictly monotone curve; ``np.interp`` inverts it
        in O(n) per draw. Cached per parameter set. Quantile accuracy in the central
        region is asserted against root-refined truth in the test suite.
        """
        key = (
            round(self.alpha, 12), round(self.beta, 12),
            round(self.loc, 12), round(self.scale, 12),
        )
        hit = _QUANTILE_LUT_CACHE.get(key)
        if hit is not None:
            return hit

        c = self.scale
        tail_right = c * (1.0 + self.beta) / np.pi
        tail_left = c * (1.0 - self.beta) / np.pi
        window = 25.0 * c

        # --- central region: numeric CDF on a grid inside the reliable window ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gx = np.linspace(self.loc - window, self.loc + window, 129)
            gf = np.array([self._cdf_scalar(v) for v in gx])
        gf = np.clip(gf, 0.0, 1.0)
        gf = np.maximum.accumulate(gf)

        spline = PchipInterpolator(gx, gf)
        fine_x = np.linspace(gx[0], gx[-1], 100_001)
        fine_f = np.maximum.accumulate(np.clip(np.asarray(spline(fine_x), dtype=float), 0.0, 1.0))

        # --- analytic tails grafted onto the central endpoints ---
        f_left_edge, f_right_edge = float(fine_f[0]), float(fine_f[-1])
        edge_lo, edge_hi = float(fine_x[0]), float(fine_x[-1])

        q_left = np.geomspace(max(f_left_edge, 1e-12), 1e-9, 400)
        x_left = self.loc - tail_left / q_left
        q_left = np.minimum(q_left, f_left_edge)  # keep strictly below central start

        q_right = np.geomspace(max(1.0 - f_right_edge, 1e-12), 1e-9, 400)
        x_right = self.loc + tail_right / q_right
        q_right = 1.0 - q_right
        q_right = np.maximum(q_right, f_right_edge)

        all_q = np.concatenate([q_left[::-1], fine_f, q_right])
        all_x = np.concatenate([x_left[::-1], fine_x, x_right])
        order = np.argsort(all_q, kind="stable")
        all_q, all_x = all_q[order], all_x[order]
        keep = np.concatenate([[True], np.diff(all_q) > 0])
        result = (all_q[keep], all_x[keep])
        _QUANTILE_LUT_CACHE[key] = result
        return result

    @classmethod
    def _initial_guess(cls, data):
        return [1.5, 0.0, float(np.median(data)), float(np.std(data)) or 1.0]

    @classmethod
    def _param_bounds(cls):
        return [(0.1, 2.0), (-1.0, 1.0), (None, None), (1e-3, None)]


class AlphaStable(StableDistribution):
    """Symmetric (beta=0) alpha-stable distribution."""

    def __init__(self, alpha=2.0, loc=0.0, scale=1.0):
        super().__init__(alpha, 0.0, loc, scale)

    @classmethod
    def _initial_guess(cls, data):
        return [1.5, float(np.median(data)), float(np.std(data)) or 1.0]

    @classmethod
    def _param_bounds(cls):
        return [(0.1, 2.0), (None, None), (1e-3, None)]

    @classmethod
    def _generic_fit(cls, data, x0, bounds=None):
        # AlphaStable's constructor takes (alpha, loc, scale); base StableDistribution._generic_fit
        # works unmodified since it just calls cls(*params).
        return super()._generic_fit(data, x0, bounds)


class LevyDistribution(Distribution):
    """Levy(loc, scale): the alpha=0.5, beta=1 stable special case, which has a simple closed
    form (used directly here rather than through StableDistribution's general numerical
    inversion, both for clarity and as a correctness check in tests)."""

    def __init__(self, loc=0.0, scale=1.0):
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.loc, self.scale = float(loc), float(scale)

    def support(self):
        return (self.loc, np.inf)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        c = self.scale
        with np.errstate(divide="ignore", invalid="ignore"):
            z = x - self.loc
            out = np.sqrt(c / (2 * np.pi)) * np.exp(-c / (2 * z)) / z**1.5
        return np.where(x > self.loc, out, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = special.erfc(np.sqrt(self.scale / (2 * (x - self.loc))))
        return np.where(x > self.loc, out, 0.0)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return self.loc + self.scale / (2 * special.erfcinv(q) ** 2)

    def mean(self):
        return np.inf

    def var(self):
        return np.inf

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        z = rng.normal(size=size)
        return self.loc + self.scale / z**2

    @classmethod
    def fit(cls, data):
        data = np.asarray(data, dtype=float)
        loc = float(np.min(data)) - 1e-6
        scale = float(np.median(data - loc) * 2 / special.erfcinv(0.5) ** 2)
        return cls(loc, scale)


class SubGaussian(Distribution):
    """Generalized normal / exponential-power distribution with shape >= 2, representing the
    sub-Gaussian tail class (shape=2 is Normal itself; shape>2 has lighter-than-Gaussian tails).
    A specific, documented parametric choice — not "the" sub-Gaussian distribution."""

    def __init__(self, mu=0.0, alpha=1.0, beta=2.0):
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if beta < 2:
            raise ValueError("beta must be >= 2 for sub-Gaussian tail behavior")
        self.mu, self.alpha, self.beta = float(mu), float(alpha), float(beta)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        mu, a, b = self.mu, self.alpha, self.beta
        return (b / (2 * a * special.gamma(1 / b))) * np.exp(-((np.abs(x - mu) / a) ** b))

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        mu, a, b = self.mu, self.alpha, self.beta
        z = np.abs(x - mu) / a
        g = special.gammainc(1 / b, z**b)
        return np.where(x >= mu, 0.5 + 0.5 * g, 0.5 - 0.5 * g)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        mu, a, b = self.mu, self.alpha, self.beta
        sign = np.where(q >= 0.5, 1.0, -1.0)
        g = special.gammaincinv(1 / b, 2 * np.abs(q - 0.5))
        return mu + sign * a * g ** (1 / b)

    def mean(self):
        return self.mu

    def var(self):
        a, b = self.alpha, self.beta
        return a**2 * special.gamma(3 / b) / special.gamma(1 / b)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        g = rng.gamma(1 / self.beta, 1.0, size=size)
        sign = rng.choice([-1.0, 1.0], size=size)
        return self.mu + self.alpha * sign * g ** (1 / self.beta)

    @classmethod
    def _initial_guess(cls, data):
        return [float(np.median(data)), float(np.std(data)) or 1.0, 2.0]

    @classmethod
    def _param_bounds(cls):
        return [(None, None), (1e-3, None), (2.0, None)]


class SubExponential(_Weibull):
    """Weibull with shape in (0, 1] — shape=1 is Exponential itself, shape<1 has
    heavier-than-exponential tails (within the sub-exponential class). A specific, documented
    parametric choice — not "the" sub-exponential distribution."""

    def __init__(self, shape=1.0, scale=1.0):
        if not (0 < shape <= 1):
            raise ValueError("shape must be in (0, 1] for sub-exponential tail behavior")
        super().__init__(shape, scale)

    @classmethod
    def _initial_guess(cls, data):
        return [0.8, float(np.mean(data))]

    @classmethod
    def _param_bounds(cls):
        return [(1e-3, 1.0), (1e-3, None)]
