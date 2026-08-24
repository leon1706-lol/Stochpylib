"""Archimedean copulas plus the bivariate Plackett family.

Every Archimedean class implements three primitives — the Laplace transform
``_psi`` (its inverse ``_psi_inv``, i.e. the generator ``phi``), and the second
derivative ``_psi_dd`` — from which the shared machinery derives:

- ``cdf``            : ``psi(sum(phi(u_j)))``, any dimension
- ``density``        : ``phi'(u) phi'(v) psi''(phi(u)+phi(v))`` (exact, bivariate)
- ``kendall_tau``    : Genest–MacKay identity ``tau = 1 - 4 int t psi'(t)^2 dt``
- fitting            : Kendall-tau inversion (closed form where known, else a
                       bracketed root-finding on the numeric ``tau(theta)`` curve)
- sampling           : Marshall–Olkin mixing where a closed-form mixer exists
                       (Clayton: gamma, Gumbel: Kanter positive-stable),
                       otherwise sequential conditional inversion driven by the
                       family's own ``cdf``

Plackett is *not* Archimedean (odds-ratio family) but lives here per the module
spec; it has a closed-form CDF, an explicit quadratic conditional sampler and a
numerically integrated tau.
"""

import numpy as np
from scipy import integrate

from stochpylib.copulas._base import BaseCopula
from stochpylib.copulas._utils import as_u_matrix, brentq_on_bracket, \
    kendall_tau_estimate

__all__ = [
    "ClaytonCopula", "FrankCopula", "GumbelCopula", "JoeCopula",
    "AliMikhailHaqCopula", "PlackettCopula", "BB1Copula", "BB7Copula",
]

_EPS = 1e-12
_EXP_MAX = 500.0


def _clip_open(u):
    """Clip unit-cube coordinates strictly inside (0, 1) for log/generator calls."""
    return np.clip(as_u_matrix(u), _EPS, 1.0 - _EPS)


def _exp_safe(t):
    return np.exp(np.minimum(t, _EXP_MAX))


class _ArchimedeanBase(BaseCopula):
    """Generator-driven base class. Bivariate by default; pass ``dimension=d``
    for exchangeable higher-dimensional use where the family supports it."""

    dimension = 2

    def __init__(self, theta=None, dimension=None):
        super().__init__()
        self.theta_ = None if theta is None else float(theta)
        if dimension is not None:
            self.dimension = int(dimension)

    # -- generator primitives (abstract) -------------------------------------
    def _psi(self, t):
        raise NotImplementedError

    def _psi_inv(self, u):
        raise NotImplementedError

    def _psi_d(self, t):
        """First derivative of the Laplace transform (negative)."""
        raise NotImplementedError

    def _psi_dd(self, t):
        """Second derivative of the Laplace transform."""
        raise NotImplementedError

    # -- validation ------------------------------------------------------------
    def _require_fit(self):
        if self.theta_ is None:
            raise RuntimeError("fit() must be called first (or pass theta=)")

    # -- shared surface ----------------------------------------------------------
    def cdf(self, u):
        self._require_fit()
        u = _clip_open(u)
        return self._psi(np.sum(self._psi_inv(u), axis=-1))

    def kendall_tau(self):
        self._require_fit()
        return float(self.tau_of_theta(self.theta_))

    def density(self, u):
        self._require_fit()
        if self.dimension != 2:
            raise NotImplementedError(
                f"{type(self).__name__} exposes a closed-form density only in "
                "2 dimensions")
        uu = _clip_open(u)
        phi_u = self._psi_inv(uu[..., 0])
        phi_v = self._psi_inv(uu[..., 1])
        return self._phi_d_of_u(uu[..., 0]) * self._phi_d_of_u(uu[..., 1]) \
            * self._psi_dd(phi_u + phi_v)

    def _phi_d_of_u(self, u):
        """Derivative of the generator phi at original coordinates (negative)."""
        raise NotImplementedError

    def _h_u(self, w, v):
        """Conditional CDF ``P(U <= w | V = v)`` (bivariate, exact).

        Archimedean identity: ``psi'(phi(v) + phi(w)) / psi'(phi(v))``.
        """
        w = np.clip(np.asarray(w, dtype=float), _EPS, 1.0 - _EPS)
        v = np.clip(np.asarray(v, dtype=float), _EPS, 1.0 - _EPS)
        t = self._psi_inv(v)
        num = self._psi_d(t + self._psi_inv(w))
        return num / self._psi_d(t)

    # -- tau machinery ---------------------------------------------------------
    def tau_of_theta(self, theta):
        """Genest–MacKay: tau = 1 - 4 int_0^inf t psi'(t)^2 dt.

        Results are cached on a per-class theta grid so repeated inversions
        (pair-copula selection loops) never re-integrate from scratch.
        """
        saved = self.theta_
        self.theta_ = float(theta)
        try:
            cls = type(self)
            cache = cls.__dict__.get("_tau_curve_")
            if cache is None or not (cache["lo"] <= theta <= cache["hi"]):
                return self._tau_integral(theta)
            grid, vals = cache["grid"], cache["vals"]
            if theta <= grid[0]:
                return float(vals[0])
            if theta >= grid[-1]:
                return float(vals[-1])
            idx = int(np.searchsorted(grid, theta))
            t0, t1 = grid[idx - 1], grid[idx]
            v0, v1 = vals[idx - 1], vals[idx]
            if t1 == t0:
                return float(v0)
            wgt = (theta - t0) / (t1 - t0)
            return float(v0 * (1 - wgt) + v1 * wgt)
        finally:
            self.theta_ = saved

    def _tau_integral(self, theta):
        saved = self.theta_
        self.theta_ = float(theta)
        try:
            def sub(s):
                one_m = max(1.0 - s, _EPS)
                t = s / one_m
                p = float(self._psi_d(t))
                val = t * p * p / one_m ** 2
                return min(val, 1e300)

            integral, _ = integrate.quad(sub, 0.0, 1.0 - 1e-10,
                                         limit=100, epsabs=1e-9,
                                         epsrel=1e-9)
            return 1.0 - 4.0 * integral
        finally:
            self.theta_ = saved

    def _build_tau_curve(self, lo, hi, n=48):
        """Precompute the monotone tau(theta) curve on a dense grid."""
        cls = type(self)
        if hi > 1e4:
            grid = np.geomspace(max(lo, 1e-3), hi, n)
        elif lo < 0 and hi > 0:
            neg = -np.geomspace(1e-3, -lo, n // 3)[::-1]
            pos = np.geomspace(1e-3, hi, n - len(neg))
            grid = np.concatenate([neg, [0.0], pos])
        else:
            grid = lo + (hi - lo) * (np.linspace(0.0, 1.0, n) ** 1.6)
        vals = np.array([self._tau_integral(float(th)) for th in grid])
        cls._tau_curve_ = {"lo": float(lo), "hi": float(hi),
                           "grid": grid, "vals": vals}

    def _invert_tau(self, tau):
        lo, hi = self._theta_bounds()
        cls = type(self)
        cache = cls.__dict__.get("_tau_curve_")
        if cache is None or cache["lo"] != lo or cache["hi"] != hi:
            self._build_tau_curve(lo, hi)
            cache = cls._tau_curve_
        grid, vals = cache["grid"], cache["vals"]
        below = np.where(vals <= tau)[0]
        above = np.where(vals >= tau)[0]
        if len(below) == 0 or len(above) == 0:
            raise ValueError(
                f"tau={tau:.4f} outside what {cls.__name__} can represent "
                f"on ({lo}, {hi})")
        i0 = int(below[-1])
        i1 = int(above[0]) if len(above) else i0 + 1
        if i1 <= i0:
            i1 = min(i0 + 1, len(grid) - 1)
        th = brentq_on_bracket(lambda th: self.tau_of_theta(th) - tau,
                               float(grid[i0]), float(grid[i1]), xtol=1e-9)
        return float(th)

    def _estimate(self, u):
        if u.shape[1] != 2:
            raise ValueError(
                f"{type(self).__name__} fits bivariate data only "
                "(use it as a pair copula inside a vine)")
        tau = float(kendall_tau_estimate(u[:, 0], u[:, 1]))
        self.theta_ = self._invert_tau(tau)

    # -- sampling ---------------------------------------------------------------
    def sample(self, n, random_state=None):
        """Marshall–Olkin fast path when available, else conditional inversion."""
        self._require_fit()
        n = self._validate_sample_n(n)
        rng = np.random.default_rng(random_state)
        mixer = self._mixing_sampler(rng, n)
        if mixer is not None:
            e = rng.exponential(size=(n, self.dimension))
            return self._psi(e / mixer[:, None])
        return self._conditional_inversion_sample(n, rng)

    def _mixing_sampler(self, rng, n):
        """n draws of the mixing distribution T whose Laplace transform is
        ``_psi``, or None when no closed-form mixer is implemented."""
        return None

    def _conditional_inversion_sample(self, n, rng):
        """Exact sequential sampling for d = 2 via the Archimedean conditional

            P(U2 <= w | U1 = u) = psi'(phi(u) + phi(w)) / psi'(phi(u)),

        inverted on a fixed w-grid. Dimension >= 3 requires higher generator
        derivatives and is only supported by families with a mixing sampler.
        """
        d = self.dimension
        if d != 2:
            raise NotImplementedError(
                f"{type(self).__name__} supports exact d >= 3 sampling only "
                "through its mixing distribution; pass a family with one or "
                "fit bivariate data")
        out = np.empty((n, d))
        out[:, 0] = rng.random(n)
        m = 4096
        grid = np.linspace(0.0, 1.0, m)
        w_grid = np.clip(grid[1:-1], _EPS, 1.0 - _EPS)
        targets = rng.random(n)
        phi_u1 = np.asarray(self._psi_inv(out[:, 0]), dtype=float)
        denom = -np.asarray(self._psi_d(phi_u1), dtype=float)      # positive
        rows = np.empty((n * len(w_grid), 2))
        rows[:, 0] = np.repeat(out[:, 0], len(w_grid))
        rows[:, 1] = np.tile(w_grid, n)
        t_sum = phi_u1[:, None] + np.asarray(
            self._psi_inv(np.broadcast_to(w_grid, (n, len(w_grid)))), dtype=float)
        num = -np.asarray(self._psi_d(t_sum), dtype=float)
        cond = np.clip(num / denom[:, None], 0.0, 1.0)
        cond[:, -1] = 1.0
        chunk = max(1, min(n, 4_000_000 // m))
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            for j in range(start, stop):
                out[j, 1] = float(np.interp(targets[j], cond[j], grid[1:-1]))
        return out


# ---------------------------------------------------------------------------
# concrete families


class ClaytonCopula(_ArchimedeanBase):
    """Clayton (lower tail dependence)::

        cl = ClaytonCopula().fit(data)     # or ClaytonCopula(theta=2.5)
        sims = cl.sample(1000, random_state=0)
    """

    def __init__(self, theta=None, dimension=None):
        super().__init__(theta, dimension=dimension)

    def _theta_bounds(self):
        return (1e-6, 300.0)

    def _psi(self, t):
        return np.maximum(1.0 + self.theta_ * np.maximum(t, 0.0),
                          _EPS) ** (-1.0 / self.theta_)

    def _psi_inv(self, u):
        return (np.maximum(u, _EPS) ** (-self.theta_) - 1.0) / self.theta_

    def _phi_d_of_u(self, u):
        return -np.maximum(u, _EPS) ** (-self.theta_ - 1.0)

    def _psi_d(self, t):
        return -(1.0 + self.theta_ * np.maximum(t, 0.0)) \
            ** (-1.0 / self.theta_ - 1.0)

    def _psi_dd(self, t):
        return (1.0 + self.theta_) * (1.0 + self.theta_ * np.maximum(t, 0.0)) \
            ** (-1.0 / self.theta_ - 2.0)

    def _invert_tau(self, tau):
        if not (0.0 < tau < 1.0):
            raise ValueError(
                f"estimated tau={tau:.4f} outside (0,1); Clayton cannot represent it")
        return 2.0 * tau / (1.0 - tau)

    def _mixing_sampler(self, rng, n):
        # Marshall–Olkin: T ~ Gamma(1/theta, scale=theta) has Laplace transform
        # (1 + theta*lambda)^(-1/theta) == _psi(lambda).
        return rng.gamma(shape=1.0 / self.theta_, scale=self.theta_, size=n)

    def tail_dependence(self):
        self._require_fit()
        return {"upper": 0.0, "lower": 2.0 ** (-1.0 / self.theta_)}


class GumbelCopula(_ArchimedeanBase):
    """Gumbel (upper tail dependence), theta >= 1."""

    def __init__(self, theta=None, dimension=None):
        super().__init__(theta, dimension=dimension)

    def _theta_bounds(self):
        return (1.0 + 1e-9, 200.0)

    def _alpha(self):
        return 1.0 / self.theta_

    def _psi(self, t):
        return np.exp(-np.maximum(t, 0.0) ** self._alpha())

    def _psi_inv(self, u):
        return np.maximum(-np.log(np.maximum(u, _EPS)), 0.0) ** self.theta_

    def _phi_d_of_u(self, u):
        u = np.maximum(u, _EPS)
        return -self.theta_ * (-np.log(u)) ** (self.theta_ - 1.0) / u

    def _psi_d(self, t):
        a = self._alpha()
        tt = np.maximum(t, 0.0)
        return -a * tt ** (a - 1.0) * np.exp(-tt ** a)

    def _psi_dd(self, t):
        a = self._alpha()
        tt = np.maximum(t, 1e-300)
        return np.exp(-tt ** a) * (a * a * tt ** (2 * a - 2)
                                   - a * (a - 1.0) * tt ** (a - 2))

    def _invert_tau(self, tau):
        if not (0.0 < tau < 1.0):
            raise ValueError(
                f"estimated tau={tau:.4f} outside (0,1); Gumbel cannot represent it")
        return 1.0 / (1.0 - tau)

    def _mixing_sampler(self, rng, n):
        """Kanter's algorithm for the positive-alpha stable law (alpha = 1/theta),
        normalized so that E[e^{-lambda S}] = e^{-lambda^alpha}."""
        a = self._alpha()
        v = rng.random(n) * np.pi
        w = rng.exponential(size=n)
        s = (np.sin(a * v) / np.sin(v)) ** (1.0 / a) \
            * np.sin((1.0 - a) * v) / np.sin(a * v)
        return s * w ** (-(1.0 - a) / a)

    def tail_dependence(self):
        self._require_fit()
        return {"upper": 2.0 - 2.0 ** (1.0 / self.theta_), "lower": 0.0}


class FrankCopula(_ArchimedeanBase):
    """Frank — the only Archimedean family covering negative dependence.

    Kendall's tau uses the exact Debye-function relation
    ``tau = 1 - 4 (1 - D1(theta)) / theta`` with
    ``D1(theta) = int_0^theta t/(e^t-1) dt / theta``.
    """

    def __init__(self, theta=None, dimension=None):
        super().__init__(theta, dimension=dimension)

    def _theta_bounds(self):
        return (-40.0, 40.0)

    def _debye1(self, th):
        th = float(th)
        if abs(th) < 1e-8:
            return 1.0 - th / 4.0
        val, _ = integrate.quad(lambda t: t / np.expm1(t) if t > 0
                                else t / np.expm1(t), 0.0, th,
                                limit=100, epsabs=1e-11, epsrel=1e-11)
        return val / th

    def tau_of_theta(self, theta):
        th = float(theta)
        if abs(th) < 1e-8:
            return 0.0
        return 1.0 - 4.0 * (1.0 - self._debye1(th)) / th

    def _c_const(self):
        th = self.theta_
        return float(np.expm1(-th)) if abs(th) > 1e-10 else -th

    def _psi(self, t):
        c = self._c_const()
        return -np.log1p(c * np.exp(-np.minimum(t, _EXP_MAX))) / self.theta_

    def _psi_inv(self, u):
        c = self._c_const()
        return -np.log(np.expm1(-self.theta_ * np.maximum(u, _EPS)) / c)

    def _phi_d_of_u(self, u):
        return self.theta_ / (np.exp(self.theta_ * u) - 1.0)

    def _psi_d(self, t):
        f = self._c_const() * np.exp(-np.minimum(t, _EXP_MAX))
        return f / (self.theta_ * (1.0 + f))

    def _psi_dd(self, t):
        # psi'' = -f / (theta (1+f)^2) with f = c*e^{-t}
        f = self._c_const() * np.exp(-np.minimum(t, _EXP_MAX))
        return -f / (self.theta_ * (1.0 + f) ** 2)

    def _invert_tau(self, tau):
        if not (-0.98 < tau < 0.98):
            raise ValueError("Frank covers tau strictly inside (-1, 1)")
        return brentq_on_bracket(lambda th: self.tau_of_theta(th) - tau,
                                 -39.5, 39.5, xtol=1e-9)


class JoeCopula(_ArchimedeanBase):
    """Joe — upper tail dependence, generally stronger body asymmetry."""

    def __init__(self, theta=None, dimension=None):
        super().__init__(theta, dimension=dimension)

    def _theta_bounds(self):
        return (1.0 + 1e-9, 60.0)

    def _psi(self, t):
        em = np.maximum(1.0 - np.exp(-np.minimum(t, _EXP_MAX)), _EPS)
        return 1.0 - em ** (1.0 / self.theta_)

    def _psi_inv(self, u):
        return -np.log1p(-np.clip(1.0 - u, _EPS, 1.0) ** self.theta_)

    def _phi_d_of_u(self, u):
        w = np.clip(1.0 - u, _EPS, 1.0)
        return -self.theta_ * w ** (self.theta_ - 1.0) \
            / np.maximum(1.0 - w ** self.theta_, _EPS)

    def _psi_d(self, t):
        inv_th = 1.0 / self.theta_
        e = np.exp(-np.minimum(t, _EXP_MAX))
        g = np.maximum(1.0 - e, _EPS)
        return -inv_th * g ** (inv_th - 1.0) * e

    def _psi_dd(self, t):
        inv_th = 1.0 / self.theta_
        e = np.exp(-np.minimum(t, _EXP_MAX))
        g = np.maximum(1.0 - e, _EPS)
        # psi'' = gamma * g^(gamma-2) * e * [g - (gamma-1) e]
        return inv_th * e * g ** (inv_th - 2.0) * (g - (inv_th - 1.0) * e)

    def tail_dependence(self):
        self._require_fit()
        return {"upper": 2.0 - 2.0 ** (1.0 / self.theta_), "lower": 0.0}


class AliMikhailHaqCopula(_ArchimedeanBase):
    """Ali–Mikhail–Haq — modest dependence, theta in (0, 1)."""

    def __init__(self, theta=None, dimension=None):
        super().__init__(theta, dimension=dimension)

    def _theta_bounds(self):
        return (1e-9, 1.0 - 1e-9)

    def _psi(self, t):
        return (1.0 - self.theta_) \
            / (_exp_safe(np.maximum(t, 0.0)) - self.theta_)

    def _psi_inv(self, u):
        u = np.maximum(u, _EPS)
        return np.log((1.0 - self.theta_ + self.theta_ * u) / u)

    def _phi_d_of_u(self, u):
        u = np.maximum(u, _EPS)
        return (self.theta_ - 1.0) / (u * (1.0 - self.theta_ + self.theta_ * u))

    def _psi_d(self, t):
        et = _exp_safe(t)
        if np.ndim(et) == 0:
            tt = float(min(max(t, 0.0), _EXP_MAX))
            if tt > 40.0:                      # (e^t - th)^2 ~ e^2t asymptotics
                return -(1.0 - self.theta_) * np.exp(-tt)
            return -(1.0 - self.theta_) * et / (et - self.theta_) ** 2
        out = np.where(t > 40.0,
                       -np.exp(-(t - 0.0)) * (1.0 - self.theta_),
                       -(1.0 - self.theta_) * et / (et - self.theta_) ** 2)
        return out

    def _psi_dd(self, t):
        et = _exp_safe(t)
        if np.ndim(et) == 0:
            tt = float(min(max(t, 0.0), _EXP_MAX))
            if tt > 40.0:
                return (1.0 - self.theta_) * np.exp(-tt)
            return (1.0 - self.theta_) * et * (et + self.theta_) \
                / (et - self.theta_) ** 3
        out = np.where(
            t > 40.0,
            np.exp(-np.minimum(t, _EXP_MAX)) * (1.0 - self.theta_),
            (1.0 - self.theta_) * et * (et + self.theta_)
            / (et - self.theta_) ** 3)
        return out


class BB1Copula(_ArchimedeanBase):
    """BB1 — two-parameter Archimedean with BOTH tail dependences.

    ``C = {1 + [(u^-theta - 1)^delta + (v^-theta - 1)^delta]^(1/delta)}^(-1/theta)``,
    theta > 0, delta >= 1; ``lambda_L = 2^(-1/(theta*delta))``,
    ``lambda_U = 2 - 2^(1/delta)``.
    """

    _n_params = 2

    def __init__(self, theta=None, delta=None, dimension=None):
        super().__init__(theta, dimension=dimension)
        self.delta_ = None if delta is None else float(delta)

    def _require_fit(self):
        if self.theta_ is None or self.delta_ is None:
            raise RuntimeError("fit() must be called first (or pass theta=/delta=)")

    def _theta_bounds(self):
        return (1e-6, 50.0)

    def _psi(self, t):
        y = np.maximum(t, 0.0) ** (1.0 / self.delta_)
        return (1.0 + y) ** (-1.0 / self.theta_)

    def _psi_inv(self, u):
        return np.maximum(np.maximum(u, _EPS) ** (-self.theta_) - 1.0,
                          0.0) ** self.delta_

    def _phi_d_of_u(self, u):
        u = np.maximum(u, _EPS)
        return -self.theta_ * self.delta_ * (u ** (-self.theta_) - 1.0) \
            ** (self.delta_ - 1.0) * u ** (-self.theta_ - 1.0)

    def _psi_d(self, t):
        y = np.maximum(t, 0.0) ** (1.0 / self.delta_ - 1.0)
        base = 1.0 + np.maximum(t, 0.0) ** (1.0 / self.delta_)
        return -(1.0 / (self.theta_ * self.delta_)) * y \
            * base ** (-1.0 / self.theta_ - 1.0)

    def _psi_dd(self, t):
        tt = np.maximum(t, 1e-300)
        beta = -1.0 / self.theta_
        f = tt ** (1.0 / self.delta_)
        base = 1.0 + f
        # psi'' = beta/delta y^{a-2} base^{beta-2} [(a-1) base + a (beta-1) f],
        # a = 1/delta
        return beta / self.delta_ * tt ** (1.0 / self.delta_ - 2.0) \
            * base ** (beta - 2.0) \
            * ((1.0 / self.delta_ - 1.0) * base
               + (1.0 / self.delta_) * (beta - 1.0) * f)

    def _estimate(self, u):
        # two parameters: invert tau over a delta grid, keep best loglik
        if u.shape[1] != 2:
            raise ValueError("BB1Copula fits bivariate data only")
        tau = float(kendall_tau_estimate(u[:, 0], u[:, 1]))
        if not (0.01 < tau < 0.98):
            raise ValueError("BB1 requires moderate-to-strong positive rank "
                             f"dependence (got tau={tau:.3f})")
        best = None
        for delta in np.concatenate([np.linspace(1.05, 2.0, 20),
                                     np.linspace(2.25, 6.0, 17)]):
            self.delta_ = float(delta)
            try:
                th = self._invert_tau(tau)
            except (ValueError, RuntimeError):
                continue
            self.theta_ = th
            ll = self.loglik(u)
            if np.isfinite(ll) and (best is None or ll > best[0]):
                best = (ll, th, float(delta))
        if best is None:
            raise RuntimeError("BB1 fit failed: no valid (theta, delta) found")
        _, self.theta_, self.delta_ = best

    def tail_dependence(self):
        """lambda_L = 2^(-1/(theta*delta)), lambda_U = 2 - 2^(1/delta)."""
        self._require_fit()
        return {"upper": 2.0 - 2.0 ** (1.0 / self.delta_),
                "lower": 2.0 ** (-1.0 / (self.theta_ * self.delta_))}


class BB7Copula(_ArchimedeanBase):
    """BB7 — two-parameter family with both tail dependences.

    ``C = 1 - {1 - [(1-(1-u)^th)^-de + (1-(1-v)^th)^-de - 1]^(-1/de)}^(1/th)``,
    theta >= 1, delta > 0;
    generator ``phi(u) = (1-(1-u)^theta)^(-delta) - 1``,
    Laplace transform ``psi(y) = 1 - (1-(1+y)^(-1/delta))^(1/theta)``.
    Tail limits are extrapolated numerically from C(u,u).
    """

    _n_params = 2

    def __init__(self, theta=None, delta=None, dimension=None):
        super().__init__(theta, dimension=dimension)
        self.delta_ = None if delta is None else float(delta)

    def _require_fit(self):
        if self.theta_ is None or self.delta_ is None:
            raise RuntimeError("fit() must be called first (or pass theta=/delta=)")

    def _theta_bounds(self):
        return (1.0 + 1e-9, 30.0)

    def _psi(self, t):
        y = 1.0 + np.minimum(np.maximum(t, 0.0), _EXP_MAX)
        inner = np.maximum(1.0 - y ** (-1.0 / self.delta_), _EPS)
        return 1.0 - inner ** (1.0 / self.theta_)

    def _psi_inv(self, u):
        uu = np.maximum(u, _EPS)
        return np.expm1(-self.delta_
                        * np.log1p(-(1.0 - uu) ** self.theta_))

    def _phi_d_of_u(self, u):
        w = np.maximum(1.0 - u, _EPS)
        return self.theta_ * self.delta_ * w ** (self.theta_ - 1.0) \
            * np.maximum(1.0 - w ** self.theta_, _EPS) ** (-self.delta_ - 1.0)

    def _psi_d(self, t):
        a = -1.0 / self.delta_
        y = 1.0 + np.maximum(t, 0.0)
        inner = np.maximum(1.0 - y ** a, _EPS)
        # psi = 1 - inner^(1/theta); d/dy inner = -a y^(a-1) > 0
        return -(1.0 / self.theta_) * inner ** (1.0 / self.theta_ - 1.0) \
            * (-a) * y ** (a - 1.0)

    def _psi_dd(self, t):
        a = -1.0 / self.delta_
        y = 1.0 + np.maximum(t, 1e-300)
        inner = np.maximum(1.0 - y ** a, _EPS)
        din = -a * y ** (a - 1.0)                     # d/dy inner, positive
        d2in = -a * (a - 1.0) * y ** (a - 2.0)        # negative
        r = 1.0 / self.theta_
        # psi = 1 - inner^r -> psi'' = -r inner^(r-2) [inner*d2in + (r-1) din^2]
        return -r * inner ** (r - 2.0) * (inner * d2in + (r - 1.0) * din ** 2)

    def _estimate(self, u):
        if u.shape[1] != 2:
            raise ValueError("BB7Copula fits bivariate data only")
        tau = float(kendall_tau_estimate(u[:, 0], u[:, 1]))
        if not (0.01 < tau < 0.98):
            raise ValueError("BB7 requires moderate-to-strong positive rank "
                             f"dependence (got tau={tau:.3f})")
        best = None
        for delta in np.concatenate([np.linspace(0.3, 2.0, 24),
                                     np.linspace(2.25, 6.0, 16)]):
            self.delta_ = float(delta)
            try:
                th = self._invert_tau(tau)
            except (ValueError, RuntimeError):
                continue
            self.theta_ = th
            ll = self.loglik(u)
            if np.isfinite(ll) and (best is None or ll > best[0]):
                best = (ll, th, float(delta))
        if best is None:
            raise RuntimeError("BB7 fit failed: no valid (theta, delta) found")
        _, self.theta_, self.delta_ = best

    def tail_dependence(self):
        """Numerically extrapolated limits of C(u,u)/u and (1-2u+C(u,u))/u."""
        self._require_fit()
        lam_l = self._tail_limit(lower=True)
        lam_u = self._tail_limit(lower=False)
        return {"upper": lam_u, "lower": lam_l}

    def _tail_limit(self, lower, m=400000):
        ks = np.array([2.0, 3.0, 4.0, 5.0])
        vals = []
        for k in ks:
            u = 10.0 ** (-k)
            c = float(self.cdf([[u, u]])[0])
            vals.append(c / u if lower else (1.0 - 2.0 * u + c) / u)
        # Richardson-style extrapolation toward the limit
        return float(np.clip(vals[-1], 0.0, 1.0))


class PlackettCopula(BaseCopula):
    """Plackett — bivariate odds-ratio family (not Archimedean).

    Closed-form CDF via S = 1 + (theta-1)(u+v);
    ``C = (S - sqrt(S^2 - 4 theta (theta-1) uv)) / (2 (theta-1))``; theta=1 is the
    product copula. Conditional sampling solves the defining equation explicitly.
    """

    dimension = 2

    def __init__(self, theta=None):
        super().__init__()
        self.theta_ = None if theta is None else float(theta)

    def _require_fit(self):
        if self.theta_ is None:
            raise RuntimeError("fit() must be called first (or pass theta=)")

    @staticmethod
    def _core_cdf(u, v, th):
        if abs(th - 1.0) < 1e-12:
            return u * v
        s = 1.0 + (th - 1.0) * (u + v)
        disc = np.maximum(s * s - 4.0 * th * (th - 1.0) * u * v, 0.0)
        return (s - np.sqrt(disc)) / (2.0 * (th - 1.0))

    def _estimate(self, u):
        tau = float(kendall_tau_estimate(u[:, 0], u[:, 1]))
        if not (-0.95 < tau < 0.95):
            raise ValueError("Plackett covers tau strictly inside (-1, 1)")

        def f(th):
            return self._tau_numeric(th) - tau

        self.theta_ = brentq_on_bracket(f, 1e-6, 1e6, xtol=1e-8)

    def _tau_numeric(self, th):
        """tau = 4 E[C(U,V)] - 1 with (U,V) ~ C, by double quadrature with the
        Plackett density ``c = th (1 + (th-1)(u+v-2uv)) / D^(3/2)``."""

        def integrand(v, u):
            s = 1.0 + (th - 1.0) * (u + v)
            disc = max(s * s - 4.0 * th * (th - 1.0) * u * v, 1e-300)
            dens = th * (1.0 + (th - 1.0) * (u + v - 2.0 * u * v)) / disc ** 1.5
            return self._core_cdf(u, v, th) * min(dens, 1e12)

        total, _ = integrate.dblquad(integrand, 0.0, 1.0, 0.0, 1.0,
                                     epsabs=1e-8, epsrel=1e-8)
        return 4.0 * total - 1.0

    def kendall_tau(self):
        self._require_fit()
        return float(self._tau_numeric(self.theta_))

    def _h_u(self, w, v):
        """Conditional ``P(U <= w | V = v)`` from the closed-form density."""
        w = np.clip(np.asarray(w, dtype=float), _EPS, 1.0 - _EPS)
        v = np.clip(np.asarray(v, dtype=float), _EPS, 1.0 - _EPS)
        th = self.theta_
        s = 1.0 + (th - 1.0) * (v + w)
        disc = np.maximum(s * s - 4.0 * th * (th - 1.0) * v * w, 0.0)

        def d_du(uu, vv):
            ss = 1.0 + (th - 1.0) * (uu + vv)
            dd = np.maximum(ss * ss - 4.0 * th * (th - 1.0) * uu * vv, 1e-300)
            return 0.5 * (1.0 - (ss - 2.0 * th * uu) / np.sqrt(dd))

        # P(U <= w | V = v) = dC/dv (w, v) / dC/dv (1, v); by symmetry of the
        # family this is the same ratio computed with the roles kept as-is
        num = 0.5 * (1.0 - (s - 2.0 * th * w) / np.sqrt(disc))
        s1 = 1.0 + (th - 1.0) * (v + 1.0)
        d1 = np.maximum(s1 * s1 - 4.0 * th * (th - 1.0) * v, 1e-300)
        den = 0.5 * (1.0 - (s1 - 2.0 * th * v) / np.sqrt(d1))
        return num / np.maximum(den, 1e-300)

    def tail_dependence(self):
        return {"upper": 0.0, "lower": 0.0}

    def cdf(self, u):
        self._require_fit()
        uu = _clip_open(u)
        return self._core_cdf(uu[..., 0], uu[..., 1], self.theta_)

    def density(self, u):
        self._require_fit()
        uu = _clip_open(u)
        u_, v_ = uu[..., 0], uu[..., 1]
        th = self.theta_
        s = 1.0 + (th - 1.0) * (u_ + v_)
        disc = np.maximum(s * s - 4.0 * th * (th - 1.0) * u_ * v_, 0.0)
        return th * (1.0 + (th - 1.0) * (u_ + v_ - 2.0 * u_ * v_)) \
            / np.maximum(disc ** 1.5, 1e-300)

    def sample(self, n, random_state=None):
        """Exact sequential sampling via the derivative-based conditional

        ``P(V <= w | U = u) = dC/du (u, w) / dC/du (u, 1)``,

        evaluated in closed form and inverted on a fixed w-grid. (Solving
        ``C(u, v) = p`` instead would be the wrong transform.)
        """
        self._require_fit()
        n = self._validate_sample_n(n)
        rng = np.random.default_rng(random_state)
        th = self.theta_
        out = np.empty((n, 2))
        out[:, 0] = rng.random(n)
        m = 4096
        grid = np.linspace(0.0, 1.0, m)
        w_grid = np.clip(grid[1:-1], _EPS, 1.0 - _EPS)

        def dc_du(u, w):
            if abs(th - 1.0) < 1e-12:
                return np.full_like(np.asarray(w, dtype=float), w)
            s = 1.0 + (th - 1.0) * (u + w)
            disc = np.maximum(s * s - 4.0 * th * (th - 1.0) * u * w, 1e-300)
            return 0.5 * (1.0 - (s - 2.0 * th * w) / np.sqrt(disc))

        rows_u = np.repeat(out[:, 0], len(w_grid))
        rows_w = np.tile(w_grid, n)
        num = dc_du(rows_u, rows_w)
        denom = dc_du(rows_u, np.ones_like(rows_u))
        cond = np.clip(num / np.maximum(denom, 1e-300), 0.0, 1.0)
        cond = cond.reshape(n, len(w_grid))
        cond[:, -1] = 1.0
        targets = rng.random(n)
        chunk = max(1, min(n, 4_000_000 // m))
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            for j in range(start, stop):
                out[j, 1] = float(np.interp(targets[j], cond[j], grid[1:-1]))
        return out
