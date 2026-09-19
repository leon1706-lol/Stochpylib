"""Spectral statistics: spacings, level repulsion, empirical spectral distributions, and
edge (largest / smallest eigenvalue) fluctuations.

Every comparison returns :class:`stochpylib.statistics.TestResult`. The unfolding-free
mean adjacent-gap ratio ``<r>`` (Atas et al. 2013) is the primary spacing statistic; the
Wigner surmise comparisons use a polynomial unfolding of the empirical staircase.
"""

import numpy as np
from scipy import integrate, special

from stochpylib.random_matrix._common import (
    _MEAN_RATIO_REFERENCE,
    _as_1d,
    _rng,
    _wigner_surmise,
    _wigner_surmise_cdf,
)
from stochpylib.random_matrix.empirical_spectra import BetaEnsemble, TracyWidomDistribution
from stochpylib.random_matrix.ensembles import MatrixEnsemble, WishartMatrix

__all__ = [
    "BulkSpectrum",
    "EigenvalueDistribution",
    "EigenvalueSpacing",
    "LargestEigenvalue",
    "LevelRepulsion",
    "SpectralEdge",
]


def _test_result(statistic, pvalue, null, method, **extras):
    from stochpylib.statistics import TestResult

    return TestResult(statistic=float(statistic), pvalue=float(pvalue), null=null,
                      method=method, extras=dict(extras))


def _ks_against_cdf(sample, cdf):
    """One-sample KS statistic + asymptotic p-value for a continuous ``cdf`` callable."""
    x = np.sort(np.asarray(sample, dtype=float))
    n = x.size
    F = np.asarray(cdf(x), dtype=float)
    i = np.arange(1, n + 1)
    d = max(float(np.max(i / n - F)), float(np.max(F - (i - 1) / n)))
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    p = 2.0 * sum((-1) ** (k - 1) * np.exp(-2.0 * k ** 2 * lam ** 2) for k in range(1, 101))
    return d, float(np.clip(p, 0.0, 1.0))


def _real_sorted(eigenvalues):
    e = _as_1d(eigenvalues)
    if np.iscomplexobj(e):
        if np.max(np.abs(e.imag)) > 1e-8 * max(1.0, float(np.max(np.abs(e)))):
            raise ValueError("expected real eigenvalues (use eigenangles for unitary spectra)")
        e = e.real
    return np.sort(np.asarray(e, dtype=float))


def _resolve(source, random_state):
    """Eigenvalue array from either an array or an ensemble (one draw)."""
    if isinstance(source, (MatrixEnsemble, BetaEnsemble)):
        return _real_sorted(source.eigenvalues(random_state)), source
    return _real_sorted(source), None


class EigenvalueSpacing:
    """Nearest-neighbour spacing statistics of one spectrum.

    ``unfold="polynomial"`` maps eigenvalues through a degree-``degree`` polynomial fit of
    the empirical staircase (so the mean local spacing is 1 everywhere); ``"none"`` only
    rescales raw spacings to unit mean. ``ratios``/``mean_ratio`` need no unfolding.
    """

    def __init__(self, eigenvalues, unfold="polynomial", degree=7, trim=0.05):
        self.eigenvalues = _real_sorted(eigenvalues)
        if self.eigenvalues.size < 4:
            raise ValueError("need at least 4 eigenvalues")
        if unfold not in ("polynomial", "none"):
            raise ValueError("unfold must be 'polynomial' or 'none'")
        self.unfold = unfold
        self.degree = int(degree)
        self.trim = float(trim)

    def unfolded(self):
        """Unfolded levels ``xi_i`` (mean spacing 1)."""
        e = self.eigenvalues
        n = e.size
        if self.unfold == "none":
            return (e - e[0]) / np.mean(np.diff(e))
        staircase = np.arange(1, n + 1, dtype=float)
        x = (e - e.mean()) / (e.std() or 1.0)
        coef = np.polynomial.polynomial.polyfit(x, staircase, self.degree)
        xi = np.polynomial.polynomial.polyval(x, coef)
        return xi

    def spacings(self):
        """Unfolded nearest-neighbour spacings with the outer ``trim`` fraction removed."""
        xi = self.unfolded()
        s = np.diff(xi)
        k = int(self.trim * s.size)
        if k > 0:
            s = s[k:-k]
        s = s[s > 0]
        return s / s.mean()

    def ratios(self):
        """Adjacent-gap ratios ``min(s_i, s_{i+1}) / max(s_i, s_{i+1})`` (unfolding-free)."""
        s = np.diff(self.eigenvalues)
        s = s[s > 0]
        a, b = s[:-1], s[1:]
        return np.minimum(a, b) / np.maximum(a, b)

    def mean_ratio(self):
        return float(np.mean(self.ratios()))

    @staticmethod
    def surmise(s, beta):
        """Wigner surmise density for ``beta`` in {1, 2, 4} (0 / 'poisson' -> exp(-s))."""
        return _wigner_surmise(s, beta)

    @staticmethod
    def mean_ratio_reference(beta):
        """Large-N ``<r>`` for ``beta`` in {1, 2, 4} or ``'poisson'``."""
        key = "poisson" if beta in (0, "poisson") else beta
        return float(_MEAN_RATIO_REFERENCE[key])

    def compare(self, beta):
        """KS test of the unfolded spacings against the Wigner surmise (or Poisson)."""
        s = self.spacings()
        d, p = _ks_against_cdf(s, lambda t: _wigner_surmise_cdf(t, beta))
        return _test_result(d, p, f"spacings follow the beta={beta} surmise",
                            "Kolmogorov-Smirnov", n=int(s.size), mean_ratio=self.mean_ratio())

    def classify(self):
        """Nearest ensemble by mean gap ratio: 'poisson', 'GOE', 'GUE' or 'GSE'."""
        r = self.mean_ratio()
        labels = {"poisson": "poisson", 1: "GOE", 2: "GUE", 4: "GSE"}
        key = min(_MEAN_RATIO_REFERENCE, key=lambda k: abs(_MEAN_RATIO_REFERENCE[k] - r))
        return labels[key]


class LevelRepulsion:
    """Level-repulsion exponent ``beta`` in ``P(s) ~ s^beta`` (``s -> 0``) of one spectrum.

    ``method="mle"`` (default) fits ``beta`` by maximum likelihood in the generalized
    Wigner family ``P_beta(s) = a s^beta exp(-b s^2)`` (``a, b`` fixed by normalization and
    unit mean), which contains the exact surmises for ``beta`` = 1, 2, 4 and uses every
    unfolded spacing. ``method="smallgap"`` is the classical log-log slope of the empirical
    spacing CDF over the lowest ``fraction`` of spacings (``P(S < s) ~ s^(beta+1)``).
    ``exponent_`` is the fitted ``beta`` (0 for Poisson, 1/2/4 for GOE/GUE/GSE).
    """

    def __init__(self, eigenvalues, method="mle", fraction=0.1, unfold="polynomial"):
        if method not in ("mle", "smallgap"):
            raise ValueError("method must be 'mle' or 'smallgap'")
        self.spacing = EigenvalueSpacing(eigenvalues, unfold=unfold)
        self.method = method
        self.fraction = float(fraction)
        s = np.sort(self.spacing.spacings())
        if method == "mle":
            self.exponent_ = self._fit_mle(s)
            self.n_used_ = int(s.size)
        else:
            self.exponent_, self.n_used_ = self._fit_smallgap(s)

    @staticmethod
    def _neg_loglik(beta, s):
        # unit-mean generalized surmise: b = [Gamma((beta+2)/2) / Gamma((beta+1)/2)]^2,
        # a = 2 b^((beta+1)/2) / Gamma((beta+1)/2)
        g1 = special.gammaln((beta + 1.0) / 2.0)
        g2 = special.gammaln((beta + 2.0) / 2.0)
        log_b = 2.0 * (g2 - g1)
        log_a = np.log(2.0) + (beta + 1.0) / 2.0 * log_b - g1
        return -float(np.sum(log_a + beta * np.log(s) - np.exp(log_b) * s * s))

    def _fit_mle(self, s):
        from scipy import optimize

        res = optimize.minimize_scalar(self._neg_loglik, bounds=(0.0, 8.0), args=(s,),
                                       method="bounded")
        return float(res.x)

    def _fit_smallgap(self, s):
        n = s.size
        k = max(5, int(self.fraction * n))
        small = s[:k]
        F = np.arange(1, k + 1) / n
        A = np.column_stack([np.ones(k), np.log(small)])
        coef, *_ = np.linalg.lstsq(A, np.log(F), rcond=None)
        return float(coef[1] - 1.0), int(k)

    def surmise(self, s):
        """The fitted generalized-surmise density evaluated at ``s``."""
        s = np.asarray(s, dtype=float)
        beta = self.exponent_
        g1 = special.gammaln((beta + 1.0) / 2.0)
        g2 = special.gammaln((beta + 2.0) / 2.0)
        b = np.exp(2.0 * (g2 - g1))
        a = 2.0 * b ** ((beta + 1.0) / 2.0) / np.exp(g1)
        return a * s ** beta * np.exp(-b * s * s)

    def classify(self):
        targets = {"poisson": 0.0, "GOE": 1.0, "GUE": 2.0, "GSE": 4.0}
        return min(targets, key=lambda k: abs(targets[k] - self.exponent_))


class EigenvalueDistribution:
    """Empirical spectral distribution of one spectrum: ``cdf``, KDE ``pdf``, moments,
    and a KS ``compare(law)`` against any distribution object with a ``cdf``."""

    def __init__(self, eigenvalues):
        self.eigenvalues = _real_sorted(eigenvalues)
        self.n = int(self.eigenvalues.size)
        self.mean_ = float(np.mean(self.eigenvalues))
        self.var_ = float(np.var(self.eigenvalues))

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        out = np.searchsorted(self.eigenvalues, x, side="right") / self.n
        return out if out.ndim else float(out)

    def pdf(self, x, bandwidth=None):
        """Gaussian kernel density (Silverman's rule by default)."""
        x = np.asarray(x, dtype=float)
        e = self.eigenvalues
        if bandwidth is None:
            iqr = np.subtract(*np.percentile(e, [75, 25]))
            sd = min(e.std(), iqr / 1.349) if iqr > 0 else e.std()
            bandwidth = 0.9 * (sd or 1.0) * self.n ** (-0.2)
        z = (x[..., None] - e) / bandwidth
        out = np.mean(np.exp(-0.5 * z * z), axis=-1) / (bandwidth * np.sqrt(2.0 * np.pi))
        return out if out.ndim else float(out)

    def moments(self, k):
        """Raw moment(s) ``mean(lambda^k)`` for an int or a sequence of ints."""
        ks = np.atleast_1d(np.asarray(k, dtype=int))
        out = np.array([np.mean(self.eigenvalues ** kk) for kk in ks])
        return out if np.ndim(k) else float(out[0])

    def compare(self, law):
        d, p = law.ks_test(self.eigenvalues) if hasattr(law, "ks_test") else \
            _ks_against_cdf(self.eigenvalues, law.cdf)
        return _test_result(d, p, f"spectrum follows {type(law).__name__}",
                            "Kolmogorov-Smirnov", n=self.n)

    def support(self):
        return float(self.eigenvalues[0]), float(self.eigenvalues[-1])


class LargestEigenvalue:
    """Largest-eigenvalue fluctuations of an ensemble and their Tracy-Widom scaling.

    ``sample(k)`` draws ``k`` largest eigenvalues; ``scaled(samples)`` applies the
    ensemble's soft-edge centering/scaling so the result converges to ``TracyWidom(beta)``:
    Hermitian Gaussian/Wigner/beta-Hermite ensembles use ``(lambda/sqrt(n) - 2) n^(2/3)``
    (Ramirez-Rider-Virag; times ``2^(1/6)`` for beta=4); real Wishart / beta-Laguerre(1)
    use Johnstone's (2001) ``mu = (sqrt(n') + sqrt(p'))^2``, ``sigma = (sqrt(n') +
    sqrt(p')) (1/sqrt(n') + 1/sqrt(p'))^(1/3)`` with Ma's (2012) ``n' = n - 1/2``,
    ``p' = p - 1/2``. Explicit ``center``/``scale``/``beta`` override the defaults.
    """

    def __init__(self, ensemble, beta=None, center=None, scale=None):
        self.ensemble = ensemble
        self.beta = beta
        self.center = center
        self.scale = scale
        self._infer()

    def _infer(self):
        e = self.ensemble
        if self.center is not None and self.scale is not None:
            if self.beta is None:
                raise ValueError("beta is required with explicit center/scale")
            self.kind_ = "custom"
            return
        if isinstance(e, WishartMatrix):
            if not np.allclose(e.sigma, np.eye(e.p)):
                raise ValueError("Tracy-Widom scaling needs an identity covariance")
            self.beta = 1 if self.beta is None else self.beta
            self.center, self.scale = self._wishart_edge(e.p, e.n)
            self.kind_ = "wishart"
            return
        if isinstance(e, BetaEnsemble) and e.kind == "laguerre":
            if e.beta != 1:
                raise ValueError("Laguerre edge scaling is implemented for beta=1 only")
            self.beta = 1
            self.center, self.scale = self._wishart_edge(e.n, e.a)
            self.kind_ = "wishart"
            return
        if isinstance(e, BetaEnsemble) or (isinstance(e, MatrixEnsemble) and e.hermitian):
            if self.beta is None:
                self.beta = int(e.beta) if e.beta in (1, 2, 4) else e.beta
            # normalized eigenvalues have their bulk edge at 2: scaled = (norm - 2) * n^(2/3)
            # (Ramirez-Rider-Virag). For beta=4 the RRV edge variable is 2^(-1/6) times the
            # classical F4 variable (Tracy-Widom / Bornemann convention, mean -2.3069) that
            # TracyWidomDistribution(4) tabulates -- verified to KS p=0.37 on 1500 draws.
            self.center = 2.0 * e._scale()
            self.scale = e._scale() * e.n ** (-2.0 / 3.0)
            if self.beta == 4:
                self.scale /= 2.0 ** (1.0 / 6.0)
            self.kind_ = "hermite"
            return
        raise ValueError("unsupported ensemble for Tracy-Widom scaling; pass center/scale/beta")

    @staticmethod
    def _wishart_edge(p, n):
        """Johnstone (2001) centering/scaling with Ma's (2012) half-integer shifts, which
        improve the finite-n error from O(n^-1/3) to O(n^-2/3)."""
        a, b = n - 0.5, p - 0.5
        center = (np.sqrt(a) + np.sqrt(b)) ** 2
        scale = (np.sqrt(a) + np.sqrt(b)) * (1.0 / np.sqrt(a) + 1.0 / np.sqrt(b)) ** (1.0 / 3.0)
        return float(center), float(scale)

    def sample(self, n_samples, random_state=None):
        rng = _rng(random_state)
        return np.array([float(np.max(self.ensemble.eigenvalues(rng)))
                         for _ in range(int(n_samples))])

    def scaled(self, samples):
        return (np.asarray(samples, dtype=float) - self.center) / self.scale

    def tracy_widom(self):
        return TracyWidomDistribution(self.beta)

    def compare(self, n_samples=200, random_state=None, samples=None):
        """KS test of the scaled largest eigenvalues against ``TracyWidom(beta)``."""
        raw = self.sample(n_samples, random_state) if samples is None else np.asarray(samples)
        z = self.scaled(raw)
        tw = self.tracy_widom()
        d, p = tw.ks_test(z)
        return _test_result(d, p, f"scaled largest eigenvalue follows Tracy-Widom beta={self.beta}",
                            "Kolmogorov-Smirnov", n=int(z.size), sample_mean=float(np.mean(z)),
                            tw_mean=tw.mean(), tw_std=float(np.sqrt(tw.var())))


class BulkSpectrum:
    """Bulk statistics of one spectrum (an eigenvalue array or an ensemble draw).

    ``density``/``moments``/``compare`` delegate to :class:`EigenvalueDistribution`;
    ``spacing()`` returns the :class:`EigenvalueSpacing`; ``number_variance(L)`` is the
    Dyson-Mehta ``Sigma^2(L)`` -- the variance of the number of unfolded levels in a window
    of length ``L`` (``= L`` for Poisson, logarithmic for the Gaussian ensembles).
    """

    def __init__(self, source, random_state=None):
        self.eigenvalues, self.ensemble = _resolve(source, random_state)
        self.esd = EigenvalueDistribution(self.eigenvalues)

    def density(self, x, bandwidth=None):
        return self.esd.pdf(x, bandwidth=bandwidth)

    def moments(self, k):
        return self.esd.moments(k)

    def compare(self, law=None):
        if law is None:
            if self.ensemble is None or self.ensemble.limit_law() is None:
                raise ValueError("no limit law available; pass one explicitly")
            law = self.ensemble.limit_law()
            return EigenvalueDistribution(self.ensemble.normalize(self.eigenvalues)).compare(law)
        return self.esd.compare(law)

    def spacing(self, **kwargs):
        return EigenvalueSpacing(self.eigenvalues, **kwargs)

    def number_variance(self, L, n_windows=None, unfold="polynomial"):
        """Dyson-Mehta ``Sigma^2(L)`` from sliding windows over the unfolded spectrum."""
        xi = EigenvalueSpacing(self.eigenvalues, unfold=unfold).unfolded()
        xi = xi[int(0.05 * xi.size): xi.size - int(0.05 * xi.size)]
        L = float(L)
        lo, hi = xi[0], xi[-1] - L
        if hi <= lo:
            raise ValueError("L too large for this spectrum")
        m = int(n_windows) if n_windows else max(50, int((hi - lo) / 0.25))
        starts = np.linspace(lo, hi, m)
        counts = np.searchsorted(xi, starts + L, side="left") - np.searchsorted(xi, starts, side="left")
        return float(np.var(counts))

    @staticmethod
    def number_variance_reference(L, beta):
        """Large-``L`` asymptotics of ``Sigma^2(L)``: ``L`` (Poisson) or
        ``(2 / (beta pi^2)) (ln(2 pi L) + gamma + 1 - ...)`` for beta=1, 2, 4 (Mehta)."""
        L = float(L)
        g = np.euler_gamma
        if beta in (0, "poisson"):
            return L
        if beta == 1:
            return (2.0 / np.pi ** 2) * (np.log(2 * np.pi * L) + g + 1.0 - np.pi ** 2 / 8.0)
        if beta == 2:
            return (1.0 / np.pi ** 2) * (np.log(2 * np.pi * L) + g + 1.0)
        if beta == 4:
            return (1.0 / (2 * np.pi ** 2)) * (np.log(4 * np.pi * L) + g + 1.0 + np.pi ** 2 / 8.0)
        raise ValueError("beta must be 1, 2, 4 or 0/'poisson'")


class SpectralEdge:
    """Edge statistics: extreme eigenvalues, the soft-edge Tracy-Widom test, and the
    hard-edge law of the smallest eigenvalue of a square real Wishart matrix.

    For ``X`` an ``n x n`` standard Gaussian matrix and ``W = X.T X``, Edelman (1988):
    ``n * lambda_min`` has density ``(1 + sqrt(x)) / (2 sqrt(x)) exp(-x/2 - sqrt(x))``.
    """

    def __init__(self, source, random_state=None):
        self.eigenvalues, self.ensemble = _resolve(source, random_state)

    def edges(self):
        """``(smallest, largest)`` eigenvalue of the spectrum."""
        return float(self.eigenvalues[0]), float(self.eigenvalues[-1])

    def largest(self):
        return float(self.eigenvalues[-1])

    def smallest(self):
        return float(self.eigenvalues[0])

    def soft_edge(self, n_samples=200, random_state=None, **kwargs):
        """Tracy-Widom test of the largest eigenvalue (needs an ensemble source)."""
        if self.ensemble is None:
            raise ValueError("soft_edge needs an ensemble source")
        return LargestEigenvalue(self.ensemble, **kwargs).compare(n_samples, random_state)

    @staticmethod
    def hard_edge_pdf(x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        pos = x > 0
        r = np.sqrt(x[pos])
        out[pos] = (1.0 + r) / (2.0 * r) * np.exp(-x[pos] / 2.0 - r)
        return out if out.ndim else float(out)

    @staticmethod
    def hard_edge_cdf(x):
        x = np.asarray(x, dtype=float)
        # closed form: 1 - exp(-x/2 - sqrt(x))  (differentiate to recover Edelman's density)
        out = np.where(x > 0, 1.0 - np.exp(-x / 2.0 - np.sqrt(np.clip(x, 0.0, None))), 0.0)
        return out if out.ndim else float(out)

    def hard_edge_scaled(self, n_samples=200, random_state=None):
        """``n * lambda_min`` samples for a square Wishart ensemble source."""
        e = self.ensemble
        if not isinstance(e, WishartMatrix) or e.p != e.n or not np.allclose(e.sigma, np.eye(e.p)):
            raise ValueError("hard_edge_scaled needs a square WishartMatrix(p=n) with identity covariance")
        rng = _rng(random_state)
        return np.array([e.n * float(np.min(e.eigenvalues(rng))) for _ in range(int(n_samples))])

    def hard_edge_compare(self, n_samples=200, random_state=None, samples=None):
        z = self.hard_edge_scaled(n_samples, random_state) if samples is None else np.asarray(samples)
        d, p = _ks_against_cdf(z, self.hard_edge_cdf)
        return _test_result(d, p, "n * lambda_min follows Edelman's hard-edge law",
                            "Kolmogorov-Smirnov", n=int(z.size))
