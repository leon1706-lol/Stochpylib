"""Information-theoretic divergence measures between probability distributions."""

import numpy as np

from stochpylib.information_theory._base import _validate_probs, _safe_log2

__all__ = [
    "KLDivergence", "RelativeEntropy", "JensenShannonDivergence",
    "WassersteinDistance", "HellingerDistance", "TotalVariation",
    "ChiSquaredDivergence", "AlphaDivergence",
]


def kl_divergence(p, q):
    """D_KL(P||Q) = sum(p_i * log2(p_i/q_i)) in bits."""
    p_v, q_v = _validate_probs(p, q)
    mask = p_v > 0
    if np.any(q_v[mask] <= 0):
        return float("inf")
    return float(np.sum(
        p_v[mask] * (_safe_log2(p_v[mask]) - _safe_log2(q_v[mask]))))


def js_divergence(p, q):
    """Jensen-Shannon divergence (symmetric, bounded by 1 bit)."""
    p_v, q_v = _validate_probs(p, q)
    m = 0.5 * (p_v + q_v)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(
            a[mask] * (_safe_log2(a[mask]) - _safe_log2(b[mask]))))

    return 0.5 * (_kl(p_v, m) + _kl(q_v, m))


class KLDivergence:
    """Kullback-Leibler divergence D_KL(P||Q) in bits.

    Non-symmetric; D_KL >= 0 with equality iff P == Q.
    """

    def fit(self, p, q):
        self.result_ = kl_divergence(p, q)
        return self

    @classmethod
    def compute(cls, p, q):
        return cls().fit(p, q).result_


class RelativeEntropy(KLDivergence):
    """Alias of :class:`KLDivergence` (spec naming)."""

    pass


class JensenShannonDivergence:
    """Jensen-Shannon divergence (symmetric, bounded by 1 bit for base 2)."""

    def fit(self, p, q):
        self.result_ = js_divergence(p, q)
        return self

    @classmethod
    def compute(cls, p, q):
        return cls().fit(p, q).result_


def js_divergence_fn(p, q):
    """Standalone JS divergence function."""
    from stochpylib.information_theory.divergences import (
        JensenShannonDivergence,
    )
    return JensenShannonDivergence.compute(p, q)


class WassersteinDistance:
    """Earth Mover's (Wasserstein-1) distance between samples."""

    def fit(self, x, y):
        from scipy.stats import wasserstein_distance
        self.result_ = float(wasserstein_distance(
            np.asarray(x, dtype=float).ravel(),
            np.asarray(y, dtype=float).ravel()))
        return self

    @classmethod
    def compute(cls, x, y):
        return cls().fit(x, y).result_


class HellingerDistance:
    """Hellinger distance H(P,Q) = (1/sqrt(2)) * ||sqrt(P) - sqrt(Q)||_2."""

    def fit(self, p, q):
        p_v, q_v = _validate_probs(p, q)
        h = np.sqrt(np.sum((np.sqrt(p_v) - np.sqrt(q_v)) ** 2)) / np.sqrt(2)
        self.result_ = float(min(max(h, 0.0), 1.0))
        return self

    @classmethod
    def compute(cls, p, q):
        return cls().fit(p, q).result_


class TotalVariation:
    """Total variation distance TV = 0.5 * sum(|p_i - q_i|)."""

    def fit(self, p, q):
        p_v, q_v = _validate_probs(p, q)
        tv = 0.5 * float(np.sum(np.abs(p_v - q_v)))
        self.result_ = min(max(tv, 0.0), 1.0)
        return self

    @classmethod
    def compute(cls, p, q):
        return cls().fit(p, q).result_


class ChiSquaredDivergence:
    """Pearson chi-squared divergence chi^2(P,Q) = sum((p_i-q_i)^2/q_i)."""

    def fit(self, p, q):
        p_v, q_v = _validate_probs(p, q)
        mask = q_v > 1e-15
        result = float(np.sum((p_v[mask] - q_v[mask]) ** 2 / q_v[mask]))
        self.result_ = max(result, 0.0)
        return self

    @classmethod
    def compute(cls, p, q):
        return cls().fit(p, q).result_


class AlphaDivergence:
    """Alpha-divergence D_alpha(P||Q) generalising KL and Hellinger.

    D_alpha = 4/(alpha^2-1) * (sum p_i^alpha q_i^(1-alpha) - 1) for alpha != 1.
    alpha=1 gives KL (up to sign convention); alpha=2 gives Pearson chi^2/2.
    """

    def __init__(self, alpha=2.0):
        self.alpha = float(alpha)

    def fit(self, p, q):
        a = self.alpha
        p_v, q_v = _validate_probs(p, q)
        mask = (q_v > 1e-15) & (p_v > 0)
        pv = p_v[mask]
        qv = q_v[mask]
        if abs(a - 1.0) < 1e-12:
            self.result_ = kl_divergence(p_v, q_v)
            return self
        s = float(np.sum(pv ** a * qv ** (1.0 - a)))
        denom = a * (a - 1.0) if abs(a) > 1e-12 else 1.0
        result = 4.0 / (a * a - 1.0) * (s - 1.0) \
            if abs(a * a - 1.0) > 1e-12 else kl_divergence(p_v, q_v)
        self.result_ = max(result, 0.0)
        return self

    @classmethod
    def compute(cls, p, q, alpha=2.0):
        return cls(alpha=alpha).fit(p, q).result_