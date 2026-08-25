"""Information-theoretic entropy measures."""

import numpy as np

from stochpylib.information_theory._base import (
    _validate_probs, _normalise, _joint_table, _safe_log2,
)

__all__ = [
    "Entropy", "JointEntropy", "ConditionalEntropy", "CrossEntropy",
    "TsallisEntropy", "RenyiEntropy", "DifferentialEntropy", "MaxEntropy",
]


def _shannon_bits(p):
    p = p[p > 0]
    return float(-np.sum(p * _safe_log2(p)))


class Entropy:
    """Shannon entropy H(p) = -sum(p_i * log2(p_i)) in bits.

    Accepts a probability vector or raw count/sample data (auto-normalised).
    For continuous data use ``DifferentialEntropy`` instead.
    """

    def __init__(self, base=2):
        self.base = base
        if base not in (2, np.e):
            raise ValueError("base must be 2 or e")

    def fit(self, data):
        data = np.asarray(data, dtype=float).ravel()
        if self.base == 2:
            self.result_ = _shannon_bits(_validate_probs(data)[0])
        else:
            p = _validate_probs(data)[0]
            p = p[p > 0]
            self.result_ = float(-np.sum(p * np.log(p)))
        return self

    @classmethod
    def compute(cls, data, **kw):
        return cls(**kw).fit(data).result_


class JointEntropy:
    """Joint entropy H(X,Y) from paired observations."""

    def __init__(self, bins=0):
        self.bins = int(bins)

    def fit(self, x, y):
        joint_flat, table = _joint_table(x, y, bins=self.bins)
        self.joint_table_ = table / max(table.sum(), 1e-300)
        self.result_ = _shannon_bits(_validate_probs(joint_flat)[0])
        return self

    @classmethod
    def compute(cls, x, y, **kw):
        return cls(**kw).fit(x, y).result_


class ConditionalEntropy:
    """Conditional entropy H(X|Y) = H(X,Y) - H(Y) from paired observations."""

    def __init__(self, bins=0):
        self.bins = int(bins)

    def fit(self, x, y):
        je = JointEntropy(bins=self.bins).fit(x, y)
        hy = Entropy().fit(np.asarray(y)) if self.bins == 0 else \
            Entropy(base=2)
        if self.bins == 0:
            _, counts = np.unique(y, return_counts=True)
            hy_val = _shannon_bits(_normalise(counts))
        else:
            hist, _ = np.histogram(y, bins=self.bins)
            hy_val = _shannon_bits(_normalise(hist))
        # H(X|Y) = H(X,Y) - H(Y)
        self.result_ = max(je.result_ - hy_val, 0.0)
        self.h_y_ = hy_val
        self.h_xy_ = je.result_
        return self

    @classmethod
    def compute(cls, x, y, **kw):
        return cls(**kw).fit(x, y).result_


class CrossEntropy:
    """Cross-entropy H(p,q) = -sum(p_i * log(q_i))."""

    def __init__(self, base=2):
        self.base = base

    def fit(self, p, q):
        p_v, q_v = _validate_probs(p, q)
        mask = p_v > 0
        if self.base == 2:
            self.result_ = float(-np.sum(p_v[mask] * _safe_log2(q_v[mask])))
        else:
            self.result_ = float(-np.sum(
                p_v[mask] * np.log(np.maximum(q_v[mask], 1e-300))))
        return self

    @classmethod
    def compute(cls, p, q, **kw):
        return cls(**kw).fit(p, q).result_


class TsallisEntropy:
    """Tsallis entropy S_q = (1/(q-1))(1 - sum(p_i^q)), q != 1."""

    def __init__(self, q=2.0):
        self.q = float(q)

    def fit(self, p):
        p_v = _normalise(p)
        q = self.q
        if abs(q - 1.0) < 1e-12:
            self.result_ = Entropy.compute(p_v)
        else:
            self.result_ = float((1.0 - np.sum(p_v ** q)) / (q - 1.0))
        return self

    @classmethod
    def compute(cls, p, q=2.0):
        return cls(q=q).fit(p).result_


class RenyiEntropy:
    """Renyi alpha-entropy H_alpha = log(sum(p^alpha))/(1-alpha), alpha != 1."""

    def __init__(self, alpha=2.0):
        self.alpha = float(alpha)

    def fit(self, p):
        p_v = _normalise(p)
        a = self.alpha
        if abs(a - 1.0) < 1e-12:
            self.result_ = Entropy.compute(p_v)
        elif abs(a) < 1e-12:
            self.result_ = float(np.log(max((p_v > 0).sum(), 1)))
        else:
            s = float(np.sum(p_v ** a))
            self.result_ = float(np.log2(max(s, 1e-300)) / (1.0 - a))
        return self

    @classmethod
    def compute(cls, p, alpha=2.0):
        return cls(alpha=alpha).fit(p).result_


class DifferentialEntropy:
    """Continuous differential entropy via histogram binning.

    Uses the plug-in estimator: h(X) ~ H(digitised) - log(n_bins).
    """

    def __init__(self, n_bins=50):
        self.n_bins = int(n_bins)

    def fit(self, samples):
        samples = np.asarray(samples, dtype=float).ravel()
        hist, edges = np.histogram(samples, bins=self.n_bins)
        p = _normalise(hist)
        h_disc = _shannon_bits(p)
        # correction term for continuous -> discrete conversion
        bin_width = edges[1] - edges[0]
        self.result_ = h_disc + np.log2(bin_width) \
            if bin_width > 0 else h_disc
        return self

    @classmethod
    def compute(cls, samples, **kw):
        return cls(**kw).fit(samples).result_


class MaxEntropy:
    """Maximum-entropy distribution subject to mean constraint(s).

    For finite support with no constraints beyond normalisation,
    the uniform distribution maximises entropy.
    """

    def __init__(self, support_size=None, mean_constraint=None,
                 lower=0.0, upper=1.0, maxiter=500):
        self.support_size = support_size
        self.mean_constraint = mean_constraint
        self.lower = lower
        self.upper = upper
        self.maxiter = maxiter

    def fit(self, data=None):
        if self.mean_constraint is None and data is None:
            # uniform over support
            k = self.support_size or 10
            self.distribution_ = np.full(k, 1.0 / k)
            self.result_ = np.log2(k)
            return self
        # exponential family with Lagrange multiplier for mean
        from scipy import optimize
        k = self.support_size or 20
        support = np.linspace(self.lower, self.upper, k)
        target_mean = self.mean_constraint
        if target_mean is None:
            target_mean = float(np.mean(data))

        def neg_entropy(lam):
            logits = lam * support
            logits -= logits.max()
            p_dist = np.exp(logits)
            p_dist /= p_dist.sum()
            mean_val = float(np.sum(support * p_dist))
            ent = -float(np.sum(p_dist * _safe_log2(p_dist)))
            return -(ent - 100.0 * (mean_val - target_mean) ** 2)

        res = optimize.minimize_scalar(neg_entropy,
                                       bounds=(-50, 50),
                                       method="bounded")
        lam_opt = res.x
        logits = lam_opt * support
        logits -= logits.max()
        p_dist = np.exp(logits)
        p_dist /= p_dist.sum()
        self.distribution_ = p_dist
        self.support_ = support
        self.result_ = float(-np.sum(p_dist[p_dist > 0] *
                                     np.log2(p_dist[p_dist > 0])))
        return self