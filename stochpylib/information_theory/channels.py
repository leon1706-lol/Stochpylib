"""Channel capacity, information gain, transfer entropy."""

import numpy as np

from stochpylib.information_theory._base import _safe_log2, _normalise
from stochpylib.information_theory.mutual_info import MutualInformation
from stochpylib.information_theory.entropy import Entropy, ConditionalEntropy

__all__ = [
    "ChannelCapacity", "InformationGain", "TransferEntropy",
    "DirectedInformation", "SymbolicTransferEntropy",
]


class ChannelCapacity:
    """Channel capacity for standard discrete memoryless channels.

    Supports binary symmetric channel (BSC), binary erasure channel (BEC),
    and Z-channel with closed forms.
    """

    def __init__(self, channel_type="BSC", crossover_prob=None,
                 erasure_prob=None):
        self.channel_type = channel_type.upper()
        self.p = float(crossover_prob) if crossover_prob is not None else None
        self.e = float(erasure_prob) if erasure_prob is not None else None

    def fit(self):
        if self.channel_type == "BSC":
            p = self.p
            self.capacity_ = 1.0 - (
                -p * np.log2(max(p, 1e-300)) -
                (1 - p) * np.log2(max(1 - p, 1e-300)))
        elif self.channel_type == "BEC":
            e = self.e
            self.capacity_ = 1.0 - e
        elif self.channel_type == "Z":
            # Z-channel: P(1->0) = p
            p = self.p
            from scipy import optimize
            def neg_cap(log_alpha):
                a = np.exp(log_alpha)
                q0 = 1 / (1 + a)
                q1 = a / (1 + a)
                out_0 = q0 * (1 - p)
                out_1 = q0 * p + q1
                total = out_0 + out_1
                if total <= 0:
                    return 0.0
                h_out = -out_0 * np.log2(max(out_0 / total, 1e-300)) \
                    - out_1 * np.log2(max(out_1 / total, 1e-300))
                return -(h_out - max(q0 * np.log2(max(1/(q0+q1*0+1e-300),1e-300)), 0))
            res = optimize.minimize_scalar(neg_cap, bounds=(-20, 20),
                                           method="bounded")
            self.capacity_ = -res.fun
        else:
            raise ValueError(f"unsupported channel type {self.channel_type!r}")
        self.capacity_ = min(max(self.capacity_, 0.0), 1.0)
        self.result_ = self.capacity_
        return self

    @classmethod
    def compute(cls, channel_type="BSC", **kw):
        return cls(channel_type=channel_type, **kw).fit().result_


class InformationGain:
    """Information gain IG(Y|X) = H(Y) - H(Y|X)."""

    def __init__(self, bins=0):
        self.bins = int(bins)

    def fit(self, x, y):
        ce = ConditionalEntropy()
        if self.bins > 0:
            ce.bins = self.bins
        ce.fit(y, x)   # H(Y|X)
        hy = Entropy.compute(np.asarray(y))
        self.result_ = max(hy - ce.result_, 0.0)
        self.h_y_ = hy
        self.h_y_given_x_ = ce.result_
        return self

    @classmethod
    def compute(cls, x, y, **kw):
        return cls(**kw).fit(x, y).result_


def _embed_series(series, lag=1, dim=2):
    """Create delay embedding: rows are [X_t-lag*(dim-1), ..., X_t]."""
    s = np.asarray(series, dtype=float).ravel()
    n = len(s)
    if n < lag * (dim - 1) + 2:
        raise ValueError("series too short for embedding")
    cols = [s[lag * (dim - 1 - j): n - lag * j] for j in range(dim)]
    return np.column_stack(cols)


class TransferEntropy:
    """Transfer entropy TE_{X->Y}(lag) measuring directed information flow.

    TE_{X→Y} = I(Y_{t+1}; X_t | Y_t) — the reduction in uncertainty about
    the future of Y given its past, attributable to X's past.

    Uses discretisation into equal-frequency bins.
    """

    def __init__(self, lag=1, n_bins=8):
        self.lag = int(lag)
        self.n_bins = int(n_bins)

    def fit(self, x, y):
        x_s = np.asarray(x, dtype=float).ravel()
        y_s = np.asarray(y, dtype=float).ravel()
        L = self.lag
        n = len(x_s) - L
        if n <= 0:
            raise ValueError("series too short")

        # discretise into bins
        xe = np.histogram_bin_edges(x_s, bins=self.n_bins)
        ye = np.histogram_bin_edges(y_s, bins=self.n_bins)
        xd = np.clip(np.searchsorted(xe[1:], x_s[:n]), 0, self.n_bins - 1)
        yd_now = np.clip(np.searchsorted(ye[1:], y_s[L:]), 0, self.n_bins - 1)
        yd_past = np.clip(np.searchsorted(ye[1:], y_s[:n]), 0, self.n_bins - 1)

        # TE = I(Y_{t+1}; X_t | Y_t) via chain rule:
        # TE = H(Y_{t+1}|Y_t) - H(Y_{t+1}|Y_t,X_t)
        # Use conditional mutual information directly:
        cmi = ConditionalMutualInfoLite()
        te_val = cmi.fit(xd, yd_now, yd_past)
        self.result_ = max(te_val, 0.0)
        return self

    @classmethod
    def compute(cls, x, y, lag=1, n_bins=8):
        return cls(lag=lag, n_bins=n_bins).fit(x, y).result_


class ConditionalMutualInfoLite:
    """Fast CMI from integer-encoded discrete variables."""

    def fit(self, xs, ys, zs):
        n = len(xs)
        uz = np.unique(zs)
        cmi = 0.0
        for zv in uz:
            mask = zs == zv
            frac = mask.sum() / n
            sub_x = xs[mask]
            sub_y = ys[mask]
            if len(sub_x) < 2:
                continue
            ux = np.unique(sub_x)
            uy = np.unique(sub_y)
            xi = np.searchsorted(ux, sub_x)
            yi = np.searchsorted(uy, sub_y)
            table = np.zeros((len(ux), len(uy)))
            np.add.at(table, (xi, yi), 1.0)
            pxy = table / table.sum()
            px = pxy.sum(axis=1)[pxy.sum(axis=1) > 0]
            py = pxy.sum(axis=0)[pxy.sum(axis=0) > 0]
            pxy_nz = pxy[pxy > 0]
            hx = -np.sum(px * np.log2(px))
            hy = -np.sum(py * np.log2(py))
            hxy = -np.sum(pxy_nz * np.log2(pxy_nz))
            cmi += frac * max(hx + hy - hxy, 0.0)
        self.result_ = cmi
        return float(cmi)


class DirectedInformation:
    """Directed information I(X^n -> Y^n) with causal conditioning.

    Sum over t of I(Y_t; X^t | Y^{t-1}), computed on discretised series.
    For two-series data this reduces to cumulative transfer entropy.
    """

    def __init__(self, lag=1, n_bins=4):
        self.lag = int(lag)
        self.n_bins = int(n_bins)

    def fit(self, x, y):
        x_s = np.asarray(x, dtype=float).ravel()
        y_s = np.asarray(y, dtype=float).ravel()
        L = max(self.lag, 1)
        n = len(x_s)
        total_di = 0.0
        for t in range(L, min(n - 1, 200)):       # cap for performance
            x_past = x_s[max(0, t - L):t]
            y_past = y_s[max(0, t - L):t]
            y_now = y_s[t]
            x_now = x_s[t]

            # I(Y_t; X_t | Y_past) — simplified single-lag version
            cmi_lite = ConditionalMutualInfoLite()
            xe = np.digitize([x_now], np.histogram_bin_edges(
                x_s, bins=self.n_bins))[0]
            ye = np.digitize([y_now], np.histogram_bin_edges(
                y_s, bins=self.n_bins))[0]
            yp = [np.digitize([v], np.histogram_bin_edges(
                y_s, bins=self.n_bins))[0] for v in y_past]

            mi_step = cmi_lite.fit(np.array([xe] * len(yp)),
                                   np.array([ye] * len(yp)),
                                   np.array(yp))
            total_di += mi_step
        self.result_ = max(total_di, 0.0)
        return self

    @classmethod
    def compute(cls, x, y, **kw):
        return cls(**kw).fit(x, y).result_


class SymbolicTransferEntropy:
    """Symbolic transfer entropy: discretise time series into ordinal
    patterns then compute standard transfer entropy on symbols."""

    def __init__(self, embedding_dim=3, lag=1):
        self.embedding_dim = int(embedding_dim)
        self.lag = int(lag)

    def _symbolise(self, s):
        """Map consecutive patterns to ordinal symbols."""
        s = np.asarray(s, dtype=float).ravel()
        d = self.embedding_dim
        lag = self.lag
        n_sym = len(s) - (d - 1) * lag
        if n_sym <= 0:
            raise ValueError("series too short for symbolic encoding")
        symbols = np.zeros(n_sym, dtype=int)
        for i in range(n_sym):
            window = s[i: i + (d - 1) * lag + 1: lag]
            symbols[i] = int(np.sum(stats_rankdata(window) *
                                    3 ** np.arange(d)))
        return symbols

    def fit(self, x, y):
        sym_x = self._symbolise(x)
        sym_y = self._symbolise(y)
        te = TransferEntropy(lag=self.lag, n_bins=max(sym_x.max() + 1, 4))
        self.result_ = te.fit(sym_x[:-self.lag or None],
                              sym_y[self.lag:]).result_
        return self

    @classmethod
    def compute(cls, x, y, **kw):
        return cls(**kw).fit(x, y).result_


def stats_rankdata(a):
    """Simple ranking (average ties broken by order)."""
    sorter = np.argsort(a)
    ranks = np.empty(len(a))
    ranks[sorter] = np.arange(len(a))
    return ranks