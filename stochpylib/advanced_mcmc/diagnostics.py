"""MCMC convergence diagnostics: R-hat, effective sample size, Gelman-Rubin/PSRF,
Geweke, Raftery-Lewis, autocorrelation time, and the :class:`TraceAnalysis` summary.

Every function accepts a single chain ``(n,)``/``(n, dim)`` or multiple chains
``(n_chains, n, dim)`` (``(n_chains, n)`` for one parameter) -- the shape a sampler's
``chains_`` attribute already has. Reference: Vehtari, Gelman, Simpson, Carpenter & Bürkner
(2021) "Rank-normalization, folding, and localization" for split/rank R-hat and bulk/tail
ESS; Gelman & Rubin (1992) and Brooks & Gelman (1998) for the classic PSRF and its
multivariate/df-corrected forms; Geweke (1992); Raftery & Lewis (1992).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import special

__all__ = ["ESS", "GelmanRubin", "PSRF", "RafteryLewisResult", "Rhat", "TraceAnalysis",
           "autocorr_time", "geweke_test", "raftery_lewis"]


# --------------------------------------------------------------------------- shape utils

def _chains_3d(chains):
    """Normalize to ``(n_chains, n, dim)`` float array."""
    arr = np.asarray(chains, dtype=float)
    if arr.ndim == 1:
        return arr[None, :, None]
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim == 3:
        return arr
    raise ValueError("chains must have 1, 2 or 3 dimensions")


def _split(chains):
    """Split each chain in half (Vehtari et al. 2021), dropping an odd trailing draw."""
    m, n, d = chains.shape
    if n < 4:
        raise ValueError("need at least 4 draws per chain to split")
    half = n // 2
    a = chains[:, :half, :]
    b = chains[:, half:2 * half, :]
    return np.concatenate([a, b], axis=0)


def _rank_normalize(x):
    """Rank-normalize a pooled sample ``x`` (any shape) via average ranks -> z-scores."""
    flat = x.ravel()
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(flat) + 1)
    # average ranks for ties
    sorted_flat = flat[order]
    i = 0
    while i < len(sorted_flat):
        j = i
        while j + 1 < len(sorted_flat) and sorted_flat[j + 1] == sorted_flat[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1
    N = len(flat)
    # Blom's normal-scores transform: (rank - c) / (N - 2c + 1) with c = 3/8.
    z = special.ndtri((ranks - 3.0 / 8.0) / (N + 1.0 / 4.0))
    return z.reshape(x.shape)


def _autocov_fft(x):
    """Biased autocovariance of a 1-D series via FFT, length matches x."""
    n = len(x)
    xm = x - x.mean()
    size = 1
    while size < 2 * n:
        size *= 2
    f = np.fft.rfft(xm, n=size)
    acov = np.fft.irfft(f * np.conj(f), n=size)[:n]
    return acov / n


def _rhat_split_one(x):
    """Split R-hat for one parameter, x shape (m, n) with m already split-doubled."""
    m, n = x.shape
    if m < 2:
        raise ValueError("Rhat needs at least 2 chains (after splitting)")
    chain_means = x.mean(axis=1)
    chain_vars = x.var(axis=1, ddof=1)
    W = chain_vars.mean()
    B = n * chain_means.var(ddof=1)
    if W <= 0 and B <= 0:
        return 1.0
    var_hat = ((n - 1) / n) * W + B / n
    return float(np.sqrt(var_hat / W)) if W > 0 else float("inf")


def _ess_one(x):
    """Stan-style bulk ESS for one parameter, x shape (m, n)."""
    m, n = x.shape
    chain_means = x.mean(axis=1)
    chain_vars = x.var(axis=1, ddof=1)
    mean_var = chain_vars.mean()
    var_plus = mean_var * (n - 1) / n
    if m > 1:
        var_plus += chain_means.var(ddof=1)
    if var_plus <= 0:
        return float(m * n)
    acov = np.array([_autocov_fft(x[c]) for c in range(m)])  # (m, n)
    mean_acov = acov.mean(axis=0)
    rho = 1.0 - (mean_var - mean_acov) / var_plus
    rho[0] = 1.0
    # Geyer's initial monotone positive sequence on paired sums
    max_lag = n - 1
    n_pairs = max_lag // 2
    tau = 1.0
    if n_pairs > 0:
        P = rho[1:1 + 2 * n_pairs:2] + rho[2:2 + 2 * n_pairs:2]
        running_min = np.inf
        cum = 0.0
        for k in range(len(P)):
            if P[k] < 0:
                break
            running_min = min(running_min, P[k])
            cum += running_min
        tau = -1.0 + 2.0 * cum
    tau = max(tau, 1.0 / n)
    ess = m * n / tau
    return float(min(ess, m * n * np.log10(max(m * n, 10))))


def Rhat(chains, method="split"):
    """Potential scale reduction factor.

    ``method="split"`` (default): classic split R-hat. ``method="classic"``: unsplit
    (needs >= 2 chains as given). ``method="rank"``: max of the rank-normalized split
    R-hat and the rank-normalized-folded (tail) split R-hat (Vehtari et al. 2021),
    detects scale differences that the mean-based classic/split statistic misses.
    Returns a float for a single parameter, else a ``(dim,)`` array.
    """
    arr = _chains_3d(chains)
    dim = arr.shape[2]
    out = np.empty(dim)
    for d in range(dim):
        col = arr[:, :, d]
        if method == "classic":
            out[d] = _rhat_split_one(col) if col.shape[0] >= 2 else _rhat_unsplit(col)
        elif method == "split":
            split = _split(col[:, :, None])[:, :, 0]
            out[d] = _rhat_split_one(split)
        elif method == "rank":
            split = _split(col[:, :, None])[:, :, 0]
            bulk = _rhat_split_one(_rank_normalize(split))
            folded = np.abs(split - np.median(split))
            tail = _rhat_split_one(_rank_normalize(folded))
            out[d] = max(bulk, tail)
        else:
            raise ValueError("method must be 'split', 'rank' or 'classic'")
    return float(out[0]) if dim == 1 else out


def _rhat_unsplit(x):
    m, n = x.shape
    if m < 2:
        raise ValueError("Rhat needs at least 2 chains")
    chain_means = x.mean(axis=1)
    chain_vars = x.var(axis=1, ddof=1)
    W = chain_vars.mean()
    B = n * chain_means.var(ddof=1)
    var_hat = ((n - 1) / n) * W + B / n
    return float(np.sqrt(var_hat / W)) if W > 0 else float("inf")


def ESS(chains, method="bulk"):
    """Effective sample size (autocorrelation-based; distinct from the importance-weight
    ESS reported by :func:`stochpylib.montecarlo.importance_sampling`).

    ``method="bulk"`` (default): rank-normalized split chains (robust to heavy tails).
    ``method="mean"``: raw split chains (classic, good for mean estimation). ``method="sd"``:
    minimum of the ESS of ``x`` and of ``(x - mean)**2`` (for variance estimation).
    ``method="tail"``: minimum of the ESS of the 5th- and 95th-percentile indicators on
    split chains. Returns a float for a single parameter, else a ``(dim,)`` array.
    """
    arr = _chains_3d(chains)
    dim = arr.shape[2]
    out = np.empty(dim)
    for d in range(dim):
        col = arr[:, :, d]
        split = _split(col[:, :, None])[:, :, 0]
        if method == "bulk":
            out[d] = _ess_one(_rank_normalize(split))
        elif method == "mean":
            out[d] = _ess_one(split)
        elif method == "sd":
            out[d] = min(_ess_one(_rank_normalize(split)),
                          _ess_one(_rank_normalize((split - split.mean()) ** 2)))
        elif method == "tail":
            q05, q95 = np.quantile(col, [0.05, 0.95])
            ind_lo = (split <= q05).astype(float)
            ind_hi = (split <= q95).astype(float)
            out[d] = min(_ess_one(ind_lo), _ess_one(ind_hi))
        else:
            raise ValueError("method must be 'bulk', 'mean', 'sd' or 'tail'")
    return float(out[0]) if dim == 1 else out


def GelmanRubin(chains):
    """Classic potential scale reduction factor with the Brooks-Gelman (1998) degrees-of-
    freedom correction. Needs at least 2 chains. Returns a float (single parameter) or
    ``(dim,)`` array.
    """
    arr = _chains_3d(chains)
    m = arr.shape[0]
    if m < 2:
        raise ValueError("GelmanRubin needs at least 2 chains")
    dim = arr.shape[2]
    out = np.empty(dim)
    for d in range(dim):
        x = arr[:, :, d]
        n = x.shape[1]
        chain_means = x.mean(axis=1)
        chain_vars = x.var(axis=1, ddof=1)
        W = chain_vars.mean()
        B = n * chain_means.var(ddof=1)
        V = (n - 1) / n * W + (m + 1) / (m * n) * B
        if W <= 0:
            out[d] = 1.0
            continue
        var_s2 = chain_vars.var(ddof=1) if m > 1 else 0.0
        xbar = chain_means.mean()
        cov_s2_xbar2 = np.cov(chain_vars, chain_means ** 2, ddof=1)[0, 1] if m > 1 else 0.0
        cov_s2_xbar = np.cov(chain_vars, chain_means, ddof=1)[0, 1] if m > 1 else 0.0
        var_V = (((n - 1) / n) ** 2 * var_s2 / m
                 + ((m + 1) / (m * n)) ** 2 * 2 * B ** 2 / (m - 1) if m > 1 else 0.0)
        var_V += 2 * (m + 1) * (n - 1) / (m * n ** 2) * (n / m) * (
            cov_s2_xbar2 - 2 * xbar * cov_s2_xbar) if m > 1 else 0.0
        df = 2 * V ** 2 / var_V if var_V > 0 else np.inf
        out[d] = float(np.sqrt((V / W) * (df + 3) / (df + 1)))
    return float(out[0]) if dim == 1 else out


def PSRF(chains):
    """Brooks & Gelman (1998) multivariate potential scale reduction factor.

    Needs at least 2 chains. For a single parameter this reduces to a scalar ratio
    (without the classic sqrt/df correction of :func:`GelmanRubin` -- the two agree only
    approximately).
    """
    arr = _chains_3d(chains)
    m, n, d = arr.shape
    if m < 2:
        raise ValueError("PSRF needs at least 2 chains")
    chain_means = arr.mean(axis=1)  # (m, d)
    W = np.zeros((d, d))
    for c in range(m):
        diff = arr[c] - chain_means[c]
        W += diff.T @ diff / (n - 1)
    W /= m
    B_n = np.cov(chain_means.T, ddof=1) if d > 1 else np.array([[chain_means[:, 0].var(ddof=1)]])
    B_n = np.atleast_2d(B_n)
    try:
        lam = np.linalg.eigvalsh(np.linalg.solve(W, B_n)).max()
    except np.linalg.LinAlgError:
        lam = np.linalg.eigvalsh(np.linalg.solve(W + 1e-10 * np.eye(d), B_n)).max()
    return float((n - 1) / n + (m + 1) / m * lam)


def autocorr_time(chain, c=5.0, method="sokal"):
    """Integrated autocorrelation time of one chain (averaged over chains if 3-D).

    ``chain``: ``(n,)``, ``(n, dim)`` or ``(n_chains, n, dim)``. ``method="sokal"``:
    self-consistent windowing, the smallest window ``M`` with ``M >= c * tau(M)``.
    ``method="geyer"``: Geyer's initial monotone positive sequence (as in :func:`ESS`).
    Returns a float or ``(dim,)`` array.
    """
    arr = np.asarray(chain, dtype=float)
    if arr.ndim == 1:
        chains_list = [arr]
        dim = 1
        squeeze = True
    elif arr.ndim == 2:
        chains_list = [arr[:, j] for j in range(arr.shape[1])]
        dim = arr.shape[1]
        squeeze = False
    elif arr.ndim == 3:
        dim = arr.shape[2]
        chains_list = None
        squeeze = False
    else:
        raise ValueError("chain must have 1, 2 or 3 dimensions")

    def _tau_1d_multi(xs):
        acovs = [_autocov_fft(x) for x in xs]
        n = min(len(a) for a in acovs)
        acov = np.mean([a[:n] for a in acovs], axis=0)
        var0 = acov[0]
        if var0 <= 0:
            return 1.0
        rho = acov / var0
        if method == "sokal":
            tau = 1.0
            for M in range(1, n):
                tau = 1.0 + 2.0 * np.sum(rho[1:M + 1])
                tau = max(tau, 1.0 / n)
                if M >= c * tau:
                    return float(tau)
            return float(max(tau, 1.0))
        elif method == "geyer":
            n_pairs = (n - 1) // 2
            if n_pairs <= 0:
                return 1.0
            P = rho[1:1 + 2 * n_pairs:2] + rho[2:2 + 2 * n_pairs:2]
            running_min = np.inf
            cum = 0.0
            for k in range(len(P)):
                if P[k] < 0:
                    break
                running_min = min(running_min, P[k])
                cum += running_min
            tau = 1.0 + 2.0 * cum
            return float(max(tau, 1.0))
        else:
            raise ValueError("method must be 'sokal' or 'geyer'")

    out = np.empty(dim)
    for d in range(dim):
        if chains_list is not None and arr.ndim <= 2:
            xs = [chains_list[d]] if dim > 1 else [chains_list[0]]
        else:
            xs = [arr[k, :, d] for k in range(arr.shape[0])]
        out[d] = _tau_1d_multi(xs)
    return float(out[0]) if squeeze or dim == 1 else out


def geweke_test(chain, first=0.1, last=0.5):
    """Geweke's (1992) convergence z-test comparing the mean of an early and a late
    segment of one chain, using spectral-density-based standard errors (Geyer's initial
    monotone sequence on each segment). Returns a ``stochpylib.statistics.TestResult``.
    """
    from stochpylib.statistics import TestResult

    arr = np.asarray(chain, dtype=float)
    if arr.ndim == 1:
        return _geweke_one(arr, first, last)
    if arr.ndim == 2:
        stats = []
        pvals = []
        for j in range(arr.shape[1]):
            r = _geweke_one(arr[:, j], first, last)
            stats.append(r.statistic)
            pvals.append(r.pvalue)
        return TestResult(statistic=np.array(stats), pvalue=np.array(pvals),
                           null="chain segments share a mean", method="Geweke",
                           extras={"first": first, "last": last})
    raise ValueError("chain must be 1-D or 2-D")


def _geweke_one(x, first, last):
    from stochpylib.statistics import TestResult

    n = len(x)
    nA = max(2, int(n * first))
    nB = max(2, int(n * last))
    A = x[:nA]
    B = x[n - nB:]
    tauA = autocorr_time(A, method="geyer")
    tauB = autocorr_time(B, method="geyer")
    varA = A.var(ddof=1) * tauA / nA
    varB = B.var(ddof=1) * tauB / nB
    denom = np.sqrt(varA + varB)
    z = float((A.mean() - B.mean()) / denom) if denom > 0 else 0.0
    p = float(2.0 * (1.0 - special.ndtr(abs(z))))
    return TestResult(statistic=z, pvalue=p, null="chain segments share a mean",
                       method="Geweke", extras={"first": first, "last": last})


@dataclass
class RafteryLewisResult:
    """Raftery & Lewis (1992) diagnostic: burn-in and run length needed to estimate the
    ``q``-quantile to within ``+-r`` with probability ``s``."""

    burn_in: int
    n_total: int
    n_min: int
    thin: int
    dependence_factor: float
    alpha: float = float("nan")
    beta: float = float("nan")
    q: float = 0.025
    r: float = 0.005
    s: float = 0.95
    extras: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"RafteryLewisResult(burn_in={self.burn_in}, n_total={self.n_total}, "
                f"n_min={self.n_min}, thin={self.thin}, "
                f"dependence_factor={self.dependence_factor:.3g})")


def raftery_lewis(chain, q=0.025, r=0.005, s=0.95, eps=0.001):
    """Raftery & Lewis (1992) run-length diagnostic for one univariate chain."""
    x = np.asarray(chain, dtype=float).ravel()
    if x.ndim != 1:
        raise ValueError("raftery_lewis takes a single univariate chain")
    n = len(x)
    thresh = np.quantile(x, q)
    z_full = (x <= thresh).astype(int)

    k = 1
    while True:
        zk = z_full[::k]
        nk = len(zk)
        if nk < 20:
            break
        # 2nd-order Markov transition counts on the thinned binary sequence
        t = np.zeros((2, 2, 2), dtype=float)
        for i in range(2, nk):
            t[zk[i - 2], zk[i - 1], zk[i]] += 1
        g2 = 0.0
        for a in range(2):
            for b in range(2):
                row_sum = t[a, b, 0] + t[a, b, 1]
                if row_sum == 0:
                    continue
                col_totals = t[:, b, :].sum(axis=0)
                col_total = col_totals.sum()
                for c in range(2):
                    obs = t[a, b, c]
                    if obs <= 0:
                        continue
                    exp = row_sum * (col_totals[c] / col_total) if col_total > 0 else 0
                    if exp > 0:
                        g2 += 2.0 * obs * np.log(obs / exp)
        bic = g2 - np.log(nk) * 2.0
        if bic < 0:
            break
        k += 1
        if k > n:
            k = max(1, k - 1)
            break

    zk = z_full[::k]
    nk = len(zk)
    t1 = np.zeros((2, 2))
    for i in range(1, nk):
        t1[zk[i - 1], zk[i]] += 1
    row0 = t1[0].sum()
    row1 = t1[1].sum()
    alpha = t1[0, 1] / row0 if row0 > 0 else 0.5
    beta = t1[1, 0] / row1 if row1 > 0 else 0.5
    alpha = min(max(alpha, 1e-8), 1 - 1e-8)
    beta = min(max(beta, 1e-8), 1 - 1e-8)

    phi = special.ndtri((1.0 + s) / 2.0)
    denom_lag = np.log(abs(1.0 - alpha - beta))
    if denom_lag == 0:
        m_burn = 1
    else:
        m_burn = int(np.ceil(np.log(eps * (alpha + beta) / max(alpha, beta)) / denom_lag))
    burn_in = k * max(m_burn, 0)

    n_needed = ((2.0 - alpha - beta) * alpha * beta / (alpha + beta) ** 3) * (phi / r) ** 2
    n_total = k * int(np.ceil(n_needed))
    n_min = int(np.ceil(phi ** 2 * q * (1.0 - q) / r ** 2))
    dep_factor = n_total / n_min if n_min > 0 else float("nan")

    return RafteryLewisResult(burn_in=burn_in, n_total=max(n_total, n_min), n_min=n_min,
                               thin=k, dependence_factor=float(dep_factor), alpha=alpha,
                               beta=beta, q=q, r=r, s=s)


class TraceAnalysis:
    """Convenience wrapper bundling every diagnostic over a set of chains.

    ``chains``: ``(n_chains, n, dim)`` (or anything :func:`ESS`/:func:`Rhat` accept).
    """

    def __init__(self, chains, names=None):
        self.chains_ = _chains_3d(chains)
        dim = self.chains_.shape[2]
        self.names = list(names) if names is not None else [f"theta{i}" for i in range(dim)]
        if len(self.names) != dim:
            raise ValueError("names must have length dim")

    @property
    def flat(self):
        m, n, d = self.chains_.shape
        return self.chains_.reshape(m * n, d)

    def mean(self):
        return self.flat.mean(axis=0)

    def std(self):
        return self.flat.std(axis=0, ddof=1)

    def quantiles(self, probs=(0.05, 0.5, 0.95)):
        return np.quantile(self.flat, probs, axis=0)

    def rhat(self, method="split"):
        return Rhat(self.chains_, method=method)

    def ess(self, method="bulk"):
        return ESS(self.chains_, method=method)

    def mcse(self):
        ess = np.atleast_1d(self.ess(method="mean"))
        return self.std() / np.sqrt(np.maximum(ess, 1.0))

    def autocorr(self, max_lag=50):
        m, n, d = self.chains_.shape
        max_lag = min(max_lag, n - 1)
        out = np.empty((d, max_lag + 1))
        for j in range(d):
            acovs = [_autocov_fft(self.chains_[c, :, j])[:max_lag + 1] for c in range(m)]
            acov = np.mean(acovs, axis=0)
            out[j] = acov / acov[0] if acov[0] > 0 else np.zeros(max_lag + 1)
        return out

    def geweke(self, first=0.1, last=0.5):
        flat_per_chain = self.chains_.reshape(-1, self.chains_.shape[2]) \
            if self.chains_.shape[0] == 1 else self.chains_[0]
        return geweke_test(flat_per_chain, first=first, last=last)

    def running_mean(self):
        return np.cumsum(self.chains_, axis=1) / np.arange(1, self.chains_.shape[1] + 1)[None, :, None]

    def thin(self, k):
        return TraceAnalysis(self.chains_[:, ::k, :], names=self.names)

    def summary(self):
        mean = self.mean()
        std = self.std()
        mcse = self.mcse()
        q = self.quantiles()
        rhat = np.atleast_1d(self.rhat())
        ess_bulk = np.atleast_1d(self.ess("bulk"))
        ess_tail = np.atleast_1d(self.ess("tail"))
        rows = []
        d = self.chains_.shape[2]
        m, n = self.chains_.shape[0], self.chains_.shape[1]
        for j in range(d):
            rows.append({
                "name": self.names[j],
                "mean": float(mean[j]),
                "sd": float(std[j]),
                "mcse": float(mcse[j]),
                "q5": float(q[0, j]),
                "median": float(q[1, j]),
                "q95": float(q[2, j]),
                "ess_bulk": float(ess_bulk[j]),
                "ess_tail": float(ess_tail[j]),
                "rhat": float(rhat[j]) if (m >= 2 and n >= 4) else float("nan"),
            })
        return rows

    def summary_table(self):
        rows = self.summary()
        header = f"{'name':>10s} {'mean':>10s} {'sd':>10s} {'mcse':>10s} {'ess_bulk':>10s} {'rhat':>8s}"
        lines = [header]
        for r in rows:
            lines.append(f"{r['name']:>10s} {r['mean']:>10.4g} {r['sd']:>10.4g} "
                         f"{r['mcse']:>10.4g} {r['ess_bulk']:>10.4g} {r['rhat']:>8.4g}")
        return "\n".join(lines)

    def __repr__(self):
        m, n, d = self.chains_.shape
        return f"TraceAnalysis(n_chains={m}, n_samples={n}, dim={d})"
