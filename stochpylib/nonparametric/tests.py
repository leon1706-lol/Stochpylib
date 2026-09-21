"""Distribution-free hypothesis tests: permutation/bootstrap resampling
tests, Mood/Kruskal-Wallis/Friedman rank tests, sign/runs tests, and the
Anderson-Darling/Cramer-von Mises goodness-of-fit family.

Every class subclasses :class:`~stochpylib.nonparametric._base.NonparametricTest`
and returns a shared :class:`stochpylib.statistics.TestResult`.
"""

import itertools
import math

import numpy as np
from scipy import special

from stochpylib.nonparametric._base import NonparametricTest
from stochpylib.nonparametric._common import (
    _chi2_ppf, _chi2_sf, _norm_cdf, _norm_ppf, _rank_ties,
)
from stochpylib.statistics._result import TestResult

__all__ = ["PermutationTest", "BootstrapTest", "MoodTest", "KruskalWallis",
           "FriedmanTest", "SignTest", "RunsTest", "WaldWolfowitz",
           "AndersenDarling", "CramerVonMises"]

_EXACT_LIMIT = 20000


# --------------------------------------------------------------------- resampling

class PermutationTest(NonparametricTest):
    """Permutation test over two independent samples (``permutation_type=
    "independent"``, the default), paired data with a permuted correspondence
    (``"pairings"``), or a single (or paired-difference) sample tested by
    random sign flips (``"samples"``).

    Exact enumeration is used whenever the number of distinct permutations is
    <= 20000 (matching ``scipy.stats.permutation_test(n_resamples=np.inf)``
    exactly for ``"independent"``); above that, Monte Carlo with
    ``n_resamples`` draws and the ``(count + 1) / (B + 1)`` p-value correction.
    """

    def __init__(self, statistic="mean_diff", n_resamples=9999,
                 permutation_type="independent", alternative="two-sided",
                 random_state=None):
        self.statistic = statistic
        self.n_resamples = n_resamples
        self.permutation_type = permutation_type
        self.alternative = alternative
        self.random_state = random_state

    def _two_sample_stat(self):
        if callable(self.statistic):
            return self.statistic
        if self.statistic == "mean_diff":
            return lambda a, b: float(np.mean(a) - np.mean(b))
        if self.statistic == "median_diff":
            return lambda a, b: float(np.median(a) - np.median(b))
        if self.statistic == "t":
            def _t(a, b):
                na, nb = len(a), len(b)
                va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
                se = math.sqrt(va / na + vb / nb)
                return float((np.mean(a) - np.mean(b)) / se) if se > 0 else 0.0
            return _t
        raise ValueError(f"unknown statistic {self.statistic!r}")

    def _one_sample_stat(self):
        if callable(self.statistic):
            return self.statistic
        return lambda a: float(np.mean(a))

    def _pvalue(self, obs, null, exact):
        """Matches ``scipy.stats.permutation_test``'s convention exactly:
        two-sided is ``2 * min(p_greater, p_less)``, not a symmetric
        ``|null| >= |obs|`` count (the two differ whenever the null
        distribution is itself asymmetric, e.g. unequal group sizes)."""
        n = len(null)
        if exact:
            p_greater = float(np.mean(null >= obs))
            p_less = float(np.mean(null <= obs))
        else:
            p_greater = float((np.sum(null >= obs) + 1) / (n + 1))
            p_less = float((np.sum(null <= obs) + 1) / (n + 1))
        if self.alternative == "greater":
            return p_greater
        if self.alternative == "less":
            return p_less
        return float(min(1.0, 2.0 * min(p_greater, p_less)))

    def _fit(self, *samples):
        rng = np.random.default_rng(self.random_state)
        B = self.n_resamples

        if self.permutation_type == "independent":
            x, y = (np.asarray(s, dtype=float) for s in samples)
            fn = self._two_sample_stat()
            pooled = np.concatenate([x, y])
            n1, n = len(x), len(pooled)
            obs = fn(x, y)
            total_exact = math.comb(n, n1)
            if total_exact <= _EXACT_LIMIT:
                null = np.empty(total_exact)
                for i, idx in enumerate(itertools.combinations(range(n), n1)):
                    mask = np.zeros(n, dtype=bool)
                    mask[list(idx)] = True
                    null[i] = fn(pooled[mask], pooled[~mask])
                exact = True
            else:
                null = np.empty(B)
                for b in range(B):
                    perm = rng.permutation(n)
                    mask = np.zeros(n, dtype=bool)
                    mask[perm[:n1]] = True
                    null[b] = fn(pooled[mask], pooled[~mask])
                exact = False

        elif self.permutation_type == "pairings":
            x, y = (np.asarray(s, dtype=float) for s in samples)
            fn = self._two_sample_stat()
            n = len(x)
            obs = fn(x, y)
            if math.factorial(n) <= _EXACT_LIMIT:
                perms = list(itertools.permutations(range(n)))
                null = np.array([fn(x, y[list(p)]) for p in perms])
                exact = True
            else:
                null = np.empty(B)
                for b in range(B):
                    null[b] = fn(x, rng.permutation(y))
                exact = False

        elif self.permutation_type == "samples":
            fn = self._one_sample_stat()
            if len(samples) == 1:
                d = np.asarray(samples[0], dtype=float)
            else:
                x, y = (np.asarray(s, dtype=float) for s in samples)
                d = x - y
            n = len(d)
            obs = fn(d)
            if 2 ** n <= _EXACT_LIMIT:
                null = np.array([fn(d * np.array(bits))
                                  for bits in itertools.product([1, -1], repeat=n)])
                exact = True
            else:
                null = np.empty(B)
                for b in range(B):
                    signs = rng.choice([-1.0, 1.0], size=n)
                    null[b] = fn(d * signs)
                exact = False
        else:
            raise ValueError("permutation_type must be 'independent', "
                              "'pairings', or 'samples'")

        pvalue = self._pvalue(obs, null, exact)
        self.null_distribution_ = null
        self.exact_ = exact
        return TestResult(float(obs), pvalue, None, "no effect", "permutation",
                           self.alternative,
                           extras={"exact": exact, "n_resamples": len(null)})


class BootstrapTest(NonparametricTest):
    """Efron-Tibshirani bootstrap hypothesis test: one-sample (``H0: statistic
    (x) == mu0``, resampling from ``x`` recentered to ``mu0``) or two-sample
    (``H0: statistic(x) == statistic(y)``, both recentered to the pooled
    value). P-value uses the ``(count + 1) / (B + 1)`` correction.
    """

    def __init__(self, statistic="mean", mu0=0.0, n_boot=2000,
                 alternative="two-sided", random_state=None):
        self.statistic = statistic
        self.mu0 = mu0
        self.n_boot = n_boot
        self.alternative = alternative
        self.random_state = random_state

    def _fit(self, *samples):
        rng = np.random.default_rng(self.random_state)
        B = self.n_boot
        n_args = len(samples)
        fn = self.statistic if callable(self.statistic) else (
            (lambda a: float(np.mean(a))) if n_args == 1
            else (lambda a, b: float(np.mean(a) - np.mean(b))))

        if n_args == 1:
            x = np.asarray(samples[0], dtype=float)
            obs = fn(x)
            x0 = x - np.mean(x) + self.mu0
            null = np.array([fn(rng.choice(x0, size=len(x0), replace=True))
                              for _ in range(B)])
            center = self.mu0
            null_op = "mean"
        else:
            x, y = (np.asarray(s, dtype=float) for s in samples)
            obs = fn(x, y)
            pooled_mean = float(np.mean(np.concatenate([x, y])))
            x0 = x - np.mean(x) + pooled_mean
            y0 = y - np.mean(y) + pooled_mean
            null = np.array([
                fn(rng.choice(x0, size=len(x0), replace=True),
                   rng.choice(y0, size=len(y0), replace=True))
                for _ in range(B)])
            center = 0.0
            null_op = "difference"

        d_obs, d_null = obs - center, null - center
        if self.alternative == "greater":
            count = np.sum(d_null >= d_obs)
        elif self.alternative == "less":
            count = np.sum(d_null <= d_obs)
        else:
            count = np.sum(np.abs(d_null) >= abs(d_obs))
        pvalue = float((count + 1) / (B + 1))
        self.null_distribution_ = null
        return TestResult(float(obs), pvalue, None, f"no {null_op}", "bootstrap",
                           self.alternative, extras={"n_boot": B})


# --------------------------------------------------------------------- rank tests

class MoodTest(NonparametricTest):
    """Mood's two-sample test: ``kind="scale"`` (rank-dispersion test for equal
    scale, matches ``scipy.stats.mood`` exactly) or ``kind="median"`` (median
    test via a 2xk contingency table with Yates' correction on ties assigned to
    the "below" group, matching ``scipy.stats.median_test(correction=True,
    ties="below")`` exactly).
    """

    def __init__(self, kind="scale", alternative="two-sided"):
        self.kind = kind
        self.alternative = alternative

    def _fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m, k = len(x), len(y)
        N = m + k
        if self.kind == "scale":
            pooled = np.concatenate([x, y])
            ranks = _rank_ties(pooled)
            Rx = ranks[:m]
            T = float(np.sum((Rx - (N + 1) / 2.0) ** 2))
            ET = m * (N ** 2 - 1) / 12.0
            VarT = m * k * (N + 1) * (N ** 2 - 4) / 180.0
            z = (T - ET) / math.sqrt(VarT)
            if self.alternative == "two-sided":
                p = 2.0 * float(_norm_cdf(-abs(z)))
            elif self.alternative == "greater":
                p = float(_norm_cdf(-z))
            else:
                p = float(_norm_cdf(z))
            return TestResult(float(z), float(np.clip(p, 0, 1)), None,
                               "equal scale", "Mood scale test", self.alternative,
                               extras={"T": T})
        if self.kind == "median":
            from stochpylib.statistics.hypothesis import chi2_test
            med = float(np.median(np.concatenate([x, y])))
            above_x, above_y = int(np.sum(x > med)), int(np.sum(y > med))
            below_x, below_y = m - above_x, k - above_y
            table = np.array([[above_x, above_y], [below_x, below_y]], dtype=float)
            res = chi2_test(table, yates=(table.shape == (2, 2)))
            res.method = "Mood median test"
            res.null = "equal medians"
            return res
        raise ValueError("kind must be 'scale' or 'median'")


class KruskalWallis(NonparametricTest):
    """Kruskal-Wallis H test (tie-corrected, matches ``scipy.stats.kruskal``
    exactly). ``posthoc_dunn(adjust=...)`` runs pairwise Dunn z-tests with a
    multiple-comparison adjustment (delegates to
    :func:`stochpylib.statistics.bonferroni`).
    """

    def _fit(self, *groups):
        groups = [np.asarray(g, dtype=float) for g in groups]
        self._groups = groups
        pooled = np.concatenate(groups)
        N = len(pooled)
        ranks = _rank_ties(pooled)
        self._pooled_ranks = ranks
        idx = np.cumsum([0] + [len(g) for g in groups])
        H = 0.0
        for i in range(len(groups)):
            ni = len(groups[i])
            Ri = ranks[idx[i]:idx[i + 1]].sum()
            H += Ri ** 2 / ni
        H = 12.0 / (N * (N + 1)) * H - 3.0 * (N + 1)
        vals, counts = np.unique(ranks, return_counts=True)
        tie_corr = 1.0 - np.sum(counts ** 3 - counts) / (N ** 3 - N)
        H = H / tie_corr if tie_corr > 0 else H
        df = len(groups) - 1
        pvalue = float(_chi2_sf(H, df))
        self._idx = idx
        return TestResult(float(H), pvalue, df, "all groups have the same distribution",
                           "Kruskal-Wallis")

    def posthoc_dunn(self, adjust="bonferroni"):
        from stochpylib.statistics.hypothesis import bonferroni
        groups, ranks, idx = self._groups, self._pooled_ranks, self._idx
        N = len(ranks)
        vals, counts = np.unique(ranks, return_counts=True)
        tie_term = np.sum(counts ** 3 - counts)
        k = len(groups)
        mean_ranks = [ranks[idx[i]:idx[i + 1]].mean() for i in range(k)]
        ns = [len(g) for g in groups]
        sigma2 = (N * (N + 1) / 12.0 - tie_term / (12.0 * (N - 1)))
        pairs, pvals, zs = [], [], []
        for i in range(k):
            for j in range(i + 1, k):
                se = math.sqrt(sigma2 * (1.0 / ns[i] + 1.0 / ns[j]))
                z = (mean_ranks[i] - mean_ranks[j]) / se if se > 0 else 0.0
                p = 2.0 * float(_norm_cdf(-abs(z)))
                pairs.append((i, j))
                zs.append(z)
                pvals.append(p)
        adj = bonferroni(pvals, method=adjust)
        adj_p = adj.extras["adjusted_pvalues"]
        table = [{"i": i, "j": j, "z": float(z), "p": float(p), "p_adj": float(pa)}
                  for (i, j), z, p, pa in zip(pairs, zs, pvals, adj_p)]
        return TestResult(float(np.max(np.abs(zs))), float(np.min(adj_p)), None,
                           "all pairwise groups are equal", f"Dunn post-hoc ({adjust})",
                           table=table)


class FriedmanTest(NonparametricTest):
    """Friedman rank test for repeated measures (matches
    ``scipy.stats.friedmanchisquare`` exactly, tie-corrected).
    ``kendall_w_`` is Kendall's coefficient of concordance.
    """

    def _fit(self, *groups):
        data = np.column_stack([np.asarray(g, dtype=float) for g in groups])
        n, k = data.shape
        ranks = np.array([_rank_ties(row) for row in data])
        Rj = ranks.sum(axis=0)
        ties = 0.0
        for row in ranks:
            vals, counts = np.unique(row, return_counts=True)
            ties += np.sum(counts ** 3 - counts)
        c = 1.0 - ties / (k * (k * k - 1) * n)
        stat = (12.0 / (n * k * (k + 1)) * np.sum(Rj ** 2) - 3 * n * (k + 1)) / c
        df = k - 1
        pvalue = float(_chi2_sf(stat, df))
        self.kendall_w_ = float(stat / (n * (k - 1))) if n * (k - 1) > 0 else float("nan")
        return TestResult(float(stat), pvalue, df, "no treatment effect", "Friedman")


# --------------------------------------------------------------------- sign / runs

class SignTest(NonparametricTest):
    """One-sample (or paired, via two equal-length arrays) exact sign test of
    ``H0: median(x) == mu0`` -- matches an exact two-sided binomial test at
    p=0.5 on the count of positive signs.
    """

    def __init__(self, mu0=0.0, alternative="two-sided"):
        self.mu0 = mu0
        self.alternative = alternative

    def _fit(self, *samples):
        if len(samples) == 1:
            d = np.asarray(samples[0], dtype=float) - self.mu0
        else:
            x, y = (np.asarray(s, dtype=float) for s in samples)
            d = x - y
        d = d[d != 0]
        n = len(d)
        k = int(np.sum(d > 0))
        from stochpylib.statistics._common import _rng  # noqa: F401 (keep import local)

        def _binom_sf_ge(k, n, p=0.5):
            i = np.arange(k, n + 1)
            return float(np.sum(special.comb(n, i) * p ** i * (1 - p) ** (n - i)))

        if self.alternative == "two-sided":
            lo, hi = min(k, n - k), max(k, n - k)
            p_lo = _binom_sf_ge(hi, n)
            pvalue = float(np.clip(2.0 * p_lo, 0.0, 1.0))
        elif self.alternative == "greater":
            pvalue = _binom_sf_ge(k, n)
        else:
            pvalue = _binom_sf_ge(n - k, n)
        return TestResult(float(k), pvalue, None, f"median = {self.mu0}", "sign test",
                           self.alternative, extras={"n_effective": n})


def _runs_test_stat(indicator, correction=True):
    """Wald-Wolfowitz normal-approximation runs-test z/p (statsmodels
    ``Runs.runs_test`` convention: continuity correction only for n < 50)."""
    indicator = np.asarray(indicator)
    change = np.flatnonzero(np.diff(indicator)) + 1
    n_runs = len(change) + 1
    n1 = int(np.sum(indicator == 1))
    n2 = int(np.sum(indicator == 0))
    n = n1 + n2
    npn = n1 * n2
    rmean = 2.0 * npn / n + 1
    rvar = 2.0 * npn * (2.0 * npn - n) / (n ** 2 * (n - 1))
    rstd = math.sqrt(rvar)
    rdemean = n_runs - rmean
    if n >= 50 or not correction:
        z = rdemean
    elif rdemean > 0.5:
        z = rdemean - 0.5
    elif rdemean < -0.5:
        z = rdemean + 0.5
    else:
        z = 0.0
    z = z / rstd
    pvalue = 2.0 * float(_norm_cdf(-abs(z)))
    return float(z), float(pvalue), n_runs


class RunsTest(NonparametricTest):
    """Wald-Wolfowitz one-sample runs test for randomness: dichotomizes ``x``
    at a cutoff (``"median"``/``"mean"``/a number; ties go to the "below"
    group) and tests whether the number of runs above/below is consistent
    with a random sequence. Matches
    ``statsmodels.sandbox.stats.runs.runstest_1samp`` exactly.
    """

    def __init__(self, cutoff="median", correction=True):
        self.cutoff = cutoff
        self.correction = correction

    def _fit(self, x):
        x = np.asarray(x, dtype=float)
        if self.cutoff == "median":
            c = float(np.median(x))
        elif self.cutoff == "mean":
            c = float(np.mean(x))
        else:
            c = float(self.cutoff)
        indicator = (x >= c).astype(int)
        z, p, n_runs = _runs_test_stat(indicator, self.correction)
        return TestResult(z, p, None, "the sequence is random", "runs test",
                           extras={"n_runs": n_runs, "cutoff": c})


class WaldWolfowitz(NonparametricTest):
    """Wald-Wolfowitz two-sample runs test: pools and sorts ``x``/``y``,
    counts runs of consecutive same-origin labels. Matches
    ``statsmodels.sandbox.stats.runs.runstest_2samp`` exactly on untied data.
    """

    def __init__(self, correction=True):
        self.correction = correction

    def _fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        pooled = np.concatenate([x, y])
        labels = np.concatenate([np.zeros(len(x)), np.ones(len(y))])
        order = np.argsort(pooled, kind="mergesort")
        indicator = labels[order].astype(int)
        z, p, n_runs = _runs_test_stat(indicator, self.correction)
        return TestResult(z, p, None, "x and y come from the same distribution",
                           "Wald-Wolfowitz", extras={"n_runs": n_runs})


# --------------------------------------------------------------------- goodness of fit

_AD_NORM_CRIT = np.array([0.561, 0.631, 0.752, 0.873, 1.035])
_AD_EXPON_CRIT = np.array([0.916, 1.062, 1.321, 1.591, 1.959])
_AD_SIG = np.array([15.0, 10.0, 5.0, 2.5, 1.0]) / 100.0
# Case-0 ("all parameters known") asymptotic critical values, D'Agostino &
# Stephens (1986) Table 4.2 -- used only for a fully-specified reference dist.
_AD_KNOWN_CRIT = np.array([1.248, 1.610, 1.933, 2.492, 3.070, 3.857, 4.500])
_AD_KNOWN_SIG = np.array([25.0, 15.0, 10.0, 5.0, 2.5, 1.0, 0.5]) / 100.0


class AndersenDarling(NonparametricTest):
    """One-sample Anderson-Darling goodness-of-fit test.

    ``dist="norm"``/``"expon"`` fit the location/scale by MLE and apply
    Stephens' finite-sample modification, matching ``scipy.stats.anderson``'s
    statistic exactly (p-value via the same critical-value-table
    interpolation as ``method="interpolate"``). ``dist=`` a
    :class:`~stochpylib.distributions._base.Distribution` instance or a plain
    CDF callable tests against a *fully specified* reference distribution
    (no fitting); its p-value is interpolated from the case-0 asymptotic
    critical-value table (D'Agostino & Stephens 1986, Table 4.2) -- an
    approximation, documented in the module README.

    ``k_sample=True`` runs the k-sample Anderson-Darling test (Scholz &
    Stephens 1987, midrank variant), matching
    ``scipy.stats.anderson_ksamp(variant="midrank")`` exactly.
    """

    def __init__(self, dist="norm", k_sample=False):
        self.dist = dist
        self.k_sample = k_sample

    def _fit(self, *samples):
        if self.k_sample:
            return self._fit_ksample(samples)
        (x,) = samples
        x = np.asarray(x, dtype=float)
        y = np.sort(x)
        n = len(y)
        xbar = float(np.mean(x))
        if self.dist == "norm":
            s = float(np.std(x, ddof=1))
            w = (y - xbar) / s
            logcdf = special.log_ndtr(w)
            logsf = special.log_ndtr(-w)
            crit = np.round(_AD_NORM_CRIT / (1.0 + 0.75 / n + 2.25 / n ** 2), 3)
            sig = _AD_SIG
        elif self.dist == "expon":
            w = y / xbar
            logcdf = np.log1p(-np.exp(-w))
            logsf = -w
            crit = np.round(_AD_EXPON_CRIT / (1.0 + 0.6 / n), 3)
            sig = _AD_SIG
        else:
            cdf_fn = self.dist.cdf if hasattr(self.dist, "cdf") else self.dist
            F = np.clip(np.asarray([cdf_fn(v) for v in y], dtype=float), 1e-300, 1 - 1e-300)
            logcdf = np.log(F)
            logsf = np.log1p(-F)
            crit, sig = _AD_KNOWN_CRIT, _AD_KNOWN_SIG

        i = np.arange(1, n + 1)
        A2 = -n - np.sum((2 * i - 1.0) / n * (logcdf + logsf[::-1]))
        pvalue = float(np.clip(np.interp(A2, crit, sig), sig.min(), sig.max()))
        return TestResult(float(A2), pvalue, None, f"data follow {self.dist!r}",
                           "Anderson-Darling", extras={"critical_values": crit,
                                                        "significance_levels": sig})

    def _fit_ksample(self, samples):
        samples = [np.asarray(s, dtype=float) for s in samples]
        k = len(samples)
        Z = np.sort(np.concatenate(samples))
        N = Z.size
        Zstar = np.unique(Z)
        n = np.array([len(s) for s in samples])

        Z_left = np.searchsorted(Z, Zstar, side="left")
        if N == Zstar.size:
            lj = 1.0
        else:
            lj = np.searchsorted(Z, Zstar, side="right") - Z_left
        Bj = Z_left + lj / 2.0

        A2akN = 0.0
        for i in range(k):
            s = np.sort(samples[i])
            right = np.searchsorted(s, Zstar, side="right").astype(float)
            left = np.searchsorted(s, Zstar, side="left")
            Mij = right - (right - left) / 2.0
            inner = lj / float(N) * (N * Mij - Bj * n[i]) ** 2 / (Bj * (N - Bj) - N * lj / 4.0)
            A2akN += inner.sum() / n[i]
        A2akN *= (N - 1.0) / N

        H = np.sum(1.0 / n)
        hs_cs = np.cumsum(1.0 / np.arange(N - 1, 1, -1))
        h = hs_cs[-1] + 1
        g = np.sum(hs_cs / np.arange(2, N))
        a = (4 * g - 6) * (k - 1) + (10 - 6 * g) * H
        b = (2 * g - 4) * k ** 2 + 8 * h * k + (2 * g - 14 * h - 4) * H - 8 * h + 4 * g - 6
        c = (6 * h + 2 * g - 2) * k ** 2 + (4 * h - 4 * g + 6) * k + (2 * h - 6) * H + 4 * h
        d = (2 * h + 6) * k ** 2 - 4 * h * k
        sigmasq = (a * N ** 3 + b * N ** 2 + c * N + d) / ((N - 1.0) * (N - 2.0) * (N - 3.0))
        m = k - 1
        A2 = (A2akN - m) / math.sqrt(sigmasq)

        b0 = np.array([0.675, 1.281, 1.645, 1.96, 2.326, 2.573, 3.085])
        b1 = np.array([-0.245, 0.25, 0.678, 1.149, 1.822, 2.364, 3.615])
        b2 = np.array([-0.105, -0.305, -0.362, -0.391, -0.396, -0.345, -0.154])
        critical = b0 + b1 / math.sqrt(m) + b2 / m
        sig = np.array([0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001])

        if A2 < critical.min():
            p = float(sig.max())
        elif A2 > critical.max():
            p = float(sig.min())
        else:
            pf = np.polyfit(critical, np.log(sig), 2)
            p = float(math.exp(np.polyval(pf, A2)))

        return TestResult(float(A2), p, m, "all k samples come from the same distribution",
                           "Anderson-Darling k-sample", extras={"critical_values": critical})


def _cdf_cvm_inf(x):
    """Asymptotic (n -> infinity) CDF of the Cramer-von Mises statistic
    (Csorgo & Faraway 1996, eq 1.2/1.3) -- a convergent series in modified
    Bessel functions of the second kind, native via scipy.special."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    tot = np.zeros_like(x)
    cond = np.ones_like(x, dtype=bool)
    k = 0
    while np.any(cond) and k < 300:
        idx = np.flatnonzero(cond)
        xc = x[idx]
        u = np.exp(special.gammaln(k + 0.5) - special.gammaln(k + 1)) / \
            (np.pi ** 1.5 * np.sqrt(xc))
        y = 4 * k + 1
        q = y ** 2 / (16.0 * xc)
        b = special.kv(0.25, q)
        z = u * math.sqrt(y) * np.exp(-q) * b
        tot[idx] += z
        cond[idx] = np.abs(z) >= 1e-7
        k += 1
    return tot


class CramerVonMises(NonparametricTest):
    """Cramer-von Mises goodness-of-fit: one-sample (``fit(x, cdf)``, statistic
    matches ``scipy.stats.cramervonmises`` exactly; p-value from the
    asymptotic null distribution, an approximation for small n -- see the
    module README) or two-sample (``fit(x, y)``, matches
    ``scipy.stats.cramervonmises_2samp(method="asymptotic")`` exactly).
    """

    def _fit(self, *args):
        x, second = args
        if hasattr(second, "cdf") or callable(second):
            return self._fit_one_sample(x, second)
        return self._fit_two_sample(np.asarray(x, dtype=float), np.asarray(second, dtype=float))

    def _fit_one_sample(self, x, cdf):
        x = np.asarray(x, dtype=float)
        n = len(x)
        y = np.sort(x)
        cdf_fn = cdf.cdf if hasattr(cdf, "cdf") else cdf
        F = np.asarray([cdf_fn(v) for v in y], dtype=float)
        i = np.arange(1, n + 1)
        u = (2 * i - 1) / (2.0 * n)
        w = 1.0 / (12 * n) + np.sum((u - F) ** 2)
        p = float(np.clip(1.0 - _cdf_cvm_inf(np.array([w]))[0], 0.0, 1.0))
        return TestResult(float(w), p, None, "data follow the specified distribution",
                           "Cramer-von Mises")

    def _fit_two_sample(self, x, y):
        nx, ny = len(x), len(y)
        N = nx + ny
        # rank the SORTED x/y concatenation (not the original order): the
        # statistic compares each sample's own order statistics to their
        # expected pooled rank, so x and y must each be sorted first.
        pooled = np.concatenate([np.sort(x), np.sort(y)])
        r = _rank_ties(pooled)
        rx, ry = r[:nx], r[nx:]
        u = nx * np.sum((rx - np.arange(1, nx + 1)) ** 2) + \
            ny * np.sum((ry - np.arange(1, ny + 1)) ** 2)
        k = nx * ny
        t = u / (k * N) - (4 * k - 1) / (6.0 * N)

        et = (1 + 1.0 / N) / 6.0
        vt = (N + 1) * (4 * k * N - 3 * (nx ** 2 + ny ** 2) - 2 * k) / (45 * N ** 2 * 4 * k)
        tn = 1.0 / 6.0 + (t - et) / math.sqrt(45 * vt)
        p = 1.0 if tn < 0.003 else float(np.clip(1.0 - _cdf_cvm_inf(np.array([tn]))[0], 0.0, None))
        return TestResult(float(t), p, None, "x and y come from the same distribution",
                           "Cramer-von Mises (two-sample)")


AndersonDarling = AndersenDarling
