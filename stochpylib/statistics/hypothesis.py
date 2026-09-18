"""Classical hypothesis tests. Every function returns a :class:`TestResult`.

``alternative`` (where defined) is one of ``"two-sided"``, ``"greater"``, ``"less"``.
"""

import math

import numpy as np
from scipy import special

from stochpylib.statistics._common import (
    _as_1d, _chi2_sf, _f_cdf, _f_sf, _kolmogorov_sf, _norm_cdf, _norm_ppf,
    _ptukey, _pvalue_from_t, _pvalue_from_z, _qtukey, _t_sf,
)
from stochpylib.statistics._result import TestResult

__all__ = [
    "ANOVA", "MANOVA", "bartlett", "bonferroni", "chi2_test", "f_test", "ks_test",
    "levene", "mann_whitney", "shapiro_wilk", "t_test", "tukey_hsd", "wilcoxon", "z_test",
]


# ---------------------------------------------------------------------------- z / t

def z_test(x, mu0=0.0, sigma=None, y=None, sigma_y=None, proportion=False,
           pooled=True, alternative="two-sided"):
    """One- or two-sample z-test on a mean (or, with ``proportion=True``, a
    proportion). ``x``/``y`` are ``(count, nobs)`` pairs when ``proportion=True``.

    Known ``sigma``/``sigma_y`` uses the exact normal test; otherwise the sample
    standard deviation is substituted (the large-sample z approximation), matching
    ``statsmodels.stats.weightstats.ztest``. The one-proportion case uses the
    classical null-hypothesis variance ``mu0(1-mu0)/n`` (a score test), matching
    ``statsmodels.stats.proportion.proportions_ztest(..., prop_var=mu0)`` -- *not*
    that function's default, which substitutes the sample proportion.
    """
    if proportion:
        count1, n1 = x
        p1 = count1 / n1
        if y is None:
            se = math.sqrt(mu0 * (1 - mu0) / n1)
            z = (p1 - mu0) / se
            estimate = p1
        else:
            count2, n2 = y
            p2 = count2 / n2
            p_pool = (count1 + count2) / (n1 + n2)
            se = math.sqrt(p_pool * (1 - p_pool) * (1.0 / n1 + 1.0 / n2))
            z = (p1 - p2 - mu0) / se
            estimate = p1 - p2
        pvalue = _pvalue_from_z(z, alternative)
        return TestResult(float(z), pvalue, None, "proportion equals the hypothesized value",
                           "z-test (proportion)", alternative, estimate, se, None, None, {})

    x = _as_1d(x)
    n = len(x)
    xbar = float(np.mean(x))
    if y is None:
        s = sigma if sigma is not None else float(np.std(x, ddof=1))
        se = s / math.sqrt(n)
        z = (xbar - mu0) / se
        estimate = xbar
        method = "one-sample z-test"
    else:
        y = _as_1d(y)
        m = len(y)
        ybar = float(np.mean(y))
        if sigma is not None and sigma_y is not None:
            se = math.sqrt(sigma ** 2 / n + sigma_y ** 2 / m)
        else:
            s1, s2 = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
            if pooled:
                var_pooled = ((n - 1) * s1 + (m - 1) * s2) / (n + m - 2)
                se = math.sqrt(var_pooled * (1.0 / n + 1.0 / m))
            else:
                se = math.sqrt(s1 / n + s2 / m)
        z = (xbar - ybar - mu0) / se
        estimate = xbar - ybar
        method = "two-sample z-test"
    pvalue = _pvalue_from_z(z, alternative)
    ci = (estimate - 1.959963984540054 * se, estimate + 1.959963984540054 * se)
    return TestResult(float(z), pvalue, None, "mean equals the hypothesized value", method,
                       alternative, estimate, se, ci, None, {})


def t_test(x, y=None, mu0=0.0, paired=False, equal_var=True, alternative="two-sided"):
    """One-sample, two-sample (pooled or Welch), or paired t-test.

    ``extras['cohens_d']`` reports the standardized effect size.
    """
    x = _as_1d(x)
    n = len(x)

    if y is None or paired:
        if paired:
            y = _as_1d(y)
            if len(y) != n:
                raise ValueError("paired samples must have equal length")
            d = x - y
        else:
            d = x
        dbar = float(np.mean(d))
        s = float(np.std(d, ddof=1))
        se = s / math.sqrt(n)
        t = (dbar - mu0) / se
        df = n - 1
        cohens_d = (dbar - mu0) / s if s > 0 else float("nan")
        method = "paired t-test" if paired else "one-sample t-test"
        estimate = dbar
    else:
        y = _as_1d(y)
        m = len(y)
        xbar, ybar = float(np.mean(x)), float(np.mean(y))
        s1, s2 = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
        if equal_var:
            sp2 = ((n - 1) * s1 + (m - 1) * s2) / (n + m - 2)
            se = math.sqrt(sp2 * (1.0 / n + 1.0 / m))
            df = n + m - 2
            pooled_std = math.sqrt(sp2)
        else:
            se = math.sqrt(s1 / n + s2 / m)
            df = (s1 / n + s2 / m) ** 2 / ((s1 / n) ** 2 / (n - 1) + (s2 / m) ** 2 / (m - 1))
            pooled_std = math.sqrt(((n - 1) * s1 + (m - 1) * s2) / (n + m - 2))
        t = (xbar - ybar - mu0) / se
        cohens_d = (xbar - ybar) / pooled_std if pooled_std > 0 else float("nan")
        method = "two-sample t-test" if equal_var else "Welch's t-test"
        estimate = xbar - ybar

    pvalue = _pvalue_from_t(t, df, alternative)
    tcrit = special.stdtrit(df, 0.975)
    ci = (estimate - tcrit * se, estimate + tcrit * se)
    return TestResult(float(t), pvalue, float(df), "mean(s) equal the hypothesized value/difference",
                       method, alternative, estimate, se, ci, None, {"cohens_d": cohens_d})


# ---------------------------------------------------------------------- chi2 / F

def chi2_test(observed, expected=None, ddof=0, yates=False):
    """Chi-squared goodness-of-fit (1-D ``observed``) or test of independence
    (2-D contingency table)."""
    observed = np.asarray(observed, dtype=float)
    if observed.ndim == 1:
        n = observed.sum()
        expected_arr = (np.full_like(observed, n / len(observed)) if expected is None
                         else np.asarray(expected, dtype=float))
        stat = float(np.sum((observed - expected_arr) ** 2 / expected_arr))
        df = len(observed) - 1 - ddof
        pvalue = float(_chi2_sf(stat, df))
        return TestResult(stat, pvalue, df, "observed frequencies match expected",
                           "chi-squared goodness-of-fit", extras={"expected": expected_arr})

    if observed.ndim != 2:
        raise ValueError("observed must be 1-D (goodness-of-fit) or 2-D (contingency table)")
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    expected_arr = row_sums @ col_sums / total
    diff = observed - expected_arr
    if yates and observed.shape == (2, 2):
        diff = np.sign(diff) * np.maximum(np.abs(diff) - 0.5, 0.0)
    stat = float(np.sum(diff ** 2 / expected_arr))
    r, c = observed.shape
    df = (r - 1) * (c - 1)
    pvalue = float(_chi2_sf(stat, df))
    cramers_v = float(np.sqrt(stat / (total * (min(r, c) - 1)))) if min(r, c) > 1 else float("nan")
    return TestResult(stat, pvalue, df, "rows and columns are independent",
                       "chi-squared test of independence",
                       extras={"expected": expected_arr, "cramers_v": cramers_v})


def f_test(x, y):
    """Variance-ratio F-test (two 1-D samples), or a nested-model F-test when both
    arguments are :class:`~stochpylib.statistics.RegressionResult` (full model vs a
    restricted sub-model fit to the same data)."""
    from stochpylib.statistics._result import RegressionResult

    if isinstance(x, RegressionResult) and isinstance(y, RegressionResult):
        full, restricted = x, y
        rss_full = float(np.sum(full.resid_ ** 2))
        rss_res = float(np.sum(restricted.resid_ ** 2))
        df_full, df_res = full.df_resid_, restricted.df_resid_
        df_diff = df_res - df_full
        stat = ((rss_res - rss_full) / df_diff) / (rss_full / df_full)
        pvalue = float(_f_sf(stat, df_diff, df_full))
        return TestResult(stat, pvalue, (df_diff, df_full), "restricted model fits equally well",
                           "nested F-test")

    x = _as_1d(x)
    y = _as_1d(y)
    n, m = len(x), len(y)
    s1, s2 = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    stat = s1 / s2
    df1, df2 = n - 1, m - 1
    p_lower = float(_f_cdf(stat, df1, df2))
    p_upper = float(_f_sf(stat, df1, df2))
    pvalue = float(min(1.0, 2.0 * min(p_lower, p_upper)))
    return TestResult(stat, pvalue, (df1, df2), "equal variances", "F variance-ratio test",
                       estimate=stat)


# ---------------------------------------------------------------------- ANOVA / MANOVA

def ANOVA(*args, factors=None, interaction=True, equal_var=True):
    """One-way ANOVA (``ANOVA(g1, g2, ...)``) or two-way Type-II ANOVA
    (``ANOVA(y, factors=[a, b], interaction=True)``).

    One-way: ``equal_var=False`` runs Welch's ANOVA (unequal variances).
    Two-way: Type-II sums of squares via nested-model comparison (coding-independent),
    exactly two factors; ``.table`` carries one row per source (A, B, A:B, Residual).
    """
    if factors is None:
        groups = [_as_1d(g) for g in args]
        if len(groups) < 2:
            raise ValueError("ANOVA needs at least two groups")
        return _welch_anova(groups) if not equal_var else _oneway_anova(groups)

    if len(factors) != 2:
        raise ValueError("two-way ANOVA requires exactly two factors")
    y = _as_1d(args[0])
    return _twoway_anova(y, factors[0], factors[1], interaction)


def _oneway_anova(groups):
    k = len(groups)
    all_x = np.concatenate(groups)
    grand_mean = float(np.mean(all_x))
    ssb = sum(len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in groups)
    ssw = sum(float(np.sum((g - np.mean(g)) ** 2)) for g in groups)
    dfb, dfw = k - 1, len(all_x) - k
    msb, msw = ssb / dfb, ssw / dfw
    F = msb / msw
    p = float(_f_sf(F, dfb, dfw))
    table = [
        {"source": "between", "ss": ssb, "df": dfb, "ms": msb, "F": F, "p": p},
        {"source": "within", "ss": ssw, "df": dfw, "ms": msw, "F": None, "p": None},
    ]
    return TestResult(F, p, (dfb, dfw), "all group means are equal", "one-way ANOVA", table=table)


def _welch_anova(groups):
    k = len(groups)
    ni = np.array([len(g) for g in groups], dtype=float)
    mi = np.array([np.mean(g) for g in groups])
    vi = np.array([np.var(g, ddof=1) for g in groups])
    wi = ni / vi
    W = np.sum(wi)
    xbar_w = np.sum(wi * mi) / W
    A = np.sum(wi * (mi - xbar_w) ** 2) / (k - 1)
    term = np.sum((1 - wi / W) ** 2 / (ni - 1))
    B = 1.0 + (2.0 * (k - 2) / (k ** 2 - 1)) * term
    F = A / B
    df1 = k - 1
    df2 = (k ** 2 - 1) / (3.0 * term)
    p = float(_f_sf(F, df1, df2))
    return TestResult(float(F), p, (df1, df2), "all group means are equal (unequal variances)",
                       "Welch's ANOVA")


def _dummies(factor):
    factor = np.asarray(factor)
    levels = np.unique(factor)
    if len(levels) < 2:
        raise ValueError("each factor must have at least 2 levels")
    D = np.zeros((len(factor), len(levels) - 1))
    for j, lv in enumerate(levels[1:]):
        D[:, j] = (factor == lv).astype(float)
    return D, levels


def _ols_rss(X, y):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(np.sum(resid ** 2))


def _twoway_anova(y, a, b, interaction):
    n = len(y)
    Da, levels_a = _dummies(a)
    Db, levels_b = _dummies(b)
    ones = np.ones((n, 1))

    Xa = np.column_stack([ones, Da])
    Xb = np.column_stack([ones, Db])
    Xab = np.column_stack([ones, Da, Db])

    rss_a = _ols_rss(Xa, y)
    rss_b = _ols_rss(Xb, y)
    rss_ab = _ols_rss(Xab, y)

    df_a = Da.shape[1]
    df_b = Db.shape[1]

    ss_a = rss_b - rss_ab   # SS(A|B)
    ss_b = rss_a - rss_ab   # SS(B|A)

    if interaction:
        Dint = np.column_stack([Da[:, i] * Db[:, j] for i in range(Da.shape[1]) for j in range(Db.shape[1])])
        Xfull = np.column_stack([ones, Da, Db, Dint])
        rss_full = _ols_rss(Xfull, y)
        df_ab = Dint.shape[1]
        ss_ab = rss_ab - rss_full
        df_resid = n - (1 + df_a + df_b + df_ab)
        ss_resid = rss_full
    else:
        df_ab = None
        ss_ab = None
        df_resid = n - (1 + df_a + df_b)
        ss_resid = rss_ab

    ms_resid = ss_resid / df_resid
    F_a = (ss_a / df_a) / ms_resid
    F_b = (ss_b / df_b) / ms_resid
    p_a = float(_f_sf(F_a, df_a, df_resid))
    p_b = float(_f_sf(F_b, df_b, df_resid))
    table = [
        {"source": "A", "ss": ss_a, "df": df_a, "F": float(F_a), "p": p_a},
        {"source": "B", "ss": ss_b, "df": df_b, "F": float(F_b), "p": p_b},
    ]
    if interaction:
        F_ab = (ss_ab / df_ab) / ms_resid
        p_ab = float(_f_sf(F_ab, df_ab, df_resid))
        table.append({"source": "A:B", "ss": ss_ab, "df": df_ab, "F": float(F_ab), "p": p_ab})
    table.append({"source": "Residual", "ss": ss_resid, "df": df_resid, "F": None, "p": None})
    return TestResult(None, None, None, "no main or interaction effects",
                       "two-way ANOVA (Type II SS)", table=table)


def MANOVA(Y, groups):
    """One-way multivariate ANOVA: Wilks' lambda, Pillai's trace, Hotelling-Lawley
    trace, and Roy's greatest root, each with its F-approximation, in ``.table``.

    The Hotelling-Lawley F-approximation is undefined when ``n = (v-p-1)/2 <= 0``
    (very small error d.o.f. relative to the number of response variables); in that
    case its row reports ``None`` and Wilks'/Pillai's/Roy's statistics remain valid.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be a (n, p) array")
    groups = np.asarray(groups)
    n, p = Y.shape
    levels = np.unique(groups)
    g = len(levels)
    q = g - 1
    v = n - g
    grand_mean = Y.mean(axis=0)
    H = np.zeros((p, p))
    E = np.zeros((p, p))
    for lv in levels:
        Yi = Y[groups == lv]
        ni = len(Yi)
        mi = Yi.mean(axis=0)
        H += ni * np.outer(mi - grand_mean, mi - grand_mean)
        Ei = Yi - mi
        E += Ei.T @ Ei

    eigvals = np.real(np.linalg.eigvals(np.linalg.solve(E, H)))
    eigvals = np.clip(eigvals, 0.0, None)
    s = min(p, q)
    eigvals = np.sort(eigvals)[::-1][:s]

    wilks_L = float(np.prod(1.0 / (1.0 + eigvals)))
    pillai_V = float(np.sum(eigvals / (1.0 + eigvals)))
    hl_U = float(np.sum(eigvals))
    roy_max = float(eigvals[0])

    m = (abs(p - q) - 1) / 2.0
    n_ = (v - p - 1) / 2.0

    if p ** 2 + q ** 2 - 5 > 0:
        t = math.sqrt((p ** 2 * q ** 2 - 4) / (p ** 2 + q ** 2 - 5))
    else:
        t = 1.0
    df1_w = p * q
    df2_w = t * (v - (p - q + 1) / 2.0) - (p * q - 2) / 2.0
    Lt = wilks_L ** (1.0 / t)
    F_wilks = ((1.0 - Lt) / Lt) * (df2_w / df1_w)
    p_wilks = float(_f_sf(F_wilks, df1_w, df2_w))

    df1_p = s * (2 * m + s + 1)
    df2_p = s * (2 * n_ + s + 1)
    F_pillai = ((2 * n_ + s + 1) / (2 * m + s + 1)) * (pillai_V / (s - pillai_V))
    p_pillai = float(_f_sf(F_pillai, df1_p, df2_p))

    if n_ > 1:
        b = (p + 2 * n_) * (q + 2 * n_) / (2.0 * (2 * n_ + 1) * (n_ - 1))
        df1_hl, df2_hl = p * q, 4.0 + (p * q + 2) / (b - 1)
        c = (df2_hl - 2) / (2 * n_)
        F_hl = (df2_hl / (p * q)) * (hl_U / c)
        p_hl = float(_f_sf(F_hl, df1_hl, df2_hl))
    else:
        F_hl = df1_hl = df2_hl = p_hl = None

    r = max(p, q)
    df1_r, df2_r = r, v - r + q
    F_roy = ((v - r + q) / r) * roy_max
    p_roy = float(_f_sf(F_roy, df1_r, df2_r))

    table = [
        {"statistic": "Wilks' lambda", "value": wilks_L, "F": float(F_wilks), "df1": df1_w, "df2": df2_w, "p": p_wilks},
        {"statistic": "Pillai's trace", "value": pillai_V, "F": float(F_pillai), "df1": df1_p, "df2": df2_p, "p": p_pillai},
        {"statistic": "Hotelling-Lawley trace", "value": hl_U, "F": F_hl, "df1": df1_hl, "df2": df2_hl, "p": p_hl},
        {"statistic": "Roy's greatest root", "value": roy_max, "F": float(F_roy), "df1": df1_r, "df2": df2_r, "p": p_roy},
    ]
    return TestResult(float(F_wilks), p_wilks, (df1_w, df2_w), "group mean vectors are equal",
                       "MANOVA", table=table, extras={"wilks_lambda": wilks_L})


# ---------------------------------------------------------------------- rank tests

def _mwu_exact_counts(n1, n2):
    """Exact null distribution of the Mann-Whitney U1 statistic (no ties): returns a
    length-``n1*n2+1`` array of Python ints, counts[u] = #arrangements with U1 = u.

    DP: think of placing n1+n2 ranked items left to right; each time a "y" item is
    placed, it adds (number of x's already placed) to U1. dp[i][j][u] = ways to place
    i x's and j y's with partial U1 = u; dp[i][j] = dp[i-1][j] + shift(dp[i][j-1], i).
    """
    maxu = n1 * n2
    prev_row = [np.zeros(maxu + 1, dtype=object) for _ in range(n1 + 1)]
    for i in range(n1 + 1):
        prev_row[i][0] = 1
    for _ in range(1, n2 + 1):
        cur_row = [np.zeros(maxu + 1, dtype=object) for _ in range(n1 + 1)]
        cur_row[0][0] = 1
        for i in range(1, n1 + 1):
            shifted = np.zeros(maxu + 1, dtype=object)
            if i <= maxu:
                shifted[i:] = prev_row[i][:maxu + 1 - i]
            cur_row[i] = cur_row[i - 1] + shifted
        prev_row = cur_row
    return prev_row[n1]


def mann_whitney(x, y, method="auto", continuity=True, alternative="two-sided"):
    """Mann-Whitney U test. ``statistic`` is U1 (the count of x_i > y_j pairs, with
    ties counted as 1/2). ``method='exact'`` uses the exact null distribution (no tie
    correction; requires no ties in the pooled sample); ``'auto'`` uses it whenever
    ``max(n1, n2) <= 50`` and there are no ties, else falls back to the tie-corrected
    normal approximation.
    """
    x = _as_1d(x)
    y = _as_1d(y)
    n1, n2 = len(x), len(y)
    pooled = np.concatenate([x, y])
    has_ties = len(np.unique(pooled)) < len(pooled)

    ranks = _rank_with_ties(pooled)
    r1 = np.sum(ranks[:n1])
    U1 = r1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1

    use_exact = method == "exact" or (method == "auto" and not has_ties and max(n1, n2) <= 50)
    if use_exact and has_ties:
        raise ValueError("exact method requires no ties in the pooled sample")

    if use_exact:
        counts = _mwu_exact_counts(n1, n2)
        total = int(np.sum(counts))
        u_int = int(round(U1))
        cdf = int(np.sum(counts[: u_int + 1]))
        sf = int(np.sum(counts[u_int:]))
        if alternative == "two-sided":
            pvalue = min(1.0, 2.0 * min(cdf, sf) / total)
        elif alternative == "greater":
            pvalue = sf / total
        else:
            pvalue = cdf / total
        method_name = "Mann-Whitney U (exact)"
    else:
        mu = n1 * n2 / 2.0
        _, tie_counts = np.unique(pooled, return_counts=True)
        tie_term = np.sum(tie_counts ** 3 - tie_counts)
        N = n1 + n2
        sigma = math.sqrt(n1 * n2 / 12.0 * ((N + 1) - tie_term / (N * (N - 1))))
        corr = 0.5 if continuity else 0.0
        if alternative == "two-sided":
            z = (U1 - mu - math.copysign(corr, U1 - mu)) / sigma
            pvalue = float(2.0 * _norm_cdf(-abs(z)))
        elif alternative == "greater":
            z = (U1 - mu - corr) / sigma
            pvalue = float(_norm_cdf(-z))
        else:
            z = (U1 - mu + corr) / sigma
            pvalue = float(_norm_cdf(z))
        method_name = "Mann-Whitney U (asymptotic)"

    return TestResult(float(U1), float(pvalue), None, "x and y have the same distribution",
                       method_name, alternative, extras={"U2": float(U2)})


def _rank_with_ties(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sorted_a = a[order]
    n = len(a)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i: j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _wilcoxon_exact_counts(n):
    """Exact null distribution of the sum of positive-signed ranks under H0 (no
    ties/zeros): counts[s] = #(2^n sign patterns) with positive-rank sum = s."""
    maxs = n * (n + 1) // 2
    dp = np.zeros(maxs + 1, dtype=object)
    dp[0] = 1
    for k in range(1, n + 1):
        new_dp = dp.copy()
        new_dp[k:] += dp[: maxs + 1 - k]
        dp = new_dp
    return dp


def wilcoxon(x, y=None, method="auto", zero_method="wilcox", correction=False,
             alternative="two-sided"):
    """Wilcoxon signed-rank test on ``x`` (one-sample, H0: median 0) or ``x - y``
    (paired). ``statistic`` is ``min(T+, T-)``. ``zero_method='wilcox'`` drops exact
    zeros before ranking (the only supported mode). ``method='exact'`` requires no
    ties/zeros in the (nonzero) differences; ``'auto'`` uses it when ``n <= 50``.
    """
    x = _as_1d(x)
    d = x if y is None else x - _as_1d(y)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        raise ValueError("all differences are zero")

    abs_d = np.abs(d)
    ranks = _rank_with_ties(abs_d)
    has_ties = len(np.unique(abs_d)) < n
    T_plus = float(np.sum(ranks[d > 0]))
    T_minus = float(np.sum(ranks[d < 0]))
    T = min(T_plus, T_minus)

    use_exact = method == "exact" or (method == "auto" and not has_ties and n <= 50)
    if use_exact and has_ties:
        raise ValueError("exact method requires no ties among the nonzero differences")

    if use_exact:
        counts = _wilcoxon_exact_counts(n)
        total = int(np.sum(counts))
        t_int = int(round(T))
        cdf = int(np.sum(counts[: t_int + 1]))
        if alternative == "two-sided":
            pvalue = min(1.0, 2.0 * cdf / total)
        else:
            pvalue = cdf / total  # both one-sided alternatives use the same min-tail statistic
        method_name = "Wilcoxon signed-rank (exact)"
    else:
        mu = n * (n + 1) / 4.0
        tie_term = 0.0
        if has_ties:
            _, tie_counts = np.unique(abs_d, return_counts=True)
            tie_term = np.sum(tie_counts ** 3 - tie_counts) / 48.0
        sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie_term)
        stat_signed = T_plus - mu
        corr = 0.5 if correction else 0.0
        if alternative == "two-sided":
            z = (stat_signed - math.copysign(corr, stat_signed)) / sigma
            pvalue = float(2.0 * _norm_cdf(-abs(z)))
        elif alternative == "greater":
            z = (stat_signed - corr) / sigma
            pvalue = float(_norm_cdf(-z))
        else:
            z = (stat_signed + corr) / sigma
            pvalue = float(_norm_cdf(z))
        method_name = "Wilcoxon signed-rank (asymptotic)"

    return TestResult(T, float(pvalue), None, "the median difference is zero", method_name,
                       alternative, extras={"T_plus": T_plus, "T_minus": T_minus})


# ---------------------------------------------------------------------- KS / normality

def ks_test(x, cdf_or_y, args=()):
    """One-sample (against a CDF callable, a fitted ``Distribution`` instance, or a
    ``stochpylib.distributions`` class name + ``args``) or two-sample Kolmogorov-Smirnov
    test. P-values use the classical asymptotic (``N -> infinity``) Kolmogorov
    distribution (``scipy.special.kolmogorov``) -- matching
    ``scipy.stats.kstest(..., method="asymp")`` exactly for the one-sample case, and
    R's ``ks.test`` for the two-sample case. This differs from
    ``scipy.stats.ks_2samp(..., method="asymp")``, which uses a more refined
    finite-sample correction (the ``kstwo`` distribution) not expressible in terms of
    ``scipy.special`` alone; the two agree closely for large samples and always agree
    on the statistic itself.
    """
    from stochpylib.distributions._base import Distribution

    x = _as_1d(x)
    n = len(x)

    if isinstance(cdf_or_y, Distribution):
        cdf_fn = cdf_or_y.cdf
    elif isinstance(cdf_or_y, str):
        import stochpylib.distributions as dist_mod
        cdf_fn = getattr(dist_mod, cdf_or_y)(*args).cdf
    elif callable(cdf_or_y):
        cdf_fn = (lambda v: cdf_or_y(v, *args)) if args else cdf_or_y
    else:
        y = _as_1d(cdf_or_y)
        return _ks_2samp(x, y)

    xs = np.sort(x)
    cdf_vals = np.asarray([float(cdf_fn(v)) for v in xs])
    i = np.arange(1, n + 1)
    d_plus = float(np.max(i / n - cdf_vals))
    d_minus = float(np.max(cdf_vals - (i - 1) / n))
    stat = max(d_plus, d_minus)
    lam = math.sqrt(n) * stat
    pvalue = float(_kolmogorov_sf(lam))
    return TestResult(stat, pvalue, None, "x is drawn from the specified distribution",
                       "one-sample Kolmogorov-Smirnov test")


def _ks_2samp(x, y):
    n, m = len(x), len(y)
    all_sorted = np.sort(np.concatenate([x, y]))
    xs, ys = np.sort(x), np.sort(y)
    cdf_x = np.searchsorted(xs, all_sorted, side="right") / n
    cdf_y = np.searchsorted(ys, all_sorted, side="right") / m
    stat = float(np.max(np.abs(cdf_x - cdf_y)))
    en = n * m / (n + m)
    lam = math.sqrt(en) * stat
    pvalue = float(_kolmogorov_sf(lam))
    return TestResult(stat, pvalue, None, "x and y are drawn from the same distribution",
                       "two-sample Kolmogorov-Smirnov test")


_SW_C1 = (0.0, 0.221157, -0.147981, -2.071190, 4.434685, -2.706056)
_SW_C2 = (0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633)
_SW_C3 = (0.544, -0.39978, 0.025054, -0.6714e-3)
_SW_C4 = (1.3822, -0.77857, 0.062767, -0.0020322)
_SW_C5 = (-1.5861, -0.31082, -0.083751, 0.0038915)
_SW_C6 = (-0.4803, -0.082676, 0.0030302)
_SW_G = (-2.273, 0.459)


def _poly(cc, x):
    return sum(c * x ** i for i, c in enumerate(cc))


def shapiro_wilk(x):
    """Shapiro-Wilk normality test (Royston 1992, Algorithm AS R94), ``3 <= n <= 5000``."""
    x = np.sort(_as_1d(x))
    n = len(x)
    if n < 3:
        raise ValueError("shapiro_wilk requires at least 3 observations")
    if n > 5000:
        raise ValueError("shapiro_wilk is only accurate for n <= 5000")

    if n == 3:
        a_full = np.array([-math.sqrt(0.5), 0.0, math.sqrt(0.5)])
    else:
        nn2 = n // 2
        i = np.arange(1, nn2 + 1)
        m = _norm_ppf((i - 0.375) / (n + 0.25))
        summ2 = 2.0 * np.sum(m ** 2)
        rsn = 1.0 / math.sqrt(n)
        a1 = _poly(_SW_C1, rsn) - m[0] / math.sqrt(summ2)

        a = np.zeros(nn2)
        if n > 5:
            a2 = -m[1] / math.sqrt(summ2) + _poly(_SW_C2, rsn)
            fac = math.sqrt((summ2 - 2 * m[0] ** 2 - 2 * m[1] ** 2) / (1.0 - 2 * a1 ** 2 - 2 * a2 ** 2))
            a[0], a[1] = a1, a2
            i1 = 2
        else:
            fac = math.sqrt((summ2 - 2 * m[0] ** 2) / (1.0 - 2 * a1 ** 2))
            a[0] = a1
            i1 = 1
        a[i1:] = -m[i1:] / fac

        a_full = np.empty(n)
        a_full[:nn2] = -a
        a_full[n - nn2:] = a[::-1]
        if n % 2 == 1:
            a_full[nn2] = 0.0

    xbar = np.mean(x)
    num = float(np.sum(a_full * x)) ** 2
    denom = float(np.sum((x - xbar) ** 2))
    w = num / denom
    w = min(w, 1.0)

    if n == 3:
        pw = 6.0 / math.pi * (math.asin(math.sqrt(w)) - math.asin(math.sqrt(0.75)))
        pw = float(np.clip(pw, 0.0, 1.0))
    else:
        an = float(n)
        y = math.log(1.0 - w) if w < 1.0 else -50.0
        if n <= 11:
            gamma = _poly(_SW_G, an)
            y = -math.log(gamma - y)
            mu = _poly(_SW_C3, an)
            sigma = math.exp(_poly(_SW_C4, an))
        else:
            xx = math.log(an)
            mu = _poly(_SW_C5, xx)
            sigma = math.exp(_poly(_SW_C6, xx))
        z = (y - mu) / sigma
        pw = float(_norm_cdf(-z))

    return TestResult(float(w), pw, None, "the sample is drawn from a normal distribution",
                       "Shapiro-Wilk test")


# ---------------------------------------------------------------------- variance tests

def levene(*groups, center="median"):
    """Levene's test for equality of variances across groups (Brown-Forsythe when
    ``center='median'``, the default and most robust choice)."""
    groups = [_as_1d(g) for g in groups]
    if center == "mean":
        centers = [float(np.mean(g)) for g in groups]
    elif center == "median":
        centers = [float(np.median(g)) for g in groups]
    elif center == "trimmed":
        from stochpylib.statistics.descriptive import mean as trimmed_mean
        centers = [trimmed_mean(g, trim=0.1) for g in groups]
    else:
        raise ValueError("center must be 'median', 'mean', or 'trimmed'")
    z_groups = [np.abs(g - c) for g, c in zip(groups, centers)]
    res = _oneway_anova(z_groups)
    return TestResult(res.statistic, res.pvalue, res.df, "group variances are equal",
                       "Levene's test", table=res.table)


def bartlett(*groups):
    """Bartlett's test for equality of variances (assumes approximate normality;
    more powerful than Levene's when that assumption holds, but sensitive to its
    violation)."""
    groups = [_as_1d(g) for g in groups]
    k = len(groups)
    ni = np.array([len(g) for g in groups], dtype=float)
    vi = np.array([np.var(g, ddof=1) for g in groups])
    N = np.sum(ni)
    sp2 = np.sum((ni - 1) * vi) / (N - k)
    stat = ((N - k) * math.log(sp2) - np.sum((ni - 1) * np.log(vi)))
    c = 1.0 + (np.sum(1.0 / (ni - 1)) - 1.0 / (N - k)) / (3.0 * (k - 1))
    stat = stat / c
    df = k - 1
    pvalue = float(_chi2_sf(stat, df))
    return TestResult(float(stat), pvalue, df, "group variances are equal", "Bartlett's test")


# ---------------------------------------------------------------------- multiple comparisons

def tukey_hsd(*groups, level=0.95):
    """Tukey-Kramer honestly-significant-difference pairwise comparisons.

    ``.table`` holds one row per pair: mean difference, standard error, studentized-range
    statistic ``q``, p-value, and the simultaneous ``level``-confidence interval.
    """
    groups = [_as_1d(g) for g in groups]
    k = len(groups)
    ni = np.array([len(g) for g in groups], dtype=float)
    mi = np.array([np.mean(g) for g in groups])
    N = np.sum(ni)
    msw = np.sum((ni - 1) * np.array([np.var(g, ddof=1) for g in groups])) / (N - k)
    df = N - k

    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    diffs = np.array([mi[i] - mi[j] for i, j in pairs])
    ses = np.array([math.sqrt(msw / 2.0 * (1.0 / ni[i] + 1.0 / ni[j])) for i, j in pairs])
    qs = np.abs(diffs) / ses
    ps = 1.0 - _ptukey(qs, k, df)
    ps = np.atleast_1d(ps)
    q_crit = _qtukey(level, k, df)
    half_width = q_crit * ses
    table = [
        {"i": i, "j": j, "diff": float(diffs[idx]), "se": float(ses[idx]), "q": float(qs[idx]),
         "p": float(ps[idx]), "conf_int": (float(diffs[idx] - half_width[idx]), float(diffs[idx] + half_width[idx]))}
        for idx, (i, j) in enumerate(pairs)
    ]
    return TestResult(float(np.max(qs)), float(np.min(ps)), df, "all pairwise means are equal",
                       "Tukey HSD", table=table, extras={"q_critical": q_crit})


def bonferroni(pvalues, alpha=0.05, method="bonferroni"):
    """Multiple-testing p-value adjustment: ``'bonferroni'``, ``'holm'``, ``'sidak'``,
    ``'holm-sidak'``, or ``'fdr_bh'`` (Benjamini-Hochberg). ``.table`` holds one row per
    test: original index, raw p, adjusted p, and the reject decision at ``alpha``."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]

    if method == "bonferroni":
        adj = np.clip(p * m, 0, 1)
    elif method == "sidak":
        adj = np.clip(1.0 - (1.0 - p) ** m, 0, 1)
    elif method == "holm":
        adj_ranked = np.maximum.accumulate(ranked * (m - np.arange(m)))
        adj = np.empty(m)
        adj[order] = np.clip(adj_ranked, 0, 1)
    elif method == "holm-sidak":
        adj_ranked = np.maximum.accumulate(1.0 - (1.0 - ranked) ** (m - np.arange(m)))
        adj = np.empty(m)
        adj[order] = np.clip(adj_ranked, 0, 1)
    elif method == "fdr_bh":
        ranks = np.arange(1, m + 1)
        adj_ranked = ranked * m / ranks
        adj_ranked = np.minimum.accumulate(adj_ranked[::-1])[::-1]
        adj = np.empty(m)
        adj[order] = np.clip(adj_ranked, 0, 1)
    else:
        raise ValueError("method must be 'bonferroni', 'holm', 'sidak', 'holm-sidak', or 'fdr_bh'")

    reject = adj <= alpha
    table = [{"index": i, "pvalue": float(p[i]), "adjusted_pvalue": float(adj[i]), "reject": bool(reject[i])}
              for i in range(m)]
    return TestResult(float(np.min(adj)), float(np.min(adj)), None,
                       "all null hypotheses are true", f"multiple-testing adjustment ({method})",
                       table=table, extras={"reject": reject, "adjusted_pvalues": adj})
