"""Distribution fitting/model-selection, goodness-of-fit, ECDF, data validation,
outlier detection, and missing-data imputation -- all built on the library's own
distributions/statistics/nonparametric/robust_statistics machinery rather than
re-deriving any of it.
"""

import math

import numpy as np
from scipy import optimize

from stochpylib.utils._results import FitResult, OutlierResult

__all__ = ["DataValidation", "ecdf", "fit", "goodness_of_fit", "missing_imputation",
           "moment_matching", "outlier_detection"]

_CONTINUOUS_DEFAULTS = ("Normal", "LogNormal", "Exponential", "Gamma", "Weibull", "Beta",
                        "Student_t", "Laplace", "Cauchy", "Gumbel", "Uniform", "InvGaussian")
_DISCRETE_DEFAULTS = ("Poisson", "Geometric", "NegBinomial")


def _resolve_candidates(distributions, discrete):
    from stochpylib import distributions as dist_mod

    if distributions is not None:
        return [getattr(dist_mod, d) if isinstance(d, str) else d for d in distributions]
    names = _DISCRETE_DEFAULTS if discrete else _CONTINUOUS_DEFAULTS
    return [getattr(dist_mod, n) for n in names]


def fit(data, distributions=None, criterion="aic", discrete=None):
    """Fit every candidate distribution to ``data`` (MLE via each class's own
    ``.fit()``) and rank by ``criterion`` (``"aic"``/``"bic"``/``"ks"``).
    Mirrors ``copulas.CopulaFit``'s AIC-table pattern. Candidates that fail to
    fit, or whose likelihood is non-finite (a support mismatch), are recorded
    in ``failed_`` rather than raising.
    """
    from stochpylib.statistics import ks_test

    x = np.asarray(data, dtype=float).ravel()
    n = len(x)
    if discrete is None:
        discrete = bool(np.all(np.equal(np.mod(x, 1), 0)))
    candidates = _resolve_candidates(distributions, discrete)

    table, failed, fitted = [], {}, {}
    for cls in candidates:
        name = cls.__name__
        try:
            inst = cls.fit(x)
            logpdf = inst.pmf if getattr(inst, "is_discrete", discrete) else inst.pdf
            loglik = float(np.sum(np.log(np.clip(
                np.asarray([logpdf(v) for v in x], dtype=float), 1e-300, None))))
            if not np.isfinite(loglik):
                failed[name] = "non-finite log-likelihood (likely support mismatch)"
                continue
            import inspect
            k = len([p for p in inspect.signature(cls.__init__).parameters if p != "self"])
            aic = 2 * k - 2 * loglik
            bic = k * math.log(n) - 2 * loglik
            row = {"name": name, "aic": aic, "bic": bic, "loglik": loglik, "k": k}
            if not discrete:
                ks_res = ks_test(x, inst)
                row["ks_stat"], row["ks_pvalue"] = ks_res.statistic, ks_res.pvalue
            table.append(row)
            fitted[name] = inst
        except Exception as exc:
            failed[name] = str(exc)

    if not table:
        raise RuntimeError(f"utils.fit: every candidate failed: {failed}")

    key = {"aic": "aic", "bic": "bic", "ks": "ks_stat"}[criterion]
    table.sort(key=lambda r: r[key])
    best_name = table[0]["name"]
    return FitResult(fitted[best_name], best_name, table, failed, criterion)


def goodness_of_fit(data, dist, tests=("ks", "ad", "cvm", "chi2"), n_params=None, bins="auto"):
    """Run KS/Anderson-Darling/Cramer-von Mises/chi-squared goodness-of-fit
    tests of ``data`` against ``dist`` (a fitted ``Distribution`` instance, a
    class, or a class name -- classes/names are fit to ``data`` first).

    P-values for KS/AD/CvM are the standard asymptotic ones and are **not**
    corrected for parameters estimated from the same data (the classical
    Lilliefors-type correction is not implemented) -- treat them as
    approximate when ``dist`` was fit to ``data`` itself.
    """
    from stochpylib import distributions as dist_mod
    from stochpylib.distributions._base import Distribution
    from stochpylib.nonparametric import AndersenDarling, CramerVonMises
    from stochpylib.statistics import chi2_test, ks_test

    x = np.asarray(data, dtype=float).ravel()
    n = len(x)

    if isinstance(dist, str):
        dist = getattr(dist_mod, dist)
    if isinstance(dist, type) and issubclass(dist, Distribution):
        import inspect
        if n_params is None:
            n_params = len([p for p in inspect.signature(dist.__init__).parameters
                           if p != "self"])
        dist = dist.fit(x)
    elif n_params is None:
        n_params = 0

    discrete = getattr(dist, "is_discrete", False)
    out = {}
    if "ks" in tests and not discrete:
        out["ks"] = ks_test(x, dist)
    if "ad" in tests and not discrete:
        out["ad"] = AndersenDarling(dist=dist).fit(x).to_result()
    if "cvm" in tests and not discrete:
        out["cvm"] = CramerVonMises().fit(x, dist).to_result()
    if "chi2" in tests:
        if discrete:
            values, counts = np.unique(x, return_counts=True)
            expected = n * np.asarray([dist.pmf(v) for v in values], dtype=float)
            observed, expected = _merge_tail(counts.astype(float), expected)
        else:
            k = max(5, math.ceil(2 * n ** 0.4))
            probs = np.linspace(0, 1, k + 1)
            edges = np.asarray([dist.ppf(p) for p in probs[1:-1]], dtype=float)
            edges = np.concatenate([[-np.inf], edges, [np.inf]])
            observed = np.histogram(x, bins=edges)[0].astype(float)
            expected = np.full(k, n / k)
        out["chi2"] = chi2_test(observed, expected, ddof=n_params)
    return out


def _merge_tail(counts, expected, min_expected=5.0):
    """Merge bins from both ends until every expected count is >= min_expected."""
    counts, expected = list(counts), list(expected)
    while len(counts) > 2 and expected[0] < min_expected:
        counts[1] += counts.pop(0)
        expected[1] += expected.pop(0)
    while len(counts) > 2 and expected[-1] < min_expected:
        counts[-2] += counts.pop(-1)
        expected[-2] += expected.pop(-1)
    return np.asarray(counts, dtype=float), np.asarray(expected, dtype=float)


def moment_matching(data=None, family="normal", *, mean=None, var=None):
    """Return a distribution matching either sample moments (``data`` given,
    delegates to ``statistics.MOM``) or explicit target ``mean``/``var``
    (closed-form inversion; ``Weibull`` uses a 1-D root solve)."""
    from stochpylib import distributions as dist_mod
    from stochpylib.statistics.estimation import MOM

    name_map = {"normal": "Normal", "gamma": "Gamma", "lognormal": "LogNormal",
               "beta": "Beta", "exponential": "Exponential", "poisson": "Poisson",
               "negbinomial": "NegBinomial", "uniform": "Uniform", "weibull": "Weibull"}
    fam = name_map.get(family.lower(), family)
    cls = getattr(dist_mod, fam)

    if data is not None:
        return MOM(cls, data)

    if mean is None or var is None:
        raise ValueError("moment_matching needs either `data` or both `mean` and `var`")
    m, v = float(mean), float(var)

    if fam == "Normal":
        return cls(m, math.sqrt(v))
    if fam == "Exponential":
        return cls(1.0 / m)
    if fam == "Gamma":
        return cls(m ** 2 / v, v / m)
    if fam == "LogNormal":
        sigma2 = math.log1p(v / m ** 2)
        return cls(math.log(m) - sigma2 / 2.0, math.sqrt(sigma2))
    if fam == "Beta":
        common = m * (1.0 - m) / v - 1.0
        return cls(m * common, (1.0 - m) * common)
    if fam == "Poisson":
        return cls(m)
    if fam == "Uniform":
        half = math.sqrt(3.0 * v)
        return cls(m - half, m + half)
    if fam == "NegBinomial":
        p = m / v
        r = m * p / (1.0 - p)
        return cls(r, p)
    if fam == "Weibull":
        cv2 = v / m ** 2

        def g(k):
            from scipy import special
            return special.gamma(1.0 + 2.0 / k) / special.gamma(1.0 + 1.0 / k) ** 2 - 1.0 - cv2

        k_hat = optimize.brentq(g, 0.05, 100.0)
        from scipy import special
        lam = m / special.gamma(1.0 + 1.0 / k_hat)
        return cls(k_hat, lam)
    raise ValueError(f"moment_matching: no closed form for family {fam!r}")


def ecdf(data):
    """Fitted empirical CDF (``nonparametric.EmpiricalCDF``), with DKW bands
    and quantiles already built in."""
    from stochpylib.nonparametric import EmpiricalCDF
    return EmpiricalCDF().fit(data)


class DataValidation:
    """Reusable data-quality checks: finiteness, NaN policy, bounds, integer-
    ness, size, and dimensionality."""

    def __init__(self, finite=True, allow_nan=False, min_size=1, bounds=None,
                integer=False, ndim=None, dtype=float):
        self.finite = finite
        self.allow_nan = allow_nan
        self.min_size = min_size
        self.bounds = bounds
        self.integer = integer
        self.ndim = ndim
        self.dtype = dtype
        self.report_ = None

    def check(self, x):
        arr = np.asarray(x, dtype=self.dtype)
        issues = []
        n_nan = int(np.sum(np.isnan(arr))) if np.issubdtype(arr.dtype, np.floating) else 0
        n_inf = int(np.sum(np.isinf(arr))) if np.issubdtype(arr.dtype, np.floating) else 0
        if n_nan and not self.allow_nan:
            issues.append(f"{n_nan} NaN value(s)")
        if n_inf and self.finite:
            issues.append(f"{n_inf} infinite value(s)")
        if self.ndim is not None and arr.ndim != self.ndim:
            issues.append(f"ndim={arr.ndim}, expected {self.ndim}")
        if arr.size < self.min_size:
            issues.append(f"size={arr.size} < min_size={self.min_size}")
        n_out = 0
        if self.bounds is not None:
            lo, hi = self.bounds
            finite_mask = np.isfinite(arr)
            n_out = int(np.sum((arr[finite_mask] < lo) | (arr[finite_mask] > hi)))
            if n_out:
                issues.append(f"{n_out} value(s) outside bounds {self.bounds}")
        n_non_int = 0
        if self.integer:
            finite_mask = np.isfinite(arr)
            n_non_int = int(np.sum(np.mod(arr[finite_mask], 1) != 0))
            if n_non_int:
                issues.append(f"{n_non_int} non-integer value(s)")
        report = {"valid": not issues, "issues": issues, "n": int(arr.size),
                 "n_nan": n_nan, "n_inf": n_inf, "n_out_of_bounds": n_out,
                 "n_non_integer": n_non_int}
        self.report_ = report
        return report

    def validate(self, x):
        """Check ``x``, then return it with any (permitted) NaNs stripped.

        ``allow_nan=False`` (the default) already rejects any NaN in ``check()``, so
        nothing is ever stripped in that case; ``allow_nan=True`` tolerates NaNs in
        the check but still hands back a clean array for convenience.
        """
        report = self.check(x)
        if not report["valid"]:
            raise ValueError(f"DataValidation failed: {'; '.join(report['issues'])}")
        arr = np.asarray(x, dtype=self.dtype)
        if self.allow_nan and np.issubdtype(arr.dtype, np.floating):
            arr = arr[~np.isnan(arr)]
        return arr


def outlier_detection(x, method="mad", threshold=None, window=7, alpha=0.05, random_state=None):
    """Flag outliers in ``x`` by one of ``"mad"``/``"iqr"``/``"zscore"``/
    ``"grubbs"``/``"hampel"``/``"mcd"`` (the last for 2-D ``x``)."""
    x_arr = np.asarray(x, dtype=float)

    if method == "mcd":
        from stochpylib.robust_statistics import MCD
        model = MCD(random_state=random_state).fit(x_arr)
        level = 1.0 - (threshold if threshold is not None else 0.025)
        mask = model.outliers(level=level)
        scores = model.mahalanobis()
        thr = float(np.sqrt(np.quantile(scores, level))) if scores.size else float("nan")
        idx = np.nonzero(mask)[0]
        return OutlierResult(mask, scores, thr, method, idx, int(mask.sum()))

    x1d = x_arr.ravel()
    n = len(x1d)

    if method == "mad":
        from stochpylib.robust_statistics._common import _mad
        thr = 3.5 if threshold is None else threshold
        med = float(np.median(x1d))
        mad = _mad(x1d)
        scores = np.abs(x1d - med) / (mad if mad > 0 else 1e-300)
        mask = scores > thr
    elif method == "iqr":
        thr = 1.5 if threshold is None else threshold
        q1, q3 = np.quantile(x1d, [0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - thr * iqr, q3 + thr * iqr
        scores = np.maximum(lo - x1d, x1d - hi)
        scores = np.maximum(scores, 0.0)
        mask = (x1d < lo) | (x1d > hi)
    elif method == "zscore":
        thr = 3.0 if threshold is None else threshold
        mu, sigma = np.mean(x1d), np.std(x1d, ddof=1)
        scores = np.abs(x1d - mu) / (sigma if sigma > 0 else 1e-300)
        mask = scores > thr
    elif method == "grubbs":
        from stochpylib.distributions import Student_t
        mask = np.zeros(n, dtype=bool)
        scores = np.zeros(n)
        remaining = np.arange(n)
        work = x1d.copy()
        thr = None
        while len(remaining) > 2:
            mu, sigma = np.mean(work), np.std(work, ddof=1)
            if sigma == 0:
                break
            dev = np.abs(work - mu)
            i_max = int(np.argmax(dev))
            g = dev[i_max] / sigma
            m = len(remaining)
            t_crit = Student_t(m - 2).ppf(1.0 - alpha / (2.0 * m))
            g_crit = ((m - 1) / math.sqrt(m)) * math.sqrt(t_crit ** 2 / (m - 2 + t_crit ** 2))
            thr = g_crit
            scores[remaining[i_max]] = g
            if g > g_crit:
                mask[remaining[i_max]] = True
                remaining = np.delete(remaining, i_max)
                work = np.delete(work, i_max)
            else:
                break
        thr = thr if thr is not None else float("nan")
    elif method == "hampel":
        from stochpylib.robust_statistics._common import _mad
        n_sigmas = 3.0 if threshold is None else threshold
        half = window // 2
        scores = np.zeros(n)
        mask = np.zeros(n, dtype=bool)
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            local = x1d[lo:hi]
            med = np.median(local)
            mad = _mad(local, center=med)
            s = abs(x1d[i] - med) / (mad if mad > 0 else 1e-300)
            scores[i] = s
            mask[i] = s > n_sigmas
        thr = n_sigmas
    else:
        raise ValueError(f"unknown method {method!r}")

    idx = np.nonzero(mask)[0]
    return OutlierResult(mask, scores, float(thr), method, idx, int(mask.sum()))


def missing_imputation(X, method="mean", *, fill_value=None, k=5, max_iter=100, tol=1e-6,
                       random_state=None, return_info=False):
    """Impute missing (``NaN``) values in ``X`` (1-D or 2-D, same shape out)."""
    from stochpylib import _rng

    arr = np.asarray(X, dtype=float)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr[:, None]
    mask = np.isnan(arr)
    n_missing = int(mask.sum())
    info = {"n_missing": n_missing, "iterations": 0, "converged": True}

    if method in ("mean", "median", "mode", "constant"):
        out = arr.copy()
        for j in range(arr.shape[1]):
            col = arr[:, j]
            m = np.isnan(col)
            if not m.any():
                continue
            if method == "mean":
                val = np.nanmean(col)
            elif method == "median":
                val = np.nanmedian(col)
            elif method == "mode":
                vals, counts = np.unique(col[~m], return_counts=True)
                val = vals[np.argmax(counts)] if len(vals) else 0.0
            else:
                val = fill_value if fill_value is not None else 0.0
            out[m, j] = val
        result = out
    elif method in ("locf", "nocb", "linear"):
        out = arr.copy()
        idx = np.arange(arr.shape[0])
        for j in range(arr.shape[1]):
            col = out[:, j]
            m = np.isnan(col)
            if not m.any():
                continue
            valid = ~m
            if method == "linear":
                if valid.sum() >= 2:
                    col[m] = np.interp(idx[m], idx[valid], col[valid])
                else:
                    col[m] = np.nanmean(col)
            elif method == "locf":
                last = None
                for i in range(len(col)):
                    if not m[i]:
                        last = col[i]
                    elif last is not None:
                        col[i] = last
                if np.isnan(col).any():
                    col[np.isnan(col)] = np.nanmean(col)
            else:  # nocb
                nxt = None
                for i in range(len(col) - 1, -1, -1):
                    if not m[i]:
                        nxt = col[i]
                    elif nxt is not None:
                        col[i] = nxt
                if np.isnan(col).any():
                    col[np.isnan(col)] = np.nanmean(col)
        result = out
    elif method == "knn":
        result = _knn_impute(arr, mask, k)
    elif method == "em":
        result, info = _em_impute(arr, mask, max_iter, tol)
    elif method == "mice":
        result, info = _mice_impute(arr, mask, max_iter, random_state)
    else:
        raise ValueError(f"unknown method {method!r}")

    if was_1d:
        result = result[:, 0]
    if return_info:
        return result, info
    return result


def _knn_impute(arr, mask, k):
    n, p = arr.shape
    out = arr.copy()
    for i in range(n):
        missing_cols = np.nonzero(mask[i])[0]
        if len(missing_cols) == 0:
            continue
        observed_cols = np.nonzero(~mask[i])[0]
        if len(observed_cols) == 0:
            for c in missing_cols:
                out[i, c] = np.nanmean(arr[:, c])
            continue
        dists = []
        for j in range(n):
            if j == i or mask[j, observed_cols].any():
                continue
            d = np.sqrt(np.sum((arr[i, observed_cols] - arr[j, observed_cols]) ** 2))
            dists.append((d, j))
        dists.sort(key=lambda t: t[0])
        donors = [j for _, j in dists[:k]]
        for c in missing_cols:
            avail = [arr[j, c] for j in donors if not mask[j, c]]
            out[i, c] = np.mean(avail) if avail else np.nanmean(arr[:, c])
    return out


def _em_impute(arr, mask, max_iter, tol):
    n, p = arr.shape
    filled = arr.copy()
    col_means = np.nanmean(arr, axis=0)
    for j in range(p):
        filled[mask[:, j], j] = col_means[j]
    mean = filled.mean(axis=0)
    cov = np.cov(filled, rowvar=False, ddof=1) if p > 1 else np.array([[filled.var(ddof=1)]])
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        prev_mean = mean.copy()
        for i in range(n):
            miss = mask[i]
            if not miss.any():
                continue
            obs = ~miss
            if obs.all():
                continue
            if obs.any():
                cov_oo = cov[np.ix_(obs, obs)]
                cov_mo = cov[np.ix_(miss, obs)]
                try:
                    sol = np.linalg.solve(cov_oo + 1e-10 * np.eye(obs.sum()),
                                          (filled[i, obs] - mean[obs]))
                    filled[i, miss] = mean[miss] + cov_mo @ sol
                except np.linalg.LinAlgError:
                    filled[i, miss] = mean[miss]
            else:
                filled[i, miss] = mean[miss]
        mean = filled.mean(axis=0)
        cov = np.cov(filled, rowvar=False, ddof=1) if p > 1 else np.array([[filled.var(ddof=1)]])
        if np.max(np.abs(mean - prev_mean)) < tol:
            converged = True
            break
    return filled, {"n_missing": int(mask.sum()), "iterations": it, "converged": converged,
                    "mean": mean, "cov": cov}


def _mice_impute(arr, mask, max_iter, random_state):
    from stochpylib import _rng

    rng = _rng.as_generator(random_state)
    n, p = arr.shape
    filled = arr.copy()
    col_means = np.nanmean(arr, axis=0)
    for j in range(p):
        filled[mask[:, j], j] = col_means[j]

    for it in range(max_iter if max_iter < 20 else 20):
        for j in range(p):
            miss_j = mask[:, j]
            if not miss_j.any():
                continue
            obs_rows = ~miss_j
            predictors = [c for c in range(p) if c != j]
            if not predictors or obs_rows.sum() < len(predictors) + 2:
                filled[miss_j, j] = col_means[j] + 0.01 * rng.standard_normal(miss_j.sum())
                continue
            Xd = np.column_stack([np.ones(obs_rows.sum()), filled[obs_rows][:, predictors]])
            y = filled[obs_rows, j]
            beta_hat, *_ = np.linalg.lstsq(Xd, y, rcond=None)
            resid = y - Xd @ beta_hat
            dof = max(len(y) - Xd.shape[1], 1)
            sigma2 = float(np.sum(resid ** 2) / dof)
            cov_beta = sigma2 * np.linalg.pinv(Xd.T @ Xd)
            beta_draw = rng.multivariate_normal(beta_hat, cov_beta + 1e-12 * np.eye(len(beta_hat)))
            X_miss = np.column_stack([np.ones(miss_j.sum()), filled[miss_j][:, predictors]])
            pred = X_miss @ beta_draw
            noise = rng.standard_normal(miss_j.sum()) * math.sqrt(max(sigma2, 1e-12))
            filled[miss_j, j] = pred + noise
    return filled, {"n_missing": int(mask.sum()), "iterations": min(max_iter, 20),
                    "converged": True, "mean": filled.mean(axis=0),
                    "cov": np.cov(filled, rowvar=False, ddof=1) if p > 1 else
                    np.array([[filled.var(ddof=1)]])}
