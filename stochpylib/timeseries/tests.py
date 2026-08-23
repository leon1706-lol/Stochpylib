"""Classical time-series hypothesis tests (native implementations).

Critical-value policy (documented per test):

- ``adf_test`` / ``pp_test``: p-values and critical values come from a cached, seeded
  Monte-Carlo simulation of the Dickey-Fuller null distribution (deterministic across
  runs; documented approximation when augmentation lags are selected by AIC).
- ``kpss_test``: published critical values from Kwiatkowski, Phillips, Schmidt & Shin
  (1992), with linearly interpolated approximate p-values.
- ``ljung_box``, ``arch_test``, ``granger_causality``: exact chi-square / F formulas.
- ``johansen_test``: trace and maximum-eigenvalue statistics; critical values from a
  seeded Monte-Carlo simulation of the unit-root VAR null (published Johansen/Osterwald-
  Lenum tables cover only specific dimension/spec combinations).

All simulations use module-level caches keyed on their configuration, so results are
reproducible bit-for-bit while paying the simulation cost once per session.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from stochpylib.timeseries._result import TestResult
from stochpylib.timeseries._utils import as_1d, as_2d, nw_lags


# ---------------------------------------------------------------------------
# shared helpers


def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid


def _schwert_maxlag(n):
    return int(np.floor(12.0 * (n / 100.0) ** 0.25))


def _nw_long_run(u, lags=None):
    """Newey-West long-run variance of a zero-mean-ish series."""
    u = np.asarray(u, dtype=float)
    T = len(u)
    if lags is None:
        lags = nw_lags(T)
    u_centered = u - u.mean()
    s0 = float(u_centered @ u_centered) / T
    s = s0
    for j in range(1, min(int(lags), T - 1) + 1):
        gamma = float(u_centered[j:] @ u_centered[:-j]) / T
        s += 2.0 * (1.0 - j / (lags + 1.0)) * gamma
    return max(s, 1e-300)


# --- Dickey-Fuller null distribution (cached Monte Carlo) --------------------


_DF_CACHE = {}


def _df_null(regression, T):
    """Sorted Monte-Carlo sample of the DF t-statistic under the unit-root null."""
    key = (regression, int(T))
    if key in _DF_CACHE:
        return _DF_CACHE[key]
    rng = np.random.default_rng(0xD1CE5E)
    n_sims, chunk = 20_000, 1000
    out = np.empty(n_sims)
    done = 0
    while done < n_sims:
        m = min(chunk, n_sims - done)
        eps = rng.standard_normal((m, T))
        y = np.cumsum(eps, axis=1)
        dy = np.diff(y, axis=1)
        ylag = y[:, :-1]
        cols = [ylag]
        if regression == "c":
            cols.insert(0, np.ones_like(ylag))
        elif regression == "ct":
            cols.insert(0, np.ones_like(ylag))
            tt = np.arange(1, T)[None, :].repeat(m, axis=0) / float(T)
            cols.insert(1, tt)
        X = np.stack(cols, axis=-1)  # (m, T-1, k)
        XtX = np.einsum("rti,rtj->rij", X, X)
        Xty = np.einsum("rti,rt->ri", X, dy)
        betas = np.linalg.solve(XtX, Xty[..., None])[..., 0]
        resid = dy - np.einsum("rti,ri->rt", X, betas)
        dof = T - 1 - X.shape[-1]
        s2 = np.sum(resid**2, axis=1) / dof
        idx_gamma = 1 if regression in ("c", "ct") else 0
        inv_diag = np.linalg.inv(XtX)[:, idx_gamma, idx_gamma]
        t_stats = betas[:, idx_gamma] / np.sqrt(s2 * inv_diag)
        out[done : done + m] = t_stats
        done += m
    result = np.sort(out)
    _DF_CACHE[key] = result
    return result


def _mc_p_and_cv(sample, stat):
    lo, hi = np.quantile(sample, [0.001, 0.999])
    stat_clamped = float(np.clip(stat, lo, hi))
    pvalue = float(np.mean(sample <= stat_clamped))
    cvs = {
        "1%": float(np.quantile(sample, 0.01)),
        "5%": float(np.quantile(sample, 0.05)),
        "10%": float(np.quantile(sample, 0.10)),
    }
    return pvalue, cvs


# --------------------------------------------------------------------------- ADF


def adf_test(x, max_lag=None, regression="c"):
    """Augmented Dickey-Fuller unit-root test.

    Null: the series contains a unit root. ``regression`` is ``'c'``, ``'ct'`` or
    ``'n'``. Lag semantics match statsmodels: an explicit ``max_lag`` is used as the
    exact augmentation order; with the default ``None`` the order is AIC-selected over
    ``0..floor(12 (n/100)^{1/4})``. p-values/critical values come from the cached DF
    Monte Carlo.
    """
    x = as_1d(x)
    if regression not in ("c", "ct", "n"):
        raise ValueError("regression must be 'c', 'ct' or 'n'")
    n = len(x)

    if max_lag is None:
        candidate_lags = list(range(_schwert_maxlag(n) + 1))  # AIC selection
        select_by_aic = True
    else:
        candidate_lags = [int(max_lag)]
        select_by_aic = False

    best = None
    for lag in candidate_lags:
        dy = np.diff(x)
        usable = len(dy) - lag
        cols = []
        if regression == "c":
            cols.append(np.ones(usable))
        elif regression == "ct":
            cols.append(np.ones(usable))
            cols.append(np.arange(lag + 1, len(dy) + 1) / float(len(dy)))
        cols.append(x[lag : -1])
        for j in range(1, lag + 1):
            cols.append(dy[lag - j : len(dy) - j])
        X = np.column_stack(cols)
        target = dy[lag:]
        beta, resid = _ols(X, target)
        dof = len(target) - X.shape[1]
        s2 = float(resid @ resid) / dof
        XtX_inv = np.linalg.inv(X.T @ X)
        idx = 1 if regression in ("c", "ct") else 0
        t_stat = float(beta[idx] / np.sqrt(s2 * XtX_inv[idx, idx]))
        aic = float(np.log(s2) + 2 * (X.shape[1]) / len(target))
        if best is None or aic < best[0]:
            best = (aic, t_stat, lag)

    _, t_stat, used_lag = best
    sample = _df_null(regression, min(len(x), 400))
    pvalue, cvs = _mc_p_and_cv(sample, t_stat)
    return TestResult(
        statistic=t_stat,
        pvalue=pvalue,
        null="the series contains a unit root",
        critical_values={**cvs, "lags": used_lag},
    )


# --------------------------------------------------------------------------- PP


def pp_test(x, regression="c", lags=None):
    """Phillips-Perron Z-tau unit-root test (Phillips 1987 correction).

    Nonparametric Newey-West adjustment of the Dickey-Fuller t-statistic; null
    distribution approximated by the cached DF Monte Carlo (documented).
    """
    x = as_1d(x)
    if regression not in ("c", "ct", "n"):
        raise ValueError("regression must be 'c', 'ct' or 'n'")
    T = len(x)
    lags = nw_lags(T) if lags is None else int(lags)

    dy = np.diff(x)
    cols = []
    if regression == "c":
        cols.append(np.ones(len(dy)))
    elif regression == "ct":
        cols.append(np.ones(len(dy)))
        cols.append(np.arange(1, len(dy) + 1) / float(len(dy)))
    cols.append(x[:-1])
    X = np.column_stack(cols)
    beta, u = _ols(X, dy)
    idx = 1 if regression in ("c", "ct") else 0
    XtX_inv = np.linalg.inv(X.T @ X)
    s2 = float(u @ u) / T
    t_stat = float(beta[idx] / np.sqrt(s2 * XtX_inv[idx, idx]))

    v = u * x[:-1]  # product series entering the long-run variance
    omega2 = _nw_long_run(v - v.mean(), lags)

    sum_y2 = float(x[:-1] @ x[:-1])
    # Phillips (1987) Z-tau: short-run-scaled t-statistic plus serial-correlation term
    z_tau = np.sqrt(s2 / omega2) * t_stat + (omega2 - s2) / (2.0 * omega2) * (
        T / np.sqrt(sum_y2)
    )

    sample = _df_null(regression, min(len(x), 400))
    pvalue, cvs = _mc_p_and_cv(sample, z_tau)
    return TestResult(
        statistic=float(z_tau),
        pvalue=pvalue,
        null="the series contains a unit root",
        critical_values={**cvs, "bandwidth": lags},
    )


# --------------------------------------------------------------------------- KPSS


_KPSS_CV = {
    "c": {"1%": 0.739, "5%": 0.463, "10%": 0.347},
    "t": {"1%": 0.216, "5%": 0.146, "10%": 0.119},
}


def kpss_test(x, regression="c", lags=None):
    """KPSS stationarity test (Kwiatkowski et al., 1992).

    Null: (trend-)stationary. Critical values are the published ones; the approximate
    p-value interpolates linearly between them and is clipped to [0.01, 0.10]
    (documented convention, mirroring common implementations).
    """
    x = as_1d(x)
    if regression not in ("c", "t"):
        raise ValueError("regression must be 'c' or 't'")
    T = len(x)
    lags = nw_lags(T) if lags is None else int(lags)

    if regression == "c":
        resid = x - x.mean()
    else:
        tt = np.arange(1, T + 1)
        slope = ((tt - tt.mean()) * (x - x.mean())).sum() / ((tt - tt.mean()) ** 2).sum()
        resid = x - (x.mean() + slope * (tt - tt.mean()))

    cum = np.cumsum(resid)
    omega2 = _nw_long_run(resid, lags)
    stat = float((cum @ cum) / (T**2 * omega2))

    cv_table = _KPSS_CV[regression]
    # np.interp needs ascending x: CVs ascend as significance level falls
    levels = np.array([0.10, 0.05, 0.01])
    cvs_asc = np.array([cv_table["10%"], cv_table["5%"], cv_table["1%"]])
    pvalue = float(np.interp(stat, cvs_asc, levels))
    pvalue = float(np.clip(pvalue, 0.01, 0.10))

    return TestResult(
        statistic=stat,
        pvalue=pvalue,
        null="the series is (trend-)stationary",
        critical_values=dict(cv_table),
    )


# --------------------------------------------------------------------------- LB / DW


@dataclass
class LjungBoxResult:
    lags: list
    statistics: list
    pvalues: list

    def __repr__(self):
        rows = ", ".join(f"{q:.2f}(p={p:.3f})" for q, p in zip(self.statistics, self.pvalues))
        return f"LjungBoxResult({rows})"


def ljung_box(x, lags=10, box_pierce=False, fit_df=0):
    """Ljung-Box (or Box-Pierce) portmanteau test for remaining autocorrelation.

    Null: no autocorrelation up to the given lag. ``fit_df`` subtracts fitted
    parameter count (e.g. pass ``p+q`` after fitting an ARMA model).
    """
    x = as_1d(x)
    T = len(x)
    xc = x - x.mean()
    denom = float(xc @ xc)
    lags_list = lags if isinstance(lags, (list, tuple)) else list(range(1, int(lags) + 1))
    q_stats, p_vals = [], []
    for k in lags_list:
        k = int(k)
        rho_sum = 0.0
        for j in range(1, k + 1):
            rho = float(xc[j:] @ xc[:-j]) / denom
            rho_sum += rho**2
        if box_pierce:
            q = T * rho_sum
        else:
            q = T * (T + 2.0) * sum(
                (float(xc[j:] @ xc[:-j]) / denom) ** 2 / (T - j) for j in range(1, k + 1)
            )
        df = max(int(k) - int(fit_df), 1)
        q_stats.append(float(q))
        p_vals.append(float(stats.chi2.sf(q, df)))
    return LjungBoxResult([int(k) for k in lags_list], q_stats, p_vals)


def durbin_watson(resid):
    """Durbin-Watson statistic: sum(diff(e)^2) / sum(e^2)."""
    e = as_1d(resid)
    return float(np.diff(e) @ np.diff(e) / (e @ e))


# --------------------------------------------------------------------------- ARCH


def arch_test(resid, lags=12):
    """Engle's Lagrange-multiplier test for ARCH effects.

    Null: no ARCH effects (conditional homoskedasticity). Regresses squared residuals
    on their own lags; ``T * R^2`` is chi-square with ``lags`` degrees of freedom.
    """
    e = as_1d(resid)
    e = e - e.mean()
    e2 = e**2
    T = len(e2)
    lags = int(lags)
    if lags >= T - 1:
        raise ValueError("lags must be smaller than the sample")
    cols = [np.ones(T - lags)]
    for j in range(1, lags + 1):
        cols.append(e2[lags - j : T - j])
    X = np.column_stack(cols)
    target = e2[lags:]
    beta, resid_fit = _ols(X, target)
    r2 = 1.0 - float(resid_fit @ resid_fit) / float((target - target.mean()) @ (target - target.mean()))
    lm = float(T * r2)
    return TestResult(
        statistic=lm,
        pvalue=float(stats.chi2.sf(lm, lags)),
        null="no ARCH effects (conditional homoskedasticity)",
    )


# --------------------------------------------------------------------------- Granger


def granger_causality(x, y, max_lag=4):
    """Pairwise Granger-causality F-tests: does ``x`` help predict ``y``?

    For each lag ``L`` in ``1..max_lag``, compares an unrestricted OLS of
    ``y_t`` on ``[const, y lags 1..max_lag, x lags 1..L]`` against the restricted
    model without ``x`` terms, using the standard F statistic.

    Returns ``{lag: TestResult}``.
    """
    xx, yy = as_1d(x), as_1d(y)
    if len(xx) != len(yy):
        raise ValueError("x and y must have equal length")
    T = len(yy)
    out = {}
    for L in range(1, int(max_lag) + 1):
        rows_u, rows_r = [], []
        tu, tr = [], []
        start = int(max_lag)
        for t in range(start, T):
            base = [1.0] + [yy[t - j] for j in range(1, max_lag + 1)]
            extra = [xx[t - j] for j in range(1, L + 1)]
            rows_u.append(base + extra)
            rows_r.append(base)
            tu.append(yy[t])
            tr.append(yy[t])
        Xu, ru = np.asarray(rows_u), np.asarray(tu)
        Xr, rr = np.asarray(rows_r), np.asarray(tr)
        bu, res_u = _ols(Xu, ru)
        br, res_r = _ols(Xr, rr)
        ssu = float(res_u @ res_u)
        ssr = float(res_r @ res_r)
        df2 = len(ru) - Xu.shape[1]
        df1 = L
        f_stat = ((ssr - ssu) / df1) / (ssu / df2)
        out[L] = TestResult(
            statistic=float(f_stat),
            pvalue=float(stats.f.sf(f_stat, df1, df2)),
            null=f"x does not Granger-cause y (lag {L})",
        )
    return out


# --------------------------------------------------------------------------- Johansen


_JOHANSEN_MC = {}


def _coint_eigen(Y, p):
    """Eigenvalues of the reduced-rank regression (shared by VECM.fit)."""
    k = Y.shape[1]
    T = Y.shape[0]
    dY = Y[1:] - Y[:-1]
    endog = dY[p:]
    levels = Y[p:-1]
    lags = np.hstack([dY[p - j : T - 1 - j] for j in range(1, p + 1)])
    Dmat = np.hstack([np.ones((len(endog), 1)), lags])

    def resid_against(M):
        b, *_ = np.linalg.lstsq(Dmat, M, rcond=None)
        return M - Dmat @ b

    R0 = resid_against(endog)
    R1 = resid_against(levels)
    S00 = R0.T @ R0 / len(R0)
    S01 = R0.T @ R1 / len(R0)
    S11 = R1.T @ R1 / len(R1)
    Mmat = np.linalg.inv(S11) @ S01.T @ np.linalg.inv(S00) @ S01
    eigvals, eigvecs = np.linalg.eig(Mmat)
    order = np.argsort(eigvals.real)[::-1]
    return eigvals.real[order]


def _johansen_mc_cv(k, p, det_order, reps=4000, sim_T=250):
    """Seeded Monte-Carlo critical values for trace / max-eigenvalue statistics."""
    key = (k, p, det_order)
    if key in _JOHANSEN_MC:
        return _JOHANSEN_MC[key]
    rng = np.random.default_rng(0x50B0C)
    trace_draws = [[] for _ in range(k)]
    maxeig_draws = [[] for _ in range(k)]
    for _ in range(reps):
        Y = np.cumsum(rng.standard_normal((sim_T, k)), axis=0)
        eigs = _coint_eigen(Y, p)
        lam1 = np.log(1.0 - eigs)
        for r in range(k):
            trace_draws[r].append(-sim_T * float(np.sum(lam1[r:])))
            maxeig_draws[r].append(-sim_T * float(lam1[r]))
    result = {}
    for r in range(k):
        tr = np.sort(trace_draws[r])
        me = np.sort(maxeig_draws[r])
        result[r] = {
            "trace": {
                lv: float(np.quantile(tr, q))
                for lv, q in (("90%", 0.90), ("95%", 0.95), ("99%", 0.99))
            },
            "max_eig": {
                lv: float(np.quantile(me, q))
                for lv, q in (("90%", 0.90), ("95%", 0.95), ("99%", 0.99))
            },
        }
    _JOHANSEN_MC[key] = result
    return result


def johansen_test(Y, p=1, det_order=0):
    """Johansen cointegration rank test (trace and maximum-eigenvalue statistics).

    Null for row ``r``: cointegration rank <= r. Only ``det_order=0`` (constant in the
    error-correction relation) is supported; critical values come from the seeded
    Monte-Carlo simulation described at module level.

    Returns ``{"trace": [TestResult per r], "max_eig": [...]}``.
    """
    if det_order != 0:
        raise NotImplementedError("only det_order=0 (constant term) is implemented")
    Y = as_2d(Y, "Y")
    k = Y.shape[1]
    eigs = _coint_eigen(Y, int(p))
    T = len(Y)
    lam1 = np.log(1.0 - np.clip(eigs, 0.0, 1.0 - 1e-12))
    cvs = _johansen_mc_cv(k, int(p), det_order)

    trace_results, me_results = [], []
    for r in range(k):
        trace_stat = -T * float(np.sum(lam1[r:]))
        max_stat = -T * float(lam1[r])
        trace_results.append(
            TestResult(
                statistic=trace_stat,
                pvalue=None,
                null=f"cointegration rank <= {r}",
                critical_values=cvs[r]["trace"],
            )
        )
        me_results.append(
            TestResult(
                statistic=max_stat,
                pvalue=None,
                null=f"cointegration rank <= {r}",
                critical_values=cvs[r]["max_eig"],
            )
        )
    return {"trace": trace_results, "max_eig": me_results}


__all__ = [
    "adf_test",
    "pp_test",
    "kpss_test",
    "ljung_box",
    "durbin_watson",
    "arch_test",
    "granger_causality",
    "johansen_test",
]
