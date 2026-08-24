"""Survival regression: Cox PH (Breslow/Efron), stratified Cox, Weibull AFT,
Aalen additive hazards and Fine-Gray subdistribution regression."""

import numpy as np
from scipy import optimize

from stochpylib.survival._base import _check_durations_events
from stochpylib.survival.nonparametric import (
    BreslowEstimator,
    KaplanMeier,
)

__all__ = [
    "CoxProportionalHazards", "StratifiedCox", "AcceleratedFailureTime",
    "AalenAdditiveModel", "FineGrayModel",
]

_EPS = 1e-12


def _as_covariates(covariates, n=None):
    X = np.asarray(covariates, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if n is not None and len(X) != n:
        raise ValueError("covariates length mismatch")
    return X


def _sorted_blocks(times, events):
    """Ascending sort order plus per-event-time bookkeeping."""
    order = np.argsort(times, kind="mergesort")
    ts, es = times[order], events[order]
    uniq = np.unique(ts[es == 1])
    starts = np.searchsorted(ts, uniq, side="left")
    ends = np.searchsorted(ts, uniq, side="right")
    dmask = (es == 1).astype(float)
    d_counts = np.add.reduceat(dmask, starts).astype(int)
    d_counts = np.minimum(d_counts, ends - starts)
    # tie-group member indices (only groups with actual ties matter)
    tie_groups = []
    for k in range(len(uniq)):
        if d_counts[k] > 1:
            idx = starts[k] + np.flatnonzero(es[starts[k]:ends[k]] == 1)
            tie_groups.append((k, idx))
    return order, ts, es, uniq, starts, d_counts, tie_groups


def _cox_neg_ll(beta, ts, es, Xs, starts, d_counts, tie_groups,
                apply_efron=True):
    eta = Xs @ beta
    eta = eta - float(eta.max())
    w = np.exp(eta)
    cs_w = np.concatenate([[0.0], np.cumsum(w)])
    risk_w = cs_w[-1] - cs_w[np.asarray(starts)]
    dmask = (es == 1).astype(float)
    ll = float(np.sum(eta * dmask))
    ll -= float(np.sum(d_counts * np.log(np.maximum(risk_w, _EPS))))
    if apply_efron:
        for k, idx in tie_groups:
            m = len(idx)
            S0D = float(w[idx].sum())
            for l in range(m):
                ll -= np.log(max(risk_w[k] - (l / m) * S0D, _EPS))
    return -ll


def _cox_grad_info(beta, ts, es, Xs, starts, d_counts, tie_groups,
                   apply_efron=True):
    p = Xs.shape[1]
    eta = Xs @ beta
    eta = eta - float(eta.max())
    w = np.exp(eta)
    df = d_counts.astype(float)

    cs_w = np.concatenate([[0.0], np.cumsum(w)])
    WX = w[:, None] * Xs
    cs_wx = np.vstack([np.zeros((1, p)), np.cumsum(WX, axis=0)])
    cs_wxx = np.vstack([np.zeros((1, p, p)),
                        np.cumsum(WX[:, :, None] * Xs[:, None, :], axis=0)])
    sel = np.asarray(starts)
    risk_w = cs_w[-1] - cs_w[sel]
    risk_wx = cs_wx[-1] - cs_wx[sel]
    risk_wxx = cs_wxx[-1] - cs_wxx[sel]

    dmask = (es == 1).astype(float)
    d_theta_x = (w * dmask) @ Xs

    xbar = risk_wx / np.maximum(risk_w, _EPS)[:, None]

    grad = d_theta_x.sum(axis=0) - (df[:, None] * xbar).sum(axis=0)
    term_a = risk_wxx / np.maximum(risk_w, _EPS)[:, None, None]
    info = np.einsum('u,uij->ij', df, term_a) \
        - np.einsum('u,ui,uj->ij', df, xbar, xbar)
    ll = float((eta * dmask).sum()) \
        - float((df * np.log(np.maximum(risk_w, _EPS))).sum())

    # Efron correction: replace the tied blocks' Breslow contribution with the
    # sum over conditional (l/m)-reduced denominators
    if apply_efron:
        for k, idx in tie_groups:
            m = len(idx)
            Sw = risk_w[k]
            SwX = risk_wx[k]
            SwXX = risk_wxx[k]
            S0D = float(w[idx].sum())
            S0DX = (w[idx, None] * Xs[idx]).sum(axis=0)
            dk = float(d_counts[k])
            xb = SwX / max(Sw, _EPS)
            grad -= dk * xb
            info -= dk * (SwXX / max(Sw, _EPS) - np.outer(xb, xb))
            ll += dk * np.log(max(Sw, _EPS))
            for l in range(m):
                cfrac = l / m
                st = max(Sw - cfrac * S0D, _EPS)
                xst = (SwX - cfrac * S0DX) / st
                ll -= np.log(st)
                grad += Xs[idx[l]] - xst
                info += SwXX / st - np.outer(xst, xst)
    # convert to the NEGATIVE log-likelihood convention
    return -grad, info, -ll


class CoxProportionalHazards:
    """Cox proportional-hazards regression (Breslow or Efron tie handling)::

        cph = CoxProportionalHazards().fit(durations, events, X)
        cph.coefficients_, cph.hazard_ratios_, cph.concordance_index_
        cph.baseline_.predict(times)     # Breslow baseline cumulative hazard

    Standard errors come from the inverse observed information with Wald
    z-tests; ``summary()`` prints the classic table.
    """

    def __init__(self, ties="efron", alpha=0.05):
        if ties not in ("breslow", "efron"):
            raise ValueError("ties must be 'breslow' or 'efron'")
        self.ties = ties
        self.alpha = float(alpha)

    def fit(self, durations, events, covariates):
        t, e = _check_durations_events(durations, events)
        X = _as_covariates(covariates, n=len(t))
        self.n_obs_ = len(t)
        self.n_features_ = X.shape[1]
        self.feature_names_ = [f"x{j}" for j in range(self.n_features_)]
        order, ts, es, uniq, starts, d_counts, tie_groups = _sorted_blocks(t, e)
        Xs = X[order]

        beta = np.zeros(self.n_features_)
        prev = np.inf
        ll = _cox_neg_ll(beta, ts, es, Xs, starts, d_counts, tie_groups,
                        self.ties == "efron")
        for _ in range(100):
            grad, info, _ll = _cox_grad_info(
                beta, ts, es, Xs, starts, d_counts, tie_groups,
                apply_efron=self.ties == "efron")
            try:
                delta = np.linalg.solve(info, grad)
            except np.linalg.LinAlgError:
                break
            # step halving on the negative log partial likelihood
            step = 1.0
            for _bt in range(30):
                cand = beta - step * delta
                cand = cand if np.all(np.isfinite(cand)) else beta
                f_new = _cox_neg_ll(cand, ts, es, Xs, starts, d_counts,
                                    tie_groups,
                                    apply_efron=self.ties == "efron")
                if f_new <= ll or step < 2 ** -30:
                    break
                step *= 0.5
            ll_new = f_new
            done = abs(ll - ll_new) < tol_scale(ll)
            beta = cand
            ll = ll_new
            if done:
                break

        self.coefficients_ = beta
        self.log_likelihood_ = -ll
        grad, info, _ = _cox_grad_info(
            beta, ts, es, Xs, starts, d_counts, tie_groups,
            apply_efron=self.ties == "efron")
        var = np.linalg.inv(info) if np.all(np.isfinite(info)) \
            else np.full_like(info, np.nan)
        self.variance_matrix_ = var
        se = np.sqrt(np.clip(np.diag(var), 0.0, np.inf))
        self.standard_errors_ = se
        from scipy import stats as sps
        self.z_scores_ = beta / np.where(se > 0, se, np.nan)
        self.p_values_ = 2.0 * sps.norm.sf(np.abs(self.z_scores_))
        self.hazard_ratios_ = np.exp(beta)
        self.confidence_intervals_ = np.exp(np.column_stack(
            [beta - sps.norm.ppf(1 - self.alpha / 2) * se,
             beta + sps.norm.ppf(1 - self.alpha / 2) * se]))
        self.baseline_ = BreslowEstimator().fit(
            t, e, np.exp(np.clip(X @ beta, -300, 300)))
        self.concordance_index_ = _concordance_index(t, e, X @ beta)
        return self

    def baseline_cumulative_hazard_(self, times=None):
        times = times if times is not None else self.baseline_.times_
        return self.baseline_.predict(times)

    def predict_partial_hazard(self, covariates):
        X = _as_covariates(covariates)
        return np.exp(X @ self.coefficients_)

    def summary(self):
        rows = []
        for j, name in enumerate(self.feature_names_):
            rows.append(
                f"{name}: HR={self.hazard_ratios_[j]:.4f} "
                f"[{self.confidence_intervals_[j, 0]:.4f}, "
                f"{self.confidence_intervals_[j, 1]:.4f}] "
                f"z={self.z_scores_[j]:.2f} p={self.p_values_[j]:.4g}")
        return "\n".join(rows)


def tol_scale(ll):
    return 1e-9 * (abs(ll) + 1e-9)


def _concordance_index(times, events, scores):
    """Harrell's C: fraction of comparable pairs ordered consistently with
    the risk score (higher score -> shorter survival). Ties contribute 0.5."""
    t = np.asarray(times, float)
    e = np.asarray(events, int)
    r = np.asarray(scores, float)
    n = len(t)
    conc = tied = total = 0
    for ii in np.argsort(t, kind="mergesort"):
        if e[ii] == 0:
            continue
        later = np.flatnonzero(
            ((t > t[ii]) | ((t == t[ii]) & (e == 0))) &
            (np.arange(n) != ii))
        if len(later) == 0:
            continue
        diff = r[later] - r[ii]
        conc += int(np.sum(diff < 0))
        tied += int(np.sum(diff == 0))
        total += len(later)
    if total == 0:
        return float("nan")
    return (conc + 0.5 * tied) / total


class StratifiedCox(CoxProportionalHazards):
    """Cox regression with stratum-specific baselines and shared betas::

        sc = StratifiedCox().fit(durations, events, X, strata)
        sc.baseline_by_stratum_['A'].predict(times)
    """

    def fit(self, durations, events, covariates, strata):
        t, e = _check_durations_events(durations, events)
        X = _as_covariates(covariates, n=len(t))
        strata = np.asarray(strata).ravel()
        if len(strata) != len(t):
            raise ValueError("strata must match durations")
        self.n_features_ = X.shape[1]
        self.strata_names_ = sorted(set(strata.tolist()), key=str)
        beta = None
        info_acc = None
        ll_total = 0.0
        parts = {}
        for s in self.strata_names_:
            mask = strata == s
            tb, eb, Xb = t[mask], e[mask], X[mask]
            order, ts, es, uniq, starts, d_counts, tie_groups = _sorted_blocks(tb, eb)
            Xs = Xb[order]
            b0 = np.zeros(self.n_features_)
            prev = np.inf
            ll = _cox_neg_ll(b0, ts, es, Xs, starts, d_counts, tie_groups,
                             apply_efron=self.ties == "efron")
            for _ in range(100):
                grad, info, _ll = _cox_grad_info(
                    b0, ts, es, Xs, starts, d_counts, tie_groups,
                    apply_efron=self.ties == "efron")
                try:
                    delta = np.linalg.solve(info, grad)
                except np.linalg.LinAlgError:
                    break
                step = 1.0
                for _bt in range(30):
                    cand = b0 - step * delta
                    f_new = _cox_neg_ll(cand, ts, es, Xs, starts, d_counts,
                                        tie_groups,
                                        apply_efron=self.ties == "efron")
                    if f_new <= ll or step < 2 ** -30:
                        break
                    step *= 0.5
                done = abs(ll - f_new) < tol_scale(ll)
                b0, prev = cand, abs(ll - f_new)
                ll = f_new
                if done:
                    break
            _grad, info, _ = _cox_grad_info(
                b0, ts, es, Xs, starts, d_counts, tie_groups,
                apply_efron=self.ties == "efron")
            beta = b0 if beta is None else beta
            info_acc = info if info_acc is None else info_acc + info
            ll_total += ll
            parts[s] = b0
        self.coefficients_ = beta
        self.log_likelihood_ = ll_total
        try:
            var = np.linalg.inv(info_acc)
            if not np.all(np.isfinite(var)):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            var = np.linalg.pinv(info_acc)
        self.variance_matrix_ = var
        se = np.sqrt(np.clip(np.diag(var), 0, np.inf))
        self.standard_errors_ = se
        from scipy import stats as sps
        self.z_scores_ = beta / np.where(se > 0, se, np.nan)
        self.p_values_ = 2.0 * sps.norm.sf(np.abs(self.z_scores_))
        self.hazard_ratios_ = np.exp(beta)
        self.confidence_intervals_ = np.exp(np.column_stack(
            [beta - 1.96 * se, beta + 1.96 * se]))
        self.concordance_index_ = _concordance_index(t, e, X @ beta)
        self.baseline_by_stratum_ = {
            s: BreslowEstimator().fit(
                t[strata == s], e[strata == s],
                np.exp(np.clip(X[strata == s] @ beta, -300, 300)))
            for s in self.strata_names_}
        self.baseline_ = self.baseline_by_stratum_[self.strata_names_[0]]
        self.n_obs_ = len(t)
        self.n_features_ = X.shape[1]
        self.feature_names_ = [f"x{j}" for j in range(self.n_features_)]
        return self


class AcceleratedFailureTime:
    """Weibull accelerated-failure-time model::

        aft = AcceleratedFailureTime().fit(durations, events, X)
        aft.coefficients_   # positive => shorter life (acceleration)

    S(t|x) = exp(-(t/lambda(x))^k), lambda(x)=exp(a - x'beta);
    fitted by maximum likelihood under right censoring.
    """

    def __init__(self):
        self.coefficients_ = None

    def fit(self, durations, events, covariates):
        t, e = _check_durations_events(durations, events)
        X = _as_covariates(covariates, n=len(t))
        self.n_obs_ = len(t)
        self.n_features_ = X.shape[1]
        self.feature_names_ = [f"x{j}" for j in range(self.n_features_)]

        def neg_ll(theta):
            a, log_k = theta[0], theta[1]
            k = float(np.exp(log_k))
            beta = theta[2:]
            lam = np.exp(np.clip(a - X @ beta, -300, 300))
            r = np.clip(t / np.maximum(lam, _EPS), _EPS, 1e300)
            dens = (k / lam) * r ** (k - 1.0) * np.exp(-np.clip(
                r ** k, -700, 700))
            surv = np.exp(-np.clip(r ** k, -700, 700))
            dens = np.clip(dens, _EPS, None)
            surv = np.clip(surv, _EPS, 1.0)
            ll = float(np.sum(e * np.log(dens) + (1 - e) * np.log(surv)))
            return -ll if np.isfinite(ll) else 1e12

        med = max(float(np.median(t)), 1e-3)
        theta0 = np.concatenate([[np.log(med), 0.0],
                                 np.zeros(self.n_features_)])
        res = optimize.minimize(neg_ll, theta0, method="Nelder-Mead",
                                options={"xatol": 1e-7, "fatol": 1e-9,
                                         "maxiter": 8000})
        th = res.x
        self.intercept_ = float(th[0])
        self.shape_ = float(np.exp(th[1]))
        self.coefficients_ = th[2:]
        self.log_likelihood_ = -float(res.fun)
        self.aic_ = -2.0 * self.log_likelihood_ + 2.0 * len(th)
        self.hazard_ratios_ = np.exp(self.coefficients_)
        return self

    def predict_median(self, covariates):
        X = _as_covariates(covariates)
        lam = np.exp(self.intercept_ - X @ self.coefficients_)
        return lam * (np.log(2.0)) ** (1.0 / self.shape_)


class AalenAdditiveModel:
    """Aalen additive hazards: dN_i(t) = Y_i(t) x_i' beta(t) dt + noise.

    Per-event-time least squares over risk-set rows, increments accumulated
    into ``cumulative_coefficients_`` (structured steps). Include a constant
    column in ``X`` to obtain the nonparametric baseline::

        am = AalenAdditiveModel().fit(durations, events, X_with_const)
        am.predict(x_row, times=[1.0, 2.0])   # cumulative hazard
    """

    def __init__(self):
        self.cumulative_coefficients_ = None

    def fit(self, durations, events, covariates):
        t, e = _check_durations_events(durations, events)
        X = _as_covariates(covariates, n=len(t))
        self.n_obs_ = len(t)
        self.n_features_ = X.shape[1]
        order = np.argsort(t, kind="mergesort")
        ts, es, Xs = t[order], e[order], X[order]
        uniq = np.unique(ts[es == 1])
        steps_time = []
        steps_coef = []
        cum = np.zeros(self.n_features_)
        last_u = -np.inf
        for u in uniq:
            start = int(np.searchsorted(ts, u, side="left"))
            Xr = Xs[start:]
            # response: dN_i(u) = 1 iff subject i fails AT u
            dr = ((ts[start:] >= u - 1e-12) &
                  (ts[start:] <= u + 1e-12) &
                  (es[start:] > 0)).astype(float)
            n_risk = Xr.shape[0]
            if n_risk < self.n_features_ or \
                    np.linalg.matrix_rank(Xr) < self.n_features_:
                continue
            b_inc, *_ = np.linalg.lstsq(Xr, dr, rcond=None)
            if not np.all(np.isfinite(b_inc)):
                continue
            cum = cum + b_inc
            steps_time.append(u)
            steps_coef.append(cum.copy())
        arr = np.empty(len(steps_time),
                       dtype=[("time", float),
                              ("coef", float, self.n_features_)])
        arr["time"] = steps_time
        arr["coef"] = steps_coef
        self.cumulative_coefficients_ = arr
        return self

    def predict(self, covariates, times):
        x = np.asarray(covariates, dtype=float).ravel()
        times = np.atleast_1d(np.asarray(times, dtype=float))
        cc = self.cumulative_coefficients_
        out = np.empty(len(times))
        for qi, tt in enumerate(times):
            idx = int(np.searchsorted(cc["time"], tt, side="right")) - 1
            out[qi] = 0.0 if idx < 0 else float(x @ cc["coef"][idx])
        return out


class FineGrayModel:
    """Fine-Gray subdistribution-hazards regression for competing risks::

        fg = FineGrayModel().fit(durations, cause, X, cause_of_interest=1)

    ``cause`` codes: 0 = censored, 1..K causes. Weighted-Cox scoring with the
    Fine-Gray time-dependent risk-set weights and inverse-KM-of-censoring
    correction; covariance is the naive inverse information (documented).
    """

    def __init__(self):
        self.coefficients_ = None

    def fit(self, durations, cause, covariates, cause_of_interest=1):
        t, _ = _check_durations_events(durations, np.ones(len(durations)))
        cause = np.asarray(cause).ravel().astype(int)
        X = _as_covariates(covariates, n=len(t))
        self.n_obs_ = len(t)
        self.n_features_ = X.shape[1]
        self.feature_names_ = [f"x{j}" for j in range(self.n_features_)]
        km_c = KaplanMeier().fit(t, (cause == 0).astype(float))
        g_vec = lambda s: np.clip(
            np.asarray(km_c.predict(np.atleast_1d(np.asarray(s, float))),
                       dtype=float), 1e-8, 1.0)
        is_int = cause == cause_of_interest
        is_comp = (cause != 0) & ~is_int

        uniq = np.unique(t[is_int])
        K = len(uniq)
        W = np.zeros((K, len(t)))
        D = np.zeros((K, len(t)), dtype=float)
        for kk, s in enumerate(uniq):
            still = t >= s
            comp_before = is_comp & (t < s)
            w_still = float(g_vec(s)[0])
            wc = np.where(comp_before,
                          1.0 / g_vec(np.minimum(t, s)), 0.0)
            W[kk] = np.where(is_int & (t >= s), w_still, 0.0) + wc
            W[kk] = np.maximum(W[kk], 0.0)
            D[kk] = (np.abs(t - s) < 1e-12) & is_int

        def score_terms(beta):
            eta = X @ beta
            theta = np.exp(np.clip(eta - eta.max(), None, 50.0))
            U = np.zeros(self.n_features_)
            I = np.zeros((self.n_features_, self.n_features_))
            ll = 0.0
            for kk in range(len(uniq)):
                sw = W[kk] * theta
                s0 = float(sw.sum())
                dk = D[kk]
                dw = float((W[kk] * dk).sum())
                if dw == 0 or s0 <= 0:
                    continue
                xbar = (sw[:, None] * X).sum(axis=0) / s0
                ll += float((dk * W[kk] * eta).sum()) - dw * np.log(s0)
                U += (dk * W[kk]) @ X - dw * xbar
                I += (X.T * sw) @ X / s0 - np.outer(xbar, xbar) * dw
            return U, I, ll

        beta = np.zeros(self.n_features_)
        ll_plus = -np.inf
        for _ in range(200):
            U, I, ll_new = score_terms(beta)
            if not np.all(np.isfinite(I)):
                break
            try:
                delta = np.linalg.solve(I, U)      # Newton ascent on +ll
            except np.linalg.LinAlgError:
                break
            step = 1.0
            for _bt in range(40):
                cand = beta + step * delta
                _, _, ll_try = score_terms(cand)
                if ll_try >= ll_plus or step < 2 ** -30:
                    break
                step *= 0.5
            done = abs(ll_new - ll_plus) < 1e-9 * (abs(ll_new) + 1e-9)
            beta = cand
            ll_plus = max(ll_plus, ll_new)
            if done:
                break
        self.coefficients_ = beta
        U, I, ll_plus_final = score_terms(beta)
        self.log_likelihood_ = ll_plus_final
        try:
            var = np.linalg.inv(I)
        except np.linalg.LinAlgError:
            var = np.linalg.pinv(I)
        self.variance_matrix_ = var
        se = np.sqrt(np.clip(np.diag(var), 0, np.inf))
        self.standard_errors_ = se
        from scipy import stats as sps
        self.z_scores_ = beta / np.where(se > 0, se, np.nan)
        self.p_values_ = 2.0 * sps.norm.sf(np.abs(self.z_scores_))
        self.hazard_ratios_ = np.exp(beta)
        self.censoring_survival_ = km_c
        return self
