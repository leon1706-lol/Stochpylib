"""High-breakdown and M-estimator robust regression.

``TheilSenRegression``/``SiegalRegression`` are exact rank-based estimators
(simple regression matches ``scipy.stats.theilslopes``/``siegelslopes``
exactly); ``RANSACRegression``/``LTS_Regression``/``MMRegression`` are
50%-breakdown estimators built on randomized subsampling with concentration
(C-)steps; ``HuberRegression`` is the classic bounded-influence M-estimator
(matches ``statsmodels.RLM`` with ``M=HuberT()``).
"""

import math

import numpy as np

from stochpylib.robust_statistics._base import RobustRegressor
from stochpylib.robust_statistics._common import (
    _mad, _m_scale, _norm_cdf, _norm_ppf, _psi_prime_of,
    _psi_value, _psi_weight, _rankdata_ties, _rng, _subsets, _weiszfeld,
)

__all__ = [
    "TheilSenRegression", "RANSACRegression", "LTS_Regression", "MMRegression",
    "HuberRegression", "SiegalRegression", "SiegelRegression",
]


# ------------------------------------------------------------------ Theil-Sen

class TheilSenRegression(RobustRegressor):
    """Theil-Sen slope estimator: the median of all pairwise slopes.

    For a single predictor this matches ``scipy.stats.theilslopes`` exactly,
    including its Sen (1968) confidence interval on the slope. For multiple
    predictors, coefficients are the spatial (geometric) median of the OLS
    fits on random ``p+1``-point subsets.
    """

    def __init__(self, fit_intercept=True, intercept_method="separate",
                max_subpopulation=10_000, random_state=None):
        super().__init__(fit_intercept)
        if intercept_method not in ("separate", "joint"):
            raise ValueError("intercept_method must be 'separate' or 'joint'")
        self.intercept_method = intercept_method
        self.max_subpopulation = int(max_subpopulation)
        self.random_state = random_state

    def _fit(self, Xd, y):
        p = Xd.shape[1] - int(self.fit_intercept)
        if p == 1:
            x = Xd[:, -1]
            self._fit_simple(x, y)
        else:
            self._fit_multiple(Xd, y)

    def _fit_simple(self, x, y, level=0.95):
        n = len(x)
        iu0, iu1 = np.triu_indices(n, k=1)
        dx = x[iu1] - x[iu0]
        dy = y[iu1] - y[iu0]
        mask = dx != 0
        slopes = dy[mask] / dx[mask]
        slope = float(np.median(slopes))
        if self.intercept_method == "separate":
            intercept = float(np.median(y) - slope * np.median(x))
        else:
            intercept = float(np.median(y - slope * x))

        z = float(_norm_ppf(0.5 * (1.0 - level))) if level > 0.5 else float(_norm_ppf(level / 2.0))
        # Sen (1968) eq. 2.6, transcribed from scipy.stats.theilslopes
        nxreps = _rankdata_ties(x)
        nyreps = _rankdata_ties(y)
        ny = float(n)
        sigsq = (ny * (ny - 1) * (2 * ny + 5)
                 - np.sum(nxreps * (nxreps - 1) * (2 * nxreps + 5))
                 - np.sum(nyreps * (nyreps - 1) * (2 * nyreps + 5))) / 18.0
        sigma = math.sqrt(max(sigsq, 0.0))
        nt = float(len(slopes))
        Ru = int(min(round((nt - z * sigma) / 2.0), nt - 1))
        Rl = int(max(round((nt + z * sigma) / 2.0) - 1, 0))
        sorted_slopes = np.sort(slopes)
        if sigsq < 0:
            low_slope, high_slope = float("nan"), float("nan")
        else:
            low_slope = float(sorted_slopes[Rl])
            high_slope = float(sorted_slopes[Ru])

        self.coef_ = np.array([intercept, slope]) if self.fit_intercept else np.array([slope])
        se = np.array([np.nan, (high_slope - low_slope) / (2.0 * abs(z))]) if abs(z) > 0 else \
            np.array([np.nan, np.nan])
        self.std_errors_ = se if self.fit_intercept else se[1:]
        self.extras_ = {"slopes": slopes, "confidence_interval": (low_slope, high_slope)}
        self.n_iter_ = None

    def _fit_multiple(self, Xd, y):
        n, p = Xd.shape
        rng = _rng(self.random_state)
        idx, exhaustive = _subsets(n, p, self.max_subpopulation, rng)
        coefs = []
        for row in idx:
            Xs, ys = Xd[row], y[row]
            if abs(np.linalg.det(Xs)) < 1e-12:
                continue
            try:
                coefs.append(np.linalg.solve(Xs, ys))
            except np.linalg.LinAlgError:
                continue
        if not coefs:
            raise ValueError("no non-degenerate p-point subset found; data may be collinear")
        coefs = np.array(coefs)
        self.coef_ = _weiszfeld(coefs)
        self.std_errors_ = np.full(p, np.nan)
        self.extras_ = {"n_subsets": len(coefs)}
        self.exhaustive_ = exhaustive
        self.n_iter_ = None


# --------------------------------------------------------------------- Siegel

class SiegalRegression(RobustRegressor):
    """Siegel (1982) repeated-medians simple regression: 50% breakdown,
    matches ``scipy.stats.siegelslopes`` exactly. Defined for one predictor
    only.
    """

    def __init__(self, fit_intercept=True, method="hierarchical"):
        super().__init__(fit_intercept)
        if method not in ("hierarchical", "separate"):
            raise ValueError("method must be 'hierarchical' or 'separate'")
        self.method = method

    def _fit(self, Xd, y):
        p = Xd.shape[1] - int(self.fit_intercept)
        if p != 1:
            raise ValueError("Siegel repeated medians is defined for one predictor")
        x = Xd[:, -1]
        n = len(x)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            slope_mat = np.where(dx != 0, dy / dx, np.nan)
        row_med = np.nanmedian(slope_mat, axis=1)
        slope = float(np.median(row_med))

        if self.method == "hierarchical":
            intercept = float(np.median(y - slope * x))
        else:
            with np.errstate(invalid="ignore"):
                inter_mat = y[:, None] - slope_mat * x[:, None]
            row_inter = np.nanmedian(inter_mat, axis=1)
            intercept = float(np.median(row_inter))

        self.coef_ = np.array([intercept, slope]) if self.fit_intercept else np.array([slope])
        self.std_errors_ = np.full(len(self.coef_), np.nan)
        self.extras_ = {}
        self.n_iter_ = None


SiegelRegression = SiegalRegression


# --------------------------------------------------------------------- RANSAC

class RANSACRegression(RobustRegressor):
    """RANSAC (Fischler & Bolles 1981): repeatedly fit a minimal random
    subset, keep the largest consensus set of inliers, refit OLS on it."""

    def __init__(self, min_samples=None, residual_threshold=None, max_trials=100,
                stop_probability=0.99, stop_n_inliers=None, fit_intercept=True,
                random_state=None):
        super().__init__(fit_intercept)
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = int(max_trials)
        self.stop_probability = float(stop_probability)
        self.stop_n_inliers = stop_n_inliers
        self.random_state = random_state

    def _fit(self, Xd, y):
        n, p = Xd.shape
        min_samples = p if self.min_samples is None else int(self.min_samples)
        if min_samples > n:
            raise ValueError("min_samples cannot exceed the number of observations")
        if self.residual_threshold is None:
            # a Theil-Sen pilot fit gives a scale-appropriate residual threshold
            # (MAD of the raw y is contaminated by the regression's own slope
            # spread, not just noise, for data whose predictors have wide range)
            pilot = TheilSenRegression(fit_intercept=False, random_state=self.random_state)
            pilot._fit(Xd, y)
            threshold = max(_mad(y - Xd @ pilot.coef_, normal=False), 1e-12)
        else:
            threshold = float(self.residual_threshold)
        stop_n = n if self.stop_n_inliers is None else int(self.stop_n_inliers)

        rng = _rng(self.random_state)
        best_mask = None
        best_ss = np.inf
        best_n_inliers = -1
        max_trials = self.max_trials
        trial = 0
        while trial < max_trials:
            trial += 1
            rows = rng.choice(n, size=min_samples, replace=False)
            Xs, ys = Xd[rows], y[rows]
            try:
                beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
            except np.linalg.LinAlgError:
                continue
            resid = y - Xd @ beta
            mask = np.abs(resid) <= threshold
            n_in = int(mask.sum())
            ss = float(np.sum(resid[mask] ** 2))
            if n_in > best_n_inliers or (n_in == best_n_inliers and ss < best_ss):
                best_mask, best_ss, best_n_inliers = mask, ss, n_in
                w = best_n_inliers / n
                if 0 < w < 1:
                    denom = math.log(max(1.0 - w ** min_samples, 1e-12))
                    if denom < 0:
                        new_max = math.ceil(math.log(1.0 - self.stop_probability) / denom)
                        max_trials = min(max_trials, max(new_max, trial))
            if best_n_inliers >= stop_n:
                break

        if best_mask is None or best_n_inliers < p:
            raise ValueError("RANSAC failed to find a valid consensus set")

        Xin, yin = Xd[best_mask], y[best_mask]
        beta, *_ = np.linalg.lstsq(Xin, yin, rcond=None)
        resid_in = yin - Xin @ beta
        n_in = len(yin)
        df_resid = max(n_in - p, 1)
        scale = float(np.std(resid_in, ddof=p)) if n_in > p else float(np.std(resid_in))
        try:
            XtX_inv = np.linalg.inv(Xin.T @ Xin)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(Xin.T @ Xin)
        cov = scale ** 2 * XtX_inv
        self.coef_ = beta
        self.std_errors_ = np.sqrt(np.clip(np.diag(cov), 0, None))
        self.inlier_mask_ = best_mask
        self.n_inliers_ = best_n_inliers
        self.n_trials_ = trial
        self.scale_ = scale
        self.df_resid_ = df_resid
        self.n_iter_ = trial
        self.extras_ = {}


# ------------------------------------------------------------------------ LTS

def _lts_h(n, p, alpha):
    h_default = (n + p + 1) // 2
    if alpha == 0.5:
        return h_default
    return int(math.floor(2 * h_default - n + 2 * (n - h_default) * alpha))


def _c_step(Xd, y, h, beta=None, max_iter=50):
    """One or more concentration steps to a local LTS optimum."""
    n, p = Xd.shape
    if beta is None:
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    obj_prev = np.inf
    subset = None
    for _ in range(max_iter):
        resid2 = (y - Xd @ beta) ** 2
        subset = np.argsort(resid2)[:h]
        Xs, ys = Xd[subset], y[subset]
        try:
            beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
        except np.linalg.LinAlgError:
            break
        obj = float(np.sum((ys - Xs @ beta) ** 2))
        if obj >= obj_prev - 1e-12:
            break
        obj_prev = obj
    resid2 = (y - Xd @ beta) ** 2
    subset = np.argsort(resid2)[:h]
    obj = float(np.sum(np.sort(resid2)[:h]))
    return beta, subset, obj


class LTS_Regression(RobustRegressor):
    """Least Trimmed Squares (Rousseeuw 1984): minimizes the sum of the
    smallest ``h`` squared residuals. FAST-LTS (Rousseeuw & Van Driessen
    2006) subsampling when the subset count is too large to enumerate.
    """

    def __init__(self, alpha=0.5, n_trials=500, n_keep=10, n_csteps=2, reweight=True,
                fit_intercept=True, random_state=None):
        super().__init__(fit_intercept)
        self.alpha = float(alpha)
        self.n_trials = int(n_trials)
        self.n_keep = int(n_keep)
        self.n_csteps = int(n_csteps)
        self.reweight = bool(reweight)
        self.random_state = random_state

    def _fit(self, Xd, y):
        n, p = Xd.shape
        h = max(_lts_h(n, p, self.alpha), p)
        rng = _rng(self.random_state)

        from scipy import special as sp
        total = sp.comb(n, h, exact=True) if n >= h else 0
        candidates = []
        if 0 < total <= 5000:
            import itertools
            for rows in itertools.combinations(range(n), h):
                rows = np.array(rows)
                Xs, ys = Xd[rows], y[rows]
                try:
                    beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                beta, subset, obj = _c_step(Xd, y, h, beta=beta, max_iter=100)
                candidates.append((obj, beta, subset))
            exhaustive = True
        else:
            starts = []
            for _ in range(self.n_trials):
                rows = rng.choice(n, size=p, replace=False)
                Xs, ys = Xd[rows], y[rows]
                if np.linalg.matrix_rank(Xs) < p:
                    continue
                beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
                beta, subset, obj = _c_step(Xd, y, h, beta=beta, max_iter=self.n_csteps)
                starts.append((obj, beta, subset))
            starts.sort(key=lambda t: t[0])
            for obj, beta, subset in starts[:self.n_keep]:
                beta, subset, obj = _c_step(Xd, y, h, beta=beta, max_iter=100)
                candidates.append((obj, beta, subset))
            exhaustive = False

        if not candidates:
            raise ValueError("LTS_Regression found no valid subset (data may be too collinear)")
        candidates.sort(key=lambda t: t[0])
        objective, raw_coef, best_subset = candidates[0]

        a = h / n
        q = float(_norm_ppf(0.5 * (1.0 + a)))
        phi_q = float(np.exp(-0.5 * q * q) / math.sqrt(2 * math.pi))
        denom = 1.0 - 2.0 * q * phi_q / a if a > 0 else 1.0
        c_alpha = 1.0 / math.sqrt(denom) if denom > 0 else 1.0
        s_raw = c_alpha * math.sqrt(objective / h)

        self.raw_coef_ = raw_coef
        self.best_subset_ = np.sort(best_subset)
        self.objective_ = objective
        self.exhaustive_ = exhaustive

        if self.reweight and s_raw > 0:
            resid = y - Xd @ raw_coef
            w = (np.abs(resid) / s_raw) <= 2.5
            n_w = int(np.sum(w))
            if n_w < p + 1:
                w = np.ones(n, dtype=bool)
                n_w = n
            Xw, yw = Xd[w], y[w]
            sw = np.sqrt(w.astype(float))
            beta, *_ = np.linalg.lstsq(Xd * sw[:, None], y * sw, rcond=None)
            resid_w = yw - Xw @ beta
            c2 = 2.5
            phi_c2 = float(np.exp(-0.5 * c2 * c2) / math.sqrt(2 * math.pi))
            Phi_c2 = float(_norm_cdf(c2))
            denom2 = 1.0 - 2.0 * c2 * phi_c2 / (2.0 * Phi_c2 - 1.0)
            c_w = 1.0 / math.sqrt(denom2) if denom2 > 0 else 1.0
            scale = c_w * math.sqrt(np.sum(resid_w ** 2) / max(n_w - p, 1))
            try:
                XtX_inv = np.linalg.inv((Xd * sw[:, None]).T @ (Xd * sw[:, None]))
            except np.linalg.LinAlgError:
                XtX_inv = np.linalg.pinv((Xd * sw[:, None]).T @ (Xd * sw[:, None]))
            self.coef_ = beta
            self.scale_ = scale
            self.std_errors_ = np.sqrt(np.clip(scale ** 2 * np.diag(XtX_inv), 0, None))
            self.weights_ = w.astype(float)
            self.df_resid_ = max(n_w - p, 1)
        else:
            self.coef_ = raw_coef
            self.scale_ = s_raw
            self.std_errors_ = np.full(p, np.nan)
            self.weights_ = None
            self.df_resid_ = max(h - p, 1)
        self.n_iter_ = None
        self.extras_ = {}


# --------------------------------------------------------------------- MM/S

class MMRegression(RobustRegressor):
    """MM-estimator (Yohai 1987): an S-estimator (50% breakdown by default)
    for the scale, followed by one M-step at fixed scale for high (default
    95%) Gaussian efficiency."""

    def __init__(self, psi="tukey", c=4.685061, s_c=1.547645, breakdown=0.5,
                n_trials=500, n_keep=5, n_isteps=2, tol=1e-8, max_iter=100,
                fit_intercept=True, random_state=None):
        super().__init__(fit_intercept)
        self.psi = psi
        self.c = c
        self.s_c = s_c
        self.breakdown = float(breakdown)
        self.n_trials = int(n_trials)
        self.n_keep = int(n_keep)
        self.n_isteps = int(n_isteps)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.random_state = random_state

    def _fit(self, Xd, y):
        n, p = Xd.shape
        rng = _rng(self.random_state)

        starts = []
        for _ in range(self.n_trials):
            rows = rng.choice(n, size=p, replace=False)
            Xs, ys = Xd[rows], y[rows]
            if np.linalg.matrix_rank(Xs) < p:
                continue
            beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
            beta, scale = self._i_steps(Xd, y, beta, self.n_isteps)
            starts.append((scale, beta))
        if not starts:
            raise ValueError("MMRegression found no valid starting subset")
        starts.sort(key=lambda t: t[0])
        candidates = starts[:self.n_keep]

        best_beta, best_scale = None, np.inf
        for _, beta0 in candidates:
            beta, scale = self._i_steps(Xd, y, beta0, 100)
            if scale < best_scale:
                best_beta, best_scale = beta, scale
        s_coef, s_scale = best_beta, best_scale
        self.s_coef_ = s_coef

        beta = s_coef.copy()
        n_iter = 0
        for _ in range(self.max_iter):
            n_iter += 1
            u = (y - Xd @ beta) / s_scale
            w = _psi_weight(self.psi, u, self.c)
            sw_sqrt = np.sqrt(np.clip(w, 0, None))
            try:
                beta_new, *_ = np.linalg.lstsq(Xd * sw_sqrt[:, None], y * sw_sqrt, rcond=None)
            except np.linalg.LinAlgError:
                break
            if np.max(np.abs(beta_new - beta)) < self.tol * max(1.0, np.max(np.abs(beta))):
                beta = beta_new
                break
            beta = beta_new

        self.coef_ = beta
        self.scale_ = s_scale
        u = (y - Xd @ beta) / s_scale
        weights_final = _psi_weight(self.psi, u, self.c)
        psi_vals = _psi_value(self.psi, u, self.c)
        psi_prime_vals = _psi_prime_of(self.psi, u, self.c)
        m = float(np.mean(psi_prime_vals))
        var_psi_prime = float(np.var(psi_prime_vals, ddof=0))
        k = 1.0 + (p / n) * var_psi_prime / m ** 2 if m != 0 else float("nan")
        ss_psi = float(np.sum(psi_vals ** 2))
        df_resid = max(n - p, 1)
        try:
            XtX_inv = np.linalg.inv(Xd.T @ Xd)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(Xd.T @ Xd)
        cov = (k ** 2 * s_scale ** 2 * (ss_psi / df_resid) / m ** 2 * XtX_inv
              if m != 0 else np.full((p, p), np.nan))
        self.std_errors_ = np.sqrt(np.clip(np.diag(cov), 0, None)) if np.all(np.isfinite(cov)) else np.full(p, np.nan)
        self.weights_ = weights_final
        self.n_iter_ = n_iter
        self.df_resid_ = df_resid
        self.extras_ = {}

    def _i_steps(self, Xd, y, beta, n_steps):
        resid = y - Xd @ beta
        scale = _m_scale(resid, c=self.s_c, b=self.breakdown)
        for _ in range(n_steps):
            if scale <= 0:
                break
            u = resid / scale
            w = _tukey_biweight_s_weight(u, self.s_c)
            sw_sqrt = np.sqrt(np.clip(w, 0, None))
            try:
                beta, *_ = np.linalg.lstsq(Xd * sw_sqrt[:, None], y * sw_sqrt, rcond=None)
            except np.linalg.LinAlgError:
                break
            resid = y - Xd @ beta
            scale = _m_scale(resid, c=self.s_c, b=self.breakdown, s0=scale)
        return beta, scale


def _tukey_biweight_s_weight(u, c):
    """Weight function for the S-scale's IRWLS step: rho_norm'(u)/u."""
    from stochpylib.robust_statistics._common import _tukey_weight
    return _tukey_weight(u, c)


# ------------------------------------------------------------------- Huber

class HuberRegression(RobustRegressor):
    """Huber's M-estimator via IRLS. Matches ``statsmodels.RLM`` with
    ``M=HuberT(t=epsilon)``, ``scale_est="mad"``, ``cov="H1"``."""

    def __init__(self, epsilon=1.345, scale="mad", cov="H1", tol=1e-8, max_iter=50,
                fit_intercept=True):
        super().__init__(fit_intercept)
        self.epsilon = float(epsilon)
        self.scale = scale
        self.cov = cov
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def _fit(self, Xd, y):
        n, p = Xd.shape
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        resid = y - Xd @ beta
        update_scale = self.scale == "mad"
        if self.scale == "mad":
            s = _mad(resid, center=0.0)
        elif self.scale == "huber":
            from stochpylib.robust_statistics._common import _huber_proposal2
            _, s = _huber_proposal2(resid)
        else:
            s = float(self.scale)
        s = s if s > 0 else 1.0

        n_iter = 0
        for _ in range(self.max_iter):
            n_iter += 1
            u = resid / s
            w = _psi_weight("huber", u, self.epsilon)
            sw_sqrt = np.sqrt(np.clip(w, 0, None))
            try:
                beta_new, *_ = np.linalg.lstsq(Xd * sw_sqrt[:, None], y * sw_sqrt, rcond=None)
            except np.linalg.LinAlgError:
                break
            resid_new = y - Xd @ beta_new
            if update_scale:
                s_new = _mad(resid_new, center=0.0)
                s = s_new if s_new > 0 else s
            converged = np.max(np.abs(beta_new - beta)) < self.tol * max(1.0, np.max(np.abs(beta)))
            beta, resid = beta_new, resid_new
            if converged:
                break

        self.coef_ = beta
        self.scale_ = s
        u = resid / s
        weights_final = _psi_weight("huber", u, self.epsilon)
        psi_vals = _psi_value("huber", u, self.epsilon)
        psi_prime_vals = _psi_prime_of("huber", u, self.epsilon)
        m = float(np.mean(psi_prime_vals))
        var_psi_prime = float(np.var(psi_prime_vals, ddof=0))
        df_resid = max(n - p, 1)
        k = 1.0 + (p / n) * var_psi_prime / m ** 2 if m != 0 else float("nan")
        ss_psi = float(np.sum(psi_vals ** 2))
        try:
            XtX_inv = np.linalg.inv(Xd.T @ Xd)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(Xd.T @ Xd)

        if self.cov == "H1" or m == 0:
            cov = (k ** 2 * s ** 2 * (ss_psi / df_resid) / m ** 2 * XtX_inv
                  if m != 0 else np.full((p, p), np.nan))
        else:
            W = (Xd * psi_prime_vals[:, None]).T @ Xd
            try:
                W_inv = np.linalg.inv(W)
            except np.linalg.LinAlgError:
                W_inv = np.linalg.pinv(W)
            if self.cov == "H2":
                cov = k * (ss_psi / df_resid) * s ** 2 / m * W_inv
            elif self.cov == "H3":
                cov = (1.0 / k) * (ss_psi / df_resid) * s ** 2 * W_inv @ (Xd.T @ Xd) @ W_inv
            else:
                raise ValueError("cov must be 'H1', 'H2', or 'H3'")

        self.std_errors_ = np.sqrt(np.clip(np.diag(cov), 0, None)) if np.all(np.isfinite(cov)) else np.full(p, np.nan)
        self.weights_ = weights_final
        self.n_iter_ = n_iter
        self.df_resid_ = df_resid
        self.extras_ = {}
