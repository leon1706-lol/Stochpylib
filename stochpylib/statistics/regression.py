"""Linear, generalized-linear, and penalized regression.

Every function returns a :class:`~stochpylib.statistics.RegressionResult`.
``X`` is ``(n, p)``; ``fit_intercept=True`` (default) prepends an intercept column, so
``coef_[0]`` is the intercept.
"""

import math

import numpy as np
from scipy import optimize, special

from stochpylib.statistics._common import _as_1d, _design, _f_sf, _t_sf
from stochpylib.statistics._result import RegressionResult

__all__ = [
    "elastic_net", "glm", "lasso", "linear_regression", "logistic_regression",
    "poisson_regression", "quantile_regression", "ridge",
]


def _finalize_ols(X, y, beta, cov, method, fit_intercept, weights=None):
    n, p_full = X.shape
    fitted = X @ beta
    resid = y - fitted
    df_model = p_full - 1
    df_resid = n - p_full
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = np.array([2.0 * float(_t_sf(abs(t), df_resid)) for t in tvals])

    w = weights if weights is not None else np.ones(n)
    y_bar = np.average(y, weights=w)
    ss_tot = np.sum(w * (y - y_bar) ** 2)
    ss_res = np.sum(w * resid ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj_r2 = 1.0 - (1.0 - r2) * (n - int(fit_intercept)) / df_resid if df_resid > 0 else float("nan")

    sigma2 = ss_res / n
    loglik = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1.0) if sigma2 > 0 else float("-inf")
    # matches statsmodels.OLS: aic/bic count only the p_full mean-function parameters,
    # not sigma^2 as an extra parameter (a documented statsmodels convention, not the
    # textbook -2*llf + 2*(k+1) form some other packages use).
    aic = -2.0 * loglik + 2.0 * p_full
    bic = -2.0 * loglik + math.log(n) * p_full

    if fit_intercept and df_model > 0:
        f_stat = (ss_tot - ss_res) / df_model / (ss_res / df_resid)
        f_pvalue = float(_f_sf(f_stat, df_model, df_resid))
    else:
        f_stat = f_pvalue = None

    res = RegressionResult(
        coef_=beta, std_errors_=se, pvalues_=pvals, tvalues_=tvals, method=method,
        fit_intercept=fit_intercept, fitted_=fitted, resid_=resid, df_model_=float(df_model),
        df_resid_=float(df_resid), loglik_=loglik, aic_=aic, bic_=bic, r2_=r2, adj_r2_=adj_r2,
        f_stat_=f_stat, f_pvalue_=f_pvalue, cov_params_=cov,
    )
    res.extras["_predict"] = lambda Xnew: _design(Xnew, fit_intercept) @ beta
    return res


def linear_regression(X, y, weights=None, cov_type="nonrobust", fit_intercept=True):
    """Ordinary (or, with ``weights``, weighted) least squares.

    ``cov_type``: ``"nonrobust"`` (homoskedastic), or heteroskedasticity-consistent
    ``"HC0"``/``"HC1"``/``"HC2"``/``"HC3"`` (White sandwich estimators).
    """
    Xd = _design(X, fit_intercept)
    y = _as_1d(y, "y")
    n, p = Xd.shape

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        sw = np.sqrt(w)
        Xw, yw = Xd * sw[:, None], y * sw
    else:
        w = np.ones(n)
        Xw, yw = Xd, y

    beta, _, rank, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    resid = y - Xd @ beta

    if cov_type == "nonrobust":
        df_resid = n - p
        sigma2 = np.sum(w * resid ** 2) / df_resid
        cov = sigma2 * XtX_inv
    else:
        u2 = w * resid ** 2
        if cov_type == "HC0":
            scale = u2
        elif cov_type == "HC1":
            scale = u2 * n / (n - p)
        elif cov_type in ("HC2", "HC3"):
            H = np.einsum("ij,jk,ik->i", Xw, XtX_inv, Xw)
            H = np.clip(H, 0, 1 - 1e-12)
            scale = u2 / (1.0 - H) if cov_type == "HC2" else u2 / (1.0 - H) ** 2
        else:
            raise ValueError("cov_type must be 'nonrobust', 'HC0', 'HC1', 'HC2', or 'HC3'")
        meat = (Xw * scale[:, None]).T @ Xw
        cov = XtX_inv @ meat @ XtX_inv

    return _finalize_ols(Xd, y, beta, cov, f"OLS ({cov_type})", fit_intercept, weights=w)


# --------------------------------------------------------------------------- GLM

_LINKS = {
    "identity": (lambda mu: mu, lambda eta: eta, lambda mu: np.ones_like(mu)),
    "log": (lambda mu: np.log(mu), lambda eta: np.exp(eta), lambda mu: 1.0 / mu),
    "logit": (lambda mu: np.log(mu / (1 - mu)), lambda eta: special.expit(eta),
              lambda mu: 1.0 / (mu * (1 - mu))),
    "probit": (lambda mu: special.ndtri(mu), lambda eta: special.ndtr(eta),
               lambda mu: 1.0 / np.clip(np.exp(-0.5 * special.ndtri(mu) ** 2) / np.sqrt(2 * np.pi), 1e-300, None)),
    "cloglog": (lambda mu: np.log(-np.log(1 - mu)), lambda eta: 1 - np.exp(-np.exp(eta)),
                lambda mu: -1.0 / ((1 - mu) * np.log(1 - mu))),
    "inverse": (lambda mu: 1.0 / mu, lambda eta: 1.0 / eta, lambda mu: -1.0 / mu ** 2),
    "sqrt": (lambda mu: np.sqrt(mu), lambda eta: eta ** 2, lambda mu: 0.5 / np.sqrt(mu)),
    "inverse_squared": (lambda mu: 1.0 / mu ** 2, lambda eta: 1.0 / np.sqrt(eta),
                         lambda mu: -2.0 / mu ** 3),
}

_DEFAULT_LINK = {
    "gaussian": "identity", "binomial": "logit", "poisson": "log",
    "gamma": "inverse", "inverse_gaussian": "inverse_squared", "negative_binomial": "log",
}


def _variance_fn(family, alpha=1.0):
    if family == "gaussian":
        return lambda mu: np.ones_like(mu)
    if family == "binomial":
        return lambda mu: mu * (1 - mu)
    if family == "poisson":
        return lambda mu: mu
    if family == "gamma":
        return lambda mu: mu ** 2
    if family == "inverse_gaussian":
        return lambda mu: mu ** 3
    if family == "negative_binomial":
        return lambda mu: mu + alpha * mu ** 2
    raise ValueError(f"unknown family: {family!r}")


_LINK_BOUNDS = {
    "identity": (-np.inf, np.inf), "log": (1e-10, np.inf), "logit": (1e-10, 1 - 1e-10),
    "probit": (1e-10, 1 - 1e-10), "cloglog": (1e-10, 1 - 1e-10), "inverse": (1e-10, np.inf),
    "sqrt": (0.0, np.inf), "inverse_squared": (1e-10, np.inf),
}


def _mu_bounds(family, link):
    """Intersection of the family's natural response range and the link's domain
    (e.g. a log link demands mu > 0 even for the otherwise-unrestricted Gaussian
    family)."""
    flo, fhi = (1e-10, 1 - 1e-10) if family == "binomial" else \
        ((-np.inf, np.inf) if family == "gaussian" else (1e-10, np.inf))
    llo, lhi = _LINK_BOUNDS[link]
    return max(flo, llo), min(fhi, lhi)


def _deviance(family, y, mu, alpha=1.0):
    eps = 1e-10
    if family == "gaussian":
        return np.sum((y - mu) ** 2)
    if family == "binomial":
        t1 = np.where(y > 0, y * np.log(np.clip(y, eps, None) / mu), 0.0)
        t2 = np.where(y < 1, (1 - y) * np.log(np.clip(1 - y, eps, None) / (1 - mu)), 0.0)
        return 2.0 * np.sum(t1 + t2)
    if family == "poisson":
        t1 = np.where(y > 0, y * np.log(np.clip(y, eps, None) / mu), 0.0)
        return 2.0 * np.sum(t1 - (y - mu))
    if family == "gamma":
        return 2.0 * np.sum(-np.log(np.clip(y, eps, None) / mu) + (y - mu) / mu)
    if family == "inverse_gaussian":
        return np.sum((y - mu) ** 2 / (mu ** 2 * y))
    if family == "negative_binomial":
        t1 = np.where(y > 0, y * np.log(np.clip(y, eps, None) / mu), 0.0)
        return 2.0 * np.sum(t1 - (y + 1.0 / alpha) * np.log((1 + alpha * y) / (1 + alpha * mu)))
    raise ValueError(family)


def _loglik(family, y, mu, scale, alpha=1.0):
    if family == "gaussian":
        return float(np.sum(-0.5 * np.log(2 * np.pi * scale) - (y - mu) ** 2 / (2 * scale)))
    if family == "binomial":
        return float(np.sum(y * np.log(np.clip(mu, 1e-300, None)) + (1 - y) * np.log(np.clip(1 - mu, 1e-300, None))))
    if family == "poisson":
        return float(np.sum(-mu + y * np.log(np.clip(mu, 1e-300, None)) - special.gammaln(y + 1)))
    if family == "gamma":
        shape = 1.0 / scale
        return float(np.sum(shape * np.log(shape / mu) - special.gammaln(shape) + (shape - 1) * np.log(y) - shape * y / mu))
    if family == "inverse_gaussian":
        return float(np.sum(-0.5 * np.log(2 * np.pi * scale * y ** 3) - (y - mu) ** 2 / (2 * scale * mu ** 2 * y)))
    if family == "negative_binomial":
        r = 1.0 / alpha
        return float(np.sum(special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1)
                             + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))))
    raise ValueError(family)


def glm(X, y, family="gaussian", link=None, weights=None, offset=None, alpha=1.0,
        fit_intercept=True, max_iter=100, tol=1e-10):
    """Generalized linear model via iteratively reweighted least squares (IRLS).

    ``family``: ``"gaussian"``, ``"binomial"``, ``"poisson"``, ``"gamma"``,
    ``"inverse_gaussian"``, or ``"negative_binomial"`` (fixed dispersion ``alpha``).
    ``link`` defaults to each family's canonical link. Matches ``statsmodels.GLM``
    conventions: dispersion (``dispersion_``) is fixed at 1 for binomial/poisson/
    negative_binomial and estimated as the Pearson chi-squared/df_resid otherwise;
    ``loglik_``/``aic_``/``bic_`` are evaluated at that dispersion.
    """
    family = family.lower()
    link = link or _DEFAULT_LINK[family]
    link_fn, inv_link, deta_dmu_fn = _LINKS[link]  # deta_dmu_fn(mu) = d(eta)/d(mu)
    var_fn = _variance_fn(family, alpha)
    lo, hi = _mu_bounds(family, link)

    Xd = _design(X, fit_intercept)
    y = _as_1d(y, "y")
    n, p = Xd.shape
    w_prior = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)

    y_clipped = np.clip(y, lo, hi)
    mu = np.clip((y_clipped + np.mean(y_clipped)) / 2.0, lo, hi)
    eta = link_fn(mu) - off
    beta = np.zeros(p)

    for _ in range(max_iter):
        mu = np.clip(inv_link(eta + off), lo, hi)
        deta_dmu = deta_dmu_fn(mu)
        var = var_fn(mu)
        W = w_prior / (deta_dmu ** 2 * var)
        z = eta + (y - mu) * deta_dmu

        sw = np.sqrt(np.clip(W, 0, None))
        Xw, zw = Xd * sw[:, None], z * sw
        beta_new, *_ = np.linalg.lstsq(Xw, zw, rcond=None)
        eta_new = Xd @ beta_new
        if np.max(np.abs(eta_new - eta)) < tol * (1.0 + np.max(np.abs(eta))):
            beta, eta = beta_new, eta_new
            break
        beta, eta = beta_new, eta_new

    mu = np.clip(inv_link(eta + off), lo, hi)
    deta_dmu = deta_dmu_fn(mu)
    var = var_fn(mu)
    W = w_prior / (deta_dmu ** 2 * var)
    sw = np.sqrt(np.clip(W, 0, None))
    Xw = Xd * sw[:, None]
    XtWX_inv = np.linalg.pinv(Xw.T @ Xw)

    df_resid = n - p
    if family in ("binomial", "poisson", "negative_binomial"):
        dispersion = 1.0
    else:
        pearson_chi2 = np.sum(w_prior * (y - mu) ** 2 / var)
        dispersion = pearson_chi2 / df_resid

    cov = dispersion * XtWX_inv
    se = np.sqrt(np.diag(cov))
    zvals = beta / se
    pvals = np.array([2.0 * float(special.ndtr(-abs(z))) for z in zvals])

    deviance = _deviance(family, y, mu, alpha)
    mu_null = np.clip(np.average(y, weights=w_prior) * np.ones(n), lo, hi)
    null_deviance = _deviance(family, y, mu_null, alpha)
    # statsmodels quirk: for Gaussian+identity specifically, llf uses the concentrated
    # (profile, n-denominator) variance SSR/n, not the SE-purpose Pearson dispersion
    # SSR/df_resid used everywhere else (including Gaussian with any other link).
    llf_scale = (np.sum(w_prior * (y - mu) ** 2) / n) if (family == "gaussian" and link == "identity") else dispersion
    loglik = _loglik(family, y, mu, llf_scale, alpha)
    # matches statsmodels.GLM: aic/bic count only the p mean-function parameters, even
    # when dispersion is separately estimated (gaussian/gamma/inverse_gaussian).
    aic = -2.0 * loglik + 2.0 * p
    bic_llf = -2.0 * loglik + math.log(n) * p
    pseudo_r2 = 1.0 - deviance / null_deviance if null_deviance > 0 else float("nan")

    res = RegressionResult(
        coef_=beta, std_errors_=se, pvalues_=pvals, zvalues_=zvals, method=f"GLM ({family}/{link})",
        fit_intercept=fit_intercept, fitted_=mu, resid_=y - mu, df_model_=float(p - 1),
        df_resid_=float(df_resid), loglik_=loglik, aic_=aic, bic_=bic_llf, cov_params_=cov,
        deviance_=deviance, null_deviance_=null_deviance, pseudo_r2_=pseudo_r2,
        dispersion_=dispersion, family=family, link=link,
    )
    res.extras["_predict"] = lambda Xnew: inv_link(_design(Xnew, fit_intercept) @ beta)
    if family == "binomial":
        res.extras["_predict_proba"] = res.extras["_predict"]
    return res


def logistic_regression(X, y, fit_intercept=True, max_iter=100, tol=1e-10):
    """Binomial GLM with the logit link (McFadden pseudo-R^2 in ``pseudo_r2_``)."""
    return glm(X, y, family="binomial", link="logit", fit_intercept=fit_intercept,
               max_iter=max_iter, tol=tol)


def poisson_regression(X, y, offset=None, exposure=None, fit_intercept=True, max_iter=100, tol=1e-10):
    """Poisson GLM with the log link. ``exposure`` is folded into the offset as
    ``log(exposure)`` (mutually exclusive with ``offset``)."""
    if exposure is not None:
        if offset is not None:
            raise ValueError("offset and exposure cannot both be given")
        offset = np.log(np.asarray(exposure, dtype=float))
    return glm(X, y, family="poisson", link="log", offset=offset, fit_intercept=fit_intercept,
               max_iter=max_iter, tol=tol)


# --------------------------------------------------------------------------- penalized

def ridge(X, y, alpha, fit_intercept=True):
    """Ridge regression (L2), closed form via SVD; the intercept (if any) is left
    unpenalized by centering ``X``/``y`` first. ``alpha="gcv"`` selects the penalty
    minimizing generalized cross-validation error over a log-spaced grid."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = _as_1d(y, "y")
    n, p = X.shape

    if fit_intercept:
        x_mean = X.mean(axis=0)
        y_mean = float(np.mean(y))
        Xc, yc = X - x_mean, y - y_mean
    else:
        x_mean, y_mean = np.zeros(p), 0.0
        Xc, yc = X, y

    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    def _solve(a):
        d = s / (s ** 2 + a)
        beta_c = Vt.T @ (d * (U.T @ yc))
        return beta_c

    if alpha == "gcv":
        grid = np.logspace(-4, 4, 100)
        best_a, best_gcv = grid[0], np.inf
        for a in grid:
            d = s ** 2 / (s ** 2 + a)
            fitted = U @ (d * (U.T @ yc))
            rss = np.sum((yc - fitted) ** 2)
            df_eff = np.sum(d)
            gcv = (rss / n) / (1.0 - df_eff / n) ** 2
            if gcv < best_gcv:
                best_gcv, best_a = gcv, a
        alpha = best_a

    beta_c = _solve(alpha)
    intercept = y_mean - x_mean @ beta_c if fit_intercept else 0.0
    beta = np.concatenate([[intercept], beta_c]) if fit_intercept else beta_c

    df_eff = float(np.sum(s ** 2 / (s ** 2 + alpha)))
    Xd = _design(X, fit_intercept)
    fitted = Xd @ beta
    resid = y - fitted
    df_resid = n - df_eff - (1 if fit_intercept else 0)
    sigma2 = np.sum(resid ** 2) / max(df_resid, 1e-8)
    d2 = s / (s ** 2 + alpha)
    cov_c = sigma2 * (Vt.T * d2 ** 2) @ Vt
    se_c = np.sqrt(np.clip(np.diag(cov_c), 0, None))
    se = np.concatenate([[float("nan")], se_c]) if fit_intercept else se_c
    tvals = beta / se
    pvals = np.array([2.0 * float(_t_sf(abs(t), max(df_resid, 1.0))) if np.isfinite(t) else float("nan") for t in tvals])

    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else float("nan")

    res = RegressionResult(
        coef_=beta, std_errors_=se, pvalues_=pvals, tvalues_=tvals, method="ridge",
        fit_intercept=fit_intercept, fitted_=fitted, resid_=resid,
        df_resid_=float(df_resid), r2_=r2, extras={"alpha": alpha, "df_effective": df_eff},
    )
    res.extras["_predict"] = lambda Xnew: _design(Xnew, fit_intercept) @ beta
    return res


def _coordinate_descent(X, y, alpha, l1_ratio, max_iter, tol):
    """Cyclic coordinate descent for elastic net: (1/2n)||y-Xb||^2 + alpha*(l1_ratio*|b|_1
    + 0.5*(1-l1_ratio)*|b|_2^2). X must already be centered/scaled (no intercept column)."""
    n, p = X.shape
    beta = np.zeros(p)
    col_norm2 = np.sum(X ** 2, axis=0) / n
    resid = y.copy()
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            if col_norm2[j] == 0:
                continue
            resid += X[:, j] * beta[j]
            rho = X[:, j] @ resid / n
            denom = col_norm2[j] + alpha * (1 - l1_ratio)
            thresh = alpha * l1_ratio
            beta[j] = np.sign(rho) * max(abs(rho) - thresh, 0.0) / denom
            resid -= X[:, j] * beta[j]
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta


def _penalized_fit(X, y, alpha, l1_ratio, fit_intercept, max_iter, tol, method):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = _as_1d(y, "y")
    n, p = X.shape

    if fit_intercept:
        x_mean, y_mean = X.mean(axis=0), float(np.mean(y))
        Xc, yc = X - x_mean, y - y_mean
    else:
        x_mean, y_mean = np.zeros(p), 0.0
        Xc, yc = X, y

    alphas = np.atleast_1d(alpha)
    results = []
    beta_c = np.zeros(p)
    for a in alphas:
        beta_c = _coordinate_descent(Xc, yc, float(a), l1_ratio, max_iter, tol) if a > 0 else \
            np.linalg.lstsq(Xc, yc, rcond=None)[0]
        results.append(beta_c.copy())

    beta_c = results[-1]
    intercept = y_mean - x_mean @ beta_c if fit_intercept else 0.0
    beta = np.concatenate([[intercept], beta_c]) if fit_intercept else beta_c

    Xd = _design(X, fit_intercept)
    fitted = Xd @ beta
    resid = y - fitted
    df_resid = n - np.sum(np.abs(beta_c) > 1e-10) - (1 if fit_intercept else 0)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else float("nan")

    se = np.full(len(beta), float("nan"))
    pvals = np.full(len(beta), float("nan"))
    res = RegressionResult(
        coef_=beta, std_errors_=se, pvalues_=pvals, method=method, fit_intercept=fit_intercept,
        fitted_=fitted, resid_=resid, df_resid_=float(df_resid), r2_=r2,
        extras={"alpha": alpha, "path": np.array(results) if len(results) > 1 else None},
    )
    res.extras["_predict"] = lambda Xnew: _design(Xnew, fit_intercept) @ beta
    return res


def lasso(X, y, alpha, fit_intercept=True, max_iter=1000, tol=1e-6):
    """Lasso (L1-penalized) regression via cyclic coordinate descent, objective
    ``(1/2n)||y - Xb||^2 + alpha*|b|_1`` (the scikit-learn convention). ``alpha`` may
    be an array for a warm-started regularization path (``extras['path']``)."""
    return _penalized_fit(X, y, alpha, l1_ratio=1.0, fit_intercept=fit_intercept,
                           max_iter=max_iter, tol=tol, method="lasso")


def elastic_net(X, y, alpha, l1_ratio=0.5, fit_intercept=True, max_iter=1000, tol=1e-6):
    """Elastic-net regression: ``l1_ratio=1`` is lasso, ``l1_ratio=0`` is ridge (in
    the sklearn parameterization, not identical to :func:`ridge`'s closed form)."""
    return _penalized_fit(X, y, alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept,
                           max_iter=max_iter, tol=tol, method="elastic net")


# --------------------------------------------------------------------------- quantile

def quantile_regression(X, y, q=0.5, fit_intercept=True):
    """Quantile regression at quantile ``q`` via linear programming (exact check-loss
    minimum). Standard errors use the Koenker-Bassett kernel sandwich with a
    Hall-Sheather bandwidth (matching ``statsmodels.QuantReg``'s default)."""
    Xd = _design(X, fit_intercept)
    y = _as_1d(y, "y")
    n, p = Xd.shape

    # min sum(q*u+ + (1-q)*u-) s.t. X b + u+ - u- = y, u+,u- >= 0, b free
    c = np.concatenate([np.zeros(2 * p), q * np.ones(n), (1 - q) * np.ones(n)])
    A_eq = np.hstack([Xd, -Xd, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * (2 * p) + [(0, None)] * (2 * n)
    res = optimize.linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    beta = res.x[:p] - res.x[p:2 * p]

    fitted = Xd @ beta
    resid = y - fitted

    z_q = float(special.ndtri(q))
    h_hs = n ** (-1.0 / 3.0) * special.ndtri(0.975) ** (2.0 / 3.0) * \
        (1.5 * (np.exp(-0.5 * z_q ** 2) / math.sqrt(2 * math.pi)) ** 2 / (2 * z_q ** 2 + 1)) ** (1.0 / 3.0)
    sd_y = float(np.std(y, ddof=0))
    q1, q3 = np.quantile(resid, [0.25, 0.75])
    iqr_e = q3 - q1
    h = min(sd_y, iqr_e / 1.34) * (special.ndtri(min(q + h_hs, 1 - 1e-6)) - special.ndtri(max(q - h_hs, 1e-6)))
    u = resid / h
    kernel = np.where(np.abs(u) < 1.0, 0.75 * (1.0 - u ** 2), 0.0)  # Epanechnikov
    f0 = float(np.sum(kernel) / (n * h))
    f0 = max(f0, 1e-8)
    d = np.where(resid > 0, (q / f0) ** 2, ((1 - q) / f0) ** 2)
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    meat = (Xd * d[:, None]).T @ Xd
    cov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    df_resid = n - p
    tvals = beta / se
    pvals = np.array([2.0 * float(_t_sf(abs(t), df_resid)) for t in tvals])

    pseudo_r2 = 1.0 - np.sum(np.where(resid >= 0, q * resid, (q - 1) * resid)) / \
        np.sum(np.where(y - np.quantile(y, q) >= 0, q * (y - np.quantile(y, q)), (q - 1) * (y - np.quantile(y, q))))

    res_obj = RegressionResult(
        coef_=beta, std_errors_=se, pvalues_=pvals, tvalues_=tvals, method=f"quantile regression (q={q})",
        fit_intercept=fit_intercept, fitted_=fitted, resid_=resid, df_resid_=float(df_resid),
        pseudo_r2_=float(pseudo_r2), cov_params_=cov, extras={"q": q, "bandwidth": h},
    )
    res_obj.extras["_predict"] = lambda Xnew: _design(Xnew, fit_intercept) @ beta
    return res_obj
