"""Model-selection criteria: classical (AIC/BIC/TIC) and Bayesian (DIC/WAIC/PSIS-LOO,
Bayes factors)."""

import numpy as np

from stochpylib.bayesian._common import _as_log_density, _logsumexp, _psis
from stochpylib.bayesian._result import ICResult

__all__ = ["bayes_factor", "BIC", "AIC", "DIC", "WAIC", "LOO_CV", "TICfit"]


def _resolve_loglik_k_n(loglik, k, n):
    """Extract (loglik value, k, n) from a float, a stochpylib fitted model (``.loglik_``),
    a statsmodels results wrapper (``.llf``), or a callable already evaluated by the
    caller."""
    if hasattr(loglik, "loglik_"):
        ll = float(loglik.loglik_)
        if k is None:
            k = int(len(np.atleast_1d(loglik.coef_))) if hasattr(loglik, "coef_") else None
        if n is None:
            n = getattr(loglik, "n_obs_", None)
        return ll, k, n
    if hasattr(loglik, "llf"):  # statsmodels-style results wrapper
        ll = float(loglik.llf)
        if k is None and hasattr(loglik, "params"):
            k = int(len(np.atleast_1d(loglik.params)))
        if n is None:
            n = getattr(loglik, "nobs", None)
        return ll, k, n
    return float(loglik), k, n


def AIC(loglik, k=None, n=None):
    """Akaike information criterion: ``-2*loglik + 2*k``. With ``n``, also reports the
    small-sample-corrected AICc in ``extras``."""
    ll, k, n = _resolve_loglik_k_n(loglik, k, n)
    if k is None:
        raise ValueError("k (number of parameters) is required")
    value = -2.0 * ll + 2.0 * k
    extras = {}
    if n is not None and n - k - 1 > 0:
        extras["aicc"] = value + (2.0 * k * (k + 1)) / (n - k - 1)
    return ICResult("AIC", value, p_eff=float(k), extras=extras)


def BIC(loglik, k=None, n=None):
    """Bayesian information criterion: ``-2*loglik + k*log(n)``."""
    ll, k, n = _resolve_loglik_k_n(loglik, k, n)
    if k is None:
        raise ValueError("k (number of parameters) is required")
    if n is None:
        raise ValueError("n (number of observations) is required")
    value = -2.0 * ll + k * np.log(n)
    return ICResult("BIC", value, p_eff=float(k))


def DIC(loglik, samples, data=None):
    """Deviance information criterion: ``D_bar + p_D`` with ``p_D = D_bar - D(theta_bar)``.

    ``loglik``: a :class:`~stochpylib.bayesian.core.Likelihood` (uses ``.loglik``), or a
    callable ``f(theta) -> float``.
    """
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    f = loglik.loglik if hasattr(loglik, "loglik") else loglik
    D = np.array([-2.0 * f(th) for th in samples])
    D_bar = float(D.mean())
    theta_bar = samples.mean(axis=0)
    D_hat = float(-2.0 * f(theta_bar))
    p_D = D_bar - D_hat
    p_V = 0.5 * float(D.var(ddof=1))
    return ICResult("DIC", D_bar + p_D, p_eff=p_D, extras={"D_bar": D_bar, "D_hat": D_hat, "p_V": p_V})


def _pointwise_matrix(log_lik, samples):
    """Normalize (log_lik, samples) into an (S, n) pointwise log-likelihood matrix."""
    if hasattr(log_lik, "pointwise") and samples is not None:
        samples = np.atleast_2d(np.asarray(samples, dtype=float))
        return np.array([log_lik.pointwise(th) for th in samples])
    return np.atleast_2d(np.asarray(log_lik, dtype=float))


def WAIC(log_lik, samples=None):
    """Widely applicable information criterion. ``log_lik``: an ``(S, n)`` pointwise
    log-likelihood matrix, or a :class:`~stochpylib.bayesian.core.Likelihood` (+
    ``samples`` ``(S, dim)``, using ``.pointwise``)."""
    ll = _pointwise_matrix(log_lik, samples)
    S, n = ll.shape
    lppd_i = _logsumexp(ll, axis=0) - np.log(S)
    p_waic_i = np.var(ll, axis=0, ddof=1)
    elpd_i = lppd_i - p_waic_i
    value = -2.0 * np.sum(elpd_i)
    se = 2.0 * np.sqrt(n * np.var(elpd_i, ddof=1))
    return ICResult("WAIC", value, se=se, p_eff=float(np.sum(p_waic_i)), pointwise=elpd_i,
                     extras={"elpd_waic": float(np.sum(elpd_i)), "lppd": float(np.sum(lppd_i)),
                             "p_waic": p_waic_i, "n_high_var": int(np.sum(p_waic_i > 0.4))})


def LOO_CV(log_lik, samples=None, method="psis"):
    """Leave-one-out cross-validation via Pareto-smoothed importance sampling (default)
    or plain (unsmoothed) importance sampling. Same inputs as :func:`WAIC`."""
    ll = _pointwise_matrix(log_lik, samples)
    S, n = ll.shape
    log_r = -ll
    if method == "psis":
        log_w, k = _psis(log_r)
    elif method == "is":
        log_w = log_r - _logsumexp(log_r, axis=0)[None, :]
        k = np.full(n, np.nan)
    else:
        raise ValueError("method must be 'psis' or 'is'")
    elpd_loo_i = _logsumexp(log_w + ll, axis=0) - _logsumexp(log_w, axis=0)
    value = -2.0 * np.sum(elpd_loo_i)
    se = 2.0 * np.sqrt(n * np.var(elpd_loo_i, ddof=1))
    lppd_i = _logsumexp(ll, axis=0) - np.log(S)
    p_eff = float(np.sum(lppd_i) - np.sum(elpd_loo_i))
    return ICResult("LOO", value, se=se, p_eff=p_eff, pointwise=elpd_loo_i,
                     extras={"elpd_loo": float(np.sum(elpd_loo_i)), "pareto_k": k,
                             "n_bad_k": int(np.sum(k > 0.7)) if method == "psis" else None,
                             "method": method})


def TICfit(loglik, theta_hat, data=None, pointwise=None):
    """Takeuchi information criterion: ``-2*loglik(theta_hat) + 2*tr(J H^-1)``, the
    sandwich-covariance generalization of AIC for a possibly misspecified model."""
    theta_hat = np.atleast_1d(np.asarray(theta_hat, dtype=float))
    if pointwise is None:
        pointwise = loglik.pointwise if hasattr(loglik, "pointwise") else None
    if pointwise is None:
        raise ValueError("pointwise log-likelihood is required (pass pointwise= or a Likelihood)")
    f = loglik.loglik if hasattr(loglik, "loglik") else loglik

    ld = _as_log_density(f)
    H = ld.hessian(theta_hat)

    eps = 1e-6 * np.maximum(1.0, np.abs(theta_hat))
    dim = len(theta_hat)
    grads = np.empty((_n_obs(pointwise, theta_hat), dim))
    for j in range(dim):
        dx = np.zeros(dim)
        dx[j] = eps[j]
        gp = np.asarray(pointwise(theta_hat + dx), dtype=float)
        gm = np.asarray(pointwise(theta_hat - dx), dtype=float)
        grads[:, j] = (gp - gm) / (2 * eps[j])
    J = grads.T @ grads

    Hinv = np.linalg.pinv(-H)
    p_eff = float(np.trace(J @ Hinv))
    ll_hat = float(f(theta_hat))
    value = -2.0 * ll_hat + 2.0 * p_eff
    return ICResult("TIC", value, p_eff=p_eff,
                     extras={"aic": -2.0 * ll_hat + 2.0 * dim, "hessian": H, "J": J})


def _n_obs(pointwise, theta_hat):
    return len(np.atleast_1d(np.asarray(pointwise(theta_hat), dtype=float)))


def _jeffreys_label(bf):
    if bf < 1.0:
        return "negative"
    if bf < 3.2:
        return "barely worth mentioning"
    if bf < 10.0:
        return "substantial"
    if bf < 100.0:
        return "strong"
    return "decisive"


def bayes_factor(m1, m2, *, data=None, method="auto", **kw):
    """Bayes factor BF_12 = p(data|M1)/p(data|M2). Each of ``m1``/``m2`` is a log-evidence
    float, an object with ``log_evidence_`` (``Posterior``/``PosteriorApproximation``/a
    fitted model), or a ``(prior, likelihood)`` tuple; ``method="bic"`` instead accepts two
    fitted models with ``loglik_``/coefficients/``n_obs_`` and uses ``exp(-0.5*delta_BIC)``.
    """
    from stochpylib.bayesian.core import evidence as _evidence

    if method == "bic":
        b1, b2 = BIC(m1), BIC(m2)
        log_bf = -0.5 * (b1.value - b2.value)
    else:
        def logZ(m):
            if isinstance(m, tuple):
                return _evidence(m[0], m[1], method=method if method != "auto" else "auto", **kw)
            if hasattr(m, "log_evidence_"):
                if m.log_evidence_ is None:
                    raise ValueError("model has no log_evidence_ (fit with an evidence-producing method)")
                return float(m.log_evidence_)
            return float(m)

        log_z1, log_z2 = logZ(m1), logZ(m2)
        log_bf = log_z1 - log_z2

    bf = float(np.exp(np.clip(log_bf, -700, 700)))
    return ICResult("BayesFactor", bf,
                     extras={"log_bf": float(log_bf), "jeffreys": _jeffreys_label(bf)})
