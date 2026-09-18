"""Point and interval estimation: maximum likelihood/moments, Bayesian conjugate
updating, classical confidence intervals, resampling (bootstrap/jackknife), the
delta method, and profile likelihood.
"""

import inspect
import math

import numpy as np
from scipy import optimize, special

from stochpylib.statistics._common import (
    _as_1d, _chi2_ppf, _norm_cdf, _norm_ppf, _numeric_hessian, _numeric_jacobian, _rng,
)
from stochpylib.statistics._result import EstimateResult

__all__ = [
    "MLE", "MOM", "bayesian_estimator", "bootstrap_ci", "confidence_interval",
    "delta_method", "jackknife", "profile_likelihood",
]


def _dist_params(dist):
    """Constructor parameter values of a fitted Distribution instance, in order."""
    names = [p for p in inspect.signature(type(dist).__init__).parameters if p != "self"]
    return np.array([getattr(dist, name) for name in names], dtype=float), names


def MLE(model, data, x0=None, bounds=None):
    """Maximum-likelihood estimation.

    ``model`` is either a :class:`~stochpylib.distributions.Distribution` subclass
    (its numeric or closed-form ``.fit()`` is used, or the generic numeric optimizer
    when ``x0``/``bounds`` are given explicitly) or a ``loglik(params, data) -> float``
    callable (``x0`` required). Standard errors are ``sqrt(diag(inv(observed
    information)))`` from a numerical Hessian of the negative log-likelihood at the
    optimum; they come back as ``nan`` when the optimum sits on a parameter bound
    (e.g. fitting ``Uniform`` -- the Hessian there is singular by construction).
    """
    from stochpylib.distributions._base import Distribution

    data_arr = _as_1d(data)
    n = len(data_arr)

    if inspect.isclass(model) and issubclass(model, Distribution):
        dist = model._generic_fit(data_arr, x0, bounds) if (x0 is not None or bounds is not None) else model.fit(data_arr)
        params, names = _dist_params(dist)

        def nll(p):
            try:
                d = model(*p)
            except (ValueError, ArithmeticError):
                return np.inf
            vals = np.atleast_1d(d.pdf(data_arr))
            vals = np.clip(vals, 1e-300, None)
            return float(-np.sum(np.log(vals)))

        loglik = -nll(params)
        extras_dist = dist
    else:
        if x0 is None:
            raise ValueError("x0 is required when model is a log-likelihood callable")
        x0 = np.asarray(x0, dtype=float)

        def nll(p):
            return -model(p, data_arr)

        method = "L-BFGS-B" if bounds is not None else "Nelder-Mead"
        res = optimize.minimize(nll, x0=x0, bounds=bounds, method=method)
        params = res.x
        loglik = -res.fun
        names = None
        extras_dist = None

    k = len(params)
    try:
        H = _numeric_hessian(nll, params)
        cov = np.linalg.inv(H)
        diag = np.diag(cov)
        se = np.sqrt(np.where(diag >= 0, diag, np.nan))
    except np.linalg.LinAlgError:
        cov = None
        se = np.full(k, np.nan)

    aic = -2.0 * loglik + 2.0 * k
    bic = -2.0 * loglik + math.log(n) * k
    extras = {"loglik": loglik, "aic": aic, "bic": bic, "cov": cov, "param_names": names, "dist": extras_dist}
    value = params[0] if k == 1 else params
    err = se[0] if k == 1 else se
    return EstimateResult(value, err, n, "MLE", extras)


def MOM(dist_class, data):
    """Method-of-moments estimation: solve sample moments = theoretical moments and
    return a fitted instance of ``dist_class``. Closed forms are used for Normal,
    Exponential, Gamma, Beta, Poisson, Uniform, LogNormal, and Weibull; any other
    :class:`Distribution` subclass falls back to a nonlinear least-squares match of
    ``dist.mean()``/``dist.var()`` to the sample moments.
    """
    data = _as_1d(data)
    m1 = float(np.mean(data))
    var = float(np.var(data, ddof=0))
    name = dist_class.__name__

    if name == "Normal":
        return dist_class(m1, math.sqrt(var))
    if name == "Exponential":
        return dist_class(1.0 / m1)
    if name == "Gamma":
        return dist_class(m1 ** 2 / var, var / m1)
    if name == "Beta":
        common = m1 * (1.0 - m1) / var - 1.0
        return dist_class(m1 * common, (1.0 - m1) * common)
    if name == "Poisson":
        return dist_class(m1)
    if name == "Uniform":
        half = math.sqrt(3.0 * var)
        return dist_class(m1 - half, m1 + half)
    if name == "LogNormal":
        cv2 = var / m1 ** 2
        sigma2 = math.log(1.0 + cv2)
        return dist_class(math.log(m1) - sigma2 / 2.0, math.sqrt(sigma2))
    if name == "Weibull":
        cv2 = var / m1 ** 2

        def g(k):
            return special.gamma(1.0 + 2.0 / k) / special.gamma(1.0 + 1.0 / k) ** 2 - 1.0 - cv2

        k_hat = optimize.brentq(g, 0.05, 100.0)
        lam = m1 / special.gamma(1.0 + 1.0 / k_hat)
        return dist_class(k_hat, lam)

    params = [p for p in inspect.signature(dist_class.__init__).parameters if p != "self"]
    x0 = np.ones(len(params))

    def resid(p):
        try:
            d = dist_class(*p)
            return [d.mean() - m1, d.var() - var]
        except (ValueError, ArithmeticError):
            return [1e6, 1e6]

    res = optimize.least_squares(resid, x0)
    return dist_class(*res.x)


def bayesian_estimator(likelihood, data, prior, level=0.95, sigma=None, grid=None):
    """Bayesian point/interval estimation.

    Conjugate families (``likelihood`` a string): ``"normal"`` (known ``sigma``,
    ``prior=(mu0, tau0)``), ``"bernoulli"``/``"binomial"`` (``prior=(a0, b0)``,
    ``data`` an array of 0/1 or a ``(successes, n)`` pair), ``"poisson"``/
    ``"exponential"`` (``prior=(alpha0, beta0)``, rate parameterization). The exact
    posterior is returned in ``extras['posterior']`` as a
    :class:`~stochpylib.distributions.Distribution`; the reported interval is the
    exact equal-tailed credible interval (``extras['credible_interval']`` and, as
    ``__float__``-compatible fields, ``estimate``/``std_error``).

    ``likelihood`` may instead be a ``loglik(theta, data)`` callable with ``prior`` a
    ``logprior(theta)`` callable and ``grid`` a 1-D array of candidate theta values;
    the posterior is then built by direct numerical integration over ``grid``
    (``extras['grid']``/``extras['density']``), with no assumption of conjugacy.
    """
    alpha2 = (1.0 - level) / 2.0

    if callable(likelihood):
        if grid is None:
            raise ValueError("grid is required when likelihood is a callable log-likelihood")
        grid = np.asarray(grid, dtype=float)
        log_post = np.array([likelihood(t, data) + prior(t) for t in grid])
        log_post -= np.max(log_post)
        w = np.exp(log_post)
        w /= np.trapezoid(w, grid)
        mean = float(np.trapezoid(w * grid, grid))
        var = float(np.trapezoid(w * (grid - mean) ** 2, grid))
        mode = float(grid[np.argmax(w)])
        cdf = np.concatenate([[0.0], np.cumsum((w[1:] + w[:-1]) / 2.0 * np.diff(grid))])
        cdf = cdf / cdf[-1]
        lo = float(np.interp(alpha2, cdf, grid))
        hi = float(np.interp(1.0 - alpha2, cdf, grid))
        n = len(_as_1d(data)) if not isinstance(data, tuple) else data[1]
        return EstimateResult(mean, math.sqrt(var), n, "Bayesian (grid posterior)",
                               {"mode": mode, "grid": grid, "density": w, "credible_interval": (lo, hi)})

    from stochpylib.distributions import Beta as _Beta, Gamma as _Gamma, Normal as _Normal

    family = likelihood.lower()
    if family == "normal":
        if sigma is None:
            raise ValueError("sigma (known observation std) is required for the normal-normal family")
        data = _as_1d(data)
        n = len(data)
        mu0, tau0 = prior
        prec0 = 1.0 / tau0 ** 2
        prec_lik = n / sigma ** 2
        post_prec = prec0 + prec_lik
        post_mean = (mu0 * prec0 + np.sum(data) / sigma ** 2) / post_prec
        posterior = _Normal(post_mean, math.sqrt(1.0 / post_prec))
    elif family in ("bernoulli", "binomial"):
        a0, b0 = prior
        if isinstance(data, tuple):
            successes, n = data
        else:
            data = _as_1d(data)
            successes, n = float(np.sum(data)), len(data)
        posterior = _Beta(a0 + successes, b0 + n - successes)
    elif family == "poisson":
        alpha0, beta0 = prior
        data = _as_1d(data)
        n = len(data)
        posterior = _Gamma(alpha0 + np.sum(data), 1.0 / (beta0 + n))
    elif family == "exponential":
        alpha0, beta0 = prior
        data = _as_1d(data)
        n = len(data)
        posterior = _Gamma(alpha0 + n, 1.0 / (beta0 + np.sum(data)))
    else:
        raise ValueError(f"unknown conjugate family: {likelihood!r}")

    mean, var = float(posterior.mean()), float(posterior.var())
    lo, hi = float(posterior.ppf(alpha2)), float(posterior.ppf(1.0 - alpha2))
    n_obs = n if not isinstance(data, tuple) else data[1]
    return EstimateResult(mean, math.sqrt(var), n_obs, f"Bayesian ({family}-conjugate)",
                           {"posterior": posterior, "credible_interval": (lo, hi)})


def confidence_interval(data, level=0.95, kind="mean", sigma=None, method=None):
    """Classical confidence interval. ``kind``: ``"mean"`` (t, or z if ``sigma`` is
    given), ``"proportion"`` (``method`` in wald/wilson/agresti-coull/clopper-pearson,
    default wilson; ``data`` an array of 0/1 or a ``(successes, n)`` pair),
    ``"variance"``/``"std"`` (chi-squared), ``"median"`` (order-statistic, binomial),
    ``"mean_diff"`` (``data=(x, y)``, ``method`` welch/pooled), or ``"correlation"``
    (``data=(x, y)``, Fisher z).
    """
    alpha2 = (1.0 - level) / 2.0

    if kind == "mean":
        x = _as_1d(data)
        n = len(x)
        xbar = float(np.mean(x))
        if sigma is not None:
            se = sigma / math.sqrt(n)
            z = float(_norm_ppf(1.0 - alpha2))
            return xbar - z * se, xbar + z * se
        se = float(np.std(x, ddof=1)) / math.sqrt(n)
        tcrit = float(special.stdtrit(n - 1, 1.0 - alpha2))
        return xbar - tcrit * se, xbar + tcrit * se

    if kind == "proportion":
        if isinstance(data, tuple):
            k, n = data
        else:
            x = _as_1d(data)
            k, n = float(np.sum(x)), len(x)
        phat = k / n
        z = float(_norm_ppf(1.0 - alpha2))
        method = method or "wilson"
        if method == "wald":
            se = math.sqrt(phat * (1 - phat) / n)
            return phat - z * se, phat + z * se
        if method == "wilson":
            denom = 1.0 + z ** 2 / n
            center = (phat + z ** 2 / (2 * n)) / denom
            half = z * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
            return center - half, center + half
        if method == "agresti-coull":
            ntil = n + z ** 2
            ptil = (k + z ** 2 / 2) / ntil
            half = z * math.sqrt(ptil * (1 - ptil) / ntil)
            return ptil - half, ptil + half
        if method == "clopper-pearson":
            lo = 0.0 if k == 0 else float(special.betaincinv(k, n - k + 1, alpha2))
            hi = 1.0 if k == n else float(special.betaincinv(k + 1, n - k, 1.0 - alpha2))
            return lo, hi
        raise ValueError("method must be 'wald', 'wilson', 'agresti-coull', or 'clopper-pearson'")

    if kind in ("variance", "std"):
        x = _as_1d(data)
        n = len(x)
        s2 = float(np.var(x, ddof=1))
        chi_lo = float(_chi2_ppf(alpha2, n - 1))
        chi_hi = float(_chi2_ppf(1.0 - alpha2, n - 1))
        lo, hi = (n - 1) * s2 / chi_hi, (n - 1) * s2 / chi_lo
        return (math.sqrt(lo), math.sqrt(hi)) if kind == "std" else (lo, hi)

    if kind == "median":
        x = np.sort(_as_1d(data))
        n = len(x)
        z = float(_norm_ppf(1.0 - alpha2))
        j = max(int(math.floor((n - z * math.sqrt(n)) / 2.0)), 0)
        k = min(n - 1 - j, n - 1)
        return float(x[j]), float(x[k])

    if kind == "mean_diff":
        x, y = data
        x, y = _as_1d(x), _as_1d(y)
        n, m = len(x), len(y)
        diff = float(np.mean(x) - np.mean(y))
        s1, s2 = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
        method = method or "welch"
        if method == "pooled":
            sp2 = ((n - 1) * s1 + (m - 1) * s2) / (n + m - 2)
            se = math.sqrt(sp2 * (1.0 / n + 1.0 / m))
            df = n + m - 2
        else:
            se = math.sqrt(s1 / n + s2 / m)
            df = (s1 / n + s2 / m) ** 2 / ((s1 / n) ** 2 / (n - 1) + (s2 / m) ** 2 / (m - 1))
        tcrit = float(special.stdtrit(df, 1.0 - alpha2))
        return diff - tcrit * se, diff + tcrit * se

    if kind == "correlation":
        x, y = data
        from stochpylib.statistics.descriptive import correlation as _corr
        res = _corr(x, y, method="pearson")
        r, n = res.estimate, res.n
        zf = math.atanh(max(min(r, 1 - 1e-12), -1 + 1e-12))
        se = 1.0 / math.sqrt(n - 3)
        z = float(_norm_ppf(1.0 - alpha2))
        return math.tanh(zf - z * se), math.tanh(zf + z * se)

    raise ValueError(f"unknown kind: {kind!r}")


def bootstrap_ci(data, statistic, n_boot=2000, level=0.95, method="percentile", random_state=None):
    """Bootstrap confidence interval for an arbitrary statistic.

    ``data`` is a single array, or a tuple of same-length arrays resampled together
    (paired bootstrap, e.g. for a correlation coefficient). ``statistic`` is called on
    plain 1-D array(s); if it also accepts ``(n_boot, n)``-shaped batched arrays and
    returns a length-``n_boot`` vector, that vectorized path is used for speed
    (falling back transparently to a per-replicate Python loop otherwise).
    ``method``: ``"percentile"``, ``"basic"``, ``"normal"``, or ``"bca"``
    (bias-corrected and accelerated, using jackknife acceleration).
    """
    rng = _rng(random_state)
    arrays = [np.asarray(a, dtype=float) for a in data] if isinstance(data, tuple) else [np.asarray(data, dtype=float)]
    n = len(arrays[0])
    theta_hat = float(statistic(*arrays))

    idx = rng.integers(0, n, size=(n_boot, n))
    batched = [a[idx] for a in arrays]
    try:
        boot_stats = np.asarray(statistic(*batched), dtype=float)
        if boot_stats.shape != (n_boot,):
            raise ValueError("statistic did not return a vectorized result")
    except Exception:
        boot_stats = np.empty(n_boot)
        for b in range(n_boot):
            boot_stats[b] = statistic(*[a[b] for a in batched])

    se = float(np.std(boot_stats, ddof=1))
    alpha2 = (1.0 - level) / 2.0

    if method == "percentile":
        lo, hi = np.quantile(boot_stats, [alpha2, 1.0 - alpha2])
    elif method == "basic":
        q_lo, q_hi = np.quantile(boot_stats, [alpha2, 1.0 - alpha2])
        lo, hi = 2 * theta_hat - q_hi, 2 * theta_hat - q_lo
    elif method == "normal":
        z = float(_norm_ppf(1.0 - alpha2))
        lo, hi = theta_hat - z * se, theta_hat + z * se
    elif method == "bca":
        jack_stats = np.array([
            statistic(*[np.delete(a, i) for a in arrays]) for i in range(n)
        ], dtype=float)
        jack_mean = np.mean(jack_stats)
        d = jack_mean - jack_stats
        denom = 6.0 * np.sum(d ** 2) ** 1.5
        a_hat = float(np.sum(d ** 3) / denom) if denom != 0 else 0.0
        prop_less = np.mean(boot_stats < theta_hat) + 0.5 * np.mean(boot_stats == theta_hat)
        z0 = float(_norm_ppf(np.clip(prop_less, 1e-10, 1 - 1e-10)))
        z_lo, z_hi = float(_norm_ppf(alpha2)), float(_norm_ppf(1.0 - alpha2))
        p_lo = float(_norm_cdf(z0 + (z0 + z_lo) / (1.0 - a_hat * (z0 + z_lo))))
        p_hi = float(_norm_cdf(z0 + (z0 + z_hi) / (1.0 - a_hat * (z0 + z_hi))))
        lo, hi = np.quantile(boot_stats, [p_lo, p_hi])
    else:
        raise ValueError("method must be 'percentile', 'basic', 'normal', or 'bca'")

    return EstimateResult(theta_hat, se, n, f"bootstrap ({method})",
                           {"distribution": boot_stats, "conf_int": (float(lo), float(hi))})


def jackknife(data, statistic):
    """Jackknife bias and standard error of a statistic; ``estimate`` is the
    bias-corrected point estimate."""
    x = _as_1d(data)
    n = len(x)
    theta_hat = float(statistic(x))
    jack_stats = np.array([statistic(np.delete(x, i)) for i in range(n)], dtype=float)
    jack_mean = float(np.mean(jack_stats))
    bias = (n - 1) * (jack_mean - theta_hat)
    se = math.sqrt((n - 1) / n * np.sum((jack_stats - jack_mean) ** 2))
    pseudovalues = n * theta_hat - (n - 1) * jack_stats
    return EstimateResult(theta_hat - bias, se, n, "jackknife",
                           {"estimate": theta_hat, "bias": bias, "pseudovalues": pseudovalues})


def delta_method(g, theta, cov, grad=None):
    """Asymptotic standard error of ``g(theta)`` given ``cov(theta)``, via the delta
    method: ``Var(g(theta)) ~= J Cov J^T`` with ``J`` the Jacobian of ``g`` (numerical
    by default, or supplied analytically via ``grad(theta)``)."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    if grad is None:
        J = _numeric_jacobian(lambda t: np.atleast_1d(g(t)), theta)
    else:
        J = np.atleast_2d(np.asarray(grad(theta), dtype=float))
    out_cov = J @ cov @ J.T
    val = np.atleast_1d(np.asarray(g(theta), dtype=float))
    se = np.sqrt(np.diag(out_cov))
    if val.size == 1:
        return EstimateResult(float(val[0]), float(se[0]), 0, "delta method", {"cov": out_cov})
    return EstimateResult(val, se, 0, "delta method", {"cov": out_cov})


def profile_likelihood(loglik, params_hat, index, data=None, grid=None, level=0.95):
    """Profile-likelihood confidence interval for one parameter.

    Other parameters are re-optimized (Nelder-Mead) at each trial value of
    ``params_hat[index]``; the interval is where the profile log-likelihood drops by
    ``chi2(level, df=1) / 2`` from its maximum (Wilks' theorem). ``grid``, if given, is
    also evaluated and returned as the profile curve (``extras['profile_grid']``/
    ``extras['profile_loglik']``) for diagnostics/plotting.
    """
    params_hat = np.atleast_1d(np.asarray(params_hat, dtype=float))
    k = len(params_hat)
    other_idx = [i for i in range(k) if i != index]

    def full_ll(p):
        return loglik(p, data) if data is not None else loglik(p)

    ll_hat = full_ll(params_hat)

    def profile_ll(theta_val):
        if not other_idx:
            return full_ll(np.array([theta_val]))

        def neg_ll_others(other_params):
            p = np.empty(k)
            p[index] = theta_val
            for oi, val in zip(other_idx, other_params):
                p[oi] = val
            return -full_ll(p)

        res = optimize.minimize(neg_ll_others, x0=params_hat[other_idx], method="Nelder-Mead")
        return -res.fun

    chi2_crit = float(_chi2_ppf(level, 1))
    target = ll_hat - 0.5 * chi2_crit
    theta_hat = params_hat[index]

    def f(theta_val):
        return profile_ll(theta_val) - target

    step = max(abs(theta_hat) * 0.1, 0.01)
    lo = theta_hat
    for _ in range(500):
        lo -= step
        if f(lo) < 0:
            break
    hi = theta_hat
    for _ in range(500):
        hi += step
        if f(hi) < 0:
            break
    ci_lo = float(optimize.brentq(f, lo, theta_hat))
    ci_hi = float(optimize.brentq(f, theta_hat, hi))

    extras = {"conf_int": (ci_lo, ci_hi), "loglik_max": float(ll_hat)}
    if grid is not None:
        grid = np.asarray(grid, dtype=float)
        extras["profile_grid"] = grid
        extras["profile_loglik"] = np.array([profile_ll(t) for t in grid])
    return EstimateResult(float(theta_hat), None, None, "profile likelihood", extras)
