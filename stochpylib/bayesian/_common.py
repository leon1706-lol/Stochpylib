"""Private helpers shared across the bayesian submodules.

Not part of the public API. Every distributional building block comes from
``stochpylib.distributions``/``scipy.special`` -- library code never wraps
``scipy.stats`` (that module is the test suite's independent oracle only, per
AGENTS.md).
"""

import numpy as np
from scipy import special


def _rng(random_state):
    return np.random.default_rng(random_state)


def _as_1d(x, name="x"):
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _as_2d(X, name="X"):
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a (n, p) array")
    return arr


def _logsumexp(a, axis=None):
    a = np.asarray(a, dtype=float)
    amax = np.max(a, axis=axis, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    out = np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True)) + amax
    if axis is None:
        return float(out.ravel()[0])
    return np.squeeze(out, axis=axis)


def _log_multibeta(alpha):
    """log B(alpha) = sum(lgamma(alpha_i)) - lgamma(sum(alpha_i))."""
    alpha = np.asarray(alpha, dtype=float)
    return float(np.sum(special.gammaln(alpha)) - special.gammaln(np.sum(alpha)))


def _cholesky_psd(A, jitter=1e-10):
    """Cholesky factor of A, escalating diagonal jitter until it succeeds."""
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    n = A.shape[0]
    eps = jitter * (np.trace(A) / n if np.trace(A) > 0 else 1.0)
    for _ in range(8):
        try:
            return np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            A = A + eps * np.eye(n)
            eps *= 10.0
    raise np.linalg.LinAlgError("matrix not positive semi-definite even with jitter")


def _mvn_logpdf(x, mean, cov):
    """log N(x | mean, cov), Cholesky-based (no raw matrix inverse)."""
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    k = len(mean)
    L = _cholesky_psd(cov)
    diff = x - mean
    y = np.linalg.solve(L, diff)
    quad = float(y @ y)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return -0.5 * (k * np.log(2.0 * np.pi) + logdet + quad)


class _LocScaleT:
    """Location-scale Student-t: X = loc + scale * T, T ~ Student_t(df).

    The library's ``Student_t`` has no loc/scale parameters; every Normal-Inverse-Gamma
    predictive needs one, so this small wrapper satisfies the same subset of the
    distribution contract that the conjugate predictives require (pdf/cdf/ppf/rvs/mean/
    var/support), built entirely on ``distributions.Student_t``.
    """

    is_discrete = False

    def __init__(self, df, loc, scale):
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.df = float(df)
        self.loc = float(loc)
        self.scale = float(scale)
        from stochpylib.distributions import Student_t
        self._t = Student_t(self.df)

    def support(self):
        return (-np.inf, np.inf)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        z = (x - self.loc) / self.scale
        return self._t.pdf(z) / self.scale

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        z = (x - self.loc) / self.scale
        return self._t.cdf(z)

    def ppf(self, q):
        return self.loc + self.scale * self._t.ppf(q)

    def rvs(self, size=1, random_state=None):
        return self.loc + self.scale * np.asarray(self._t.rvs(size, random_state=random_state))

    def mean(self):
        return self.loc if self.df > 1 else np.nan

    def var(self):
        if self.df > 2:
            return self.scale ** 2 * self.df / (self.df - 2)
        return np.inf

    def std(self):
        return np.sqrt(self.var())

    def __repr__(self):
        return f"_LocScaleT(df={self.df:.3g}, loc={self.loc:.3g}, scale={self.scale:.3g})"


class _NormalInverseGamma:
    """Normal-Inverse-Gamma(mu, kappa, alpha, beta): mu | sigma2 ~ N(mu0, sigma2/kappa),
    sigma2 ~ InvGamma(alpha, scale=beta). The conjugate joint prior/posterior for a normal
    likelihood with unknown mean and variance."""

    def __init__(self, mu, kappa, alpha, beta):
        if kappa <= 0 or alpha <= 0 or beta <= 0:
            raise ValueError("kappa, alpha, beta must be > 0")
        self.mu = float(mu)
        self.kappa = float(kappa)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def rvs(self, size=1, random_state=None):
        rng = _rng(random_state)
        n = size if isinstance(size, int) else int(np.prod(size))
        sigma2 = self.beta / rng.gamma(self.alpha, 1.0, size=n)  # InvGamma(alpha, beta)
        mu = self.mu + rng.standard_normal(n) * np.sqrt(sigma2 / self.kappa)
        out = np.column_stack([mu, sigma2])
        return out[0] if size == 1 else out

    def mean(self):
        sigma2_mean = self.beta / (self.alpha - 1) if self.alpha > 1 else np.inf
        return np.array([self.mu, sigma2_mean])

    def marginal_mu(self):
        return _LocScaleT(2 * self.alpha, self.mu, np.sqrt(self.beta / (self.alpha * self.kappa)))

    def marginal_var(self):
        from stochpylib.distributions import InvGamma
        return InvGamma(self.alpha, self.beta)

    def logpdf(self, mu, sigma2):
        from scipy.special import gammaln
        a, b, k, m0 = self.alpha, self.beta, self.kappa, self.mu
        lp = (a * np.log(b) - gammaln(a) - 0.5 * np.log(2 * np.pi / k)
              - (a + 1.5) * np.log(sigma2) - (b + 0.5 * k * (mu - m0) ** 2) / sigma2)
        return float(lp)

    def __repr__(self):
        return (f"_NormalInverseGamma(mu={self.mu:.3g}, kappa={self.kappa:.3g}, "
                f"alpha={self.alpha:.3g}, beta={self.beta:.3g})")


def _as_log_density(target):
    """Normalize any of (callable, LogDensity, (Prior, Likelihood), Posterior) to a
    stochpylib.advanced_mcmc.LogDensity -- the single choke point every computation/
    selection function goes through for gradients/Hessians without autodiff."""
    from stochpylib.advanced_mcmc import LogDensity

    if isinstance(target, LogDensity):
        return target
    if callable(target):
        return LogDensity(target)
    if isinstance(target, tuple) and len(target) == 2:
        prior, lik = target

        def log_post(theta):
            return float(prior.logpdf(theta) + lik.loglik(theta))

        return LogDensity(log_post)
    if hasattr(target, "logpdf"):
        return LogDensity(lambda theta: float(target.logpdf(theta)))
    raise TypeError("target must be a callable, LogDensity, (Prior, Likelihood) tuple, "
                     "or an object exposing logpdf(theta)")


def _slice_step_1d(logf, x, w=1.0, m=50, rng=None):
    """Neal (2003) stepping-out + shrinkage slice update for one scalar."""
    rng = rng if rng is not None else np.random.default_rng()
    fx = logf(x)
    log_y = fx - rng.exponential(1.0)
    u = rng.uniform()
    L = x - w * u
    R = L + w
    j = int(np.floor(m * rng.uniform()))
    k = m - 1 - j
    while j > 0 and logf(L) > log_y:
        L -= w
        j -= 1
    while k > 0 and logf(R) > log_y:
        R += w
        k -= 1
    for _ in range(200):
        x1 = L + rng.uniform() * (R - L)
        if logf(x1) > log_y:
            return x1
        if x1 < x:
            L = x1
        else:
            R = x1
    return x


def _gpd_fit(x):
    """Zhang & Stephens (2009) profile-likelihood estimator for a generalized Pareto's
    (k, sigma), with the standard weakly-informative-prior adjustment on k. ``x`` are the
    (positive) excesses over a threshold."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 5:
        return 0.0, float(np.mean(x)) if n else 1.0
    m = 20 + int(np.floor(np.sqrt(n)))
    xstar = x[int(n / 4 + 0.5) - 1] if n >= 4 else x[0]
    theta = 1.0 / x[-1] + (1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))) / (3.0 * xstar)
    theta = theta[theta != 0]

    def profile(th):
        k = np.mean(np.log1p(-th[:, None] * x[None, :]), axis=1)
        return n * (np.log(-th / k) - k - 1.0)

    logL = profile(theta)
    w = 1.0 / np.sum(np.exp(np.clip(logL[None, :] - logL[:, None], -700, 700)), axis=1)
    w = w / np.sum(w)
    theta_hat = float(np.sum(w * theta))
    k = float(np.mean(np.log1p(-theta_hat * x)))
    sigma = -k / theta_hat if theta_hat != 0 else float(np.mean(x))
    k = (n * k + 10.0 * 0.5) / (n + 10.0)
    return float(k), float(max(sigma, 1e-12))


def _psis(log_ratios):
    """Pareto-smoothed importance sampling (Vehtari, Simpson, Gelman, Yao & Gabry).

    ``log_ratios``: (S,) or (S, m) array of raw log importance ratios, S draws per
    column (one column per observation for LOO, or a single column for a global
    importance-sampling posterior). Returns ``(log_smoothed_weights, pareto_k)``,
    shaped like the input / ``(m,)`` respectively (scalar k for a 1-D input).
    """
    is_1d = np.ndim(log_ratios) == 1
    lr = np.asarray(log_ratios, dtype=float)
    if is_1d:
        lr = lr[:, None]
    S, n_obs = lr.shape
    M = max(int(min(0.2 * S, 3.0 * np.sqrt(S))), 5)
    out = lr.copy()
    ks = np.empty(n_obs)
    for j in range(n_obs):
        col = lr[:, j]
        order = np.argsort(col)
        tail_idx = order[-M:]
        cutoff = col[order[-M - 1]] if S > M else -np.inf
        excess = np.clip(np.exp(col[tail_idx]) - np.exp(cutoff), 1e-300, None)
        k, sigma = _gpd_fit(excess)
        ks[j] = k
        probs = (np.arange(1, M + 1) - 0.5) / M
        if abs(k) < 1e-8:
            q = -sigma * np.log(1 - probs)
        else:
            q = sigma / k * ((1 - probs) ** (-k) - 1)
        smoothed = np.log(np.exp(cutoff) + q)
        out[tail_idx, j] = smoothed  # tail_idx is already ascending in col value, as is q
    log_max_w = np.log(S ** 0.75) + _logsumexp(out, axis=0) - np.log(S)
    out = np.minimum(out, log_max_w[None, :])
    out = out - _logsumexp(out, axis=0)[None, :]
    if is_1d:
        return out[:, 0], float(ks[0])
    return out, ks
