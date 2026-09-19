"""Private helpers shared across the advanced_mcmc submodules.

Not part of the public API (no ``__all__``). Everything here is numpy/scipy.special only --
library code never wraps ``scipy.stats`` (test oracle only).
"""

import numpy as np
from scipy import special


def _rng(random_state):
    return np.random.default_rng(random_state)


def _as_1d(x, name="x"):
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_2d(X, name="X"):
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a (n, p) array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _spawn_rngs(rng, n):
    """Independent child generators, seeded off ``rng`` so a run is reproducible."""
    seeds = rng.integers(0, np.iinfo(np.int64).max, size=n)
    return [np.random.default_rng(int(s)) for s in seeds]


def _broadcast_init(theta_init, n_chains):
    """``(dim,)`` shared by every chain, or ``(n_chains, dim)`` -> ``(n_chains, dim)``."""
    arr = np.atleast_1d(np.asarray(theta_init, dtype=float))
    if arr.ndim == 1:
        return np.tile(arr, (n_chains, 1))
    if arr.ndim == 2 and arr.shape[0] == n_chains:
        return arr
    raise ValueError("theta_init must be (dim,) or (n_chains, dim)")


def _numeric_grad(f, x, eps=1e-6):
    """Central-difference gradient of a scalar function f(x) -> float."""
    x = np.asarray(x, dtype=float)
    g = np.empty_like(x)
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = eps * max(1.0, abs(x[i]))
        g[i] = (f(x + dx) - f(x - dx)) / (2 * dx[i])
    return g


def _numeric_hessian(f, x, eps=1e-4):
    """Central-difference Hessian of a scalar function f(x) -> float."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.empty((n, n))
    steps = eps * np.maximum(1.0, np.abs(x))
    f0 = f(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = steps[i]
        for j in range(i, n):
            if i == j:
                H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / steps[i] ** 2
            else:
                ej = np.zeros(n)
                ej[j] = steps[j]
                fpp = f(x + ei + ej)
                fpm = f(x + ei - ej)
                fmp = f(x - ei + ej)
                fmm = f(x - ei - ej)
                H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * steps[i] * steps[j])
    return H


def _cholesky_psd(A, jitter=1e-10):
    """Cholesky factor of A, adding diagonal jitter if not exactly PSD."""
    A = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
    n = A.shape[0]
    eps = jitter * (np.trace(A) / n if np.trace(A) > 0 else 1.0)
    for _ in range(7):
        try:
            return np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            A = A + eps * np.eye(n)
            eps *= 2.0
    raise np.linalg.LinAlgError("matrix not positive semi-definite after jittering")


def _logsumexp(a, axis=None):
    return special.logsumexp(np.asarray(a, dtype=float), axis=axis)


def _systematic_resample(w, rng):
    """Systematic resampling: returns indices of length len(w) drawn according to w."""
    n = len(w)
    positions = (rng.uniform() + np.arange(n)) / n
    idx = np.searchsorted(np.cumsum(w), positions)
    return np.clip(idx, 0, n - 1)


def _log_accept(u_log, log_alpha):
    """Whether a Metropolis proposal is accepted, given log(u) and the log acceptance ratio."""
    return np.isfinite(log_alpha) and u_log < log_alpha


class _DualAveraging:
    """Hoffman & Gelman (2014) primal-dual step-size adaptation toward a target accept stat."""

    def __init__(self, step_size0, target, gamma=0.05, t0=10.0, kappa=0.75):
        self.mu = np.log(10.0 * step_size0)
        self.target = float(target)
        self.gamma = float(gamma)
        self.t0 = float(t0)
        self.kappa = float(kappa)
        self.h_bar = 0.0
        self.log_eps = np.log(step_size0)
        self.log_eps_bar = 0.0
        self.t = 0

    def update(self, accept_stat):
        self.t += 1
        a = min(1.0, float(accept_stat))
        eta_h = 1.0 / (self.t + self.t0)
        self.h_bar = (1.0 - eta_h) * self.h_bar + eta_h * (self.target - a)
        self.log_eps = self.mu - np.sqrt(self.t) / self.gamma * self.h_bar
        eta = self.t ** (-self.kappa)
        self.log_eps_bar = eta * self.log_eps + (1.0 - eta) * self.log_eps_bar
        return np.exp(self.log_eps)

    def final(self):
        return float(np.exp(self.log_eps_bar))

    def current(self):
        return float(np.exp(self.log_eps))


class _Welford:
    """Running mean / covariance (ddof=1) for online adaptation."""

    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim)
        self.m2 = np.zeros((dim, dim))

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean = self.mean + delta / self.n
        delta2 = x - self.mean
        self.m2 = self.m2 + np.outer(delta, delta2)

    def cov(self):
        if self.n < 2:
            return np.eye(len(self.mean))
        return self.m2 / (self.n - 1)

    def var(self):
        return np.diag(self.cov())

    def reset(self):
        self.n = 0
        self.mean = np.zeros_like(self.mean)
        self.m2 = np.zeros_like(self.m2)


class _Adam:
    """Adam optimizer step for gradient ascent on a flat parameter vector."""

    def __init__(self, shape, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, params, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params + self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _robbins_monro_scale(log_scale, accepted, target, k, floor=-10.0, ceil=10.0):
    """Robbins-Monro update of a log proposal-scale toward a target acceptance rate."""
    step = min(0.5, 1.0 / np.sqrt(k + 1))
    new = log_scale + step * (float(accepted) - target)
    return float(np.clip(new, floor, ceil))
