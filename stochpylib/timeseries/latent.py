"""Latent-variable and regime-switching time-series models.

- :class:`HiddenMarkovModel` — Gaussian-emission HMM fit by Baum-Welch EM.
- :class:`SwitchingRegression` — Markov-switching linear regression (Hamilton-style EM
  with smoothed regime probabilities; weighted least squares per regime).
- :class:`RegimeSwitching` — Markov-switching autoregression AR(p) (thin wrapper that
  builds the lag design and delegates to the switching core).
- :class:`MixtureAutoregressive` — mixture of AR(p) components with iid mixing weights
  (no Markov persistence across regimes).

Documented simplifications: Gaussian emissions with diagonal covariance for the HMM;
regime probabilities come from a Hamilton filter + backward smoothing (not Kim's full
collapsed filter); mixture components share no cross-component structure.
"""

import numpy as np
from scipy import stats
from scipy.special import logsumexp

from stochpylib.timeseries._utils import as_1d, as_2d, lag_matrix

__all__ = [
    "HiddenMarkovModel",
    "SwitchingRegression",
    "RegimeSwitching",
    "MixtureAutoregressive",
]


# ---------------------------------------------------------------------------
# helpers


def _weighted_ols(X, y, w):
    sw = np.sqrt(np.clip(w, 1e-300, None))
    Xw = X * sw[:, None]
    yw = y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ beta
    return beta, resid


def _forward_backward_markov(log_emis, A, pi):
    """Scaled forward-backward over a Markov chain of hidden regimes.

    Returns ``(gamma, xi_sum, loglik)`` where ``gamma`` is the smoothed posterior
    (T, K), ``xi_sum`` holds expected transition counts, and ``loglik`` is the exact
    likelihood of the observed sequence under the current parameters.
    """
    emis = np.exp(log_emis)
    T, K = emis.shape
    A = np.asarray(A, dtype=float)

    alpha_hat = np.empty((T, K))
    scales = np.empty(T)
    a = pi * emis[0]
    scales[0] = max(a.sum(), 1e-300)
    alpha_hat[0] = a / scales[0]
    for t in range(1, T):
        a = (alpha_hat[t - 1] @ A) * emis[t]
        scales[t] = max(a.sum(), 1e-300)
        alpha_hat[t] = a / scales[t]

    beta_hat = np.empty((T, K))
    beta_hat[-1] = 1.0
    for t in range(T - 2, -1, -1):
        b = A @ (emis[t + 1] * beta_hat[t + 1])
        s = b.sum()
        beta_hat[t] = b / s if s > 1e-300 else b

    gamma = alpha_hat * beta_hat
    gamma /= gamma.sum(axis=1, keepdims=True)

    xi_sum = np.zeros((K, K))
    for t in range(T - 1):
        M = (alpha_hat[t][:, None] * A) * (emis[t + 1] * beta_hat[t + 1])[None, :]
        xi_sum += M / max(scales[t + 1], 1e-300)

    loglik = float(np.log(scales).sum())
    return gamma, xi_sum, loglik


def _viterbi(log_emis, A, pi):
    T, K = log_emis.shape
    delta = pi + log_emis[0]
    psi = np.empty((T, K), dtype=int)
    log_A = np.log(np.clip(A, 1e-300, None))
    for t in range(1, T):
        scores = delta[:, None] + np.log(np.clip(A, 1e-300, None))
        psi[t] = np.argmax(scores, axis=0)
        delta = np.max(scores, axis=0) + log_emis[t]
    states = np.empty(T, dtype=int)
    states[-1] = int(np.argmax(delta))
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


# ---------------------------------------------------------------------------
# HiddenMarkovModel


class HiddenMarkovModel:
    """Gaussian-emission hidden Markov model fit by Baum-Welch EM.

    Emissions are univariate Gaussians with per-state mean and standard deviation.
    ``decode`` runs Viterbi; ``score`` returns the log-likelihood;
    ``predict_proba`` gives smoothed state responsibilities.
    """

    def __init__(self, n_states=2, max_iter=200, tol=1e-6, random_state=None):
        self.n_states = int(n_states)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

    def _emissions(self, y):
        return stats.norm.logpdf(y[:, None], self.means_[None, :], self.stds_[None, :])

    def fit(self, y):
        y = as_1d(y)
        K = self.n_states
        if K < 2 or len(y) < 4 * K:
            raise ValueError("need n_states >= 2 and at least 4*K observations")
        stick = 0.7
        self.transition_ = stick * np.eye(K) + (1.0 - stick) / K
        self.startprob_ = np.full(K, 1.0 / K)
        self.means_ = np.quantile(y, (np.arange(K) + 0.5) / K)
        self.stds_ = np.full(K, y.std() + 1e-6)

        prev_ll = -np.inf
        converged = False
        n_iter = 0
        gamma_last = None
        for n_iter in range(1, self.max_iter + 1):
            emis = self._emissions(y)
            gamma, xi_sum, ll = _forward_backward_markov(emis, self.transition_, self.startprob_)
            w = gamma.sum(axis=0) + 1e-12
            self.means_ = (gamma * y[:, None]).sum(axis=0) / w
            var = (gamma * (y[:, None] - self.means_[None, :]) ** 2).sum(axis=0) / w
            self.stds_ = np.sqrt(var + 1e-9)
            rows = xi_sum.sum(axis=1, keepdims=True) + 1e-12
            self.transition_ = xi_sum / rows
            self.startprob_ = gamma[0] + 1e-12
            self.startprob_ /= self.startprob_.sum()
            if abs(ll - prev_ll) < self.tol * (1.0 + abs(prev_ll)):
                converged = True
                break
            prev_ll = ll
            gamma_last = gamma

        # final pass so stored quantities match final parameters
        emis = self._emissions(y)
        gamma_final, _, ll_final = _forward_backward_markov(
            emis, self.transition_, self.startprob_
        )
        self.loglik_ = ll_final
        self.filtered_probs_ = gamma_final
        self.converged_ = converged
        self.n_iter_ = n_iter
        self._y = y
        return self

    def score(self, y=None):
        """Log-likelihood of ``y`` (defaults to the fitted series)."""
        data = as_1d(self._y if y is None else y)
        emis = stats.norm.logpdf(data[:, None], self.means_[None, :], self.stds_[None, :])
        _, _, ll = _forward_backward_markov(emis, self.transition_, self.startprob_)
        return ll

    def predict_proba(self, y=None):
        """Smoothed state probabilities ``(T, K)``."""
        data = as_1d(self._y if y is None else y)
        emis = stats.norm.logpdf(data[:, None], self.means_[None, :], self.stds_[None, :])
        gamma, _, _ = _forward_backward_markov(emis, self.transition_, self.startprob_)
        return gamma

    def decode(self, y=None):
        """Most likely hidden-state path (Viterbi)."""
        data = as_1d(self._y if y is None else y)
        emis = stats.norm.logpdf(data[:, None], self.means_[None, :], self.stds_[None, :])
        return _viterbi(emis, self.transition_, self.startprob_)


# ---------------------------------------------------------------------------
# switching core shared by the three regression-family models


def _fit_switching_core(design, target, n_regimes, markov=True,
                        max_iter=200, tol=1e-7, random_state=None):
    design = np.asarray(design, dtype=float)
    target = as_1d(target)
    T, d = design.shape
    rng = np.random.default_rng(random_state)
    K = int(n_regimes)

    # initialization: OLS on contiguous segments keeps coefficients distinct
    coefs, sigmas = [], []
    edges = np.linspace(0, T, K + 1).astype(int)
    pooled_var = float(np.var(target)) + 1e-9
    for i in range(K):
        seg = design[edges[i]:edges[i + 1]]
        tgt = target[edges[i]:edges[i + 1]]
        if len(tgt) >= d + 1:
            b, r = _weighted_ols(seg, tgt, np.ones(len(tgt)))
            s2 = max(float(r @ r) / len(r), pooled_var * 1e-3)
        else:
            b = np.zeros(d)
            s2 = pooled_var
        coefs.append(b)
        sigmas.append(np.sqrt(s2))

    if markov:
        A = 0.85 * np.eye(K) + 0.15 / K
        pi = np.full(K, 1.0 / K)
    else:
        weights = np.full(K, 1.0 / K)

    prev_ll = -np.inf
    converged = False
    n_iter = 0
    gamma = None
    for n_iter in range(1, max_iter + 1):
        e = target[:, None] - design @ np.column_stack(coefs)   # (T, K)
        sig = np.asarray(sigmas)
        lik = stats.norm.logpdf(e, scale=sig[None, :])          # (T, K)

        if markov:
            gamma, xi_sum, ll = _forward_backward_markov(lik, A, pi)
        else:
            logw = np.log(weights + 1e-300)
            joint = logw[None, :] + lik
            norm = logsumexp(joint, axis=1)
            gamma = np.exp(joint - norm[:, None])
            ll = float(norm.sum())
            weights = gamma.mean(axis=0)

        # M-step: weighted least squares per regime
        for r in range(K):
            wr = np.clip(gamma[:, r], 1e-12, None)
            coef_r, resid_r = _weighted_ols(design, target, wr)
            coefs[r] = coef_r
            sigmas[r] = float(
                np.sqrt(max(float(wr @ resid_r**2) / max(float(wr.sum()), 1e-12), 1e-10))
            )
        if markov:
            rows = xi_sum.sum(axis=1, keepdims=True) + 1e-12
            A = xi_sum / rows
            pi = gamma[0] + 1e-12
            pi /= pi.sum()

        if abs(ll - prev_ll) < tol * (1.0 + abs(prev_ll)):
            converged = True
            break
        prev_ll = ll

    return {
        "coefs": coefs,
        "sigmas": np.asarray(sigmas),
        "transition": A.copy() if markov else None,
        "startprob": gamma[0].copy() if markov else None,
        "weights": gamma.mean(axis=0) if not markov else None,
        "regime_probs": gamma,
        "loglik": float(ll),
        "converged": converged,
        "n_iter": n_iter,
    }


# ---------------------------------------------------------------------------
# public classes


class SwitchingRegression:
    """Markov-switching linear regression.

    Fits ``y_t = x_t' beta_{s_t} + eps_t`` where the latent regime ``s_t`` follows a
    first-order Markov chain (Hamilton-style EM with smoothed regime probabilities).
    A constant column is prepended to the design internally.
    """

    def __init__(self, n_regimes=2, max_iter=200, tol=1e-7, random_state=None):
        self.n_regimes = int(n_regimes)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

    def fit(self, X, y):
        X = as_2d(X, "X")
        y = as_1d(y)
        if len(X) != len(y):
            raise ValueError("X and y must have equal length")
        design = np.column_stack([np.ones(len(X)), X])
        out = _fit_switching_core(design, y, self.n_regimes, markov=True,
                                  max_iter=self.max_iter, tol=self.tol,
                                  random_state=self.random_state)
        self.coefficients_ = out["coefs"]          # list of (d+1,) arrays, intercept first
        self.sigmas_ = out["sigmas"]
        self.transition_ = out["transition"]
        self.regime_probs_ = out["regime_probs"]
        self.loglik_ = out["loglik"]
        self.converged_ = out["converged"]
        self.n_iter_ = out["n_iter"]
        self._design_cols = design.shape[1]
        return self

    def predict(self, X_new):
        """Expected value using the final filtered regime weights."""
        X_new = as_2d(X_new, "X_new")
        design = np.column_stack([np.ones(len(X_new)), X_new])
        weights = self.regime_probs_[-1]
        preds = np.column_stack([design @ b for b in self.coefficients_])
        return preds @ weights


class RegimeSwitching(SwitchingRegression):
    """Markov-switching autoregression AR(p).

    Builds the lag design internally; ``ar_coefficients_`` exposes the per-regime AR
    coefficients (intercept excluded, which lives in ``coefficients_[r][0]``).
    """

    def __init__(self, p=1, n_states=2, max_iter=200, tol=1e-7, random_state=None):
        SwitchingRegression.__init__(
            self,
            n_regimes=n_states,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.p = int(p)

    def fit(self, y):
        y = as_1d(y)
        X_lags, target = lag_matrix(y, self.p)
        base = super().fit(X_lags, target)
        self.ar_coefficients_ = [b[1:] for b in self.coefficients_]
        self.intercepts_ = [b[0] for b in self.coefficients_]
        self._y = y
        return self


class MixtureAutoregressive(SwitchingRegression):
    """Mixture of k autoregressive components with iid mixing weights.

    Identical estimation machinery to the Markov-switching variant but without regime
    persistence: each time point draws its regime independently from fixed weights.
    """

    def __init__(self, k=2, p=1, max_iter=200, tol=1e-7, random_state=None):
        # bypass SwitchingRegression's own init (it sets n_regimes semantics)
        SwitchingRegression.__init__(self, n_regimes=k, max_iter=max_iter, tol=tol,
                                     random_state=random_state)
        self.p = int(p)

    def fit(self, y):
        y = as_1d(y)
        X_lags, target = lag_matrix(y, self.p)
        design = np.column_stack([np.ones(len(X_lags)), X_lags])
        out = _fit_switching_core(design, target, self.n_regimes, markov=False,
                                  max_iter=self.max_iter, tol=self.tol,
                                  random_state=self.random_state)
        self.coefficients_ = out["coefs"]
        self.sigmas_ = out["sigmas"]
        self.weights_ = out["weights"]
        self.regime_probs_ = out["regime_probs"]
        self.loglik_ = out["loglik"]
        self.converged_ = out["converged"]
        self.n_iter_ = out["n_iter"]
        self.ar_coefficients_ = [b[1:] for b in self.coefficients_]
        self.intercepts_ = [b[0] for b in self.coefficients_]
        self._y = y
        return self
