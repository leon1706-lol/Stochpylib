"""State-space models and the Kalman filter family.

Linear-Gaussian convention::

    x_{t+1} = F x_t + w_t,   w_t ~ N(0, Q)
    y_t     = H x_t + v_t,   v_t ~ N(0, R)

``KalmanFilter.fit(observations)`` runs the forward recursion and stores the filtered
paths plus the exact log-likelihood; ``smooth()`` adds the Rauch-Tung-Striebel backward
pass. The nonlinear variants (EKF, UKF) accept arbitrary transition/observation maps;
``ParticleFilter`` is a bootstrap SIR sampler; ``RaoBlackwellFilter`` is a GPB(1)-style
conditionally-Gaussian mixture over discrete modes (documented approximation).
"""

from dataclasses import dataclass

import numpy as np

from stochpylib.timeseries._result import ForecastResult


def _mat(M, r, c, name):
    M = np.asarray(M, dtype=float)
    if M.shape != (r, c):
        raise ValueError(f"{name} must have shape ({r}, {c}), got {M.shape}")
    if not np.all(np.isfinite(M)):
        raise ValueError(f"{name} contains non-finite values")
    return M


def _vec(v, k, name):
    v = np.asarray(v, dtype=float).ravel()
    if v.size != k or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must have length {k}")
    return v


@dataclass
class FilterPath:
    """Filtered/smoothed paths from a Kalman-type estimator."""

    means: np.ndarray            # (T, k)
    covs: np.ndarray             # (T, k, k)
    loglik: float = float("nan")
    smoothed_means: np.ndarray | None = None
    smoothed_covs: np.ndarray | None = None


class StateSpaceModel:
    """Container/validator for a linear-Gaussian state-space system."""

    def __init__(self, F, H, Q, R, x0=None, P0=None):
        F = _mat(np.asarray(F, dtype=float), *(np.asarray(F).shape), "F")
        if F.shape[0] != F.shape[1]:
            raise ValueError("F must be square")
        k = F.shape[0]
        self.F = F
        H_arr = np.asarray(H, dtype=float)
        if H_arr.ndim == 1:
            H_arr = H_arr[None, :]
        m = H_arr.shape[0]
        self.H = _mat(H_arr, m, k, "H")
        self.Q = _mat(Q, k, k, "Q")
        self.R = _mat(R, m, m, "R")
        self.k, self.m = k, m
        self.x0 = _vec(x0, k, "x0") if x0 is not None else np.zeros(k)
        self.P0 = _mat(P0, k, k, "P0") if P0 is not None else 10.0 * np.eye(k)

    def _validate_obs(self, observations):
        obs = np.asarray(observations, dtype=float)
        if obs.ndim == 1:
            obs = obs[:, None]
        if obs.shape[0] == 0 or obs.shape[1] != self.m:
            raise ValueError(f"observations must be (T, {self.m})")
        return obs


class KalmanFilter(StateSpaceModel):
    """Exact Kalman filter for the linear-Gaussian model.

    ``fit(observations)`` stores filtered/predicted paths, innovation covariances and
    the exact Gaussian log-likelihood. ``smooth()`` runs the RTS backward pass.
    """

    def fit(self, observations):
        obs = self._validate_obs(observations)
        T = len(obs)
        k = self.k
        I_k = np.eye(k)

        x_pred = np.empty((T + 1, k))
        P_pred = np.empty((T + 1, k, k))
        x_filt = np.empty((T, k))
        P_filt = np.empty((T, k, k))
        innov = np.empty((T, self.m))
        S_hist = np.empty((T, self.m, self.m))

        x, P = self.x0.copy(), self.P0.copy()
        loglik = 0.0
        for t in range(T):
            if t > 0:
                x = self.F @ x_filt[t - 1]
                P = self.F @ P_filt[t - 1] @ self.F.T + self.Q
            x_pred[t], P_pred[t] = x, P
            S = self.H @ P @ self.H.T + self.R
            K = P @ self.H.T @ np.linalg.inv(S)
            innov_t = obs[t] - self.H @ x
            x = x + K @ innov_t
            P = (I_k - K @ self.H) @ P
            P = 0.5 * (P + P.T)  # numerical symmetry
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:
                raise np.linalg.LinAlgError("innovation covariance not positive definite")
            loglik += -0.5 * (self.m * np.log(2 * np.pi) + logdet + innov_t @ np.linalg.solve(S, innov_t))
            x_filt[t], P_filt[t] = x, P
            innov[t], S_hist[t] = innov_t, S

        x_pred[T], P_pred[T] = self.F @ x_filt[-1], self.F @ P_filt[-1] @ self.F.T + self.Q
        self.filtered_means_, self.filtered_covs_ = x_filt, P_filt
        self.predicted_means_, self.predicted_covs_ = x_pred[:T], P_pred[:T]
        self.innovations_, self.innovation_covs_ = innov, S_hist
        self.loglik_ = float(loglik)
        self._obs = obs
        self._pred_next = (x_pred[T], P_pred[T])
        return self

    def filter(self):
        """Filtered state means ``(T, k)``."""
        if not hasattr(self, "filtered_means_"):
            raise RuntimeError("fit() must be called first")
        return self.filtered_means_

    def residuals(self):
        """One-step-ahead observation innovations."""
        if not hasattr(self, "innovations_"):
            raise RuntimeError("fit() must be called first")
        return self.innovations_

    def smooth(self):
        """Rauch-Tung-Striebel backward pass; returns :class:`FilterPath`."""
        if not hasattr(self, "filtered_means_"):
            raise RuntimeError("fit() must be called first")
        return KalmanSmoother().apply(self)

    def apply(self):
        """Compatibility alias for :meth:`fit` chains in pipelines."""
        return self

    def forecast(self, horizon=1):
        """Latent-state and observation forecasts after filtering."""
        if not hasattr(self, "_pred_next"):
            raise RuntimeError("fit() must be called first")
        horizon = int(horizon)
        x, P = self._pred_next
        xs = []
        Ps = []
        for _ in range(horizon):
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q
            xs.append(x.copy())
            Ps.append(P.copy())
        xs_arr, Ps_arr = np.array(xs), np.array(Ps)
        ys = xs_arr @ self.H.T
        ys_std = np.array([
            np.sqrt(np.clip(np.diag(self.H @ Ps_arr[h] @ self.H.T + self.R), 0, None))
            for h in range(horizon)
        ])
        return ForecastResult(ys, ys_std)


class KalmanSmoother:
    """Rauch-Tung-Striebel fixed-interval smoother."""

    def apply(self, kf: KalmanFilter) -> FilterPath:
        F = kf.F
        x_smooth = np.empty_like(kf.filtered_means_)
        P_smooth = np.empty_like(kf.filtered_covs_)
        x_smooth[-1], P_smooth[-1] = kf.filtered_means_[-1], kf.filtered_covs_[-1]
        for t in range(len(x_smooth) - 2, -1, -1):
            P_pred = F @ kf.filtered_covs_[t] @ F.T + kf.Q
            G = kf.filtered_covs_[t] @ F.T @ np.linalg.inv(P_pred)
            x_smooth[t] = kf.filtered_means_[t] + G @ (x_smooth[t + 1] - F @ kf.filtered_means_[t])
            P_smooth[t] = kf.filtered_covs_[t] + G @ (P_smooth[t + 1] - P_pred) @ G.T
            P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)
        return FilterPath(means=x_smooth, covs=P_smooth, loglik=kf.loglik_,
                          smoothed_means=x_smooth, smoothed_covs=P_smooth)


# --------------------------------------------------------------------------- EKF / UKF


def _numeric_jacobian(f, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    f0 = np.atleast_1d(f(x))
    J = np.empty((len(f0), len(x)))
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = eps * max(1.0, abs(x[i]))
        J[:, i] = (np.atleast_1d(f(x + dx)) - np.atleast_1d(f(x - dx))) / (2 * dx[i])
    return J


class ExtendedKalmanFilter(StateSpaceModel):
    """EKF for nonlinear x' = f(x), y = h(x) with additive Gaussian noise.

    Jacobians default to central finite differences.
    """

    def __init__(self, f, h, Q, R, x0, P0, jf=None, jh=None):
        self.f_fn, self.h_fn = f, h
        x0 = np.asarray(x0, dtype=float).ravel()
        k = len(x0)
        probe_h = np.atleast_1d(h(x0))
        m = len(probe_h)
        Q = _mat(Q, k, k, "Q")
        R = _mat(R, m, m, "R")
        self.k, self.m = k, m
        self.Q, self.R = Q, R
        self.x0, self.P0 = x0, _mat(P0, k, k, "P0")
        self.jf_fn = jf or (lambda x: _numeric_jacobian(f, x))
        self.jh_fn = jh or (lambda x: _numeric_jacobian(h, x))

    def _validate_obs(self, observations):
        obs = np.asarray(observations, dtype=float)
        if obs.ndim == 1:
            obs = obs[:, None]
        return obs

    def fit(self, observations):
        obs = self._validate_obs(observations)
        T = len(obs)
        k = self.k
        I_k = np.eye(k)
        x, P = self.x0.copy(), self.P0.copy()
        xf = np.empty((T, k)); Pf = np.empty((T, k, k))
        loglik = 0.0
        for t in range(T):
            F = self.jf_fn(x)
            x = self.f_fn(x)
            P = F @ P @ F.T + self.Q
            H = self.jh_fn(x)
            S = H @ P @ H.T + self.R
            K = P @ H.T @ np.linalg.inv(S)
            innov = obs[t] - np.atleast_1d(self.h_fn(x))
            x = x + K @ innov
            P = (I_k - K @ H) @ P
            P = 0.5 * (P + P.T)
            sign, logdet = np.linalg.slogdet(S)
            loglik += -0.5 * (self.m * np.log(2 * np.pi) + logdet + innov @ np.linalg.solve(S, innov))
            xf[t], Pf[t] = x, P
        self.filtered_means_, self.filtered_covs_ = xf, Pf
        self.loglik_ = float(loglik)
        return self

    def filter(self):
        return self.filtered_means_

    def smooth(self):
        raise NotImplementedError("EKF smoothing requires iterated methods; not implemented")


class UnscentedKalmanFilter(StateSpaceModel):
    """Scaled unscented transform KF for nonlinear x' = f(x), y = h(x).

    Bypasses the linear container: only Q/R/x0/P0 and dimension bookkeeping come from
    the base class; prediction/update go through the sigma-point transform.
    """

    def __init__(self, f, h, Q, R, x0, P0, alpha=1e-3, beta=2.0, kappa=0.0):
        x0 = np.asarray(x0, dtype=float).ravel()
        k = len(x0)
        m = len(np.atleast_1d(h(x0)))
        self.k, self.m = k, m
        self.Q = _mat(Q, k, k, "Q")
        self.R = _mat(R, m, m, "R")
        self.x0, self.P0 = x0, _mat(P0, k, k, "P0")
        self.f_fn, self.h_fn = f, h
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        self._lambda = alpha**2 * (k + kappa) - k
        w_m = np.full(2 * k + 1, 1.0 / (2.0 * (k + self._lambda)))
        w_c = w_m.copy()
        w_m[0] = self._lambda / (k + self._lambda)
        w_c[0] = w_m[0] + beta
        self.w_m, self.w_c = w_m, w_c

    def _validate_obs(self, observations):
        obs = np.asarray(observations, dtype=float)
        if obs.ndim == 1:
            obs = obs[:, None]
        if obs.shape[1] != self.m:
            raise ValueError(f"observations must have {self.m} columns")
        return obs

    def _sigma_points(self, x, P):
        S = np.linalg.cholesky((self.k + self._lambda) * P)
        pts = [x]
        for i in range(self.k):
            pts.append(x + S[:, i])
            pts.append(x - S[:, i])
        return np.array(pts)

    def fit(self, observations):
        obs = self._validate_obs(observations)
        T = len(obs)
        k = self.k
        x, P = self.x0.copy(), self.P0.copy()
        xf = np.empty((T, k)); Pf = np.empty((T, k, k))
        loglik = 0.0
        I_k = np.eye(k)
        for t in range(T):
            sig = self._sigma_points(x, P)
            sig_f = np.array([self.f_fn(s) for s in sig])
            x_pred = self.w_m @ sig_f
            d = sig_f - x_pred
            P_pred = self.Q + sum(self.w_c[i] * np.outer(d[i], d[i]) for i in range(2 * k + 1))

            sig_p = self._sigma_points(x_pred, P_pred)
            sig_h = np.array([np.atleast_1d(self.h_fn(s)) for s in sig_p])
            y_pred = self.w_m @ sig_h
            dy = sig_h - y_pred
            S = self.R + sum(self.w_c[i] * np.outer(dy[i], dy[i]) for i in range(2 * k + 1))
            Cxy = sum(self.w_c[i] * np.outer(sig_p[i] - x_pred, dy[i]) for i in range(2 * k + 1))
            K = Cxy @ np.linalg.inv(S)
            innov = obs[t] - y_pred
            x = x_pred + K @ innov
            P = P_pred - K @ S @ K.T
            P = 0.5 * (P + P.T)
            sign, logdet = np.linalg.slogdet(S)
            loglik += -0.5 * (self.m * np.log(2 * np.pi) + logdet + innov @ np.linalg.solve(S, innov))
            xf[t], Pf[t] = x, P
        self.filtered_means_, self.filtered_covs_ = xf, Pf
        self.loglik_ = float(loglik)
        return self

    def filter(self):
        return self.filtered_means_

    def smooth(self):
        raise NotImplementedError("UKF smoothing (UKS) not implemented")


# --------------------------------------------------------------------------- particle


class ParticleFilter:
    """Bootstrap sequential-importance-resampling particle filter.

    Constructor callables:
      ``transition_sampler(particles, rng) -> new particles``
      ``observation_logpdf(particles, y) -> log p(y | x)`` (vectorized)
      ``initial_sampler(rng) -> initial particle array``
    """

    def __init__(self, n_particles=2000, transition_sampler=None, observation_logpdf=None,
                 initial_sampler=None, resample_threshold=0.5, random_state=None):
        if transition_sampler is None or observation_logpdf is None or initial_sampler is None:
            raise ValueError("transition_sampler, observation_logpdf and initial_sampler are required")
        self.n = int(n_particles)
        self.transition_sampler = transition_sampler
        self.observation_logpdf = observation_logpdf
        self.initial_sampler = initial_sampler
        self.resample_threshold = float(resample_threshold)
        self.rng = np.random.default_rng(random_state)

    def fit(self, observations):
        obs = np.asarray(observations, dtype=float).ravel()
        T = len(obs)
        particles = np.asarray(self.initial_sampler(self.rng), dtype=float)
        if particles.ndim == 0:
            particles = particles[None]
        elif particles.ndim == 1:
            particles = particles[:, None]
        log_w = np.full(len(particles), -np.log(len(particles)))
        means = np.empty((T, particles.shape[1]))
        ess_path = np.empty(T)
        loglik = 0.0
        for t in range(T):
            particles = np.asarray(self.transition_sampler(particles, self.rng), dtype=float)
            if particles.ndim == 1:
                particles = particles[:, None]
            # user callables may return (n,), (n,1), or scalars — flatten defensively
            obs_ll = np.asarray(
                self.observation_logpdf(particles, obs[t]), dtype=float
            ).reshape(-1)
            lw = (log_w + obs_ll).ravel()
            lw -= lw.max()
            w = np.exp(lw)
            total = w.sum()
            loglik += np.log(total) + lw.max() if total > 0 else -np.inf
            w /= total
            means[t] = w @ particles
            ess = 1.0 / np.sum(w**2)
            ess_path[t] = ess
            if ess < self.resample_threshold * len(particles):
                idx = self._systematic_resample(w)
                particles = particles[idx]
                log_w = np.full(len(particles), -np.log(len(particles)))
            else:
                log_w = np.log(w + 1e-300)
        self.particles_last_ = particles
        self.weights_last_ = w
        self.filtered_means_ = means
        self.effective_sample_sizes_ = ess_path
        self.loglik_ = float(loglik)
        return self

    def _systematic_resample(self, w):
        n = len(w)
        positions = (self.rng.uniform() + np.arange(n)) / n
        return np.searchsorted(np.cumsum(w), positions)

    def filter(self):
        return self.filtered_means_

    def smooth(self):
        raise NotImplementedError("particle smoothing (FFBS) not implemented")


# --------------------------------------------------------------------------- Rao-Blackwell


class RaoBlackwellFilter:
    """GPB(1)-style conditionally-Gaussian mixture filter over discrete modes.

    Each mode carries its own linear-Gaussian dynamics; per-mode Kalman predictions are
    combined through exact mode likelihoods, and the reported state estimate is the
    probability-weighted collapse of the mode Gaussians (documented approximation —
    full GPB merges mode histories rather than collapsing them every step).
    """

    def __init__(self, modes, initial_weights=None):
        self.modes = []
        for spec in modes:
            self.modes.append({
                "F": np.asarray(spec["F"], dtype=float),
                "H": np.atleast_2d(np.asarray(spec["H"], dtype=float)),
                "Q": np.asarray(spec["Q"], dtype=float),
                "R": np.asarray(spec["R"], dtype=float),
            })
        k = self.modes[0]["F"].shape[0]
        self.M = len(self.modes)
        for spec in self.modes:
            _mat(spec["F"], k, k, "mode F")
            _mat(spec["H"], spec["H"].shape[0], k, "mode H")
            _mat(spec["Q"], k, k, "mode Q")
            _mat(spec["R"], spec["H"].shape[0], spec["H"].shape[0], "mode R")
        self.initial_weights = (
            np.asarray(initial_weights, dtype=float) if initial_weights is not None
            else np.full(self.M, 1.0 / self.M)
        )

    def fit(self, observations):
        obs = np.asarray(observations, dtype=float)
        if obs.ndim == 1:
            obs = obs[:, None]
        T = len(obs)
        k = self.modes[0]["F"].shape[0]
        mus = [np.zeros(k) for _ in range(self.M)]
        Ps = [10.0 * np.eye(k) for _ in range(self.M)]
        w = self.initial_weights.copy()

        means = np.empty((T, k))
        weights_path = np.empty((T, self.M))
        loglik = 0.0
        I_k = np.eye(k)
        for t in range(T):
            new_mus, new_Ps, liks = [], [], []
            for m_spec, mu_prev, P_prev in zip(self.modes, mus, Ps):
                F, H, Q, R = m_spec["F"], m_spec["H"], m_spec["Q"], m_spec["R"]
                mu_pred = F @ mu_prev
                P_pred = F @ P_prev @ F.T + Q
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T @ np.linalg.inv(S)
                innov = obs[t] - H @ mu_pred
                mu_new = mu_pred + K @ innov
                P_new = (I_k - K @ H) @ P_pred
                P_new = 0.5 * (P_new + P_new.T)
                sign, logdet = np.linalg.slogdet(S)
                ll = -0.5 * (obs.shape[1] * np.log(2 * np.pi) + logdet
                             + innov @ np.linalg.solve(S, innov))
                new_mus.append(mu_new)
                new_Ps.append(P_new)
                liks.append(ll)
            liks = np.asarray(liks)
            log_w = np.log(w + 1e-300) + liks - liks.max()
            w = np.exp(log_w)
            w_sum = w.sum()
            w /= w_sum
            loglik += float(np.log(w_sum) + liks.max())
            mus, Ps = new_mus, new_Ps
            means[t] = sum(wi * mi for wi, mi in zip(w, mus))
            weights_path[t] = w
        self.mode_weights_ = weights_path
        self.filtered_means_ = means
        self.loglik_ = float(loglik)
        return self

    def filter(self):
        return self.filtered_means_

    def smooth(self):
        raise NotImplementedError("RaoBlackwell smoothing not implemented")
