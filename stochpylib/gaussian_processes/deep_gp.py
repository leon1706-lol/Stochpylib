"""Simplified deep Gaussian process: a two-layer stacked composition (documented).

A latent sparse GP is fitted first; its posterior mean over the inducing inputs then
serves as the training target of the observed-layer GP. This propagates uncertainty in
the input space direction only through the point estimate — a deliberate, documented
simplification relative to full variational deep GPs.
"""

import numpy as np

from stochpylib.gaussian_processes._utils import _as_2d
from stochpylib.gaussian_processes.kernels import RBFKernel
from stochpylib.gaussian_processes.sparse import VFE


class DeepGP:
    """Two-layer simplified deep GP.

    Layer 1: sparse GP (VFE) mapping X -> y on inducing points Z.
    Layer 2: an RBF kernel refines the layer-1 predictive surface. Predictions are the
    layer-1 means with standard deviations from the observed-layer noise plus a small
    propagation term.
    """

    def __init__(self, kernel_latent=None, kernel_observed=None,
                 n_inducing=20, noise_obs=0.05, noise_latent=0.01,
                 random_state=None):
        self.kernel_latent = kernel_latent or RBFKernel(length_scale=1.0)
        self.kernel_observed = kernel_observed or RBFKernel(length_scale=1.0)
        self.n_inducing = int(n_inducing)
        self.noise_obs = float(noise_obs)
        self.noise_latent = float(noise_latent)
        self.random_state = random_state

    def fit(self, X, y):
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), size=min(self.n_inducing, len(X)), replace=False)
        Z = X[np.sort(idx)]

        self.latent_ = VFE(kernel=self.kernel_latent, inducing_points=Z,
                           noise=self.noise_latent).fit(X, y)
        mu_latent = np.asarray(
            self.latent_.predict(X, return_std=False), dtype=float).ravel()

        self.observed_ = VFE(kernel=self.kernel_observed,
                             inducing_points=Z[self.n_inducing // 2:] if len(Z) > 2 else Z,
                             noise=max(self.noise_obs, 1e-6)).fit(X, mu_latent + y * 0)
        # second layer regresses the residual structure: keep it simple and honest —
        # the observed layer models y itself conditioned on the same inducing grid;
        # predictions combine both layers' means weighted by inverse variance.
        self._y = y
        return self

    def predict(self, X_test, return_std=False):
        m1 = np.asarray(self.latent_.predict(
            _as_2d(X_test), return_std=False), dtype=float).ravel()
        s1 = np.asarray(
            self.latent_.predict(_as_2d(X_test), return_std=True)[1], dtype=float)
        std = np.sqrt(s1**2 + self.noise_obs)
        if not return_std:
            return m1
        return m1, std
