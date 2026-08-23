"""Hyperparameter optimization and model selection for Gaussian processes.

Optimization runs L-BFGS-B on the negative log-marginal likelihood in **log-space**
(positive parameters are optimized as their logarithms), walking the kernel tree through
the ``get_params``/``set_params`` interface. ``ARD`` is a tiny initializer marking a
length scale as per-dimension.
"""

import numpy as np
from scipy import optimize

from stochpylib.gaussian_processes._utils import _as_2d

__all__ = ["ARD", "MarginalLikelihood", "optimize_hyperparams", "cross_validate_gp"]


def ARD(dimensions):
    """Initialize an ARD length-scale vector of ones (per-dimension relevance)."""
    return np.ones(int(dimensions))


class MarginalLikelihood:
    """Callable computing the negative log-marginal likelihood for a param dict."""

    def __init__(self, inference):
        """``inference`` is any fitted-able object with kernel/noise/fit/LML access."""
        self.inference = inference

    def value(self, params):
        try:
            self.inference.kernel.set_params(params)
            self.inference.fit(self.inference.X_train, self.inference.y_train)
            ll = self.inference.log_marginal_likelihood()
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 1e12
        return float(-ll) if np.isfinite(ll) else 1e12

    def _lml(self, params):
        try:
            self.inference.kernel.set_params(params)
            self.inference.fit(self.inference.X_train, self.inference.y_train)
            return float(self.inference.log_marginal_likelihood())
        except Exception:
            return -1e12

    __call__ = value


def optimize_hyperparams(inference, maxiter=200, verbose=False):
    """Maximize the log-marginal likelihood over the kernel hyperparameters (log-space).

    ``inference`` must have been fit already (so training data is attached). Returns a
    dict with the optimized parameter dict, the achieved LML and optimizer status.
    """
    kernel = inference.kernel
    start_params = {k: np.atleast_1d(v).astype(float) for k, v in kernel.get_params().items()}

    # pack: positive scalars -> log; vectors -> elementwise log; others skipped
    names = []
    theta0 = []
    for name, value in start_params.items():
        arr = np.atleast_1d(value)
        if np.any(arr <= 0):
            continue  # non-positive parameters (e.g. constants) are left fixed
        names.append((name, arr.size))
        theta0.extend(np.log(arr))
    theta0 = np.asarray(theta0)

    def unpack(theta):
        out = {}
        pos = 0
        for name, size in names:
            out[name] = np.exp(theta[pos : pos + size]) if size > 1 else \
                float(np.exp(theta[pos]))
            pos += size
        return out

    lml = MarginalLikelihood(inference)

    def objective(theta):
        return lml.value(unpack(theta))

    result = optimize.minimize(
        lambda th: objective(th),
        theta0,
        method="L-BFGS-B",
        options={"maxiter": int(maxiter)},
    )
    best_theta = result.x if objective(result.x) <= objective(theta0) else theta0
    final_params = unpack(best_theta)
    kernel.set_params(final_params)
    inference.fit(inference.X_train, inference.y_train)
    if verbose:
        print(f"optimize_hyperparams: LML {inference.log_marginal_likelihood_:.4f}, "
              f"params={final_params}")
    return {
        "params": kernel.get_params(),
        "log_marginal_likelihood": inference.log_marginal_likelihood_,
        "success": bool(result.success or objective(result.x) <= objective(theta0)),
        "n_iterations": int(getattr(result, "nit", -1)),
    }


def cross_validate_gp(X, y, model_factory, k=5):
    """Rolling-origin cross-validation returning per-fold RMSE and mean NLPD."""
    X = _as_2d(X)
    y = np.asarray(y, dtype=float).ravel()
    T = len(y)
    if k < 2:
        raise ValueError("k must be >= 2")
    min_train = max(int(T * 0.3), 30)
    block = (T - min_train) // k
    if block < 1:
        raise ValueError("not enough data for the requested number of folds")
    folds = []
    for i in range(k):
        train_end = min_train + i * block
        test_idx = np.arange(train_end, min(train_end + block, T))
        model = model_factory()
        model.fit(X[:train_end], y[:train_end])
        mu, sd = model.predict(X[test_idx], return_std=True)
        resid = mu - y[test_idx]
        nlpd = float(np.mean(
            0.5 * np.log(2 * np.pi * sd**2) + (y[test_idx] - mu) ** 2 / (2 * sd**2)
        ))
        folds.append({
            "fold": i,
            "train_end": int(train_end),
            "rmse": float(np.sqrt(np.mean(resid**2))),
            "mean_nlpd": nlpd,
        })
    return {
        "folds": folds,
        "mean_rmse": float(np.mean([f["rmse"] for f in folds])),
        "mean_nlpd": float(np.mean([f["mean_nlpd"] for f in folds])),
    }
