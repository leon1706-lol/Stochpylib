"""Numerical plumbing for Gaussian processes."""

import numpy as np


def _as_2d(X, name="X"):
    """Coerce input to a (n, d) float array; 1-D input becomes a column."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty (n, d) array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _sqdist(X, Y=None, length_scale=1.0):
    """Pairwise squared distances of ARD-scaled inputs.

    ``length_scale`` may be a scalar (isotropic) or a per-dimension vector (ARD).
    """
    X = _as_2d(X)
    ls = np.atleast_1d(np.asarray(length_scale, dtype=float))
    if ls.size == 1:
        Xs = X / ls[0]
    else:
        if ls.size != X.shape[1]:
            raise ValueError(
                f"ARD length scale has {ls.size} entries but X has {X.shape[1]} dimensions"
            )
        Xs = X / ls[None, :]
    X2 = np.sum(Xs**2, axis=1)[:, None]
    if Y is None:
        Y2 = X2.T                                   # row so broadcasting pairs i vs j
        cross = Xs @ Xs.T
    else:
        Y = _as_2d(Y)
        if ls.size == 1:
            Ys = Y / ls[0]
        else:
            Ys = Y / ls[None, :]
        Y2 = np.sum(Ys**2, axis=1)[None, :]
        cross = Xs @ Ys.T
    D2 = X2 - 2.0 * cross + Y2
    # numerical guard: self-distances must be exactly zero
    return np.maximum(D2, 0.0)


def cholesky_with_jitter(A, max_tries=8):
    """Cholesky with escalating diagonal jitter; returns (L, jitter_used)."""
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    jitter = 0.0
    for i in range(max_tries):
        try:
            return np.linalg.cholesky(A + jitter * np.eye(len(A))), jitter
        except np.linalg.LinAlgError:
            jitter = 10.0 ** (i - 6) if i > 0 else 1e-12
    raise np.linalg.LinAlgError(
        "matrix not positive definite even with jitter "
        f"(max jitter tried: {jitter:.1e}); check kernel hyperparameters and duplicated inputs"
    )
