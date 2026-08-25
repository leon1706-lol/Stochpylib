"""Shared conventions for information-theoretic measures."""

import numpy as np

__all__ = ["_validate_probs", "_normalise", "_joint_table"]


def _validate_probs(p, q=None):
    """Validate probability vector(s): finite, non-negative, sums to 1."""
    p = np.asarray(p, dtype=float).ravel()
    if not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite")
    if np.any(p < 0):
        raise ValueError("probabilities must be non-negative")
    total = p.sum()
    if abs(total - 1.0) > 1e-6:
        p = p / total  # auto-normalise
    if q is not None:
        q = np.asarray(q, dtype=float).ravel()
        if q.shape != p.shape:
            raise ValueError(f"p and q must have same shape: {p.shape} vs {q.shape}")
        if np.any(q < 0):
            raise ValueError("q probabilities must be non-negative")
        qt = q.sum()
        if abs(qt - 1.0) > 1e-6:
            q = q / qt
    return p, q


def _normalise(v):
    """Normalise count vector to a probability distribution."""
    v = np.asarray(v, dtype=float)
    return v / max(v.sum(), 1e-300)


def _joint_table(x, y, bins=0):
    """Build joint contingency table from paired observations.

    If ``bins`` > 0 and data is continuous, discretise into equal-width bins.
    Otherwise assume discrete/categorical labels.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    if bins > 0:
        # continuous -> discretise
        xe = np.histogram_bin_edges(x, bins=bins)
        ye = np.histogram_bin_edges(y, bins=bins)
        xi = np.clip(np.searchsorted(xe[1:], x), 0, bins - 1)
        yi = np.clip(np.searchsorted(ye[1:], y), 0, bins - 1)
    else:
        # discrete/categorical
        ux = np.unique(x)
        uy = np.unique(y)
        xi = np.searchsorted(ux, x)
        yi = np.searchsorted(uy, y)
    table = np.zeros((xi.max() + 1, yi.max() + 1))
    np.add.at(table, (xi, yi), 1.0)
    return _normalise(table.ravel()), table


def _safe_log2(x):
    return np.log2(np.maximum(np.asarray(x, float), 1e-300))


def _safe_log(x):
    return np.log(np.maximum(np.asarray(x, float), 1e-300))
