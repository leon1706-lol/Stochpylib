"""Walk-forward evaluation and forecast-interval utilities.

These are the honest way to score a time-series model: refit on a growing history,
predict the unseen next steps, and aggregate out-of-sample errors by horizon.
"""

from dataclasses import dataclass

import numpy as np

from stochpylib.timeseries._result import ForecastResult
from stochpylib.timeseries._utils import as_1d


# --------------------------------------------------------------------------- dispatchers


def forecast(model, horizon=10):
    """Dispatch to ``model.forecast(horizon)`` for any fitted timeseries model."""
    if hasattr(model, "forecast"):
        return model.forecast(horizon)
    raise TypeError(f"{type(model).__name__} does not implement .forecast(horizon)")


def predict(model, horizon=10):
    """Alias of :func:`forecast` (both names are part of the public spec)."""
    return forecast(model, horizon)


def confidence_bands(result, level=0.95):
    """Normal-approximation prediction bands ``(low, high)`` for a forecast.

    Accepts a :class:`ForecastResult` or a ``(mean, std)`` pair of equal-length arrays.
    """
    if isinstance(result, ForecastResult):
        low, high = result.confidence_interval(level)
        return low, high
    mean, std = result
    from scipy import optimize, special

    if level == 0.95:
        z = 1.959963984540054
    else:
        alpha2 = (1.0 - level) / 2.0
        z = float(optimize.brentq(lambda q: 2.0 * special.ndtr(q) - 1.0 - 2.0 * alpha2, 0.0, 50.0))
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    return mean - z * std, mean + z * std


# --------------------------------------------------------------------------- backtesting


@dataclass
class BacktestResult:
    """Aggregated walk-forward errors by horizon."""

    horizons: list
    rmse: list
    mae: list
    n_windows: int

    def __repr__(self):
        rows = ", ".join(
            f"h{h}: rmse={r:.4g}/mae={m:.4g}" for h, r, m in zip(self.horizons, self.rmse, self.mae)
        )
        return f"BacktestResult(n={self.n_windows}; {rows})"


def _walk_forward_errors(y, model_factory, horizon, origins, window=None):
    errors = {h: [] for h in range(1, horizon + 1)}
    n_done = 0
    for o in origins:
        train = y[:o] if window is None else y[o - window : o]
        try:
            model = model_factory()
            model.fit(train)
            fc = model.forecast(horizon)
        except Exception:
            continue  # a degenerate training window must not kill the whole backtest
        means = np.atleast_1d(np.asarray(fc.mean, dtype=float))
        for h in range(1, horizon + 1):
            if o + h - 1 < len(y):
                errors[h].append(means[h - 1] - y[o + h - 1])
        n_done += 1
    return errors, n_done


def backtesting(y, model_factory, horizon=1, min_train=None, expanding=True, step=1):
    """Expanding- (or rolling-) window walk-forward evaluation.

    ``model_factory`` is a zero-argument callable returning an unfitted model
    (``lambda: AR(2)``). At every origin ``o`` (spaced by ``step``), the model is fit on
    everything up to ``o`` (expanding) or the last ``min_train`` points (rolling), then
    forecasts ``horizon`` steps ahead; errors are aggregated per horizon as RMSE/MAE.
    """
    y = as_1d(y)
    T = len(y)
    horizon = int(horizon)
    step = max(int(step), 1)
    if min_train is None:
        min_train = max(30, T // 4)
    min_train = int(min_train)
    if min_train + horizon > T:
        raise ValueError("not enough data: need min_train + horizon <= len(y)")
    origins = range(min_train, T - horizon + 1, step)
    errors, n_done = _walk_forward_errors(y, model_factory, horizon, origins,
                                          window=None if expanding else min_train)
    horizons = sorted(errors)
    rmse = [float(np.sqrt(np.mean(np.square(errors[h])))) if errors[h] else float("nan") for h in horizons]
    mae = [float(np.mean(np.abs(errors[h]))) if errors[h] else float("nan") for h in horizons]
    return BacktestResult(horizons=horizons, rmse=rmse, mae=mae, n_windows=n_done)


def cross_validation_ts(y, model_factory, n_splits=5, horizon=1, min_train=None):
    """Rolling-origin cross-validation with ``n_splits`` contiguous test blocks.

    Returns one record per fold::

        [{"fold": 0, "train_end": o, "test_start": ..., "rmse": ..., "mae": ...}, ...]
    """
    y = as_1d(y)
    T = len(y)
    horizon = int(horizon)
    n_splits = int(n_splits)
    if min_train is None:
        min_train = max(30, T // 4)
    min_train = int(min_train)
    block = max((T - min_train) // n_splits, 1)
    folds = []
    for i in range(n_splits):
        start = min_train + i * block          # first one-step-ahead test index
        stop = start + block                    # exclusive
        errors, preds, actuals = [], [], []
        for t in range(start, min(stop, T)):
            model = model_factory()
            model.fit(y[:t])
            fc = np.atleast_1d(np.asarray(model.forecast(horizon).mean, dtype=float))
            preds.append(float(fc[0]))
            actuals.append(float(y[t]))
            errors.append(float(fc[0]) - float(y[t]))
        errs = np.asarray(errors)
        folds.append({
            "fold": i,
            "train_end": start,
            "test_points": len(errs),
            "rmse": float(np.sqrt(np.mean(errs**2))),
            "mae": float(np.mean(np.abs(errs))),
            "predictions": np.asarray(preds),
            "actuals": np.asarray(actuals),
        })
    return folds
