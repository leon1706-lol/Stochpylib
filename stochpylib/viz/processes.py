"""Time-series and stochastic-process plots: sample paths, ACF/PACF, spectral views,
and phase portraits.
"""

import numpy as np

from stochpylib.viz._common import _as_1d, _durbin_levinson, _get_fig_ax

__all__ = ["plot_process", "plot_acf", "plot_pacf", "plot_periodogram", "plot_spectrogram",
           "plot_wavelet", "plot_trajectory"]


def plot_process(paths, t=None, max_paths=50, band=(0.05, 0.95), ax=None, title=None,
                 color=None, label=None, random_state=None):
    """Sample paths of a stochastic process, with a mean line and quantile band when
    more than one path is given.

    Accepts a 1-D path, a ``(n_paths, n_steps)`` array, or a ``(t, paths)`` tuple.
    """
    fig, axes = _get_fig_ax(ax)
    if isinstance(paths, tuple) and len(paths) == 2:
        t, paths = paths
    paths = np.asarray(paths, dtype=float)
    if paths.ndim == 1:
        paths = paths[None, :]
    if paths.ndim != 2 or paths.size == 0:
        raise ValueError("paths must be 1-D or a (n_paths, n_steps) array")
    n_paths, n_steps = paths.shape
    t = np.asarray(t, dtype=float) if t is not None else np.arange(n_steps)
    if t.shape[0] != n_steps:
        raise ValueError("t must have one entry per step")

    rng = np.random.default_rng(random_state)
    if n_paths > max_paths:
        idx = rng.choice(n_paths, size=max_paths, replace=False)
        idx.sort()
    else:
        idx = np.arange(n_paths)
    for i in idx:
        axes.line(t, paths[i], color=color or "#4E79A7", alpha=0.4 if n_paths > 1 else 1.0,
                  label=label if i == idx[0] and n_paths == 1 else None)

    out = {"t": t, "paths": paths}
    if n_paths > 1:
        mean = paths.mean(axis=0)
        lower = np.quantile(paths, band[0], axis=0)
        upper = np.quantile(paths, band[1], axis=0)
        axes.line(t, mean, color="#E15759", width=2.0, label="mean")
        axes.band(t, lower, upper, color="#E15759", label=f"{band[0]:.0%}-{band[1]:.0%}")
        out.update(mean=mean, lower=lower, upper=upper)
        axes.set(legend="best")
    axes.set(title=title, xlabel="t", ylabel="x(t)")
    fig.data = out
    return fig


def _acf_from_data(x, nlags):
    from stochpylib.advanced_mcmc.diagnostics import _autocov_fft

    acov = _autocov_fft(x)
    acf = acov[:nlags + 1] / acov[0]
    return acf


def plot_acf(x, nlags=None, alpha=0.05, ax=None, title=None, color=None):
    """Sample autocorrelation function with a Bartlett confidence band."""
    from stochpylib.statistics._common import _norm_ppf

    fig, axes = _get_fig_ax(ax)
    x = _as_1d(x, "x")
    n = len(x)
    if nlags is None:
        nlags = min(int(10 * np.log10(max(n, 2))), n - 1)
    nlags = max(int(nlags), 1)
    acf = _acf_from_data(x, nlags)
    z = _norm_ppf(1.0 - alpha / 2.0)
    varacf = np.ones(nlags + 1) / n
    varacf[0] = 0.0
    if nlags >= 1:
        varacf[1] = 1.0 / n
    for k in range(2, nlags + 1):
        varacf[k] = (1.0 + 2.0 * np.sum(acf[1:k] ** 2)) / n
    band = z * np.sqrt(varacf)
    lags = np.arange(nlags + 1)
    axes.stem(lags, acf, color=color)
    axes.line(lags, band, color="#999999", dash="dash", label=f"{int((1 - alpha) * 100)}% band")
    axes.line(lags, -band, color="#999999", dash="dash")
    axes.set(title=title, xlabel="lag", ylabel="ACF")
    fig.data = {"lags": lags, "acf": acf, "band": band}
    return fig


def plot_pacf(x, nlags=None, alpha=0.05, ax=None, title=None, color=None):
    """Sample partial autocorrelation function (Durbin-Levinson) with a confidence band."""
    from stochpylib.statistics._common import _norm_ppf

    fig, axes = _get_fig_ax(ax)
    x = _as_1d(x, "x")
    n = len(x)
    if nlags is None:
        nlags = min(int(10 * np.log10(max(n, 2))), n - 1)
    nlags = max(int(nlags), 1)
    acf = _acf_from_data(x, nlags)
    pacf = _durbin_levinson(acf)
    z = _norm_ppf(1.0 - alpha / 2.0)
    band_val = z / np.sqrt(n)
    lags = np.arange(nlags + 1)
    axes.stem(lags, pacf, color=color)
    axes.hline(band_val, dash="dash", color="#999999", label=f"{int((1 - alpha) * 100)}% band")
    axes.hline(-band_val, dash="dash", color="#999999")
    axes.set(title=title, xlabel="lag", ylabel="PACF")
    fig.data = {"lags": lags, "pacf": pacf, "band": np.full(nlags + 1, band_val)}
    return fig


def plot_periodogram(x, fs=1.0, method="periodogram", log=True, nperseg=256, ax=None,
                     title=None, color=None):
    """Power spectrum of a series: raw periodogram or Welch-averaged."""
    from stochpylib.timeseries.spectral import Periodogram, PowerSpectrum

    fig, axes = _get_fig_ax(ax)
    x = _as_1d(x, "x")
    if method == "periodogram":
        freqs, power = Periodogram(x, fs=fs)
    elif method == "welch":
        freqs, power = PowerSpectrum(x, fs=fs, nperseg=nperseg)
    else:
        raise ValueError("method must be 'periodogram' or 'welch'")
    plot_freqs, plot_power = freqs[1:], power[1:]  # skip DC for a log axis
    if log:
        axes.line(plot_freqs, plot_power, color=color)
        axes.set(yscale="log")
    else:
        axes.line(freqs, power, color=color)
    axes.set(title=title, xlabel="frequency", ylabel="power")
    fig.data = {"freqs": freqs, "power": power}
    return fig


def plot_spectrogram(x, fs=1.0, window_len=256, hop=128, db=True, ax=None, title=None):
    """Time-frequency magnitude via the short-time Fourier transform."""
    from stochpylib.timeseries.spectral import STFT

    fig, axes = _get_fig_ax(ax)
    x = _as_1d(x, "x")
    freqs, times, spec = STFT(x, fs=fs, window_len=window_len, hop=hop)
    mag = 20.0 * np.log10(spec + 1e-300) if db else spec
    if times.size < 2:
        t_edges = np.array([times[0] - 0.5, times[0] + 0.5]) if times.size else np.array([0, 1])
    else:
        dt = times[1] - times[0]
        t_edges = np.concatenate([times - dt / 2, [times[-1] + dt / 2]])
    if freqs.size < 2:
        f_edges = np.array([0.0, 1.0])
    else:
        df = freqs[1] - freqs[0]
        f_edges = np.concatenate([freqs - df / 2, [freqs[-1] + df / 2]])
    axes.heatmap(mag, x_edges=t_edges, y_edges=f_edges, cmap="viridis")
    axes.colorbar = {"cmap": "viridis", "vmin": float(np.min(mag)), "vmax": float(np.max(mag))}
    axes.set(title=title, xlabel="time", ylabel="frequency")
    fig.data = {"times": times, "freqs": freqs, "magnitude": spec}
    return fig


def plot_wavelet(x, scales=None, fs=1.0, kind="cwt", ax=None, title=None):
    """Continuous wavelet scalogram (Morlet), or stacked discrete-wavelet detail levels."""
    from stochpylib.timeseries.spectral import CWTTransform, DWTTransform

    fig, axes = _get_fig_ax(ax)
    x = _as_1d(x, "x")
    if kind == "cwt":
        used_scales, coeffs = CWTTransform(x, scales=scales, fs=fs)
        power = np.abs(coeffs) ** 2
        t = np.arange(len(x)) / fs
        axes.heatmap(power, x_edges=np.concatenate([t - 0.5 / fs, [t[-1] + 0.5 / fs]])
                     if len(t) > 1 else np.array([0.0, 1.0]),
                     y_edges=np.concatenate([used_scales, [used_scales[-1] * 1.1]])
                     if len(used_scales) > 1 else np.array([used_scales[0], used_scales[0] * 2]),
                     cmap="viridis")
        axes.colorbar = {"cmap": "viridis", "vmin": float(power.min()), "vmax": float(power.max())}
        axes.set(title=title, xlabel="time", ylabel="scale", yscale="log")
        fig.data = {"scales": used_scales, "power": power}
        return fig
    if kind == "dwt":
        result = DWTTransform(x)
        details = result["details"]
        n_levels = len(details)
        fig2 = fig if fig.nrows * fig.ncols >= n_levels + 1 else None
        if fig2 is None:
            from stochpylib.viz._figure import Figure

            fig = Figure(nrows=n_levels + 1, ncols=1, height=120 * (n_levels + 1))
        for i, d in enumerate(details):
            fig.axes[i].line(np.arange(len(d)), d, label=f"detail {i + 1}")
            fig.axes[i].set(ylabel=f"D{i + 1}")
        fig.axes[-1].line(np.arange(len(result["approx"])), result["approx"], label="approx",
                          color="#E15759")
        fig.axes[-1].set(ylabel="A", xlabel="index")
        fig.data = {"details": details, "approx": result["approx"]}
        return fig
    raise ValueError("kind must be 'cwt' or 'dwt'")


def plot_trajectory(path, y=None, lag=None, markers=True, ax=None, title=None, color=None):
    """A 2-D trajectory, or a lag-``k`` phase portrait of a 1-D series."""
    fig, axes = _get_fig_ax(ax)
    if y is not None:
        x_arr = _as_1d(path, "path")
        y_arr = _as_1d(y, "y")
        if x_arr.shape != y_arr.shape:
            raise ValueError("path and y must have the same length")
    else:
        arr = np.asarray(path, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            x_arr, y_arr = arr[:, 0], arr[:, 1]
        elif arr.ndim == 1 and lag:
            k = int(lag)
            x_arr, y_arr = arr[:-k], arr[k:]
        else:
            raise ValueError("path must be (n, 2), or 1-D with a lag= given")
    axes.line(x_arr, y_arr, color=color)
    if markers:
        axes.scatter(x_arr[:1], y_arr[:1], color="#59A14F", size=6, label="start")
        axes.scatter(x_arr[-1:], y_arr[-1:], color="#E15759", size=6, label="end")
        axes.set(legend="best")
    axes.set(title=title, xlabel="x(t)" if lag is None else "x(t)",
             ylabel="y(t)" if lag is None else f"x(t+{lag})", aspect="equal")
    fig.data = {"x": x_arr, "y": y_arr}
    return fig
