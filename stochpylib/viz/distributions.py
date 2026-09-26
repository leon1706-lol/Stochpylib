"""Distribution- and sample-level plots: pdf/pmf/cdf/survival/hazard, QQ/PP-plots,
histograms and KDE.

Every function computes its numbers with the library's own distribution/nonparametric
code (never re-derives them) and returns a :class:`~stochpylib.viz._figure.Figure`.
"""

import numpy as np

from stochpylib.viz._common import (
    _as_1d, _dist_grid, _dist_int_grid, _filliben_pp, _get_fig_ax,
)

__all__ = ["plot_pdf", "plot_pmf", "plot_cdf", "plot_survival", "plot_hazard",
           "plot_qqplot", "plot_ppplot", "plot_histogram", "plot_kde"]


def _is_dist(obj):
    return hasattr(obj, "cdf") and callable(obj.cdf) and hasattr(obj, "pdf")


def plot_pdf(dist, x=None, n_points=200, ax=None, title=None, color=None, label=None):
    """Density (or list of densities, overlaid) of one or more distributions.

    Discrete distributions (``dist.is_discrete``) delegate to :func:`plot_pmf`.
    """
    dists = dist if isinstance(dist, (list, tuple)) else [dist]
    if any(getattr(d, "is_discrete", False) for d in dists):
        if not all(getattr(d, "is_discrete", False) for d in dists):
            raise ValueError("cannot mix discrete and continuous distributions")
        return plot_pmf(dist, k=x, ax=ax, title=title, color=color, label=label)

    fig, axes = _get_fig_ax(ax)
    last_x, last_pdf = None, None
    for i, d in enumerate(dists):
        grid = np.asarray(x, dtype=float) if x is not None else _dist_grid(d, n_points)
        pdf = np.asarray(d.pdf(grid), dtype=float)
        lab = label if (label and len(dists) == 1) else (
            type(d).__name__ if len(dists) > 1 else label)
        axes.line(grid, pdf, color=color if len(dists) == 1 else None, label=lab)
        last_x, last_pdf = grid, pdf
    axes.set(title=title, xlabel="x", ylabel="density", legend="best" if len(dists) > 1 else None)
    fig.data = {"x": last_x, "pdf": last_pdf}
    return fig


def plot_pmf(dist, k=None, ax=None, title=None, color=None, label=None):
    """Stem plot of a discrete distribution's probability mass function."""
    if getattr(dist, "is_discrete", False) is False and not isinstance(dist, (list, tuple)):
        raise ValueError("plot_pmf needs a discrete distribution (dist.is_discrete)")
    fig, axes = _get_fig_ax(ax)
    ks = np.asarray(k, dtype=float) if k is not None else _dist_int_grid(dist)
    pmf = np.asarray(dist.pmf(ks), dtype=float)
    axes.stem(ks, pmf, color=color, label=label)
    axes.set(title=title, xlabel="k", ylabel="probability")
    fig.data = {"k": ks, "pmf": pmf}
    return fig


def plot_cdf(obj, dist=None, confidence=None, n_points=200, ax=None, title=None,
             color=None, label=None):
    """CDF of a distribution, or the empirical CDF of a data sample.

    With sample data, ``confidence`` (e.g. ``0.95``) adds a Dvoretzky-Kiefer-Wolfowitz
    simultaneous confidence band; ``dist`` (a fitted distribution) overlays its CDF.
    """
    fig, axes = _get_fig_ax(ax)
    if _is_dist(obj):
        grid = _dist_int_grid(obj) if getattr(obj, "is_discrete", False) else \
            _dist_grid(obj, n_points)
        cdf = np.asarray(obj.cdf(grid), dtype=float)
        if getattr(obj, "is_discrete", False):
            axes.step(grid, cdf, where="post", color=color, label=label)
        else:
            axes.line(grid, cdf, color=color, label=label)
        axes.set(title=title, xlabel="x", ylabel="F(x)")
        fig.data = {"x": grid, "cdf": cdf}
        return fig

    from stochpylib.nonparametric.empirical import EmpiricalCDF

    x = _as_1d(obj, "data")
    ecdf = EmpiricalCDF().fit(x)
    grid = ecdf.sample_points_
    cdf = ecdf.evaluate(grid)
    axes.step(grid, cdf, where="post", color=color, label=label or "empirical")
    out = {"x": grid, "cdf": cdf}
    if confidence is not None:
        eps = ecdf.confidence_band(confidence)
        lower = np.clip(cdf - eps, 0.0, 1.0)
        upper = np.clip(cdf + eps, 0.0, 1.0)
        axes.band(grid, lower, upper, label=f"{int(confidence * 100)}% DKW band")
        out["lower"], out["upper"] = lower, upper
    if dist is not None:
        dgrid = _dist_grid(dist, n_points)
        axes.line(dgrid, np.asarray(dist.cdf(dgrid), dtype=float), label=type(dist).__name__)
    axes.set(title=title, xlabel="x", ylabel="F(x)", legend="best")
    fig.data = out
    return fig


def plot_survival(dist, x=None, n_points=200, ax=None, title=None, color=None, label=None):
    """Survival function ``S(x) = 1 - F(x)`` (or ``dist.sf(x)`` when available)."""
    fig, axes = _get_fig_ax(ax)
    grid = np.asarray(x, dtype=float) if x is not None else (
        _dist_int_grid(dist) if getattr(dist, "is_discrete", False) else
        _dist_grid(dist, n_points))
    survival = (np.asarray(dist.sf(grid), dtype=float) if hasattr(dist, "sf") else
                1.0 - np.asarray(dist.cdf(grid), dtype=float))
    if getattr(dist, "is_discrete", False):
        axes.step(grid, survival, where="post", color=color, label=label)
    else:
        axes.line(grid, survival, color=color, label=label)
    axes.set(title=title, xlabel="x", ylabel="S(x)")
    fig.data = {"x": grid, "survival": survival}
    return fig


def plot_hazard(dist, x=None, n_points=200, cumulative=False, ax=None, title=None,
                color=None, label=None):
    """Hazard rate ``h(x) = f(x) / S(x)`` (or cumulative hazard ``H(x) = -log S(x)``)."""
    fig, axes = _get_fig_ax(ax)
    grid = np.asarray(x, dtype=float) if x is not None else _dist_grid(dist, n_points)
    pdf = np.asarray(dist.pdf(grid), dtype=float)
    survival = (np.asarray(dist.sf(grid), dtype=float) if hasattr(dist, "sf") else
                1.0 - np.asarray(dist.cdf(grid), dtype=float))
    keep = survival > 1e-12
    grid, pdf, survival = grid[keep], pdf[keep], survival[keep]
    if cumulative:
        y = -np.log(np.clip(survival, 1e-300, None))
        ylabel = "H(x)"
    else:
        y = pdf / survival
        ylabel = "h(x)"
    axes.line(grid, y, color=color, label=label)
    axes.set(title=title, xlabel="x", ylabel=ylabel)
    fig.data = {"x": grid, "hazard": y}
    return fig


def plot_qqplot(data, dist=None, line="fit", ax=None, title=None, color=None, label=None):
    """Q-Q plot of sample vs. theoretical quantiles (Filliben plotting positions).

    ``dist=None`` fits a Normal to ``data``'s mean/std. Also accepts a fitted
    :class:`stochpylib.experimental_design.NormalPlot` (renders its half-normal
    effects plot with margin-of-error reference lines instead).
    """
    fig, axes = _get_fig_ax(ax)
    if hasattr(data, "theoretical_quantiles_") and hasattr(data, "sorted_effects_"):
        labels = [lab for lab, _ in data.sorted_effects_]
        vals = np.array([v for _, v in data.sorted_effects_])
        q = data.theoretical_quantiles_
        axes.scatter(q, vals, color=color, label=label)
        for lab, qi, vi in zip(labels, q, vals):
            if lab in data.active_:
                axes.text(qi, vi, lab, anchor="start", size=8)
        axes.hline(0.0, dash="dash", color="#999999")
        axes.hline(data.me_, dash="dash", color="#E15759", label="ME")
        axes.hline(-data.me_, dash="dash", color="#E15759")
        axes.hline(data.sme_, dash="dash", color="#B07AA1", label="SME")
        axes.hline(-data.sme_, dash="dash", color="#B07AA1")
        axes.set(title=title or "Normal plot of effects", xlabel="theoretical quantile",
                 ylabel="effect", legend="best")
        fig.data = {"theoretical": q, "sample": vals, "active": data.active_}
        return fig

    x = _as_1d(data, "data")
    n = len(x)
    pp = _filliben_pp(n)
    sample = np.sort(x)
    if dist is None:
        mean, std = float(x.mean()), float(x.std(ddof=1))
        from stochpylib.statistics._common import _norm_ppf

        theoretical = mean + std * _norm_ppf(pp)
    else:
        theoretical = np.asarray(dist.ppf(pp), dtype=float)
    axes.scatter(theoretical, sample, color=color, label=label)
    slope = intercept = r = None
    if line in ("fit", "45") and n >= 2:
        if line == "45":
            slope, intercept = 1.0, 0.0
        else:
            slope, intercept = np.polyfit(theoretical, sample, 1)
        xs = np.array([theoretical.min(), theoretical.max()])
        axes.line(xs, slope * xs + intercept, color="#E15759", dash="dash",
                  label=f"{line} line")
        ss_res = float(np.sum((sample - (slope * theoretical + intercept)) ** 2))
        ss_tot = float(np.sum((sample - sample.mean()) ** 2)) or 1.0
        r = float(np.sign(slope) * np.sqrt(max(1.0 - ss_res / ss_tot, 0.0)))
    axes.set(title=title, xlabel="theoretical quantiles", ylabel="sample quantiles")
    fig.data = {"theoretical": theoretical, "sample": sample, "slope": slope,
                "intercept": intercept, "r": r}
    return fig


def plot_ppplot(data, dist, ax=None, title=None, color=None, label=None):
    """P-P plot: empirical CDF vs. a hypothesized distribution's CDF."""
    fig, axes = _get_fig_ax(ax)
    x = _as_1d(data, "data")
    n = len(x)
    sample = np.sort(x)
    empirical = (np.arange(1, n + 1) - 0.5) / n
    theoretical = np.asarray(dist.cdf(sample), dtype=float)
    axes.scatter(theoretical, empirical, color=color, label=label)
    axes.line(np.array([0.0, 1.0]), np.array([0.0, 1.0]), color="#999999", dash="dash",
             label="45 deg")
    axes.set(title=title, xlabel="theoretical CDF", ylabel="empirical CDF",
             xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    fig.data = {"empirical": empirical, "theoretical": theoretical}
    return fig


def plot_histogram(data, bins="auto", density=True, dist=None, kde=False, ax=None,
                   title=None, color=None, label=None):
    """Histogram of a data sample, with optional fitted-distribution and/or KDE overlay."""
    fig, axes = _get_fig_ax(ax)
    x = _as_1d(data, "data")
    heights, edges = np.histogram(x, bins=bins, density=density)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    axes.bar(centers, heights, width=widths, color=color, label=label or "data")
    if dist is not None:
        grid = _dist_grid(dist, 300)
        axes.line(grid, np.asarray(dist.pdf(grid), dtype=float), color="#E15759",
                 label=type(dist).__name__)
    if kde:
        kfig = plot_kde(x)
        axes.line(kfig.data["x"], kfig.data["density"], color="#59A14F", label="KDE")
    axes.set(title=title, xlabel="x", ylabel="density" if density else "count",
             legend="best" if (dist is not None or kde) else None)
    fig.data = {"edges": edges, "heights": heights}
    return fig


def plot_kde(data, bandwidth="silverman", kernel="gaussian", rug=False, n_points=200,
            ax=None, title=None, color=None, label=None):
    """Kernel density estimate of a data sample."""
    from stochpylib.nonparametric.density import KernelDensityEstimate

    fig, axes = _get_fig_ax(ax)
    x = _as_1d(data, "data")
    kde_est = KernelDensityEstimate(kernel=kernel, bandwidth=bandwidth).fit(x)
    h = float(np.ravel(kde_est.bandwidth_)[0])
    lo, hi = x.min() - 3 * h, x.max() + 3 * h
    grid = np.linspace(lo, hi, n_points)
    density = np.asarray(kde_est.pdf(grid), dtype=float)
    axes.line(grid, density, color=color, label=label)
    if rug:
        axes.scatter(x, np.zeros_like(x), color="#333333", size=2, label=None)
    axes.set(title=title, xlabel="x", ylabel="density")
    fig.data = {"x": grid, "density": density, "bandwidth": h}
    return fig
