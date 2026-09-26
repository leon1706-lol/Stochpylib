"""Model-diagnostic plots: MCMC trace/posterior/pair plots, regression residual/
leverage/influence plots, and meta-analysis funnel plots.
"""

import numpy as np

from stochpylib.viz._common import _as_1d, _extract_chains, _extract_samples, _get_fig_ax, \
    _leverage_stats

__all__ = ["trace_plot", "posterior_plot", "pair_plot", "residual_plot", "leverage_plot",
           "influence_plot", "funnel_plot"]


def trace_plot(chains, param_names=None, ax=None, title=None):
    """One trace panel per parameter (one line per chain), titled with R-hat/ESS."""
    from stochpylib.advanced_mcmc.diagnostics import ESS, Rhat
    from stochpylib.viz._figure import Figure

    arr = _extract_chains(chains)
    n_chains, n_samples, dim = arr.shape
    rhat = Rhat(arr)
    ess = ESS(arr)
    names = param_names or [f"param[{i}]" for i in range(dim)]

    if ax is not None:
        fig = ax.figure
        panels = [ax]
    else:
        fig = Figure(nrows=dim, ncols=1, height=max(200, 150 * dim))
        panels = fig.axes
    for j in range(dim):
        panel = panels[min(j, len(panels) - 1)]
        for c in range(n_chains):
            panel.line(np.arange(n_samples), arr[c, :, j], label=f"chain {c}" if j == 0 else None)
        panel.set(title=f"{names[j]}  (R-hat={rhat[j]:.3f}, ESS={ess[j]:.0f})" if dim > 1 or ax is None else title,
                  xlabel="iteration", ylabel=names[j])
    if dim > 1 and ax is None:
        fig.title = title
    fig.data = {"chains": arr, "rhat": rhat, "ess": ess}
    return fig


def posterior_plot(samples, param_names=None, hdi_prob=0.94, ref_val=None,
                   point_estimate="mean", ax=None, title=None):
    """Marginal posterior density per parameter, with an HDI band and a point estimate."""
    from stochpylib.bayesian._result import Posterior
    from stochpylib.nonparametric.density import KernelDensityEstimate
    from stochpylib.viz._figure import Figure

    arr = _extract_samples(samples)
    n, dim = arr.shape
    names = param_names or [f"param[{i}]" for i in range(dim)]

    if ax is not None:
        fig = ax.figure
        panels = [ax]
    else:
        fig = Figure(nrows=1, ncols=dim, width=max(400, 320 * dim), height=320)
        panels = fig.axes
    means, hdis = [], []
    for j in range(dim):
        col = arr[:, j]
        kde = KernelDensityEstimate().fit(col)
        h = float(np.ravel(kde.bandwidth_)[0])
        grid = np.linspace(col.min() - 3 * h, col.max() + 3 * h, 200)
        density = np.asarray(kde.pdf(grid), dtype=float)
        lo, hi = Posterior._hpd_from_samples(col, hdi_prob)
        panel = panels[min(j, len(panels) - 1)]
        panel.line(grid, density, label=names[j] if dim == 1 else None)
        inband = (grid >= lo) & (grid <= hi)
        panel.band(grid[inband], np.zeros(inband.sum()), density[inband],
                   label=f"{int(hdi_prob * 100)}% HDI")
        point = float(np.mean(col)) if point_estimate == "mean" else float(np.median(col))
        panel.vline(point, color="#333333", label=point_estimate)
        if ref_val is not None:
            rv = ref_val[j] if isinstance(ref_val, (list, tuple, np.ndarray)) else ref_val
            panel.vline(rv, color="#E15759", dash="dash", label="ref")
        panel.set(title=names[j], xlabel="value", ylabel="density", legend="best")
        means.append(point)
        hdis.append((lo, hi))
    if title and ax is None:
        fig.title = title
    fig.data = {"mean": np.array(means), "hdi": hdis}
    return fig


def pair_plot(samples, names=None, max_points=2000, random_state=None, ax=None, title=None):
    """k-by-k grid: histogram diagonal, scatter below, correlation text above."""
    from stochpylib.viz._common import _scatter_grid

    arr = _extract_samples(samples)
    n, dim = arr.shape
    if dim < 2:
        raise ValueError("pair_plot needs at least 2 parameters")
    labels = names or [f"p{i}" for i in range(dim)]
    if n > max_points:
        idx = np.random.default_rng(random_state).choice(n, size=max_points, replace=False)
        idx.sort()
        arr = arr[idx]
    fig, corr = _scatter_grid(arr, labels, diagonal="hist", title=title)
    fig.data = {"corr": corr}
    return fig


def residual_plot(model, residuals=None, kind="fitted", smooth=True, ax=None, title=None):
    """Residuals vs. fitted values (``kind="fitted"``), a QQ-plot of residuals
    (``kind="qq"``), or a scale-location plot (``kind="scale-location"``)."""
    from stochpylib.nonparametric.regression import LocalPolynomialReg
    from stochpylib.viz.distributions import plot_qqplot

    if residuals is not None:
        fitted, resid = np.asarray(model, dtype=float), np.asarray(residuals, dtype=float)
    else:
        fitted, resid = np.asarray(model.fitted_, dtype=float), np.asarray(model.resid_, dtype=float)

    if kind == "qq":
        return plot_qqplot(resid, ax=ax, title=title or "Residual QQ-plot")

    fig, axes = _get_fig_ax(ax)
    if kind == "fitted":
        axes.scatter(fitted, resid)
        axes.hline(0.0, dash="dash", color="#999999")
        if smooth and len(fitted) >= 4:
            order = np.argsort(fitted)
            lp = LocalPolynomialReg(degree=1).fit(fitted[order], resid[order])
            axes.line(fitted[order], lp.predict(fitted[order][:, None]), color="#E15759",
                      label="smoother")
        axes.set(title=title, xlabel="fitted values", ylabel="residuals")
        fig.data = {"fitted": fitted, "residuals": resid}
    elif kind == "scale-location":
        std = resid / (resid.std(ddof=0) or 1.0)
        y = np.sqrt(np.abs(std))
        axes.scatter(fitted, y)
        if smooth and len(fitted) >= 4:
            order = np.argsort(fitted)
            lp = LocalPolynomialReg(degree=1).fit(fitted[order], y[order])
            axes.line(fitted[order], lp.predict(fitted[order][:, None]), color="#E15759",
                      label="smoother")
        axes.set(title=title, xlabel="fitted values", ylabel="sqrt(|standardized residual|)")
        fig.data = {"fitted": fitted, "residuals": resid, "smooth": y}
    else:
        raise ValueError("kind must be 'fitted', 'qq', or 'scale-location'")
    return fig


def leverage_plot(X, model_or_residuals, fit_intercept=True, ax=None, title=None):
    """Hat-value leverage vs. internally studentized residuals, with Cook's-distance
    contours at 0.5 and 1."""
    fig, axes = _get_fig_ax(ax)
    resid = (model_or_residuals.resid_ if hasattr(model_or_residuals, "resid_")
             else np.asarray(model_or_residuals, dtype=float))
    stats = _leverage_stats(X, resid, fit_intercept=fit_intercept)
    hat, resid_int, p = stats["hat"], stats["resid_internal"], stats["p"]
    axes.scatter(hat, resid_int)
    axes.hline(0.0, color="#999999")
    h_grid = np.linspace(max(1e-4, hat.min() * 0.5), min(0.999, hat.max() * 1.2), 100)
    for D, style in ((0.5, "#F28E2B"), (1.0, "#E15759")):
        with np.errstate(invalid="ignore"):
            y = np.sqrt(np.clip(D * p * (1.0 - h_grid) / h_grid, 0.0, None))
        axes.line(h_grid, y, color=style, dash="dash", label=f"Cook's D={D}")
        axes.line(h_grid, -y, color=style, dash="dash")
    axes.set(title=title, xlabel="leverage (hat value)", ylabel="studentized residual",
             legend="best")
    fig.data = {"leverage": hat, "student_resid": resid_int, "cooks": stats["cooks"]}
    return fig


def influence_plot(X, model_or_residuals, fit_intercept=True, n_labels=3, ax=None, title=None):
    """Bubble plot of leverage vs. externally studentized residuals, sized by Cook's D."""
    fig, axes = _get_fig_ax(ax)
    resid = (model_or_residuals.resid_ if hasattr(model_or_residuals, "resid_")
             else np.asarray(model_or_residuals, dtype=float))
    stats = _leverage_stats(X, resid, fit_intercept=fit_intercept)
    hat, resid_ext, cooks = stats["hat"], stats["resid_external"], stats["cooks"]
    size = 4.0 + 20.0 * (cooks / (cooks.max() or 1.0))
    top = np.argsort(cooks)[::-1][:n_labels]
    for i in range(len(hat)):
        axes.scatter(np.array([hat[i]]), np.array([resid_ext[i]]), size=float(size[i]),
                    color="#4E79A7", label=None)
    for i in top:
        axes.text(hat[i], resid_ext[i], str(int(i)), anchor="start", size=8)
    axes.hline(0.0, color="#999999", dash="dash")
    axes.set(title=title, xlabel="leverage (hat value)",
             ylabel="externally studentized residual")
    fig.data = {"leverage": hat, "student_resid_ext": resid_ext, "cooks": cooks}
    return fig


def funnel_plot(effects, std_errors, pooled=None, ax=None, title=None):
    """Meta-analysis funnel plot: effect size vs. standard error, with a pooled estimate,
    pseudo-95% funnel and Egger's regression test for small-study effects/publication bias."""
    from stochpylib.statistics._common import _pvalue_from_t

    fig, axes = _get_fig_ax(ax)
    effects = _as_1d(effects, "effects")
    se = _as_1d(std_errors, "std_errors")
    if effects.shape != se.shape:
        raise ValueError("effects and std_errors must have the same length")
    if pooled is None:
        w = 1.0 / se ** 2
        pooled = float(np.sum(w * effects) / np.sum(w))

    axes.scatter(effects, se)
    axes.vline(0.0, color="#999999", label="null effect")
    se_grid = np.linspace(0.0, se.max() * 1.1, 50)
    axes.line(pooled + 1.96 * se_grid, se_grid, color="#999999", dash="dash", label="95% funnel")
    axes.line(pooled - 1.96 * se_grid, se_grid, color="#999999", dash="dash")
    axes.vline(pooled, color="#E15759", label="pooled")
    axes.set(title=title, xlabel="effect size", ylabel="standard error", legend="best")
    axes.ylim = (se.max() * 1.15, 0.0)  # invert: precise studies (small SE) at the top

    # Egger's regression: SND_i = a + b * precision_i, precision = 1/se
    precision = 1.0 / se
    snd = effects / se
    design = np.column_stack([np.ones_like(precision), precision])
    coef, *_ = np.linalg.lstsq(design, snd, rcond=None)
    resid = snd - design @ coef
    n, p = design.shape
    dof = max(n - p, 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(design.T @ design)
    intercept, intercept_se = float(coef[0]), float(np.sqrt(cov[0, 0]))
    t_stat = intercept / intercept_se if intercept_se > 0 else 0.0
    pvalue = _pvalue_from_t(t_stat, dof)
    fig.data = {"pooled": pooled, "egger_intercept": intercept, "egger_pvalue": pvalue}
    return fig
