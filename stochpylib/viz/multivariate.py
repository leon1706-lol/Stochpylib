"""Multivariate plots: heatmaps, correlation matrices, copula views, scatter matrices,
PCA biplots and dendrograms.
"""

import numpy as np

from stochpylib.viz._common import _as_2d, _get_fig_ax, _scatter_grid

__all__ = ["plot_heatmap", "plot_correlation", "plot_copula", "plot_scatter_matrix",
           "plot_biplot", "plot_dendrogram"]


def plot_heatmap(Z, row_labels=None, col_labels=None, cmap="viridis", annotate=False,
                 vmin=None, vmax=None, ax=None, title=None):
    """Generic matrix heatmap with a colorbar."""
    fig, axes = _get_fig_ax(ax)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError("Z must be a 2-D matrix")
    axes.heatmap(Z, cmap=cmap, vmin=vmin, vmax=vmax, annotate=annotate)
    vmin = vmin if vmin is not None else float(np.nanmin(Z))
    vmax = vmax if vmax is not None else float(np.nanmax(Z))
    axes.colorbar = {"cmap": cmap, "vmin": vmin, "vmax": vmax}
    ny, nx = Z.shape
    if col_labels is not None:
        axes.xticklabels = (np.arange(nx), list(col_labels))
    if row_labels is not None:
        axes.yticklabels = (np.arange(ny), list(row_labels))
    axes.set(title=title, grid=False)
    fig.data = {"Z": Z}
    return fig


def plot_correlation(X, method="pearson", names=None, is_corr=False, annotate=True,
                     ax=None, title=None):
    """Correlation matrix heatmap (pearson/spearman/kendall), or a given matrix directly."""
    fig, axes = _get_fig_ax(ax)
    if is_corr:
        corr = np.asarray(X, dtype=float)
        if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
            raise ValueError("is_corr=True needs a square matrix")
    else:
        from stochpylib.statistics.descriptive import correlation

        corr = np.asarray(correlation(_as_2d(X, "X"), method=method), dtype=float)
    p = corr.shape[0]
    labels = names or [f"x{i}" for i in range(p)]
    axes.heatmap(corr, cmap="RdBu", vmin=-1.0, vmax=1.0, annotate=annotate)
    axes.colorbar = {"cmap": "RdBu", "vmin": -1.0, "vmax": 1.0}
    axes.xticklabels = (np.arange(p), labels)
    axes.yticklabels = (np.arange(p), labels)
    axes.set(title=title, grid=False)
    fig.data = {"corr": corr}
    return fig


def plot_copula(copula_or_u, n=1000, kind="scatter", random_state=None, n_grid=60,
               ax=None, title=None):
    """Scatter of copula draws (``kind="scatter"``) or the copula's density on
    ``[0, 1]^2`` (``kind="density"``). ``copula_or_u`` may also be raw bivariate data,
    which is converted to pseudo-observations by rank."""
    fig, axes = _get_fig_ax(ax)
    if hasattr(copula_or_u, "sample"):
        cop = copula_or_u
        u = np.asarray(cop.sample(n, random_state=random_state), dtype=float)
        if kind == "density":
            grid = np.linspace(0.01, 0.99, n_grid)
            U1, U2 = np.meshgrid(grid, grid)
            uu = np.column_stack([U1.ravel(), U2.ravel()])
            dens = np.asarray(cop.density(uu), dtype=float).reshape(U1.shape)
            edges = np.concatenate([grid - (grid[1] - grid[0]) / 2,
                                    [grid[-1] + (grid[1] - grid[0]) / 2]])
            axes.heatmap(dens, x_edges=edges, y_edges=edges, cmap="viridis")
            axes.colorbar = {"cmap": "viridis", "vmin": float(dens.min()), "vmax": float(dens.max())}
            axes.set(title=title, xlabel="u1", ylabel="u2", grid=False)
            fig.data = {"u": u, "density": dens}
            return fig
    else:
        data = np.asarray(copula_or_u, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("copula_or_u must be a fitted copula or (n, 2) data")
        n_obs = data.shape[0]
        ranks = np.apply_along_axis(lambda c: c.argsort().argsort(), 0, data)
        u = (ranks + 1.0) / (n_obs + 1.0)

    axes.scatter(u[:, 0], u[:, 1], size=2.5, alpha=0.5, color="#4E79A7")
    axes.set(title=title, xlabel="u1", ylabel="u2", xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    fig.data = {"u": u}
    return fig


def plot_scatter_matrix(X, names=None, diagonal="hist", ax=None, title=None):
    """Scatter-matrix grid: pairwise scatter plots below the diagonal, a density
    (``diagonal="hist"``/``"kde"``) on it, correlation text above."""
    X = _as_2d(X, "X")
    labels = names or [f"x{i}" for i in range(X.shape[1])]
    fig, corr = _scatter_grid(X, labels, diagonal=diagonal, title=title)
    fig.data = {"corr": corr}
    return fig


def plot_biplot(X_or_pca, names=None, standardize=False, scale=1.0, ax=None, title=None):
    """PCA biplot: PC1/PC2 scores plus loading-vector arrows for each variable."""
    fig, axes = _get_fig_ax(ax)
    if hasattr(X_or_pca, "scores_"):
        result = X_or_pca
    else:
        from stochpylib.statistics.multivariate import PCA

        X = _as_2d(X_or_pca, "X")
        result = PCA(X, n_components=2, standardize=standardize)
    scores = np.asarray(result.scores_, dtype=float)[:, :2]
    loadings = np.asarray(result.components_, dtype=float)[:2, :]
    ratio = np.asarray(result.explained_variance_ratio_, dtype=float)
    p = loadings.shape[1]
    labels = names or [f"x{i}" for i in range(p)]

    axes.scatter(scores[:, 0], scores[:, 1], size=3.0, alpha=0.6)
    span = float(np.max(np.abs(scores))) or 1.0
    load_scale = span * scale / (np.max(np.abs(loadings)) or 1.0)
    for j in range(p):
        lx, ly = loadings[0, j] * load_scale, loadings[1, j] * load_scale
        axes.arrow(0.0, 0.0, lx, ly, color="#E15759")
        axes.text(lx * 1.1, ly * 1.1, labels[j], anchor="middle", size=9, color="#E15759")
    axes.hline(0.0, color="#CCCCCC")
    axes.vline(0.0, color="#CCCCCC")
    axes.set(title=title, xlabel=f"PC1 ({ratio[0]:.1%})", ylabel=f"PC2 ({ratio[1]:.1%})")
    fig.data = {"scores": scores, "loadings": loadings, "explained": ratio[:2]}
    return fig


def plot_dendrogram(X_or_Z, method="ward", labels=None, orientation="top", ax=None, title=None):
    """Hierarchical-clustering dendrogram from data (agglomerative linkage) or a
    precomputed scipy-format linkage matrix."""
    fig, axes = _get_fig_ax(ax)
    if isinstance(X_or_Z, np.ndarray) and X_or_Z.ndim == 2 and X_or_Z.shape[1] == 4:
        Z = X_or_Z
        n_leaves = Z.shape[0] + 1
    else:
        from stochpylib.statistics.multivariate import _agglomerative

        X = _as_2d(X_or_Z, "X")
        n_leaves = X.shape[0]
        Z, _ = _agglomerative(X, 1, method)

    labels = list(labels) if labels is not None else [str(i) for i in range(n_leaves)]
    segs, leaf_order = _dendrogram_layout(Z, n_leaves)
    axes.segments(segs, color="#4E79A7")
    # leaf_order[k] is the original leaf id drawn at rank position k (0..n_leaves-1)
    tick_positions = list(range(n_leaves))
    tick_labels = [labels[i] for i in leaf_order]
    if orientation == "top":
        axes.xticklabels = (tick_positions, tick_labels)
        axes.set(ylabel="distance")
    else:
        axes.yticklabels = (tick_positions, tick_labels)
        axes.set(xlabel="distance")
    axes.set(title=title, grid=False)
    fig.data = {"Z": Z, "leaf_order": np.asarray(leaf_order)}
    return fig


def _dendrogram_layout(Z, n_leaves):
    """Recursive leaf ordering + U-shaped line segments for a scipy-format linkage
    matrix ``Z`` (each row: ``[left, right, height, size]``). Leaves are assigned
    integer x-positions in left-to-right visiting order; internal nodes sit midway
    between their two children's positions."""
    order = []
    segs = []

    def visit(node_id):
        if node_id < n_leaves:
            order.append(node_id)
            return float(len(order) - 1), 0.0
        left, right, height, _ = Z[node_id - n_leaves]
        lx, lh = visit(int(left))
        rx, rh = visit(int(right))
        segs.append(((lx, lh), (lx, height)))
        segs.append(((lx, height), (rx, height)))
        segs.append(((rx, height), (rx, rh)))
        return (lx + rx) / 2.0, height

    visit(n_leaves + len(Z) - 1)
    return np.array(segs), order
