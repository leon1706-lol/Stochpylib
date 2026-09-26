"""Optional matplotlib backend: same scene graph, drawn with matplotlib instead of SVG.

matplotlib is never imported at module import time -- only inside these functions, and
only from this file (guarded by ``tests/library/tests.py``). Import it only when a caller
actually wants a matplotlib figure or a raster/PDF save; the native SVG path
(``Figure.to_svg()``/``.save("*.svg")``) never touches this module.
"""

import numpy as np

from stochpylib.viz._colors import colormap
from stochpylib.viz._figure import (
    Arrow, Band, Bars, Circle, Heatmap, Line, RefLine, Scatter, Segments, Step, Stem, Text,
)
from stochpylib.viz._svg import _step_path

__all__ = []  # private module

_MISSING_MPL_MSG = (
    "stochpylib.viz: matplotlib is optional and not installed; use Figure.to_svg() / "
    "Figure.save('*.svg') for the native zero-dependency renderer, or `pip install "
    "matplotlib` to enable Figure.to_matplotlib()/save('*.png' etc.)."
)


def is_available():
    """Whether matplotlib is importable right now (no import kept afterward)."""
    import importlib.util

    return importlib.util.find_spec("matplotlib") is not None


def _import_mpl():
    import importlib

    try:
        matplotlib = importlib.import_module("matplotlib")
        pyplot_figure = importlib.import_module("matplotlib.figure")
    except ImportError as exc:
        raise ImportError(_MISSING_MPL_MSG) from exc
    return matplotlib, pyplot_figure


def draw_axes(mpl_ax, axes):
    """Draw one :class:`stochpylib.viz._figure.Axes` scene into a matplotlib Axes-like
    object. Only calls a fixed, documented method surface, so a recording fake object
    (no real matplotlib) can stand in for ``mpl_ax`` in tests."""
    for a in axes.artists:
        _draw_one(mpl_ax, a)

    if axes.title:
        mpl_ax.set_title(axes.title)
    if axes.xlabel:
        mpl_ax.set_xlabel(axes.xlabel)
    if axes.ylabel:
        mpl_ax.set_ylabel(axes.ylabel)
    xlim, ylim = axes.data_limits()
    mpl_ax.set_xlim(xlim)
    mpl_ax.set_ylim(ylim)
    mpl_ax.set_xscale(axes.xscale)
    mpl_ax.set_yscale(axes.yscale)
    mpl_ax.grid(axes.grid)
    if axes.xticklabels is not None:
        pos, lab = axes.xticklabels
        mpl_ax.set_xticks(pos)
        mpl_ax.set_xticklabels(lab)
    if axes.yticklabels is not None:
        pos, lab = axes.yticklabels
        mpl_ax.set_yticks(pos)
        mpl_ax.set_yticklabels(lab)
    if axes.aspect:
        mpl_ax.set_aspect(axes.aspect)
    if axes.legend and any(getattr(a, "label", None) for a in axes.artists):
        mpl_ax.legend(loc=axes.legend if axes.legend != "best" else "best")


def _draw_one(mpl_ax, a):
    if isinstance(a, Line):
        style = "--" if a.dash else "-"
        mpl_ax.plot(a.x, a.y, style, color=a.color, label=a.label, linewidth=a.width,
                    marker=a.marker, alpha=a.alpha)
    elif isinstance(a, Step):
        x, y = _step_path(a.x, a.y, a.where)
        mpl_ax.plot(x, y, "-", color=a.color, label=a.label, linewidth=a.width, alpha=a.alpha)
    elif isinstance(a, Scatter):
        mpl_ax.scatter(a.x, a.y, s=a.size ** 2, color=a.color, label=a.label,
                       marker=a.marker, alpha=a.alpha)
    elif isinstance(a, Bars):
        if a.orient == "v":
            mpl_ax.bar(a.x, a.height, width=a.width, bottom=a.bottom, color=a.color,
                      label=a.label, alpha=a.alpha)
        else:
            mpl_ax.barh(a.x, a.height, height=a.width, left=a.bottom, color=a.color,
                       label=a.label, alpha=a.alpha)
    elif isinstance(a, Band):
        mpl_ax.fill_between(a.x, a.lower, a.upper, color=a.color, label=a.label, alpha=a.alpha)
    elif isinstance(a, Stem):
        mpl_ax.stem(a.x, a.y, bottom=a.baseline, linefmt=a.color, markerfmt="o",
                   basefmt=" ", label=a.label)
    elif isinstance(a, RefLine):
        if a.orient == "h":
            mpl_ax.axhline(a.value, color=a.color, label=a.label,
                          linestyle="--" if a.dash else "-", alpha=a.alpha)
        else:
            mpl_ax.axvline(a.value, color=a.color, label=a.label,
                          linestyle="--" if a.dash else "-", alpha=a.alpha)
    elif isinstance(a, Segments):
        import matplotlib.collections as mcoll

        lc = mcoll.LineCollection(a.segs, colors=a.color, linewidths=a.width, alpha=a.alpha,
                                  label=a.label)
        mpl_ax.add_collection(lc)
    elif isinstance(a, Heatmap):
        cmap = colormap(a.cmap)
        vmin = a.vmin if a.vmin is not None else float(np.nanmin(a.Z))
        vmax = a.vmax if a.vmax is not None else float(np.nanmax(a.Z))
        n = 256
        from matplotlib.colors import LinearSegmentedColormap

        mpl_cmap = LinearSegmentedColormap.from_list(
            f"stochpylib_{a.cmap}", [cmap(t) for t in np.linspace(0, 1, n)])
        mesh = mpl_ax.pcolormesh(a.x_edges, a.y_edges, a.Z, cmap=mpl_cmap, vmin=vmin, vmax=vmax)
        mpl_ax.figure.colorbar(mesh, ax=mpl_ax)
        if a.annotate:
            ny, nx = a.Z.shape
            xc = 0.5 * (a.x_edges[:-1] + a.x_edges[1:])
            yc = 0.5 * (a.y_edges[:-1] + a.y_edges[1:])
            for r in range(ny):
                for c in range(nx):
                    mpl_ax.text(xc[c], yc[r], format(a.Z[r, c], a.fmt), ha="center", va="center",
                               fontsize=8)
    elif isinstance(a, Text):
        mpl_ax.annotate(a.s, (a.x, a.y), ha=("center" if a.anchor == "middle" else
                        ("right" if a.anchor == "end" else "left")), fontsize=a.size,
                        color=a.color or "black")
    elif isinstance(a, Arrow):
        rad = a.curved
        mpl_ax.annotate("", xy=(a.x1, a.y1), xytext=(a.x0, a.y0),
                        arrowprops=dict(arrowstyle="-|>", color=a.color, alpha=a.alpha,
                                        linewidth=a.width,
                                        connectionstyle=f"arc3,rad={rad}"))
    elif isinstance(a, Circle):
        import matplotlib.patches as mpatches

        patch = mpatches.Circle((a.x, a.y), a.r, fill=a.fill, color=a.color, alpha=a.alpha)
        mpl_ax.add_patch(patch)


def to_matplotlib(fig):
    """Build a real ``matplotlib.figure.Figure`` from a stochpylib ``Figure`` scene.

    Uses the object-oriented API (``Figure`` + ``subplots``), never ``pyplot`` -- no
    global figure state, no GUI backend requirement.
    """
    _, mpl_figure_mod = _import_mpl()

    mfig = mpl_figure_mod.Figure(figsize=(fig.width / 100.0, fig.height / 100.0),
                                 layout="constrained")
    mpl_axes = mfig.subplots(nrows=fig.nrows, ncols=fig.ncols, squeeze=False)
    for i, axes in enumerate(fig.axes):
        row, col = divmod(i, fig.ncols)
        draw_axes(mpl_axes[row][col], axes)
    if fig.title:
        mfig.suptitle(fig.title)
    return mfig


def save(fig, path):
    """Render ``fig`` with matplotlib and save it to ``path`` (any format matplotlib
    supports by extension, e.g. ``.png``/``.pdf``/``.jpg``)."""
    mfig = to_matplotlib(fig)
    mfig.savefig(path)
    return path


_NON_INTERACTIVE_BACKENDS = ("agg", "cairo", "pdf", "pgf", "ps", "svg", "template")


def show(fig):
    """Display ``fig`` with matplotlib.

    A ``matplotlib.figure.Figure`` built directly (as ``to_matplotlib()`` does, by
    design, to avoid pyplot's global state) has no canvas manager, so ``Figure.show()``
    on it always raises -- that call only works for figures pyplot itself created. With
    a genuinely interactive backend, build the figure through ``pyplot`` instead so
    ``show()`` is supported; with a non-interactive one (Agg, PDF, ...) there is no
    window to pop up regardless, so fall back to writing a temp image and opening it.
    """
    matplotlib, _ = _import_mpl()
    backend = matplotlib.get_backend().lower()
    if any(name in backend for name in _NON_INTERACTIVE_BACKENDS):
        import os
        import tempfile
        import webbrowser

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        save(fig, path)
        webbrowser.open(f"file://{path}")
        return path

    import matplotlib.pyplot as plt

    pfig = plt.figure(figsize=(fig.width / 100.0, fig.height / 100.0), layout="constrained")
    mpl_axes = pfig.subplots(nrows=fig.nrows, ncols=fig.ncols, squeeze=False)
    for i, axes in enumerate(fig.axes):
        row, col = divmod(i, fig.ncols)
        draw_axes(mpl_axes[row][col], axes)
    if fig.title:
        pfig.suptitle(fig.title)
    plt.show()
    return pfig
