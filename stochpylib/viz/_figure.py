"""The viz scene graph: ``Figure``/``Axes`` plus artist dataclasses.

Every ``plot_*()`` function builds one of these instead of drawing directly, so the same
scene renders to native SVG (``_svg.py``, always available) or to matplotlib
(``_mpl.py``, optional). All numeric fields are float ndarrays -- tests assert on these
numbers directly, independent of any rendering backend.
"""

from dataclasses import dataclass, field

import numpy as np

from stochpylib.viz._colors import palette

__all__ = ["Figure", "Axes"]


def _as_xy(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    return x, y


# --------------------------------------------------------------------------- artists


@dataclass
class Line:
    x: np.ndarray
    y: np.ndarray
    color: str = None
    label: str = None
    width: float = 1.5
    dash: str = None
    marker: str = None
    alpha: float = 1.0
    kind: str = field(default="line", init=False)


@dataclass
class Scatter:
    x: np.ndarray
    y: np.ndarray
    color: str = None
    label: str = None
    size: float = 4.0
    marker: str = "o"
    alpha: float = 1.0
    kind: str = field(default="scatter", init=False)


@dataclass
class Bars:
    x: np.ndarray
    height: np.ndarray
    width: np.ndarray
    bottom: float = 0.0
    color: str = None
    label: str = None
    orient: str = "v"
    alpha: float = 1.0
    kind: str = field(default="bars", init=False)


@dataclass
class Band:
    x: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    color: str = None
    label: str = None
    alpha: float = 0.25
    kind: str = field(default="band", init=False)


@dataclass
class Step:
    x: np.ndarray
    y: np.ndarray
    color: str = None
    label: str = None
    width: float = 1.5
    where: str = "post"
    alpha: float = 1.0
    kind: str = field(default="step", init=False)


@dataclass
class Stem:
    x: np.ndarray
    y: np.ndarray
    baseline: float = 0.0
    color: str = None
    label: str = None
    alpha: float = 1.0
    kind: str = field(default="stem", init=False)


@dataclass
class RefLine:
    value: float
    orient: str = "h"
    color: str = None
    label: str = None
    dash: str = "dash"
    alpha: float = 1.0
    kind: str = field(default="refline", init=False)


@dataclass
class Segments:
    segs: np.ndarray  # shape (k, 2, 2): k segments of (x, y) endpoint pairs
    color: str = None
    label: str = None
    width: float = 1.0
    alpha: float = 1.0
    kind: str = field(default="segments", init=False)


@dataclass
class Heatmap:
    Z: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    cmap: str = "viridis"
    vmin: float = None
    vmax: float = None
    annotate: bool = False
    fmt: str = ".2f"
    label: str = None
    color: str = None
    alpha: float = 1.0
    kind: str = field(default="heatmap", init=False)


@dataclass
class Text:
    x: float
    y: float
    s: str
    anchor: str = "start"
    size: float = 11.0
    color: str = None
    label: str = None
    alpha: float = 1.0
    kind: str = field(default="text", init=False)


@dataclass
class Arrow:
    x0: float
    y0: float
    x1: float
    y1: float
    curved: float = 0.0
    color: str = None
    label: str = None
    width: float = 1.2
    alpha: float = 1.0
    kind: str = field(default="arrow", init=False)


@dataclass
class Circle:
    x: float
    y: float
    r: float
    color: str = None
    label: str = None
    fill: bool = True
    alpha: float = 1.0
    kind: str = field(default="circle", init=False)


_LINELIKE = (Line, Step)


def _finite_extent(*arrays):
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays if
                           a is not None and np.size(a)])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(vals.min()), float(vals.max())


def _artist_extent(artist):
    """Return ``((xlo, xhi), (ylo, yhi))`` or ``None`` if the artist has no finite data."""
    if isinstance(artist, (Line, Scatter, Step)):
        xe = _finite_extent(artist.x)
        ye = _finite_extent(artist.y)
    elif isinstance(artist, Bars):
        w = np.atleast_1d(artist.width)
        if artist.orient == "v":
            xe = _finite_extent(artist.x - w / 2, artist.x + w / 2)
            ye = _finite_extent(artist.height, np.full_like(np.atleast_1d(artist.height),
                                                              artist.bottom))
        else:
            ye = _finite_extent(artist.x - w / 2, artist.x + w / 2)
            xe = _finite_extent(artist.height, np.full_like(np.atleast_1d(artist.height),
                                                              artist.bottom))
    elif isinstance(artist, Band):
        xe = _finite_extent(artist.x)
        ye = _finite_extent(artist.lower, artist.upper)
    elif isinstance(artist, Stem):
        xe = _finite_extent(artist.x)
        ye = _finite_extent(artist.y, np.full_like(np.atleast_1d(artist.y), artist.baseline))
    elif isinstance(artist, RefLine):
        if artist.orient == "h":
            xe, ye = None, _finite_extent(np.array([artist.value]))
        else:
            xe, ye = _finite_extent(np.array([artist.value])), None
    elif isinstance(artist, Segments):
        segs = np.asarray(artist.segs, dtype=float)
        xe = _finite_extent(segs[..., 0])
        ye = _finite_extent(segs[..., 1])
    elif isinstance(artist, Heatmap):
        xe = _finite_extent(artist.x_edges)
        ye = _finite_extent(artist.y_edges)
    elif isinstance(artist, Text):
        xe = _finite_extent(np.array([artist.x]))
        ye = _finite_extent(np.array([artist.y]))
    elif isinstance(artist, Arrow):
        xe = _finite_extent(np.array([artist.x0, artist.x1]))
        ye = _finite_extent(np.array([artist.y0, artist.y1]))
    elif isinstance(artist, Circle):
        xe = _finite_extent(np.array([artist.x - artist.r, artist.x + artist.r]))
        ye = _finite_extent(np.array([artist.y - artist.r, artist.y + artist.r]))
    else:
        return None
    return xe, ye


# --------------------------------------------------------------------------- Axes


class Axes:
    """One plotting panel: a list of artists plus titles/limits/scale."""

    def __init__(self, figure=None):
        self.figure = figure
        self.title = None
        self.xlabel = None
        self.ylabel = None
        self.xlim = None
        self.ylim = None
        self.xscale = "linear"
        self.yscale = "linear"
        self.grid = True
        self.legend = None
        self.aspect = None
        self.xticklabels = None  # (positions, labels) override for categorical axes
        self.yticklabels = None
        self.colorbar = None
        self.artists = []
        self._color_i = 0

    def _next_color(self):
        c = palette(self._color_i)
        self._color_i += 1
        return c

    def _add(self, artist):
        if getattr(artist, "color", None) is None and not isinstance(artist, Heatmap):
            artist.color = self._next_color()
        self.artists.append(artist)
        return artist

    # ---- artist-adding methods, each returns the artist ----

    def line(self, x, y, **kw):
        x, y = _as_xy(x, y)
        return self._add(Line(x, y, **kw))

    def scatter(self, x, y, **kw):
        x, y = _as_xy(x, y)
        return self._add(Scatter(x, y, **kw))

    def bar(self, x, height, width=0.8, bottom=0.0, orient="v", **kw):
        x = np.asarray(x, dtype=float).ravel()
        height = np.asarray(height, dtype=float).ravel()
        return self._add(Bars(x, height, width, bottom=bottom, orient=orient, **kw))

    def band(self, x, lower, upper, **kw):
        x = np.asarray(x, dtype=float).ravel()
        lower = np.asarray(lower, dtype=float).ravel()
        upper = np.asarray(upper, dtype=float).ravel()
        return self._add(Band(x, lower, upper, **kw))

    def step(self, x, y, where="post", **kw):
        x, y = _as_xy(x, y)
        return self._add(Step(x, y, where=where, **kw))

    def stem(self, x, y, baseline=0.0, **kw):
        x, y = _as_xy(x, y)
        return self._add(Stem(x, y, baseline=baseline, **kw))

    def hline(self, value, **kw):
        return self._add(RefLine(float(value), orient="h", **kw))

    def vline(self, value, **kw):
        return self._add(RefLine(float(value), orient="v", **kw))

    def segments(self, segs, **kw):
        segs = np.asarray(segs, dtype=float)
        if segs.ndim != 3 or segs.shape[1:] != (2, 2):
            raise ValueError("segments must have shape (k, 2, 2)")
        return self._add(Segments(segs, **kw))

    def heatmap(self, Z, x_edges=None, y_edges=None, **kw):
        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2:
            raise ValueError("heatmap Z must be 2-D")
        ny, nx = Z.shape
        if x_edges is None:
            x_edges = np.arange(nx + 1) - 0.5
        if y_edges is None:
            y_edges = np.arange(ny + 1) - 0.5
        return self._add(Heatmap(Z, np.asarray(x_edges, dtype=float),
                                  np.asarray(y_edges, dtype=float), **kw))

    def text(self, x, y, s, **kw):
        return self._add(Text(float(x), float(y), str(s), **kw))

    def arrow(self, x0, y0, x1, y1, **kw):
        return self._add(Arrow(float(x0), float(y0), float(x1), float(y1), **kw))

    def circle(self, x, y, r, **kw):
        return self._add(Circle(float(x), float(y), float(r), **kw))

    def set(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self

    def data_limits(self):
        """Auto axis limits: union of finite artist extents + 5% padding."""
        xs, ys = [], []
        for a in self.artists:
            xe, ye = _artist_extent(a)
            if xe is not None:
                xs.append(xe)
            if ye is not None:
                ys.append(ye)

        def _pad(pairs, log):
            if not pairs:
                return (0.0, 1.0) if not log else (0.1, 10.0)
            lo = min(p[0] for p in pairs)
            hi = max(p[1] for p in pairs)
            if log:
                lo = max(lo, 1e-300)
                hi = max(hi, lo * 10)
                if lo <= 0:
                    lo = hi / 100.0
                pad = (np.log10(hi) - np.log10(lo)) * 0.05 or 0.5
                return 10 ** (np.log10(lo) - pad), 10 ** (np.log10(hi) + pad)
            if lo == hi:
                return lo - 0.5, hi + 0.5
            pad = (hi - lo) * 0.05
            return lo - pad, hi + pad

        xlim = self.xlim if self.xlim is not None else _pad(xs, self.xscale == "log")
        ylim = self.ylim if self.ylim is not None else _pad(ys, self.yscale == "log")
        return xlim, ylim

    def __repr__(self):
        return f"Axes(artists={len(self.artists)}, title={self.title!r})"


# --------------------------------------------------------------------------- Figure


class Figure:
    """A grid of :class:`Axes` plus the numeric ``data`` each plot function exposes."""

    def __init__(self, nrows=1, ncols=1, width=640, height=480, title=None):
        self.nrows = int(nrows)
        self.ncols = int(ncols)
        self.width = int(width)
        self.height = int(height)
        self.title = title
        self.axes = [Axes(figure=self) for _ in range(self.nrows * self.ncols)]
        self.data = {}

    @property
    def ax(self):
        return self.axes[0]

    def to_svg(self):
        from stochpylib.viz._svg import render_svg
        return render_svg(self)

    def _repr_svg_(self):
        return self.to_svg()

    def to_matplotlib(self):
        from stochpylib.viz._mpl import to_matplotlib
        return to_matplotlib(self)

    def save(self, path):
        path = str(path)
        if path.lower().endswith(".svg"):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.to_svg())
            return path
        from stochpylib.viz._mpl import save
        return save(self, path)

    def show(self):
        from stochpylib.viz._mpl import is_available

        if not is_available():
            import tempfile
            import webbrowser

            fd, path = tempfile.mkstemp(suffix=".svg")
            import os

            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(self.to_svg())
            webbrowser.open(f"file://{path}")
            return path
        fig = self.to_matplotlib()
        fig.show()
        return fig

    def __repr__(self):
        n_artists = sum(len(a.artists) for a in self.axes)
        return (f"Figure(nrows={self.nrows}, ncols={self.ncols}, "
                f"axes={len(self.axes)}, artists={n_artists})")
