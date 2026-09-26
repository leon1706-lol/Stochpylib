"""Zero-dependency scene graph -> SVG renderer.

Deterministic (same ``Figure`` -> byte-identical SVG), and every element stays inside its
axes' clip rectangle. This is the only rendering path with no optional dependency; the
matplotlib backend (``_mpl.py``) is a separate, optional translation of the same scene.
"""

import math
from xml.sax.saxutils import escape

import numpy as np

from stochpylib.viz._colors import colormap
from stochpylib.viz._figure import (
    Arrow, Band, Bars, Circle, Heatmap, Line, RefLine, Scatter, Segments, Step, Stem, Text,
)
from stochpylib.viz._ticks import fmt, log_ticks, nice_ticks

__all__ = []  # private module

_MARGIN_L = 60
_MARGIN_B = 45
_MARGIN_T = 30
_MARGIN_R = 20
_TITLE_BAND = 34
_COLORBAR_W = 60


class _Scale:
    """Data <-> pixel affine map for one axis (linear or log10)."""

    def __init__(self, lo, hi, px_lo, px_hi, log=False):
        self.log = bool(log)
        if self.log:
            lo = max(lo, 1e-300)
            hi = max(hi, lo * 10)
            self.lo, self.hi = math.log10(lo), math.log10(hi)
        else:
            self.lo, self.hi = lo, hi
        self.px_lo, self.px_hi = px_lo, px_hi
        span = (self.hi - self.lo) or 1.0
        self._scale = (px_hi - px_lo) / span

    def __call__(self, v):
        v = math.log10(max(v, 1e-300)) if self.log else v
        return self.px_lo + (v - self.lo) * self._scale


def _pt(sx, sy, x, y):
    return sx(x), sy(y)


def _poly_points(xs, ys, sx, sy):
    return " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, ys))


def _split_finite_runs(x, y):
    """Split (x, y) into contiguous runs with no NaN/inf, for lines that shouldn't
    connect across missing data."""
    finite = np.isfinite(x) & np.isfinite(y)
    runs = []
    start = None
    for i, ok in enumerate(finite):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(finite)))
    return runs


def _step_path(x, y, where):
    """Expand (x, y) into a piecewise-constant polyline's vertex list."""
    xs, ys = [x[0]], [y[0]]
    if where == "post":
        for i in range(1, len(x)):
            xs.append(x[i])
            ys.append(y[i - 1])
            xs.append(x[i])
            ys.append(y[i])
    else:  # pre
        for i in range(1, len(x)):
            xs.append(x[i - 1])
            ys.append(y[i])
            xs.append(x[i])
            ys.append(y[i])
    return xs, ys


def _cell_rect(fig, row, col):
    plot_w = (fig.width - _MARGIN_L - _MARGIN_R) / fig.ncols
    plot_h = (fig.height - _TITLE_BAND - _MARGIN_T - _MARGIN_B) / fig.nrows
    x0 = _MARGIN_L + col * plot_w
    y0 = _TITLE_BAND + row * plot_h
    return x0, y0, x0 + plot_w - (_MARGIN_L + _MARGIN_R) * 0.0, y0 + plot_h


def render_svg(fig):
    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{fig.width}" height="{fig.height}" '
        f'viewBox="0 0 {fig.width} {fig.height}" font-family="sans-serif">'
    )
    parts.append(f'<rect x="0" y="0" width="{fig.width}" height="{fig.height}" fill="#FFFFFF"/>')
    if fig.title:
        parts.append(
            f'<text x="{fig.width / 2:.2f}" y="22" font-size="15" font-weight="bold" '
            f'text-anchor="middle">{escape(str(fig.title))}</text>'
        )

    plot_w = (fig.width - _MARGIN_L - _MARGIN_R) / fig.ncols
    plot_h = (fig.height - _TITLE_BAND - _MARGIN_T - _MARGIN_B) / fig.nrows

    for i, axes in enumerate(fig.axes):
        row, col = divmod(i, fig.ncols)
        px0 = _MARGIN_L + col * plot_w + _MARGIN_L * 0.0
        py0 = _TITLE_BAND + row * plot_h + _MARGIN_T
        cell_left = _MARGIN_L if col == 0 else 0
        # Each cell reserves the same fixed margins around its own plot rectangle.
        rx0 = col * plot_w + _MARGIN_L
        ry0 = _TITLE_BAND + row * plot_h + _MARGIN_T
        rx1 = (col + 1) * plot_w - _MARGIN_R
        ry1 = _TITLE_BAND + (row + 1) * plot_h - _MARGIN_B
        has_cbar = axes.colorbar is not None
        if has_cbar:
            rx1 -= _COLORBAR_W
        parts.append(_render_axes(axes, rx0, ry0, rx1, ry1, clip_id=f"clip{i}"))

    parts.append("</svg>")
    return "".join(parts)


def _render_axes(axes, x0, y0, x1, y1, clip_id):
    xlim, ylim = axes.data_limits()
    sx = _Scale(xlim[0], xlim[1], x0, x1, log=(axes.xscale == "log"))
    sy = _Scale(ylim[0], ylim[1], y1, y0, log=(axes.yscale == "log"))  # y flipped

    out = [f'<clipPath id="{clip_id}"><rect x="{x0:.2f}" y="{y0:.2f}" '
           f'width="{x1 - x0:.2f}" height="{y1 - y0:.2f}"/></clipPath>']

    # frame
    out.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{x1 - x0:.2f}" '
               f'height="{y1 - y0:.2f}" fill="none" stroke="#333333" stroke-width="1"/>')

    # ticks + gridlines
    if axes.xticklabels is not None:
        xt_pos, xt_lab = axes.xticklabels
    else:
        xt_pos = (log_ticks(*xlim) if axes.xscale == "log" else nice_ticks(*xlim))
        xt_pos = [t for t in xt_pos if xlim[0] - 1e-9 <= t <= xlim[1] + 1e-9]
        xt_lab = [fmt(t) for t in xt_pos]
    if axes.yticklabels is not None:
        yt_pos, yt_lab = axes.yticklabels
    else:
        yt_pos = (log_ticks(*ylim) if axes.yscale == "log" else nice_ticks(*ylim))
        yt_pos = [t for t in yt_pos if ylim[0] - 1e-9 <= t <= ylim[1] + 1e-9]
        yt_lab = [fmt(t) for t in yt_pos]

    for t, lab in zip(xt_pos, xt_lab):
        px = sx(t)
        if axes.grid:
            out.append(f'<line x1="{px:.2f}" y1="{y0:.2f}" x2="{px:.2f}" y2="{y1:.2f}" '
                       f'stroke="#E0E0E0" stroke-width="1"/>')
        out.append(f'<line x1="{px:.2f}" y1="{y1:.2f}" x2="{px:.2f}" y2="{y1 + 5:.2f}" '
                   f'stroke="#333333"/>')
        out.append(f'<text x="{px:.2f}" y="{y1 + 18:.2f}" font-size="10" '
                   f'text-anchor="middle">{escape(str(lab))}</text>')
    for t, lab in zip(yt_pos, yt_lab):
        pyv = sy(t)
        if axes.grid:
            out.append(f'<line x1="{x0:.2f}" y1="{pyv:.2f}" x2="{x1:.2f}" y2="{pyv:.2f}" '
                       f'stroke="#E0E0E0" stroke-width="1"/>')
        out.append(f'<line x1="{x0 - 5:.2f}" y1="{pyv:.2f}" x2="{x0:.2f}" y2="{pyv:.2f}" '
                   f'stroke="#333333"/>')
        out.append(f'<text x="{x0 - 8:.2f}" y="{pyv + 3:.2f}" font-size="10" '
                   f'text-anchor="end">{escape(str(lab))}</text>')

    if axes.title:
        out.append(f'<text x="{(x0 + x1) / 2:.2f}" y="{y0 - 8:.2f}" font-size="12" '
                   f'text-anchor="middle">{escape(str(axes.title))}</text>')
    if axes.xlabel:
        out.append(f'<text x="{(x0 + x1) / 2:.2f}" y="{y1 + 36:.2f}" font-size="11" '
                   f'text-anchor="middle">{escape(str(axes.xlabel))}</text>')
    if axes.ylabel:
        cy = (y0 + y1) / 2
        out.append(f'<text x="{x0 - 42:.2f}" y="{cy:.2f}" font-size="11" '
                   f'text-anchor="middle" transform="rotate(-90 {x0 - 42:.2f} {cy:.2f})">'
                   f'{escape(str(axes.ylabel))}</text>')

    out.append(f'<g clip-path="url(#{clip_id})">')
    for artist in axes.artists:
        out.append(_render_artist(artist, sx, sy))
    out.append("</g>")

    labelled = [a for a in axes.artists if getattr(a, "label", None)]
    if axes.legend and labelled:
        lx, ly = x1 - 8, y0 + 8
        box_h = 14 * len(labelled) + 6
        box_w = max(60, 8 * max(len(a.label) for a in labelled))
        out.append(f'<rect x="{lx - box_w:.2f}" y="{ly:.2f}" width="{box_w:.2f}" '
                   f'height="{box_h:.2f}" fill="#FFFFFF" stroke="#CCCCCC" opacity="0.9"/>')
        for k, a in enumerate(labelled):
            yy = ly + 12 + 14 * k
            out.append(f'<line x1="{lx - box_w + 6:.2f}" y1="{yy - 3:.2f}" '
                       f'x2="{lx - box_w + 20:.2f}" y2="{yy - 3:.2f}" '
                       f'stroke="{a.color}" stroke-width="2"/>')
            out.append(f'<text x="{lx - box_w + 24:.2f}" y="{yy:.2f}" font-size="9">'
                       f'{escape(str(a.label))}</text>')

    if axes.colorbar is not None:
        cb = axes.colorbar
        cx0, cx1 = x1 + 15, x1 + 30
        cmap = colormap(cb.get("cmap", "viridis"))
        stops = "".join(
            f'<stop offset="{(1 - t) * 100:.1f}%" stop-color="{cmap(t)}"/>'
            for t in np.linspace(0, 1, 9)
        )
        out.append(f'<defs><linearGradient id="{clip_id}cbar" x1="0" y1="0" x2="0" y2="1">'
                   f'{stops}</linearGradient></defs>')
        out.append(f'<rect x="{cx0:.2f}" y="{y0:.2f}" width="{cx1 - cx0:.2f}" '
                   f'height="{y1 - y0:.2f}" fill="url(#{clip_id}cbar)" '
                   f'stroke="#333333" stroke-width="0.5"/>')
        vmin, vmax = cb["vmin"], cb["vmax"]
        for t in np.linspace(0, 1, 5):
            yy = y1 - t * (y1 - y0)
            out.append(f'<text x="{cx1 + 4:.2f}" y="{yy + 3:.2f}" font-size="9">'
                       f'{fmt(vmin + t * (vmax - vmin))}</text>')

    return "".join(out)


def _render_artist(a, sx, sy):
    if isinstance(a, (Line, Step)):
        x, y = a.x, a.y
        if isinstance(a, Step):
            x, y = _step_path(x, y, a.where)
            x, y = np.asarray(x), np.asarray(y)
        segs = []
        for s, e in _split_finite_runs(x, y):
            if e - s < 2:
                continue
            pts = _poly_points(x[s:e], y[s:e], sx, sy)
            dash = ' stroke-dasharray="6,4"' if getattr(a, "dash", None) else ""
            segs.append(f'<polyline points="{pts}" fill="none" stroke="{a.color}" '
                        f'stroke-width="{a.width}" opacity="{a.alpha}"{dash}/>')
        return "".join(segs)
    if isinstance(a, Scatter):
        out = []
        for x, y in zip(a.x, a.y):
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            px, py = sx(x), sy(y)
            out.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{a.size:.2f}" '
                       f'fill="{a.color}" opacity="{a.alpha}"/>')
        return "".join(out)
    if isinstance(a, Bars):
        out = []
        widths = np.broadcast_to(np.atleast_1d(a.width), a.x.shape)
        for xv, hv, wv in zip(a.x, a.height, widths):
            if a.orient == "v":
                px0, px1 = sx(xv - wv / 2), sx(xv + wv / 2)
                py0, py1 = sy(a.bottom), sy(hv)
                rx, ry = min(px0, px1), min(py0, py1)
                rw, rh = abs(px1 - px0), abs(py1 - py0)
            else:
                py0, py1 = sy(xv - wv / 2), sy(xv + wv / 2)
                px0, px1 = sx(a.bottom), sx(hv)
                rx, ry = min(px0, px1), min(py0, py1)
                rw, rh = abs(px1 - px0), abs(py1 - py0)
            out.append(f'<rect x="{rx:.2f}" y="{ry:.2f}" width="{rw:.2f}" height="{rh:.2f}" '
                       f'fill="{a.color}" opacity="{a.alpha}"/>')
        return "".join(out)
    if isinstance(a, Band):
        pts_top = [(x, u) for x, u in zip(a.x, a.upper) if math.isfinite(u)]
        pts_bot = [(x, lo) for x, lo in zip(a.x, a.lower) if math.isfinite(lo)][::-1]
        pts = pts_top + pts_bot
        if len(pts) < 3:
            return ""
        d = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in pts)
        return f'<polygon points="{d}" fill="{a.color}" opacity="{a.alpha}" stroke="none"/>'
    if isinstance(a, Stem):
        out = []
        yb = sy(a.baseline)
        for xv, yv in zip(a.x, a.y):
            if not math.isfinite(yv):
                continue
            px, py = sx(xv), sy(yv)
            out.append(f'<line x1="{px:.2f}" y1="{yb:.2f}" x2="{px:.2f}" y2="{py:.2f}" '
                       f'stroke="{a.color}" stroke-width="1.5" opacity="{a.alpha}"/>')
            out.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3" fill="{a.color}" '
                       f'opacity="{a.alpha}"/>')
        return "".join(out)
    if isinstance(a, RefLine):
        dash = ' stroke-dasharray="6,4"' if a.dash else ""
        if a.orient == "h":
            py = sy(a.value)
            return (f'<line x1="{sx.px_lo:.2f}" y1="{py:.2f}" x2="{sx.px_hi:.2f}" y2="{py:.2f}" '
                    f'stroke="{a.color}" stroke-width="1" opacity="{a.alpha}"{dash}/>')
        px = sx(a.value)
        return (f'<line x1="{px:.2f}" y1="{sy.px_lo:.2f}" x2="{px:.2f}" y2="{sy.px_hi:.2f}" '
                f'stroke="{a.color}" stroke-width="1" opacity="{a.alpha}"{dash}/>')
    if isinstance(a, Segments):
        out = []
        for (x0, y0), (x1, y1) in a.segs:
            out.append(f'<line x1="{sx(x0):.2f}" y1="{sy(y0):.2f}" x2="{sx(x1):.2f}" '
                       f'y2="{sy(y1):.2f}" stroke="{a.color}" stroke-width="{a.width}" '
                       f'opacity="{a.alpha}"/>')
        return "".join(out)
    if isinstance(a, Heatmap):
        cmap = colormap(a.cmap)
        vmin = a.vmin if a.vmin is not None else float(np.nanmin(a.Z))
        vmax = a.vmax if a.vmax is not None else float(np.nanmax(a.Z))
        span = (vmax - vmin) or 1.0
        out = []
        ny, nx = a.Z.shape
        for r in range(ny):
            for c in range(nx):
                v = a.Z[r, c]
                if not math.isfinite(v):
                    continue
                t = (v - vmin) / span
                px0, px1 = sx(a.x_edges[c]), sx(a.x_edges[c + 1])
                py0, py1 = sy(a.y_edges[r]), sy(a.y_edges[r + 1])
                rx, ry = min(px0, px1), min(py0, py1)
                rw, rh = abs(px1 - px0), abs(py1 - py0)
                out.append(f'<rect x="{rx:.2f}" y="{ry:.2f}" width="{rw:.2f}" '
                           f'height="{rh:.2f}" fill="{cmap(np.clip(t, 0, 1))}"/>')
                if a.annotate:
                    cx, cy = rx + rw / 2, ry + rh / 2
                    out.append(f'<text x="{cx:.2f}" y="{cy + 3:.2f}" font-size="8" '
                               f'text-anchor="middle">{format(v, a.fmt)}</text>')
        return "".join(out)
    if isinstance(a, Text):
        px, py = sx(a.x), sy(a.y)
        return (f'<text x="{px:.2f}" y="{py:.2f}" font-size="{a.size}" '
                f'text-anchor="{a.anchor}" fill="{a.color or "#000000"}">'
                f'{escape(str(a.s))}</text>')
    if isinstance(a, Arrow):
        px0, py0 = sx(a.x0), sy(a.y0)
        px1, py1 = sx(a.x1), sy(a.y1)
        if a.curved:
            mx, my = (px0 + px1) / 2, (py0 + py1) / 2
            dx, dy = px1 - px0, py1 - py0
            nx_, ny_ = -dy, dx
            norm = math.hypot(nx_, ny_) or 1.0
            cx = mx + nx_ / norm * a.curved * math.hypot(dx, dy)
            cy = my + ny_ / norm * a.curved * math.hypot(dx, dy)
            path = f'M {px0:.2f} {py0:.2f} Q {cx:.2f} {cy:.2f} {px1:.2f} {py1:.2f}'
            # tangent of the quadratic bezier at t=1: 2*(P1 - C)
            tx, ty = px1 - cx, py1 - cy
        else:
            path = f'M {px0:.2f} {py0:.2f} L {px1:.2f} {py1:.2f}'
            tx, ty = px1 - px0, py1 - py0
        tnorm = math.hypot(tx, ty) or 1.0
        tx, ty = tx / tnorm, ty / tnorm
        # manually drawn arrowhead triangle at the endpoint (no <marker>, so a single
        # color/opacity always applies cleanly regardless of renderer support)
        hx, hy = -ty, tx  # perpendicular
        size = 6.0
        p_tip = (px1, py1)
        p_a = (px1 - tx * size + hx * size * 0.5, py1 - ty * size + hy * size * 0.5)
        p_b = (px1 - tx * size - hx * size * 0.5, py1 - ty * size - hy * size * 0.5)
        head = (f'<polygon points="{p_tip[0]:.2f},{p_tip[1]:.2f} {p_a[0]:.2f},{p_a[1]:.2f} '
                f'{p_b[0]:.2f},{p_b[1]:.2f}" fill="{a.color}" opacity="{a.alpha}"/>')
        return (f'<path d="{path}" fill="none" stroke="{a.color}" stroke-width="{a.width}" '
                f'opacity="{a.alpha}"/>{head}')
    if isinstance(a, Circle):
        px, py = sx(a.x), sy(a.y)
        pr = abs(sx(a.x + a.r) - px)
        fill = a.color if a.fill else "none"
        return (f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{pr:.2f}" fill="{fill}" '
                f'stroke="{a.color}" opacity="{a.alpha}"/>')
    return ""
