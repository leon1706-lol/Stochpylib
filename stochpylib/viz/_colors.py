"""Color palettes and colormaps: a fixed categorical cycle plus two continuous maps
(viridis-like sequential, RdBu-like diverging). Native numpy only, no matplotlib.

Anchor colors are short, well-known approximations (not byte-identical to matplotlib's
built-in tables) -- good enough for a from-scratch renderer; linear RGB interpolation
between anchors keeps them monotone in perceived lightness for the sequential map.
"""

import numpy as np

__all__ = []  # private module

# Tableau-10 categorical palette.
_CATEGORICAL = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]

# Sequential (viridis-like): dark purple -> teal -> yellow, 9 anchors.
_VIRIDIS_ANCHORS = np.array([
    [0.267, 0.005, 0.329],
    [0.283, 0.141, 0.458],
    [0.254, 0.265, 0.530],
    [0.207, 0.372, 0.553],
    [0.164, 0.471, 0.558],
    [0.128, 0.567, 0.551],
    [0.135, 0.659, 0.518],
    [0.267, 0.749, 0.441],
    [0.478, 0.821, 0.318],
    [0.741, 0.873, 0.150],
    [0.993, 0.906, 0.144],
])

# Diverging (RdBu-like): blue -> white -> red, symmetric about 0.5.
_RDBU_ANCHORS = np.array([
    [0.020, 0.188, 0.380],
    [0.130, 0.400, 0.674],
    [0.418, 0.682, 0.839],
    [0.780, 0.898, 0.941],
    [0.969, 0.969, 0.969],
    [0.992, 0.859, 0.780],
    [0.957, 0.647, 0.510],
    [0.839, 0.376, 0.302],
    [0.647, 0.059, 0.082],
])


def palette(i):
    """Cycle through the categorical palette by integer index."""
    return _CATEGORICAL[int(i) % len(_CATEGORICAL)]


def _interp_anchors(anchors, t):
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    n = len(anchors) - 1
    pos = t * n
    lo = np.clip(np.floor(pos).astype(int), 0, n - 1)
    hi = lo + 1
    frac = (pos - lo)[..., None]
    rgb = anchors[lo] * (1 - frac) + anchors[hi] * frac
    return rgb


def colormap(name):
    """Return ``f(t) -> "#rrggbb"`` for ``t`` in ``[0, 1]`` (scalar or array)."""
    name = str(name).lower()
    if name in ("viridis", "sequential"):
        anchors = _VIRIDIS_ANCHORS
    elif name in ("rdbu", "diverging", "coolwarm"):
        anchors = _RDBU_ANCHORS
    else:
        raise ValueError(f"unknown colormap {name!r}; use 'viridis' or 'RdBu'")

    def f(t):
        scalar = np.isscalar(t)
        rgb = _interp_anchors(anchors, t)
        if scalar:
            return to_hex(rgb)
        return [to_hex(c) for c in np.atleast_2d(rgb)]

    return f


def to_hex(rgb):
    r, g, b = (int(round(np.clip(c, 0.0, 1.0) * 255)) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"
