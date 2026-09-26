"""Nice-number tick placement (Heckbert 1990) and number formatting.

Native numpy only. Log-scale ticks use decades (powers of ten) instead.
"""

import math

import numpy as np

__all__ = []  # private module


def _nice_num(span, round_):
    """The 'nicest' number close to ``span``: 1/2/5 times a power of ten."""
    if span <= 0:
        return 1.0
    exponent = math.floor(math.log10(span))
    fraction = span / 10.0**exponent
    if round_:
        if fraction < 1.5:
            nice_fraction = 1.0
        elif fraction < 3.0:
            nice_fraction = 2.0
        elif fraction < 7.0:
            nice_fraction = 5.0
        else:
            nice_fraction = 10.0
    else:
        if fraction <= 1.0:
            nice_fraction = 1.0
        elif fraction <= 2.0:
            nice_fraction = 2.0
        elif fraction <= 5.0:
            nice_fraction = 5.0
        else:
            nice_fraction = 10.0
    return nice_fraction * 10.0**exponent


def nice_ticks(lo, hi, n=5):
    """``n``-ish evenly spaced 'nice' tick values covering ``[lo, hi]``."""
    lo, hi = float(lo), float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.array([0.0, 1.0])
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    if lo > hi:
        lo, hi = hi, lo
    span = _nice_num(hi - lo, False)
    step = _nice_num(span / max(n - 1, 1), True)
    nice_lo = math.floor(lo / step) * step
    nice_hi = math.ceil(hi / step) * step
    count = int(round((nice_hi - nice_lo) / step)) + 1
    ticks = nice_lo + step * np.arange(count)
    # clip tiny float noise (e.g. -1e-16 instead of 0.0)
    ticks = np.where(np.abs(ticks) < step * 1e-9, 0.0, ticks)
    return ticks


def log_ticks(lo, hi):
    """Decade ticks (powers of ten) covering a positive ``[lo, hi]`` range."""
    lo, hi = float(lo), float(hi)
    if lo <= 0 or hi <= 0 or not (np.isfinite(lo) and np.isfinite(hi)):
        return np.array([1.0])
    if lo > hi:
        lo, hi = hi, lo
    e_lo = math.floor(math.log10(lo))
    e_hi = math.ceil(math.log10(hi))
    if e_hi <= e_lo:
        e_hi = e_lo + 1
    return 10.0 ** np.arange(e_lo, e_hi + 1)


def fmt(v, sig=4):
    """Format a tick/label value compactly; never prints ``-0``."""
    v = float(v)
    if v == 0.0:
        return "0"
    s = f"{v:.{sig}g}"
    if s in ("-0", "-0.0"):
        s = "0"
    if "e" in s:
        mantissa, exp = s.split("e")
        exp_i = int(exp)
        s = f"{mantissa}e{exp_i:+03d}"
    return s
