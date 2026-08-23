"""Frequency-domain tools: periodogram, Welch PSD, CWT/SCALOGRAM, DWT, STFT, Hilbert.

All functions assume a uniform sampling rate ``fs`` (default 1.0 = samples per unit).
The discrete wavelet transform implements compactly supported orthonormal wavelets
(Haar and Daubechies-4 "db2") through their QMF filter pairs with periodic extension,
and provides exact perfect-reconstruction inversion.
"""

import numpy as np

from stochpylib.timeseries._utils import as_1d

__all__ = [
    "SpectralAnalysis",
    "Periodogram",
    "PowerSpectrum",
    "WaveletTransform",
    "CWTTransform",
    "DWTTransform",
    "IDWTTransform",
    "STFT",
    "Hilbert",
]


def _detrended_tapered(x):
    x = as_1d(x)
    n = len(x)
    t = np.arange(n)
    slope, intercept = np.polyfit(t, x, 1)
    z = x - (intercept + slope * t)
    w = np.hanning(n)
    return z * w


# --------------------------------------------------------------------------- periodogram


def Periodogram(x, fs=1.0, detrend=True):
    """One-sided periodogram ``(frequencies, power)``.

    Uses a rectangular window (linear detrending only) so that Parseval's identity
    holds exactly against the detrended series' variance. For leakage control use
    :func:`PowerSpectrum` (Welch with Hann windows).
    """
    x = as_1d(x)
    n = len(x)
    t = np.arange(n)
    if detrend:
        slope, intercept = np.polyfit(t, x, 1)
        z = x - (intercept + slope * t)
    else:
        z = x - x.mean()
    Xf = np.fft.rfft(z)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # one-sided density normalization: integrating power * df reproduces the
    # detrended series' variance exactly (Parseval)
    power = (np.abs(Xf) ** 2) * 2.0 / (n * fs)
    power[0] /= 2.0
    if n % 2 == 0:
        power[-1] /= 2.0
    return freqs, power


def PowerSpectrum(x, fs=1.0, nperseg=256, overlap=0.5, detrend=True):
    """Welch-averaged power spectral density ``(frequencies, psd)``."""
    x = as_1d(x)
    nperseg = int(min(nperseg, len(x)))
    hop = max(int(nperseg * (1.0 - overlap)), 1)
    w = np.hanning(nperseg)
    win_power = float(w @ w)
    acc = None
    count = 0
    for start in range(0, len(x) - nperseg + 1, hop):
        seg = x[start : start + nperseg]
        z = _detrended_tapered(seg) if detrend else (seg - seg.mean()) * w
        Xf = np.fft.rfft(z)
        psd_seg = (np.abs(Xf) ** 2) / win_power / fs
        acc = psd_seg if acc is None else acc + psd_seg
        count += 1
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    psd = acc / count
    psd *= 2.0
    psd[0] /= 2.0
    if nperseg % 2 == 0:
        psd[-1] /= 2.0
    return freqs, psd


class SpectralAnalysis:
    """Convenience wrapper bundling the frequency-domain views of one series."""

    def __init__(self, x, fs=1.0):
        self.x = as_1d(x)
        self.fs = float(fs)

    def periodogram(self):
        return Periodogram(self.x, fs=self.fs)

    def power_spectrum(self, nperseg=256, overlap=0.5):
        return PowerSpectrum(self.x, fs=self.fs, nperseg=nperseg, overlap=overlap)

    def dominant_frequency(self):
        freqs, psd = self.power_spectrum(min(256, max(64, len(self.x) // 4)))
        return float(freqs[int(np.argmax(psd[1:]) + 1)])  # skip DC bin

    def total_power(self):
        """Parseval check: time-domain variance ≈ integrated one-sided spectrum."""
        centered = self.x - self.x.mean()
        time_power = float(centered @ centered) / len(self.x)
        freqs, power = self.periodogram()
        df = self.fs / len(self.x)
        spec_power = float(np.sum(power) * df)
        return {"time_domain": time_power, "frequency_domain": spec_power}


# --------------------------------------------------------------------------- CWT


def _morlet(scale, omega0=6.0):
    """Normalized Morlet wavelet sampled on its effective support."""
    length = int(np.ceil(10.0 * scale))
    t = np.arange(-length, length + 1) / scale
    psi = np.pi ** (-0.25) * np.exp(1j * omega0 * t) * np.exp(-(t**2) / 2.0)
    return psi * scale ** (-0.5)


def CWTTransform(x, scales=None, fs=1.0, wavelet="morlet", omega0=6.0):
    """Continuous wavelet transform (Morlet default).

    Returns ``(scales, coefficients)`` where ``coefficients`` has shape
    ``(len(scales), len(x))``; use ``np.abs`` for the scaleogram.
    """
    x = as_1d(x)
    if scales is None:
        scales = np.geomspace(2.0 / fs, len(x) / 8.0, 24)
    scales = np.asarray(scales, dtype=float)

    def morlet(scale):
        length = int(np.ceil(10.0 * scale))
        tt = (np.arange(-length, length + 1)) / scale
        psi = np.pi ** (-0.25) * np.exp(1j * omega0 * tt) * np.exp(-(tt**2) / 2.0)
        return psi / np.sqrt(scale)

    if str(wavelet).lower() != "morlet":
        raise NotImplementedError("only the Morlet wavelet is implemented")

    n = len(x)
    padded = np.concatenate([x[::-1], x, x[::-1]])
    P_f = np.fft.rfft(padded)
    coeffs = np.empty((len(scales), n), dtype=complex)
    for i, s in enumerate(scales):
        psi = morlet(s)
        conv = np.convolve(padded, np.conj(psi[::-1]), mode="valid")
        coeffs[i] = conv[:n]
    return scales, coeffs


def WaveletTransform(x, scales=None, fs=1.0, kind="cwt"):
    """Facade: ``kind='cwt'`` delegates to :func:`CWTTransform` (documented)."""
    kind = str(kind).lower()
    if kind in ("cwt", "continuous"):
        return CWTTransform(x, scales=scales, fs=fs)
    if kind in ("dwt", "discrete"):
        coeffs = DWTTransform(x, level=None)
        return coeffs
    raise ValueError("kind must be 'cwt' or 'dwt'")


# --------------------------------------------------------------------------- DWT


_HAAR_LO = np.array([1.0, 1.0]) / np.sqrt(2.0)
# Daubechies-4 ("db2") scaling filter, normalized so that sum(h^2) = 1
_D4_RAW = np.array([
    (1 + np.sqrt(3)) / 4,
    (3 + np.sqrt(3)) / 4,
    (3 - np.sqrt(3)) / 4,
    (1 - np.sqrt(3)) / 4,
])
_DAUBECHIES4 = _D4_RAW / np.sqrt(np.sum(_D4_RAW**2))


def _qmf_filters(wavelet):
    wl = str(wavelet).lower()
    if wl in ("haar", "db1"):
        lo = _HAAR_LO
    elif wl in ("db2", "daubechies4", "d4"):
        lo = _DAUBECHIES4
    else:
        raise ValueError(f"unsupported wavelet {wavelet!r}; use 'haar' or 'db2'")
    hi = lo[::-1].copy() * np.array([(-1.0) ** j for j in range(len(lo))])
    return lo, hi


def _analysis_pass(x, lo, hi):
    """One-level analysis with periodic extension: a[i] = sum lo[k] x[(2i+k) mod n]."""
    n = len(x)
    L = len(lo)
    half = n // 2
    approx = np.empty(half)
    detail = np.empty(half)
    for i in range(half):
        idx = (2 * i + np.arange(L)) % n
        approx[i] = float(lo @ x[idx])
        detail[i] = float(hi @ x[idx])
    return approx, detail


def _synthesis_pass(approx, detail, lo, hi):
    """Matched synthesis: for orthonormal QMF pairs the synthesis operator is the
    transpose of the analysis operator — apply the same filters at the same taps."""
    n = 2 * len(approx)
    L = len(lo)
    x = np.zeros(n)
    for i in range(len(approx)):
        idx = (2 * i + np.arange(L)) % n
        x[idx] += lo * approx[i] + hi * detail[i]
    return x


def DWTTransform(x, wavelet="haar", level=None):
    """Multi-level discrete wavelet transform (periodic extension).

    Requires the series length to stay even through ``level`` halvings (pad beforehand
    otherwise — documented). Returns a dict with ``details`` (finest first),
    ``approx``, ``wavelet``, ``levels``.
    """
    x = as_1d(x)
    lo, hi = _qmf_filters(wavelet)
    max_levels = max(int(np.log2(len(x))), 1)
    level = int(level) if level is not None else min(max_levels, 6)
    level = max(level, 1)
    details = []
    current = x.copy()
    for lv in range(level):
        if len(current) // 2 < 1 or len(current) % 2 != 0:
            raise ValueError(
                f"series length {len(current)} is odd at level {lv}; pad to an even length"
            )
        approx, detail = _analysis_pass(current, lo, hi)
        details.append(detail)
        current = approx
    return {
        "details": details,
        "approx": current,
        "wavelet": str(wavelet),
        "levels": len(details),
    }


def IDWTTransform(coeffs):
    """Perfect-reconstruction inverse of :func:`DWTTransform` (same wavelet)."""
    lo, hi = _qmf_filters(coeffs["wavelet"])
    current = coeffs["approx"]
    for detail in reversed(coeffs["details"]):
        current = _synthesis_pass(current, detail, lo, hi)
    return current


# --------------------------------------------------------------------------- STFT / Hilbert


def STFT(x, fs=1.0, window_len=256, hop=128, detrend=True):
    """Short-time Fourier transform: returns ``(freqs, times, spectrogram)`` where the
    spectrogram holds magnitudes with shape ``(len(freqs), n_frames)``."""
    x = as_1d(x)
    window_len = int(min(window_len, len(x)))
    hop = max(int(hop), 1)
    w = np.hanning(window_len)
    freqs = np.fft.rfftfreq(window_len, d=1.0 / fs)
    frames = []
    times = []
    start = 0
    while start + window_len <= len(x):
        seg = x[start : start + window_len]
        z = _detrended_tapered(seg) if detrend else (seg - seg.mean()) * w
        frames.append(np.abs(np.fft.rfft(z * w)))
        times.append((start + window_len / 2) / fs)
        start += hop
    spec = np.array(frames).T if frames else np.empty((len(freqs), 0))
    return freqs, np.array(times), spec


def Hilbert(x):
    """Analytic signal via FFT phase shift: ``z = x + i*Hilbert_transform(x)``.

    For a pure cosine this returns a complex exponential (imaginary part = sine).
    """
    x = as_1d(x)
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(X * h)
