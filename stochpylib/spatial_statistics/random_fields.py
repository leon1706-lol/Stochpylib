"""Gaussian and related random fields: covariance-driven sampling on scattered points or
regular grids (exact Cholesky, exact circulant embedding, or per-axis AR(1) recursions where
the structure allows it), plus Brownian/fractional-Brownian sheets.

``spatial_statistics.GaussianRandomField`` is a separate, covariance-driven class from
``levy_processes.GaussianRandomField`` (FFT spectral synthesis on a grid) -- they only meet
when ``method="spectral"`` is requested, which delegates to the levy class.
"""

import numpy as np
from scipy import special

from stochpylib.gaussian_processes._utils import cholesky_with_jitter
from stochpylib.spatial_statistics._common import _as_coords, _pairwise
from stochpylib.spatial_statistics.variogram import SpatialCovariance, Variogram, _matern_corr

__all__ = [
    "GaussianRandomField", "MaternField", "OrnsteinUhlenbeckField", "BrownianSheet",
    "FractionalBrownianSheet",
]


def _grid_points(shape, spacing):
    spacing = np.broadcast_to(np.atleast_1d(spacing).astype(float), (len(shape),))
    axes = [np.arange(n) * spacing[i] for i, n in enumerate(shape)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh]), spacing


def _as_covariance_fn(covariance):
    """Normalize covariance/kernel/variogram/callable input to a callable C(h)."""
    if isinstance(covariance, SpatialCovariance):
        return covariance
    if isinstance(covariance, Variogram):
        return SpatialCovariance(covariance)
    if hasattr(covariance, "_shape_fn"):
        kernel = covariance
        ls = float(np.atleast_1d(kernel.length_scale)[0])

        def cov_from_kernel(h):
            return kernel.variance * kernel._shape_fn(np.asarray(h, dtype=float) / ls)
        return cov_from_kernel
    if callable(covariance):
        return covariance
    raise TypeError("covariance must be a SpatialCovariance, a bounded Variogram, a "
                     "stationary gaussian_processes kernel, or a callable C(h)")


class GaussianRandomField:
    """A Gaussian random field with a prescribed covariance (not a fixed spectrum).

    ``covariance`` is a ``SpatialCovariance``, a bounded ``Variogram``, a
    ``gaussian_processes`` kernel, or a callable ``C(h)``. ``sample`` handles arbitrary
    points via Cholesky; ``sample_grid`` uses exact circulant embedding on a regular grid
    (falls back to Cholesky if the embedding cannot be made positive semi-definite);
    ``method="spectral"`` (with ``spectrum=``) delegates to
    ``levy_processes.GaussianRandomField``'s FFT synthesis instead.
    """

    def __init__(self, covariance=None, mean=0.0, method="auto", spectrum=None):
        if method not in ("auto", "cholesky", "circulant", "spectral"):
            raise ValueError("method must be 'auto', 'cholesky', 'circulant' or 'spectral'")
        if method == "spectral" and spectrum is None:
            raise ValueError("method='spectral' needs spectrum=")
        if method != "spectral" and covariance is None:
            raise ValueError("covariance= is required unless method='spectral'")
        self.covariance = covariance
        self.mean = float(mean)
        self.method = method
        self.spectrum = spectrum
        self._cov_fn = _as_covariance_fn(covariance) if covariance is not None else None

    def sample(self, coords, n_samples=1, random_state=None):
        X = _as_coords(coords)
        C = self._cov_fn(_pairwise(X))
        L, _ = cholesky_with_jitter(C)
        rng = np.random.default_rng(random_state)
        Z = rng.standard_normal((len(X), int(n_samples)))
        out = self.mean + (L @ Z).T
        return out[0] if n_samples == 1 else out

    def sample_grid(self, shape, spacing=1.0, n_samples=1, random_state=None):
        if self.method == "spectral":
            from stochpylib.levy_processes import GaussianRandomField as _LevyGRF

            grf = _LevyGRF(self.spectrum, shape=tuple(shape), length=float(np.atleast_1d(spacing)[0]))
            rng = np.random.default_rng(random_state)
            samples = [self.mean + grf.sample(random_state=rng) for _ in range(int(n_samples))]
            return samples[0] if n_samples == 1 else np.stack(samples)

        shape = tuple(int(s) for s in shape)
        pts, sp = _grid_points(shape, spacing)
        use_circulant = self.method in ("auto", "circulant")
        if use_circulant:
            try:
                out = self._sample_circulant(shape, sp, n_samples, random_state)
                return out
            except np.linalg.LinAlgError:
                if self.method == "circulant":
                    raise
        flat = self.sample(pts, n_samples=n_samples, random_state=random_state)
        return flat.reshape((int(n_samples),) + shape) if n_samples > 1 else flat.reshape(shape)

    def _sample_circulant(self, shape, spacing, n_samples, random_state):
        """Exact circulant embedding: pad each axis to 2x, embed by symmetric wrap-around."""
        d = len(shape)
        pad = tuple(2 * s for s in shape)
        axes = [np.concatenate([np.arange(s), np.arange(s - pad[i], 0)]) * spacing[i]
                for i, s in enumerate(shape)]
        mesh = np.meshgrid(*axes, indexing="ij")
        h = np.sqrt(sum(m ** 2 for m in mesh))
        C_row = self._cov_fn(h)
        eig = np.fft.fftn(C_row).real
        if eig.min() < -1e-8 * max(eig.max(), 1e-300):
            raise np.linalg.LinAlgError("circulant embedding is not positive semi-definite; "
                                         "use method='cholesky' instead")
        eig = np.maximum(eig, 0.0)
        rng = np.random.default_rng(random_state)
        n = int(n_samples)
        samples = np.empty((n,) + shape)
        for s in range(n):
            noise = rng.standard_normal(pad) + 1j * rng.standard_normal(pad)
            field = np.fft.ifftn(np.sqrt(eig) * noise).real * np.sqrt(np.prod(pad))
            slicer = tuple(slice(0, sz) for sz in shape)
            samples[s] = self.mean + field[slicer]
        return samples[0] if n == 1 else samples

    def condition(self, coords, values, noise=0.0):
        """A new field that samples conditionally on ``values`` observed at ``coords``."""
        X = _as_coords(coords)
        z = np.asarray(values, dtype=float).ravel()
        C = self._cov_fn(_pairwise(X)) + float(noise) * np.eye(len(X))
        C_inv = np.linalg.inv(C)
        base = self

        class _Conditional:
            def sample(inner_self, coords2, n_samples=1, random_state=None):
                X2 = _as_coords(coords2)
                rng = np.random.default_rng(random_state)
                unc = base.sample(np.vstack([X, X2]), n_samples=n_samples, random_state=rng)
                unc = np.atleast_2d(unc)
                unc_data, unc_new = unc[:, :len(X)], unc[:, len(X):]
                Cx0 = base._cov_fn(_pairwise(X2, X))
                correction = (Cx0 @ C_inv @ (z[None, :] - unc_data).T).T
                out = unc_new + correction
                return out[0] if n_samples == 1 else out

        return _Conditional()

    def __repr__(self):
        return f"GaussianRandomField(method={self.method!r})"


class MaternField(GaussianRandomField):
    """A Gaussian random field with a general-nu Matern covariance.

    Uses its own general-nu correlation (``scipy.special.kv``), matching
    ``gaussian_processes.MaternKernel`` exactly at nu in {0.5, 1.5, 2.5} (its only closed
    forms) but not limited to them.
    """

    def __init__(self, nu=1.5, length_scale=1.0, variance=1.0, mean=0.0, method="auto"):
        self.nu = float(nu)
        self.length_scale = float(length_scale)
        self.variance = float(variance)

        def cov_fn(h):
            return self.variance * _matern_corr(np.asarray(h, dtype=float), self.nu,
                                                 self.length_scale)

        super().__init__(covariance=cov_fn, mean=mean, method=method)
        self._cov_fn = cov_fn

    def __repr__(self):
        return f"MaternField(nu={self.nu}, length_scale={self.length_scale})"


class OrnsteinUhlenbeckField:
    """A multi-parameter OU sheet: separable exponential covariance, or isotropic (a GRF).

    On a grid with ``separable=True`` (default), sampling uses per-axis AR(1) recursions --
    exact and O(n) -- since a separable exponential covariance is a Kronecker product of
    1-D OU covariances. ``separable=False`` gives the isotropic exponential covariance
    (a plain ``GaussianRandomField``), and arbitrary (non-grid) points always fall back to
    Cholesky.
    """

    def __init__(self, length_scale=1.0, variance=1.0, separable=True):
        self.length_scale = float(length_scale)
        self.variance = float(variance)
        self.separable = bool(separable)

    def _cov_fn(self, h):
        return self.variance * np.exp(-h / self.length_scale)

    def sample(self, coords, n_samples=1, random_state=None):
        X = _as_coords(coords)
        if self.separable:
            H = np.abs(X[:, None, :] - X[None, :, :])          # per-axis |h_i|
            C = self.variance * np.exp(-H.sum(axis=2) / self.length_scale)
        else:
            C = self._cov_fn(_pairwise(X))
        L, _ = cholesky_with_jitter(C)
        rng = np.random.default_rng(random_state)
        Z = rng.standard_normal((len(X), int(n_samples)))
        out = (L @ Z).T
        return out[0] if n_samples == 1 else out

    def sample_grid(self, shape, spacing=1.0, n_samples=1, random_state=None):
        shape = tuple(int(s) for s in shape)
        sp = np.broadcast_to(np.atleast_1d(spacing).astype(float), (len(shape),))
        if not self.separable:
            pts, _ = _grid_points(shape, sp)
            flat = self.sample(pts, n_samples=n_samples, random_state=random_state)
            return flat.reshape((int(n_samples),) + shape) if n_samples > 1 else flat.reshape(shape)

        rng = np.random.default_rng(random_state)
        n = int(n_samples)
        out = np.empty((n,) + shape)
        var_axis = self.variance ** (1.0 / len(shape))
        for s in range(n):
            field = None
            for axis, (size, step) in enumerate(zip(shape, sp)):
                rho = np.exp(-step / self.length_scale)
                innov = rng.standard_normal(size) * np.sqrt(var_axis * (1.0 - rho ** 2))
                axis_path = np.empty(size)
                axis_path[0] = rng.standard_normal() * np.sqrt(var_axis)
                for t in range(1, size):
                    axis_path[t] = rho * axis_path[t - 1] + innov[t]
                shape_i = [1] * len(shape)
                shape_i[axis] = size
                field = axis_path.reshape(shape_i) if field is None else field * axis_path.reshape(shape_i)
            out[s] = field
        return out[0] if n == 1 else out

    def __repr__(self):
        return f"OrnsteinUhlenbeckField(length_scale={self.length_scale}, separable={self.separable})"


class BrownianSheet:
    """The Chentsov / Brownian sheet: ``Cov(W(s), W(t)) = prod_i min(s_i, t_i)``."""

    def __init__(self, extent=(1.0, 1.0)):
        self.extent = tuple(float(e) for e in extent)

    def sample_grid(self, shape, random_state=None):
        shape = tuple(int(s) for s in shape)
        if len(shape) != len(self.extent):
            raise ValueError("shape must have one entry per extent dimension")
        rng = np.random.default_rng(random_state)
        field = rng.standard_normal(shape)
        # scale by sqrt(cell) and zero each axis's origin slice *before* cumulative-summing
        # any axis -- W must vanish whenever any coordinate is 0, and zeroing only after
        # cumsum leaves earlier terms contaminated by the discarded origin draw
        for axis, (size, ext) in enumerate(zip(shape, self.extent)):
            cell = ext / (size - 1) if size > 1 else ext
            field = field * np.sqrt(cell)
            idx = [slice(None)] * len(shape)
            idx[axis] = slice(0, 1)
            field[tuple(idx)] = 0.0
        for axis in range(len(shape)):
            field = np.cumsum(field, axis=axis)
        return field

    def __repr__(self):
        return f"BrownianSheet(extent={self.extent})"


class FractionalBrownianSheet:
    """Fractional Brownian sheet: ``Cov = prod_i 0.5*(s_i^2H + t_i^2H - |s_i-t_i|^2H)``.

    Exact on a grid via the separable Kronecker structure ``X = L_1 Z L_2^T ...`` (each
    ``L_i`` the Cholesky factor of a 1-D fBm covariance). ``hurst=0.5`` reduces to a plain
    ``BrownianSheet`` in distribution.
    """

    def __init__(self, hurst=(0.5, 0.5), extent=(1.0, 1.0)):
        self.hurst = tuple(float(h) for h in hurst)
        self.extent = tuple(float(e) for e in extent)
        if len(self.hurst) != len(self.extent):
            raise ValueError("hurst and extent must have equal length")
        if any(not (0.0 < h < 1.0) for h in self.hurst):
            raise ValueError("every hurst exponent must be in (0, 1)")

    def _axis_cholesky(self, size, ext, H):
        t = np.linspace(0.0, ext, size)
        C = 0.5 * (t[:, None] ** (2 * H) + t[None, :] ** (2 * H)
                   - np.abs(t[:, None] - t[None, :]) ** (2 * H))
        L, _ = cholesky_with_jitter(C)
        return L

    def sample_grid(self, shape, random_state=None):
        shape = tuple(int(s) for s in shape)
        if len(shape) != len(self.extent):
            raise ValueError("shape must have one entry per extent dimension")
        rng = np.random.default_rng(random_state)
        Ls = [self._axis_cholesky(s, e, h) for s, e, h in zip(shape, self.extent, self.hurst)]
        field = rng.standard_normal(shape)
        # apply each axis's Cholesky factor via tensordot, moving the transformed axis back
        for axis, L in enumerate(Ls):
            field = np.moveaxis(np.tensordot(L, field, axes=([1], [axis])), 0, axis)
        return field

    def __repr__(self):
        return f"FractionalBrownianSheet(hurst={self.hurst})"
