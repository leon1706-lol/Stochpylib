"""Haar-distributed random matrices on the compact classical groups.

``HaarMeasure(group, n)`` samples ``O(n)`` / ``U(n)`` / ``Sp(n)`` (the compact symplectic
group ``USp(2n)``, as ``2n x 2n`` unitary matrices) by Gaussian QR with Mezzadri's (2007)
diagonal correction, which is what makes the result exactly Haar rather than merely
orthogonal/unitary. ``RandomOrthogonalMatrix`` / ``RandomUnitaryMatrix`` /
``RandomSymplectic`` are the per-group facades.
"""

import numpy as np

from stochpylib.random_matrix._common import _rng

__all__ = [
    "HaarMeasure",
    "RandomOrthogonalMatrix",
    "RandomSymplectic",
    "RandomUnitaryMatrix",
]


def _haar_qr(Z):
    """Haar element from a Ginibre matrix: Q of Z = QR with R's diagonal made positive."""
    Q, R = np.linalg.qr(Z)
    d = np.diagonal(R)
    ph = d / np.abs(d)
    return Q * ph  # multiply column j by phase_j


def _symplectic_form(n):
    """J = [[0, I], [-I, 0]] of size 2n."""
    J = np.zeros((2 * n, 2 * n))
    J[:n, n:] = np.eye(n)
    J[n:, :n] = -np.eye(n)
    return J


def _haar_symplectic(n, rng):
    """Haar element of USp(2n) via quaternionic Gram-Schmidt on a quaternion Ginibre matrix.

    A quaternion ``q = a + b j`` acting on ``C^2`` is the block ``[[a, b], [-conj(b), conj(a)]]``;
    a ``2n x 2n`` matrix with that block structure commutes with ``J`` conjugation. Columns
    come in pairs ``(v, J conj(v))`` and orthonormalizing each pair against the previous ones
    with the *complex* inner product yields a unitary matrix that also preserves ``J`` --
    hence lies in ``USp(2n)``. Haar-ness follows from the invariance of the Gaussian input.
    """
    A = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2.0)
    B = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2.0)
    Z = np.block([[A, B], [-B.conj(), A.conj()]])
    J = _symplectic_form(n)
    Q = np.zeros((2 * n, 2 * n), dtype=complex)
    k = 0
    for j in range(n):
        v = Z[:, j].copy()
        for i in range(k):
            v -= Q[:, i] * (Q[:, i].conj() @ v)
        v /= np.linalg.norm(v)
        w = -(J @ v.conj())  # the quaternionic partner, automatically orthogonal to v
        for i in range(k):
            w -= Q[:, i] * (Q[:, i].conj() @ w)
        w -= v * (v.conj() @ w)
        w /= np.linalg.norm(w)
        Q[:, k], Q[:, k + 1] = v, w
        k += 2
    # reorder columns so that Q has the (v_1..v_n | w_1..w_n) block layout matching J
    order = np.concatenate([np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2)])
    return Q[:, order]


class HaarMeasure:
    """Haar (uniform) measure on ``O(n)``, ``U(n)`` or ``Sp(n)`` (``USp(2n)``).

    ``sample(random_state)`` returns one matrix; ``samples(k, random_state)`` a stack
    ``(k, m, m)`` with ``m = n`` (O/U) or ``2n`` (Sp). ``eigenangles`` are the arguments of
    the (unit-modulus) eigenvalues in ``(-pi, pi]``: exactly uniform for ``U(n)`` (the CUE);
    ``O(n)`` carries eigenvalue atoms at ``+-1`` and ``Sp(n)`` conjugate pairs with repulsion
    from ``0`` and ``pi``, so their angle marginals are *not* uniform.
    """

    def __init__(self, group="U", n=2):
        group = str(group).upper()
        if group not in ("O", "U", "SP"):
            raise ValueError("group must be 'O', 'U' or 'Sp'")
        if n < 1:
            raise ValueError("n must be >= 1")
        self.group = "Sp" if group == "SP" else group
        self.n = int(n)
        self.dim = 2 * self.n if self.group == "Sp" else self.n

    def sample(self, random_state=None):
        rng = _rng(random_state)
        n = self.n
        if self.group == "O":
            return _haar_qr(rng.standard_normal((n, n)))
        if self.group == "U":
            Z = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2.0)
            return _haar_qr(Z)
        return _haar_symplectic(n, rng)

    def samples(self, n_samples, random_state=None):
        rng = _rng(random_state)
        return np.stack([self.sample(rng) for _ in range(int(n_samples))])

    def eigenangles(self, random_state=None):
        """Arguments of the eigenvalues of one draw, sorted, in ``(-pi, pi]``."""
        return np.sort(np.angle(np.linalg.eigvals(self.sample(random_state))))

    @staticmethod
    def symplectic_form(n):
        """The standard symplectic form ``J`` of size ``2n`` preserved by ``Sp(n)`` draws."""
        return _symplectic_form(n)


class RandomOrthogonalMatrix(HaarMeasure):
    """Haar-random ``n x n`` orthogonal matrix; ``det=+1`` restricts to ``SO(n)``."""

    def __init__(self, n, det=None):
        super().__init__("O", n)
        if det not in (None, 1, -1):
            raise ValueError("det must be None, +1 or -1")
        self.det = det

    def sample(self, random_state=None):
        Q = super().sample(random_state)
        if self.det is not None and np.sign(np.linalg.det(Q)) != self.det:
            Q = Q.copy()
            Q[:, 0] = -Q[:, 0]  # flipping one column keeps Haar on the target component
        return Q


class RandomUnitaryMatrix(HaarMeasure):
    """Haar-random ``n x n`` unitary matrix (the CUE matrix distribution)."""

    def __init__(self, n):
        super().__init__("U", n)


class RandomSymplectic(HaarMeasure):
    """Haar-random element of ``USp(2n)`` as a ``2n x 2n`` unitary preserving ``J``."""

    def __init__(self, n):
        super().__init__("Sp", n)
