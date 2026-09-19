"""Numerical linear algebra: matrix exponential/logarithm, Cholesky, eigendecomposition,
SVD, QR, and the real Schur form.

Every class is a fluent ``.compute()`` object (mirroring the library's fit/predict
convention): construct with the matrix, call ``.compute()``, then read the ``_``-suffixed
attributes. Native algorithms are the default; ``method='numpy'`` (where offered) uses
``numpy.linalg`` as a fast path -- both are tested against each other and against
``scipy.linalg`` as an independent oracle.
"""

import numpy as np

from stochpylib.numerical_methods._common import _hessenberg, _householder_vector
from stochpylib.numerical_methods.integration import GaussLegendre

__all__ = [
    "MatrixExponential", "MatrixLogarithm", "CholeskyDecomp", "EigenDecomp",
    "SVD", "QRDecomp", "Schur",
]


def _require_square(A, name="A"):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be a square 2-D array")
    return A


def _not_computed():
    raise RuntimeError("call .compute() first")


# ------------------------------------------------------------- MatrixExponential

_PADE13_B = np.array([
    64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
    1187353796428800.0, 129060195264000.0, 10559470521600.0,
    670442572800.0, 33522128640.0, 1323241920.0, 40840800.0,
    960960.0, 16380.0, 182.0, 1.0,
])
_PADE13_THETA = 5.371920351148152


class MatrixExponential:
    """``exp(A)`` via scaling-and-squaring with the degree-13 Pade approximant
    (Higham 2005), or a Taylor/eigendecomposition fallback.
    """

    def __init__(self, A):
        self.A = _require_square(A)

    def compute(self, method="pade"):
        if method not in ("pade", "eig", "taylor"):
            raise ValueError("method must be 'pade', 'eig' or 'taylor'")
        if method == "pade":
            self.expm_ = _expm_pade13(self.A)
        elif method == "eig":
            self.expm_ = _expm_eig(self.A)
        else:
            self.expm_ = _expm_taylor(self.A)
        self.method_ = method
        return self

    def at(self, t):
        return _expm_pade13(t * self.A)

    def apply(self, v, t=1.0):
        return _expm_pade13(t * self.A) @ np.asarray(v, dtype=float)

    def __repr__(self):
        return f"MatrixExponential(n={self.A.shape[0]})"


def _expm_pade13(A):
    n = A.shape[0]
    norm1 = np.max(np.sum(np.abs(A), axis=0))
    s = max(0, int(np.ceil(np.log2(norm1 / _PADE13_THETA)))) if norm1 > 0 else 0
    As = A / (2.0 ** s)
    b = _PADE13_B
    I = np.eye(n)
    A2 = As @ As
    A4 = A2 @ A2
    A6 = A2 @ A4
    U = As @ (A6 @ (b[13] * A6 + b[11] * A4 + b[9] * A2)
             + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * I)
    V = A6 @ (b[12] * A6 + b[10] * A4 + b[8] * A2) + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * I
    P = V + U
    Q = V - U
    R = np.linalg.solve(Q, P)
    for _ in range(s):
        R = R @ R
    return R


def _expm_eig(A):
    if not np.allclose(A, A.T):
        raise ValueError("method='eig' requires a symmetric matrix")
    w, V = np.linalg.eigh(A)
    return (V * np.exp(w)) @ V.T


def _expm_taylor(A, terms=60):
    n = A.shape[0]
    norm1 = np.max(np.sum(np.abs(A), axis=0))
    s = max(0, int(np.ceil(np.log2(max(norm1, 1e-300))))) if norm1 > 1 else 0
    As = A / (2.0 ** s)
    term = np.eye(n)
    total = np.eye(n)
    for k in range(1, terms + 1):
        term = term @ As / k
        total = total + term
    for _ in range(s):
        total = total @ total
    return total


# ------------------------------------------------------------- MatrixLogarithm

class MatrixLogarithm:
    """``log(A)`` (principal branch) via inverse scaling-and-squaring (Denman-Beavers
    square roots + a Gauss-Legendre partial-fraction Pade evaluation of ``log(I+X)``),
    or an eigendecomposition fallback for symmetric positive-definite ``A``.
    """

    def __init__(self, A):
        self.A = _require_square(A)

    def compute(self, method="inverse_scaling"):
        if method not in ("inverse_scaling", "eig"):
            raise ValueError("method must be 'inverse_scaling' or 'eig'")
        A = self.A
        if method == "eig":
            if not np.allclose(A, A.T):
                raise ValueError("method='eig' requires a symmetric matrix")
            w, V = np.linalg.eigh(A)
            if np.any(w <= 0):
                raise ValueError("A has non-positive eigenvalues; log is undefined (eig path)")
            self.logm_ = (V * np.log(w)) @ V.T
            self.method_ = method
            return self
        eig = np.linalg.eigvals(A)
        if np.any((eig.real <= 0) & (np.abs(eig.imag) < 1e-10 * np.maximum(np.abs(eig.real), 1))):
            raise ValueError("A has a non-positive real eigenvalue; principal log is undefined")
        self.logm_ = _logm_inverse_scaling(A)
        self.method_ = method
        return self

    def __repr__(self):
        return f"MatrixLogarithm(n={self.A.shape[0]})"


def _matrix_sqrt_denman_beavers(A, max_iter=100, tol=1e-13):
    n = A.shape[0]
    Y = A.copy()
    Z = np.eye(n)
    for _ in range(max_iter):
        Y_inv = np.linalg.inv(Y)
        Z_inv = np.linalg.inv(Z)
        Y_new = 0.5 * (Y + Z_inv)
        Z_new = 0.5 * (Z + Y_inv)
        if np.linalg.norm(Y_new - Y, 1) < tol * np.linalg.norm(Y_new, 1):
            Y, Z = Y_new, Z_new
            break
        Y, Z = Y_new, Z_new
    return Y


def _logm_inverse_scaling(A):
    n = A.shape[0]
    I = np.eye(n)
    k = 0
    X = A.copy()
    while np.linalg.norm(X - I, 1) > 0.25 and k < 50:
        X = _matrix_sqrt_denman_beavers(X)
        k += 1
    Y = X - I
    gl = GaussLegendre(8)
    x_nodes, w_nodes = gl.nodes_weights(0.0, 1.0)
    log_Y = np.zeros((n, n))
    for xi, wi in zip(x_nodes, w_nodes):
        log_Y += wi * np.linalg.solve(I + xi * Y, Y)
    return log_Y * (2 ** k)


# ------------------------------------------------------------- CholeskyDecomp

class CholeskyDecomp:
    """Native Cholesky decomposition ``A = L L^T``, with optional escalating jitter."""

    def __init__(self, A, jitter=False):
        A = _require_square(A)
        if not np.allclose(A, A.T, atol=1e-10 * max(1.0, np.max(np.abs(A)))):
            raise ValueError("A must be symmetric")
        self.A = A
        self.jitter = bool(jitter)
        self._compute()

    def _compute(self):
        A = self.A
        n = A.shape[0]
        jitter_used = 0.0
        if self.jitter:
            eps = 1e-12
            for attempt in range(8):
                try:
                    self.L_ = _cholesky_native(A + jitter_used * np.eye(n))
                    self.jitter_used_ = jitter_used
                    return
                except np.linalg.LinAlgError:
                    jitter_used = eps if attempt == 0 else jitter_used * 10.0
                    eps = jitter_used
            raise np.linalg.LinAlgError("matrix not positive definite even with jitter")
        self.L_ = _cholesky_native(A)
        self.jitter_used_ = 0.0

    def solve(self, b):
        b = np.asarray(b, dtype=float)
        L = self.L_
        y = _forward_sub(L, b)
        return _back_sub(L.T, y)

    def logdet(self):
        return 2.0 * float(np.sum(np.log(np.diag(self.L_))))

    def inverse(self):
        n = self.L_.shape[0]
        return self.solve(np.eye(n))

    def reconstruct(self):
        return self.L_ @ self.L_.T

    def __repr__(self):
        return f"CholeskyDecomp(n={self.A.shape[0]}, jitter_used={self.jitter_used_:.2g})"


def _cholesky_native(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        s = A[i, i] - np.dot(L[i, :i], L[i, :i])
        if s <= 0:
            raise np.linalg.LinAlgError("matrix is not positive definite")
        L[i, i] = np.sqrt(s)
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - np.dot(L[j, :i], L[i, :i])) / L[i, i]
    return L


def _forward_sub(L, b):
    n = L.shape[0]
    vec = b.ndim == 1
    B = b[:, None] if vec else b
    X = np.empty_like(B)
    for i in range(n):
        X[i] = (B[i] - L[i, :i] @ X[:i]) / L[i, i]
    return X[:, 0] if vec else X


def _back_sub(U, b):
    n = U.shape[0]
    vec = b.ndim == 1
    B = b[:, None] if vec else b
    X = np.empty_like(B)
    for i in range(n - 1, -1, -1):
        X[i] = (B[i] - U[i, i + 1:] @ X[i + 1:]) / U[i, i]
    return X[:, 0] if vec else X


# ------------------------------------------------------------- EigenDecomp

class EigenDecomp:
    """Eigendecomposition: cyclic Jacobi (symmetric) or Hessenberg + shifted QR (general)."""

    def __init__(self, A, symmetric=None):
        self.A = _require_square(A)
        self.symmetric = (bool(np.allclose(self.A, self.A.T)) if symmetric is None
                          else bool(symmetric))

    def compute(self, method="auto"):
        if method not in ("auto", "jacobi", "qr", "numpy"):
            raise ValueError("method must be 'auto', 'jacobi', 'qr' or 'numpy'")
        if method == "auto":
            method = "jacobi" if self.symmetric else "qr"
        if method == "numpy":
            if self.symmetric:
                w, V = np.linalg.eigh(self.A)
            else:
                w, V = np.linalg.eig(self.A)
            order = np.argsort(w.real if not self.symmetric else w)
            self.eigenvalues_, self.eigenvectors_ = w[order], V[:, order]
        elif method == "jacobi":
            if not self.symmetric:
                raise ValueError("method='jacobi' requires a symmetric matrix")
            self.eigenvalues_, self.eigenvectors_ = _jacobi_eigh(self.A)
        else:
            self.eigenvalues_, self.eigenvectors_ = _qr_eig(self.A)
        self.method_ = method
        return self

    def reconstruct(self):
        return (self.eigenvectors_ * self.eigenvalues_) @ np.linalg.inv(self.eigenvectors_)

    def condition_number(self):
        w = np.abs(self.eigenvalues_)
        return float(np.max(w) / np.min(w[w > 0])) if np.any(w > 0) else float("inf")

    def spectral_radius(self):
        return float(np.max(np.abs(self.eigenvalues_)))

    def power_iteration(self, n_iter=1000, tol=1e-12, random_state=None):
        rng = np.random.default_rng(random_state)
        n = self.A.shape[0]
        v = rng.standard_normal(n)
        v /= np.linalg.norm(v)
        lam_old = 0.0
        for _ in range(int(n_iter)):
            w = self.A @ v
            nrm = np.linalg.norm(w)
            if nrm < 1e-300:
                break
            v = w / nrm
            lam = float(v @ self.A @ v)
            if abs(lam - lam_old) < tol:
                lam_old = lam
                break
            lam_old = lam
        return lam_old, v

    def __repr__(self):
        return f"EigenDecomp(n={self.A.shape[0]}, symmetric={self.symmetric})"


def _jacobi_eigh(A, max_sweeps=100, tol=1e-14):
    A = A.copy()
    n = A.shape[0]
    V = np.eye(n)
    for _ in range(max_sweeps):
        off = np.sqrt(np.sum(A[np.triu_indices(n, 1)] ** 2))
        if off < tol * max(1.0, np.linalg.norm(np.diag(A))):
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p, q]) < 1e-300:
                    continue
                theta = (A[q, q] - A[p, p]) / (2.0 * A[p, q])
                t = np.sign(theta) / (abs(theta) + np.sqrt(1 + theta ** 2)) if theta != 0 else 1.0
                c = 1.0 / np.sqrt(1 + t ** 2)
                s = t * c
                Ap, Aq = A[:, p].copy(), A[:, q].copy()
                A[:, p] = c * Ap - s * Aq
                A[:, q] = s * Ap + c * Aq
                Ap, Aq = A[p, :].copy(), A[q, :].copy()
                A[p, :] = c * Ap - s * Aq
                A[q, :] = s * Ap + c * Aq
                Vp, Vq = V[:, p].copy(), V[:, q].copy()
                V[:, p] = c * Vp - s * Vq
                V[:, q] = s * Vp + c * Vq
    eigvals = np.diag(A)
    order = np.argsort(eigvals)
    return eigvals[order], V[:, order]


def _qr_eig(A, max_iter=500, tol=1e-12):
    H, Q0 = _hessenberg(A)
    n = H.shape[0]
    Hk = H.copy()
    Qtot = Q0.copy()
    m = n
    it = 0
    while m > 1 and it < max_iter * n:
        it += 1
        # deflation check
        if abs(Hk[m - 1, m - 2]) < tol * (abs(Hk[m - 1, m - 1]) + abs(Hk[m - 2, m - 2]) + 1e-300):
            Hk[m - 1, m - 2] = 0.0
            m -= 1
            continue
        mu = Hk[m - 1, m - 1]
        Q, R = _qr_householder(Hk[:m, :m] - mu * np.eye(m))
        Hk[:m, :m] = R @ Q + mu * np.eye(m)
        Qtot[:, :m] = Qtot[:, :m] @ Q
    eigvals = np.diag(Hk).copy()
    # handle any un-deflated 2x2 real blocks (complex pair) by their trace/2 real part
    i = 0
    while i < n - 1:
        if abs(Hk[i + 1, i]) > 1e-8 * (abs(Hk[i, i]) + abs(Hk[i + 1, i + 1]) + 1e-300):
            a, b_, c_, d = Hk[i, i], Hk[i, i + 1], Hk[i + 1, i], Hk[i + 1, i + 1]
            tr, det = a + d, a * d - b_ * c_
            disc = tr ** 2 - 4 * det
            if disc < 0:
                eigvals = eigvals.astype(complex)
                eigvals[i] = (tr + np.sqrt(complex(disc))) / 2
                eigvals[i + 1] = (tr - np.sqrt(complex(disc))) / 2
            else:
                eigvals[i] = (tr + np.sqrt(disc)) / 2
                eigvals[i + 1] = (tr - np.sqrt(disc)) / 2
            i += 2
        else:
            i += 1
    order = np.argsort(eigvals.real if np.iscomplexobj(eigvals) else eigvals)
    eigvals = eigvals[order]
    V = _eigenvectors_inverse_iteration(A, eigvals)
    return eigvals, V


def _qr_householder(A):
    A = A.copy()
    n = A.shape[0]
    Q = np.eye(n)
    for k in range(n - 1):
        x = A[k:, k]
        v, beta = _householder_vector(x)
        if beta == 0.0:
            continue
        Hk = np.eye(n - k) - beta * np.outer(v, v)
        A[k:, :] = Hk @ A[k:, :]
        Qk = np.eye(n)
        Qk[k:, k:] = Hk
        Q = Q @ Qk
    return Q, A


def _eigenvectors_inverse_iteration(A, eigvals, n_iter=3, random_state=0):
    n = A.shape[0]
    rng = np.random.default_rng(random_state)
    V = np.zeros((n, n), dtype=eigvals.dtype)
    Ac = A.astype(eigvals.dtype)
    for j, lam in enumerate(eigvals):
        shifted = Ac - (lam + 1e-9 * (abs(lam) + 1)) * np.eye(n)
        v = rng.standard_normal(n).astype(eigvals.dtype)
        v /= np.linalg.norm(v)
        for _ in range(n_iter):
            try:
                w = np.linalg.solve(shifted, v)
            except np.linalg.LinAlgError:
                shifted = shifted + 1e-6 * np.eye(n)
                w = np.linalg.solve(shifted, v)
            nrm = np.linalg.norm(w)
            if nrm < 1e-300:
                break
            v = w / nrm
        V[:, j] = v
    return V


# ------------------------------------------------------------- SVD

class SVD:
    """Singular value decomposition via one-sided Jacobi (Hestenes), or ``numpy.linalg.svd``."""

    def __init__(self, A):
        self.A = np.asarray(A, dtype=float)
        if self.A.ndim != 2:
            raise ValueError("A must be a 2-D array")

    def compute(self, method="jacobi"):
        if method not in ("jacobi", "numpy"):
            raise ValueError("method must be 'jacobi' or 'numpy'")
        if method == "numpy":
            U, s, Vt = np.linalg.svd(self.A, full_matrices=False)
        else:
            U, s, Vt = _svd_one_sided_jacobi(self.A)
        self.U_, self.s_, self.Vt_ = U, s, Vt
        self.method_ = method
        return self

    def rank(self, tol=None):
        s = self.s_
        tol = tol if tol is not None else np.max(self.A.shape) * np.finfo(float).eps * np.max(s)
        return int(np.sum(s > tol))

    def pinv(self, tol=None):
        s = self.s_
        tol = tol if tol is not None else np.max(self.A.shape) * np.finfo(float).eps * np.max(s)
        s_inv = np.where(s > tol, 1.0 / np.where(s > tol, s, 1.0), 0.0)
        return (self.Vt_.T * s_inv) @ self.U_.T

    def low_rank(self, k):
        k = int(k)
        return (self.U_[:, :k] * self.s_[:k]) @ self.Vt_[:k, :]

    def condition_number(self):
        s = self.s_
        return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")

    def nuclear_norm(self):
        return float(np.sum(self.s_))

    def reconstruct(self):
        return (self.U_ * self.s_) @ self.Vt_

    def __repr__(self):
        return f"SVD(shape={self.A.shape})"


def _svd_one_sided_jacobi(A, max_sweeps=60, tol=1e-14):
    m, n = A.shape
    transposed = m < n
    B = A.T.copy() if transposed else A.copy()
    mB, nB = B.shape
    V = np.eye(nB)
    for _ in range(max_sweeps):
        off = 0.0
        for p in range(nB - 1):
            for q in range(p + 1, nB):
                alpha = float(B[:, p] @ B[:, p])
                beta = float(B[:, q] @ B[:, q])
                gamma = float(B[:, p] @ B[:, q])
                off += gamma ** 2
                if abs(gamma) < tol * np.sqrt(alpha * beta + 1e-300):
                    continue
                zeta = (beta - alpha) / (2.0 * gamma)
                t = np.sign(zeta) / (abs(zeta) + np.sqrt(1 + zeta ** 2)) if zeta != 0 else 1.0
                c = 1.0 / np.sqrt(1 + t ** 2)
                s = c * t
                Bp, Bq = B[:, p].copy(), B[:, q].copy()
                B[:, p] = c * Bp - s * Bq
                B[:, q] = s * Bp + c * Bq
                Vp, Vq = V[:, p].copy(), V[:, q].copy()
                V[:, p] = c * Vp - s * Vq
                V[:, q] = s * Vp + c * Vq
        if off < tol:
            break
    sing = np.linalg.norm(B, axis=0)
    order = np.argsort(sing)[::-1]
    sing = sing[order]
    V = V[:, order]
    nz = sing > 1e-300
    U = np.zeros((mB, nB))
    U[:, nz] = B[:, order][:, nz] / sing[nz]
    if not np.all(nz):
        # fill zero-singular-value columns with an orthonormal complement
        Q, _ = np.linalg.qr(np.eye(mB))
        free = list(np.where(~nz)[0])
        for j in free:
            U[:, j] = Q[:, j % mB]
    if transposed:
        return V, sing, U.T
    return U, sing, V.T


# ------------------------------------------------------------- QRDecomp

class QRDecomp:
    """QR decomposition via Householder reflections (default), modified Gram-Schmidt,
    or Givens rotations.
    """

    def __init__(self, A, mode="reduced"):
        self.A = np.asarray(A, dtype=float)
        if self.A.ndim != 2:
            raise ValueError("A must be a 2-D array")
        if mode not in ("reduced", "complete"):
            raise ValueError("mode must be 'reduced' or 'complete'")
        self.mode = mode

    def compute(self, method="householder"):
        if method not in ("householder", "gram_schmidt", "givens", "numpy"):
            raise ValueError("method must be 'householder', 'gram_schmidt', 'givens' or 'numpy'")
        A = self.A
        m, n = A.shape
        if method == "numpy":
            Q, R = np.linalg.qr(A, mode=self.mode)
        elif method == "householder":
            Q, R = _qr_householder_rect(A)
        elif method == "givens":
            Q, R = _qr_givens(A)
        else:
            Q, R = _qr_mgs(A)
        if self.mode == "reduced" and method != "numpy":
            k = min(m, n)
            Q, R = Q[:, :k], R[:k, :]
        self.Q_, self.R_ = Q, R
        self.method_ = method
        return self

    def solve_least_squares(self, b):
        b = np.asarray(b, dtype=float)
        Qtb = self.Q_.T @ b
        k = self.R_.shape[0]
        return _back_sub(self.R_[:k, :k], Qtb[:k] if Qtb.ndim == 1 else Qtb[:k, :])

    def reconstruct(self):
        return self.Q_ @ self.R_

    def __repr__(self):
        return f"QRDecomp(shape={self.A.shape}, mode={self.mode!r})"


def _qr_householder_rect(A):
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    for k in range(min(m - 1, n)):
        x = R[k:, k]
        v, beta = _householder_vector(x)
        if beta == 0.0:
            continue
        Hk = np.eye(m - k) - beta * np.outer(v, v)
        R[k:, :] = Hk @ R[k:, :]
        Qk = np.eye(m)
        Qk[k:, k:] = Hk
        Q = Q @ Qk
    return Q, R


def _qr_mgs(A):
    m, n = A.shape
    V = A.copy()
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        for i in range(j):
            R[i, j] = Q[:, i] @ V[:, j]
            V[:, j] -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(V[:, j])
        Q[:, j] = V[:, j] / R[j, j] if R[j, j] > 1e-300 else V[:, j]
    return Q, R


def _qr_givens(A):
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    for j in range(min(n, m - 1)):
        for i in range(m - 1, j, -1):
            a, b_ = R[i - 1, j], R[i, j]
            if abs(b_) < 1e-300:
                continue
            r = np.hypot(a, b_)
            c, s = a / r, -b_ / r
            G = np.eye(m)
            G[i - 1, i - 1], G[i - 1, i] = c, -s
            G[i, i - 1], G[i, i] = s, c
            R = G @ R
            Q = Q @ G.T
    return Q, R


# ------------------------------------------------------------- Schur

class Schur:
    """Real or complex Schur decomposition ``A = Z T Z^*`` via Hessenberg reduction
    followed by the (double-shift Francis / single-shift Wilkinson) QR algorithm.
    """

    def __init__(self, A):
        self.A = _require_square(A)

    def compute(self, output="real"):
        if output not in ("real", "complex"):
            raise ValueError("output must be 'real' or 'complex'")
        if output == "real":
            self.T_, self.Z_ = _real_schur(self.A)
        else:
            self.T_, self.Z_ = _complex_schur(self.A)
        self.output_ = output
        return self

    def eigenvalues(self):
        T = self.T_
        n = T.shape[0]
        if self.output_ == "complex":
            return np.diag(T).copy()
        eig = np.zeros(n, dtype=complex)
        i = 0
        while i < n:
            if i < n - 1 and abs(T[i + 1, i]) > 1e-10 * (abs(T[i, i]) + abs(T[i + 1, i + 1]) + 1e-300):
                a, b_, c_, d = T[i, i], T[i, i + 1], T[i + 1, i], T[i + 1, i + 1]
                tr, det = a + d, a * d - b_ * c_
                disc = tr ** 2 - 4 * det
                sq = np.sqrt(complex(disc))
                eig[i] = (tr + sq) / 2
                eig[i + 1] = (tr - sq) / 2
                i += 2
            else:
                eig[i] = T[i, i]
                i += 1
        if np.allclose(eig.imag, 0.0):
            return eig.real
        return eig

    def reconstruct(self):
        Z = self.Z_
        return Z @ self.T_ @ (Z.conj().T if self.output_ == "complex" else Z.T)

    def __repr__(self):
        return f"Schur(n={self.A.shape[0]})"


def _real_schur(A, max_iter=500, tol=1e-13):
    """Francis implicit double-shift QR (Golub & Van Loan Alg. 7.5.1/7.5.2), 0-indexed.

    Deflates from the bottom: ``m`` tracks the size of the still-active leading
    ``H[:m, :m]`` block. Each outer step either deflates (a negligible last subdiagonal
    entry) or applies one double-shift bulge-chase sweep across the whole active block.
    Householder reflectors are applied to full matrix rows/columns (scoped to the active
    range) so ``H`` stays consistent for the next iteration.
    """
    H, Q0 = _hessenberg(A)
    n = H.shape[0]
    Hk = H.copy()
    Z = Q0.copy()
    m = n
    it = 0
    while m > 2 and it < max_iter * n:
        it += 1
        if abs(Hk[m - 1, m - 2]) < tol * (abs(Hk[m - 1, m - 1]) + abs(Hk[m - 2, m - 2]) + 1e-300):
            Hk[m - 1, m - 2] = 0.0
            m -= 1
            continue
        if abs(Hk[m - 2, m - 3]) < tol * (abs(Hk[m - 2, m - 2]) + abs(Hk[m - 3, m - 3]) + 1e-300):
            Hk[m - 2, m - 3] = 0.0
            m -= 2
            continue
        # double-shift values from the trailing 2x2 of the active block
        a, b_, c_, d = Hk[m - 2, m - 2], Hk[m - 2, m - 1], Hk[m - 1, m - 2], Hk[m - 1, m - 1]
        s = a + d
        p = a * d - b_ * c_
        x = Hk[0, 0] ** 2 + Hk[0, 1] * Hk[1, 0] - s * Hk[0, 0] + p
        y = Hk[1, 0] * (Hk[0, 0] + Hk[1, 1] - s)
        z = Hk[2, 1] * Hk[1, 0] if m > 2 else 0.0
        for k in range(m - 1):
            last = (k == m - 2)
            col = np.array([x, y]) if last else np.array([x, y, z])
            v, beta = _householder_vector(col)
            r = k + len(col)
            Hb = np.eye(len(col)) - beta * np.outer(v, v)
            lo = max(k - 1, 0)
            Hk[k:r, lo:] = Hb @ Hk[k:r, lo:]
            hi = min(k + 4, m)
            Hk[:hi, k:r] = Hk[:hi, k:r] @ Hb
            Z[:, k:r] = Z[:, k:r] @ Hb
            if not last:
                x, y = Hk[k + 1, k], Hk[k + 2, k]
                z = Hk[k + 3, k] if k + 3 < m else 0.0
    # remaining m<=2 block is left as-is: a real pair (both deflate to 1x1 via the
    # abs(subdiag) check above on the next outer call site) or a genuine complex-
    # conjugate 2x2 block, exactly what the real Schur form is meant to carry.
    for i in range(2, n):
        Hk[i, :i - 1] = 0.0
    return Hk, Z


def _complex_schur(A, max_iter=1000, tol=1e-12):
    """Complex single-shift QR. The Wilkinson shift is the trailing-2x2 eigenvalue
    closest to ``H[m-1,m-1]`` (generally complex, via the quadratic formula) -- a real
    shift alone does not converge on genuinely complex eigenvalues.
    """
    n = A.shape[0]
    H, Q0 = _hessenberg(A)
    Hk = H.astype(complex)
    Z = Q0.astype(complex)
    m = n
    it = 0
    while m > 1 and it < max_iter * n:
        it += 1
        if abs(Hk[m - 1, m - 2]) < tol * (abs(Hk[m - 1, m - 1]) + abs(Hk[m - 2, m - 2]) + 1e-300):
            Hk[m - 1, m - 2] = 0.0
            m -= 1
            continue
        if m == 1:
            break
        a, b_, c_, d = Hk[m - 2, m - 2], Hk[m - 2, m - 1], Hk[m - 1, m - 2], Hk[m - 1, m - 1]
        tr, det = a + d, a * d - b_ * c_
        disc = np.sqrt((tr / 2) ** 2 - det + 0j)
        lam1, lam2 = tr / 2 + disc, tr / 2 - disc
        mu = lam1 if abs(lam1 - d) <= abs(lam2 - d) else lam2
        Q, R = _qr_householder_complex(Hk[:m, :m] - mu * np.eye(m))
        Hk[:m, :m] = R @ Q + mu * np.eye(m)
        if m < n:
            Hk[:m, m:] = Q.conj().T @ Hk[:m, m:]  # full similarity: Qfull = diag(Q, I)
        Z[:, :m] = Z[:, :m] @ Q
    for i in range(1, n):
        Hk[i, :i - 1] = 0.0
    return Hk, Z


def _qr_householder_complex(A):
    A = A.copy()
    n = A.shape[0]
    Q = np.eye(n, dtype=complex)
    for k in range(n - 1):
        x = A[k:, k]
        nrm = np.linalg.norm(x)
        if nrm < 1e-300:
            continue
        phase = x[0] / abs(x[0]) if abs(x[0]) > 1e-300 else 1.0
        v = x.copy()
        v[0] = x[0] + phase * nrm
        vnorm = np.linalg.norm(v)
        if vnorm < 1e-300:
            continue
        v = v / vnorm
        Hk = np.eye(len(x), dtype=complex) - 2.0 * np.outer(v, v.conj())
        A[k:, :] = Hk @ A[k:, :]
        Qk = np.eye(n, dtype=complex)
        Qk[k:, k:] = Hk
        Q = Q @ Qk.conj().T
    return Q, A
