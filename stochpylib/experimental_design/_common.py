"""Shared numerics for :mod:`stochpylib.experimental_design`.

Model-term expansion, finite-field (GF(q)) arithmetic behind the combinatorial designs,
mutually orthogonal Latin squares, the four L2-type discrepancies, distance metrics and
small correlation/rank helpers. Everything is native numpy; scipy.stats is the test
suite's oracle only.
"""

import itertools
import math
from functools import lru_cache

import numpy as np

# DOE convention: "I" is the identity word in a defining relation, never a factor name.
_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def _rng(random_state):
    return np.random.default_rng(random_state)


def _factor_names(k):
    return [(_LETTERS[i] if i < len(_LETTERS) else f"x{i + 1}") for i in range(k)]


def _as_2d(X, name="X"):
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty (n, k) array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _normalize_bounds(bounds, k):
    b = np.asarray(bounds, dtype=float)
    if b.ndim == 1 and b.size == 2:
        b = np.tile(b, (k, 1))
    if b.shape != (k, 2):
        raise ValueError(f"bounds must be (lo, hi) or a ({k}, 2) array")
    if np.any(b[:, 1] <= b[:, 0]):
        raise ValueError("every bound needs hi > lo")
    return b


# ------------------------------------------------------------------------ model terms

def _model_terms(k, model):
    """Exponent tuples (intercept first) for a named polynomial model over k factors."""
    zero = (0,) * k

    def unit(*idx):
        e = [0] * k
        for i in idx:
            e[i] += 1
        return tuple(e)

    if isinstance(model, str):
        mains = [unit(i) for i in range(k)]
        pairs = [unit(i, j) for i, j in itertools.combinations(range(k), 2)]
        squares = [unit(i, i) for i in range(k)]
        if model == "linear":
            return [zero] + mains
        if model == "interaction":
            return [zero] + mains + pairs
        if model == "quadratic":
            return [zero] + mains + pairs + squares
        if model == "full":
            terms = [zero]
            for r in range(1, k + 1):
                terms += [unit(*c) for c in itertools.combinations(range(k), r)]
            return terms
        if model == "cubic":
            terms = [zero]
            for deg in (1, 2, 3):
                for c in itertools.combinations_with_replacement(range(k), deg):
                    terms.append(unit(*c))
            return terms
        raise ValueError("model must be 'linear', 'interaction', 'quadratic', 'cubic', "
                         "'full', a list of exponent tuples, or a callable")
    terms = [tuple(int(v) for v in t) for t in model]
    if any(len(t) != k for t in terms):
        raise ValueError(f"every exponent tuple must have length {k}")
    return terms


def _term_name(exponents, names):
    parts = []
    for name, e in zip(names, exponents):
        if e == 1:
            parts.append(name)
        elif e > 1:
            parts.append(f"{name}^{e}")
    return "*".join(parts) if parts else "1"


def _model_matrix(X, model="linear", names=None):
    """Model matrix ``F`` and its column labels for a design ``X``."""
    X = _as_2d(X)
    n, k = X.shape
    if callable(model):
        F = np.asarray(model(X), dtype=float)
        if F.ndim == 1:
            F = F[:, None]
        return F, [f"f{j}" for j in range(F.shape[1])]
    terms = _model_terms(k, model)
    names = names or _factor_names(k)
    F = np.empty((n, len(terms)))
    for j, t in enumerate(terms):
        F[:, j] = np.prod(X ** np.asarray(t, dtype=float), axis=1)
    return F, [_term_name(t, names) for t in terms]


def _logdet(M):
    sign, val = np.linalg.slogdet(M)
    return float(val) if sign > 0 else -np.inf


# ---------------------------------------------------------------------- finite fields

def _is_prime(n):
    if n < 2:
        return False
    for p in range(2, int(math.isqrt(n)) + 1):
        if n % p == 0:
            return False
    return True


def _is_prime_power(q):
    """``(p, m)`` with ``q == p**m`` for prime p, else None."""
    if q < 2:
        return None
    for p in range(2, q + 1):
        if q % p == 0:
            if not _is_prime(p):
                return None
            m, r = 0, q
            while r % p == 0:
                r //= p
                m += 1
            return (p, m) if r == 1 else None
    return None


def _poly_mulmod(a, b, mod, p):
    """Multiply coefficient lists (lowest degree first) mod p, reduced by monic ``mod``."""
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] = (out[i + j] + ai * bj) % p
    m = len(mod) - 1
    for deg in range(len(out) - 1, m - 1, -1):
        c = out[deg]
        if c:
            for i in range(m + 1):
                out[deg - m + i] = (out[deg - m + i] - c * mod[i]) % p
    return (out + [0] * m)[:m]


def _irreducible(p, m):
    """First monic irreducible polynomial of degree m over GF(p) (lowest degree first)."""
    if m == 1:
        return [0, 1]
    for tail in itertools.product(range(p), repeat=m):
        poly = list(tail) + [1]
        if poly[0] == 0:
            continue
        # irreducible iff it has no monic factor of degree 1..m//2
        reducible = False
        for d in range(1, m // 2 + 1):
            for ftail in itertools.product(range(p), repeat=d):
                f = list(ftail) + [1]
                if _poly_divides(f, poly, p):
                    reducible = True
                    break
            if reducible:
                break
        if not reducible:
            return poly
    raise RuntimeError(f"no irreducible polynomial of degree {m} over GF({p})")


def _poly_divides(f, g, p):
    r = list(g)
    df = len(f) - 1
    inv_lead = pow(f[-1], p - 2, p)
    while len(r) - 1 >= df and any(r):
        c = (r[-1] * inv_lead) % p
        shift = len(r) - 1 - df
        for i, fi in enumerate(f):
            r[shift + i] = (r[shift + i] - c * fi) % p
        while r and r[-1] == 0:
            r.pop()
    return not any(r)


@lru_cache(maxsize=None)
def _gf_tables(q):
    """Addition and multiplication tables of GF(q), elements encoded as ints 0..q-1."""
    pm = _is_prime_power(q)
    if pm is None:
        raise ValueError(f"GF({q}) does not exist: {q} is not a prime power")
    p, m = pm
    if m == 1:
        e = np.arange(q)
        return (e[:, None] + e[None, :]) % q, (e[:, None] * e[None, :]) % q
    mod = _irreducible(p, m)
    digits = [[(v // p ** i) % p for i in range(m)] for v in range(q)]

    def encode(d):
        return int(sum(di * p ** i for i, di in enumerate(d)))

    add = np.empty((q, q), dtype=int)
    mul = np.empty((q, q), dtype=int)
    for a in range(q):
        for b in range(q):
            add[a, b] = encode([(x + y) % p for x, y in zip(digits[a], digits[b])])
            mul[a, b] = encode(_poly_mulmod(digits[a], digits[b], mod, p))
    return add, mul


def _gf_squares(q):
    _, mul = _gf_tables(q)
    return {int(mul[a, a]) for a in range(1, q)}


def _gf_neg(q):
    add, _ = _gf_tables(q)
    return np.array([int(np.where(add[a] == 0)[0][0]) for a in range(q)])


# ------------------------------------------------------ mutually orthogonal Latin squares

def _mols_prime_power(q):
    """Two orthogonal Latin squares of prime-power order q: L_a(i, j) = a*i + j."""
    add, mul = _gf_tables(q)
    i = np.arange(q)
    L1 = add[mul[1][i][:, None], i[None, :]]
    L2 = add[mul[2 if q > 2 else 1][i][:, None], i[None, :]]
    return L1, L2


def _prime_power_factors(n):
    out, r, p = [], n, 2
    while r > 1:
        if r % p == 0:
            q = 1
            while r % p == 0:
                r //= p
                q *= p
            out.append(q)
        p += 1
    return out


def _mols_pair(n):
    """A pair of orthogonal Latin squares of order n (n not congruent to 2 mod 4)."""
    if n < 3 or n % 4 == 2:
        raise ValueError(
            f"no Graeco-Latin square of order {n} is constructed here: orders 2 and 6 are "
            "impossible (Euler/Tarry), and orders 10, 14, ... (n = 2 mod 4) exist but need the "
            "Bose-Shrikhande-Parker construction, which is not implemented")
    A = np.zeros((1, 1), dtype=int)
    B = np.zeros((1, 1), dtype=int)
    for q in _prime_power_factors(n):
        a, b = _mols_prime_power(q)
        # Kronecker product of MOLS is MOLS: entry = outer * q + inner
        A = (A[:, None, :, None] * q + a[None, :, None, :]).reshape(len(A) * q, -1)
        B = (B[:, None, :, None] * q + b[None, :, None, :]).reshape(len(B) * q, -1)
    return A, B


def _is_latin(L):
    n = len(L)
    target = set(range(n))
    return (all(set(row) == target for row in L.tolist())
            and all(set(col) == target for col in L.T.tolist()))


def _are_orthogonal(A, B):
    n = len(A)
    return len({(int(a), int(b)) for a, b in zip(A.ravel(), B.ravel())}) == n * n


# ------------------------------------------------------------------ space-filling metrics

def _discrepancy(U, method="CD"):
    """L2-type discrepancy of points in [0, 1]^d.

    CD/WD/MD return the squared discrepancy and L2-star its square root -- the same
    conventions as ``scipy.stats.qmc.discrepancy``.
    """
    U = _as_2d(U)
    if np.any(U < 0) or np.any(U > 1):
        raise ValueError("discrepancy needs points in the unit cube")
    n, d = U.shape
    a = np.abs(U - 0.5)
    D = np.abs(U[:, None, :] - U[None, :, :])
    if method == "CD":
        t1 = np.prod(1 + 0.5 * a - 0.5 * a ** 2, axis=1).sum()
        t2 = np.prod(1 + 0.5 * a[:, None, :] + 0.5 * a[None, :, :] - 0.5 * D, axis=2).sum()
        return float((13 / 12) ** d - 2 / n * t1 + t2 / n ** 2)
    if method == "WD":
        return float(-(4 / 3) ** d + np.prod(1.5 - D * (1 - D), axis=2).sum() / n ** 2)
    if method == "MD":
        t1 = np.prod(5 / 3 - 0.25 * a - 0.25 * a ** 2, axis=1).sum()
        t2 = np.prod(15 / 8 - 0.25 * a[:, None, :] - 0.25 * a[None, :, :]
                     - 0.75 * D + 0.5 * D ** 2, axis=2).sum()
        return float((19 / 12) ** d - 2 / n * t1 + t2 / n ** 2)
    if method == "L2-star":
        mx = np.maximum(U[:, None, :], U[None, :, :])
        val = ((1 / 3) ** d - 2 ** (1 - d) / n * np.prod(1 - U ** 2, axis=1).sum()
               + np.prod(1 - mx, axis=2).sum() / n ** 2)
        return float(np.sqrt(max(val, 0.0)))
    raise ValueError("method must be 'CD', 'WD', 'MD' or 'L2-star'")


def _pairwise(X, Y=None):
    Y = X if Y is None else Y
    return np.sqrt(np.maximum(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1), 0.0))


def _min_distance(X):
    X = _as_2d(X)
    if len(X) < 2:
        return np.inf
    d = _pairwise(X)
    return float(d[np.triu_indices(len(X), 1)].min())


def _nearest_distance(ref, X):
    """Distance from every reference point to its nearest point of X."""
    out = np.empty(len(ref))
    for s in range(0, len(ref), 2048):
        out[s:s + 2048] = _pairwise(ref[s:s + 2048], X).min(axis=1)
    return out


def _sobol_reference(dim, n, random_state=0):
    from stochpylib.montecarlo import SobolSequence

    return SobolSequence(dim=dim, random_state=random_state).generate(int(n))


def _is_lhs(U):
    """Whether every column hits each of the n equal strata exactly once."""
    U = _as_2d(U)
    n = len(U)
    strata = np.clip(np.floor(U * n).astype(int), 0, n - 1)
    return all(sorted(strata[:, j]) == list(range(n)) for j in range(U.shape[1]))


# ---------------------------------------------------------------- correlation / ranks

def _ranks(x):
    """Average ranks (1-based) with ties sharing their mean rank."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _pearson(x, y):
    x = np.asarray(x, dtype=float) - np.mean(x)
    y = np.asarray(y, dtype=float) - np.mean(y)
    den = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / den) if den > 0 else 0.0


def _spearman(x, y):
    return _pearson(_ranks(x), _ranks(y))


def _max_abs_correlation(X):
    X = _as_2d(X)
    if X.shape[1] < 2:
        return 0.0
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C)
    return float(np.max(np.abs(C[np.triu_indices(X.shape[1], 1)])))
