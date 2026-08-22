"""Quasi-random / low-discrepancy sequences — the foundation everything else here samples from.

All sequence classes share the same shape::

    seq = SobolSequence(dim=5, random_state=None)
    pts = seq.generate(n)   # -> ndarray of shape (n, dim), values in [0, 1)

Implemented natively (no scipy dependency at runtime):

- :class:`HaltonSequence` — radical inverse in per-dimension prime bases.
- :class:`FaureSequence` — base ``b >= dim``; dimension ``j`` applies the ``j``-th power of
  the Pascal matrix (mod ``b``) to the base-``b`` digits — the distinct powers are what make
  the coordinates mutually low-discrepancy.
- :class:`SobolSequence` — base-2 digital net from *primitive* polynomials over GF(2).
- :class:`NiederreiterSequence` — base-2 net from *irreducible* polynomials.
- :class:`DigitalNetBase2` — general configurable GF(2) digital-net engine.
- :class:`LowDiscrepancy` — factory/facade picking a sequence by name.

Direction-number note: :class:`SobolSequence` embeds the standard Joe-Kuo lineage
direction-number table (64 dimensions, 30 columns) — identical to the data used by
``scipy.stats.qmc.Sobol``. Consequences, verified in tests:

- Every ``generate_block(m)`` (the first ``2**m`` points including the origin) is an
  *exactly* balanced ``(t,m,s)``-net block: perfect half/quarter/eighth stratification.
- As a set, ``generate_block(m)`` equals scipy's block bitwise; enumeration order differs
  (natural order here, Gray-code order in scipy).
- Streaming mode still skips the origin point, so a plain ``generate(2**m)`` window is
  misaligned by one position and shows +-1 imbalance — use ``generate_block`` when exact
  balance matters.

:class:`NiederreiterSequence` and custom-polynomial nets beyond dimension 64 keep the
GF(2)-recurrence machinery with simple canonical initial values (balance within +-1,
documented). Scrambling uses a seeded digital shift (base 2), which preserves net structure,
or an additive shift mod 1 (Halton/Faure).
"""

import numpy as np

from stochpylib.montecarlo import _direction_numbers as _dn

__all__ = [
    "HaltonSequence",
    "FaureSequence",
    "SobolSequence",
    "NiederreiterSequence",
    "DigitalNetBase2",
    "LowDiscrepancy",
    "radical_inverse",
    "first_primes",
]

_BITS = 53  # stay inside float64 exact-integer range


# ---------------------------------------------------------------------------
# helpers


def first_primes(count):
    """First ``count`` prime numbers."""
    primes = []
    candidate = 2
    while len(primes) < count:
        if all(candidate % p != 0 for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 1 if candidate == 2 else 2
    return primes


def radical_inverse(index, base):
    """van der Corput radical inverse of ``index`` in ``base``."""
    result = 0.0
    factor = 1.0 / base
    index = int(index)
    while index > 0:
        result += factor * (index % base)
        index //= base
        factor /= base
    return result


def _smallest_prime_geq(x):
    p = max(int(x), 2)
    while True:
        if all(p % d != 0 for d in range(2, int(p**0.5) + 1)):
            return p
        p += 1


def _to_digits_lsb(index, base):
    """Digits of ``index`` in ``base``, least-significant first."""
    digits = []
    while index > 0:
        digits.append(index % base)
        index //= base
    return digits


# GF(2) polynomial machinery ------------------------------------------------


def _is_irreducible_gf2(degree, inner_bits):
    """Is ``x^degree + inner_bits-poly + 1`` irreducible? (trial division)."""
    poly = (1 << degree) | inner_bits | 1

    def divides(q):  # does q divide poly?
        rem = poly
        while rem.bit_length() >= q.bit_length():
            rem ^= q << (rem.bit_length() - q.bit_length())
        return rem == 0

    if degree == 1:
        return True
    for d in range(1, degree // 2 + 1):
        for inner in range(1 << d):
            if divides((1 << d) | (inner << 1) | 1):
                return False
    return True


def _first_irreducible_polys(count):
    """First ``count`` irreducible polys over GF(2), by (degree, inner_bits)."""
    out = []
    degree = 1
    while len(out) < count:
        found = None
        for inner in range(1 << max(degree - 1, 0)):
            bits = 0 if degree == 1 else (inner << 1)
            if _is_irreducible_gf2(degree, bits):
                found = bits
                break
        if found is not None:
            out.append((degree, found))
        degree += 1
    return out


def _primitive_poly_per_degree(max_degree):
    """One *primitive* polynomial per degree 1..max_degree: list of (degree, inner_bits)."""
    out = []

    def multiplicative_order(poly, degree):
        # order of x modulo poly; == 2^degree - 1 iff primitive
        def mul(a, b):
            res = 0
            while b:
                if b & 1:
                    res ^= a
                b >>= 1
                a <<= 1
                if (a >> degree) & 1:
                    a ^= poly
            return res

        target = (1 << degree) - 1
        power = 2  # element "x"
        order = 1
        while power != 1 and order < target:
            power = mul(power, 2)
            order += 1
        return order == target

    degree = 1
    while degree <= max_degree:
        for inner in range(1 << max(degree - 1, 0)):
            bits = 0 if degree == 1 else (inner << 1)
            if multiplicative_order((1 << degree) | bits | 1, degree):
                out.append((degree, bits))
                break
        degree += 1
    return out


# ---------------------------------------------------------------------------
# sequence classes


class _LowDiscrepancyBase:
    """Shared plumbing: dim validation, generate() contract."""

    def __init__(self, dim=1):
        dim = int(dim)
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim

    def _raw(self, n):
        raise NotImplementedError

    def generate(self, n):
        """Return the next ``n`` points as an ``(n, dim)`` float array in [0, 1)."""
        n = int(n)
        if n < 0:
            raise ValueError("n must be >= 0")
        pts = np.asarray(self._raw(n), dtype=float).reshape(n, self.dim)
        return np.clip(pts, 0.0, 1.0 - 1e-15)


class HaltonSequence(_LowDiscrepancyBase):
    """Halton sequence: radical inverse in the first ``dim`` prime bases."""

    def __init__(self, dim=1, random_state=None):
        super().__init__(dim)
        self.bases = first_primes(dim)
        # additive shift mod 1 (seeded -> reproducible); unshifted when no seed given
        self._shifts = (
            np.random.default_rng(random_state).uniform(size=dim) if random_state is not None else None
        )

    def _raw(self, n):
        cols = [[radical_inverse(i, b) for i in range(1, n + 1)] for b in self.bases]
        pts = np.array(cols, dtype=float).T
        if self._shifts is not None:
            pts = (pts + self._shifts) % 1.0
        return pts


class FaureSequence(_LowDiscrepancyBase):
    """Faure sequence: base ``b`` = smallest prime >= dim; coordinate ``j`` applies
    ``Pascal^j mod b`` to the base-b digits so all pairs of coordinates are balanced."""

    def __init__(self, dim=1, random_state=None):
        super().__init__(dim)
        self.base = _smallest_prime_geq(dim)
        self._shifts = (
            np.random.default_rng(random_state).uniform(size=dim) if random_state is not None else None
        )
        b = self.base
        size = max(dim, 16)
        pascal = np.zeros((size, size), dtype=np.int64)
        from math import comb

        for i in range(size):
            for j in range(i, size):
                pascal[i, j] = comb(j, i) % b
        # per-coordinate transform matrices: identity, P, P^2, ... (mod b)
        self._mats = [np.eye(size, dtype=np.int64)]
        for _ in range(1, dim):
            nxt = (self._mats[-1] @ pascal) % b
            self._mats.append(nxt)

    def _point(self, index, coord):
        b = self.base
        digits = _to_digits_lsb(index, b)
        mat = self._mats[coord]
        m = len(digits)
        vec = np.array(digits, dtype=np.int64)
        transformed = (mat[:m, :m] @ vec) % b
        value = 0.0
        factor = 1.0 / b
        for dig in transformed:
            value += factor * float(dig)
            factor /= b
        return value

    def _raw(self, n):
        pts = np.empty((n, self.dim), dtype=float)
        for coord in range(self.dim):
            pts[:, coord] = [self._point(i, coord) for i in range(1, n + 1)]
        if self._shifts is not None:
            pts = (pts + self._shifts) % 1.0
        return pts


class DigitalNetBase2(_LowDiscrepancyBase):
    """General base-2 digital net.

    Two operating modes:

    - **Standard table** (default for :class:`SobolSequence` with ``dim <= 64``): uses the
      embedded Joe-Kuo lineage direction numbers (:mod:`stochpylib.montecarlo._direction_numbers`,
      30 columns). With these, every ``generate_block(m)`` is an *exactly* balanced
      ``(t,m,s)``-net block and the points match ``scipy.stats.qmc.Sobol`` as a set.
    - **GF(2) machinery**: generator polynomials + direction-number recurrence (53 columns);
      used by :class:`NiederreiterSequence`, custom ``polys``, or dims beyond the table
      (balance then documented within +-1).

    Points advance in natural order: ``x_i = XOR of V[:, b] over the set bits b of i``
    (i = 1, 2, 3, ...; the origin point i = 0 is skipped in streaming mode).
    """

    BITS = _BITS  # default; instances override per operating mode

    def __init__(self, dim=1, polys=None, random_state=None, init="canonical",
                 direction_matrix=None):
        super().__init__(dim)
        self.init = init
        if direction_matrix is not None:
            # precomputed standard table (exact net, scipy-compatible)
            self.BITS = int(direction_matrix.shape[1])
            self._V = np.asarray(direction_matrix, dtype=np.int64)
            self.polys = None
            self.uses_standard_table = True
        elif polys is None and dim <= _dn.MAX_TABLE_DIM:
            self.BITS = _dn.TABLE_WIDTH
            self._V = np.array(_dn.DIRECTION_NUMBERS[:dim], dtype=np.int64)
            self.polys = None
            self.uses_standard_table = True
        else:
            if polys is None:
                polys = _primitive_poly_per_degree(dim)
            if len(polys) < dim:
                raise ValueError(f"need {dim} generator polynomials, got {len(polys)}")
            self.polys = [(int(d), int(b)) for d, b in polys[:dim]]
            self.BITS = 53
            self._V = self._build_direction_numbers()
            self.uses_standard_table = False
        rng = np.random.default_rng(random_state) if random_state is not None else None
        self._shift_mask = (
            rng.integers(0, 1 << self.BITS, size=self.dim, dtype=np.int64)
            if rng is not None
            else None
        )
        self.reset()

    def _build_direction_numbers(self):
        V = np.zeros((self.dim, self.BITS), dtype=np.int64)
        for row, (degree, inner_bits) in enumerate(self.polys):
            # Dimension 1 is exactly van der Corput: m_j = 1 for EVERY level, no
            # recurrence (the degree-1 polynomial x+1 would otherwise double even
            # integers through the generic rule and corrupt the net structure).
            if row == 0:
                V[row, :] = np.left_shift(1, self.BITS - 1 - np.arange(self.BITS))
                continue
            # initial direction numbers m_{i,j}, j = 1..degree (odd, < 2^j required)
            for j in range(1, degree + 1):
                if self.init == "ones":
                    m = 1
                elif self.init == "canonical":
                    m = 2 * j - 1
                else:
                    raise ValueError(f"unknown init {self.init!r}")
                V[row, j - 1] = m << (self.BITS - j)
            # recurrence for j > degree (Bratley-Fox / Antonov-Saleev form):
            #   v_j = v_{j-degree} ^ (v_{j-degree} >> degree) ^ XOR_{a_k=1} v_{j-k}
            for j in range(degree + 1, self.BITS + 1):
                newv = V[row, j - degree - 1] ^ (V[row, j - degree - 1] >> degree)
                for k in range(1, degree):
                    if (inner_bits >> (degree - 1 - k)) & 1:
                        newv ^= V[row, j - k - 1]
                V[row, j - 1] = newv
        return V

    def reset(self):
        """Restart the stream at the beginning."""
        self._index = 0

    def _points_for_indices(self, idx):
        acc = np.zeros((len(idx), self.dim), dtype=np.int64)
        for b in range(self.BITS):
            sel = ((idx >> b) & 1).astype(bool)
            if sel.any():
                acc[sel] ^= self._V[:, b]
        out = acc.astype(float) / float(1 << self.BITS)
        if self._shift_mask is not None:  # seeded digital shift preserves net structure
            ints = (out * (1 << self.BITS)).astype(np.int64)
            out = (ints ^ self._shift_mask) / float(1 << self.BITS)
        return out

    def generate_block(self, m):
        """Return exactly the first ``2**m`` points INCLUDING the origin (stream indices
        ``0 .. 2**m - 1``) — the aligned block for which the net's balance is certified
        exact when the standard table is used. Does not disturb the stream position.
        """
        m = int(m)
        if not (0 < m <= self.BITS):
            raise ValueError(f"m must be in 1..{self.BITS}")
        return self._points_for_indices(np.arange(0, 1 << m, dtype=np.int64))

    def _raw(self, n):
        idx = np.arange(self._index + 1, self._index + n + 1, dtype=np.int64)
        out = self._points_for_indices(idx)
        self._index += n
        return out


def _sobol_polys(dim):
    return _primitive_poly_per_degree(dim)


def _niederreiter_polys(dim):
    return _first_irreducible_polys(dim)


class SobolSequence(DigitalNetBase2):
    """Sobol' (t,s)-sequence in base 2.

    Uses the embedded standard (Joe-Kuo lineage) direction-number table for
    ``dim <= 64`` — exact ``(t,m,s)``-net balance per ``generate_block`` and set-identical
    to ``scipy.stats.qmc.Sobol``. Beyond 64 dimensions it falls back to the GF(2)
    machinery with canonical initial values.
    """

    def __init__(self, dim=1, random_state=None, init="canonical", standard_table=True,
                 polys=None):
        dm = None
        if standard_table and polys is None and dim <= _dn.MAX_TABLE_DIM:
            dm = np.array(_dn.DIRECTION_NUMBERS[:dim], dtype=np.int64)
        super().__init__(dim, polys=polys, random_state=random_state, init=init,
                         direction_matrix=dm)


class NiederreiterSequence(DigitalNetBase2):
    """Niederreiter-style base-2 net from irreducible generator polynomials.

    Documented simplification: shares the Sobol engine's direction-number recursion,
    keyed on irreducible rather than primitive polynomials.
    """

    def __init__(self, dim=1, random_state=None, init="canonical"):
        super().__init__(
            dim, polys=_niederreiter_polys(dim), random_state=random_state, init=init
        )


_SEQUENCE_REGISTRY = {
    "sobol": SobolSequence,
    "halton": HaltonSequence,
    "faure": FaureSequence,
    "niederreiter": NiederreiterSequence,
}


def LowDiscrepancy(sequence="sobol", dim=1, random_state=None, **kwargs):
    """Factory/facade: ``LowDiscrepancy("sobol", dim=4).generate(1024)``."""
    try:
        impl = _SEQUENCE_REGISTRY[str(sequence).lower()]
    except KeyError:
        raise ValueError(
            f"unknown sequence {sequence!r}; choose from {sorted(_SEQUENCE_REGISTRY)}"
        ) from None
    return impl(dim=dim, random_state=random_state, **kwargs)
