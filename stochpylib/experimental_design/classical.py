"""Classical designs: full and fractional factorials, Plackett-Burman screening designs,
central composite and Box-Behnken response-surface designs, and Latin / Graeco-Latin
squares for blocking in two (three) directions.

Two-level designs are in coded units (``-1``/``+1``) and in **Yates standard order** (the
first factor changes fastest); call :meth:`Design.randomized` for a randomized run order.
"""

import itertools
import math

import numpy as np

from stochpylib.experimental_design._base import DesignGenerator
from stochpylib.experimental_design._common import (
    _LETTERS,
    _factor_names,
    _gf_neg,
    _gf_squares,
    _gf_tables,
    _is_prime,
    _is_prime_power,
    _mols_pair,
)
from stochpylib.experimental_design._design import Design

__all__ = ["FullFactorial", "FractionalFactorial", "Plackett_Burman", "CCD", "BoxBehnken",
           "LatinSquare", "GraecoLatin"]


def _yates(axes):
    """Cartesian product with the first factor changing fastest."""
    rows = list(itertools.product(*axes[::-1]))
    return np.array(rows, dtype=float)[:, ::-1] if rows else np.zeros((0, len(axes)))


def _effect_label(indices, names):
    parts = [names[i] for i in sorted(indices)]
    sep = "" if all(len(p) == 1 for p in parts) else "*"
    return sep.join(parts)


# ---------------------------------------------------------------------- full factorial

class FullFactorial(DesignGenerator):
    """Full factorial design.

    ``levels`` is an int (with ``n_factors``: every factor gets that many levels), a list
    of per-factor level counts, or a list of explicit level-value sequences. Level counts
    are coded ``linspace(-1, 1, m)`` (two levels -> ``-1, +1``; three -> ``-1, 0, 1``);
    explicit values are used as given and the design is then in natural units.
    """

    kind = "full factorial"

    def __init__(self, levels=2, n_factors=None, factor_names=None, bounds=None):
        self.levels = levels
        self.n_factors = n_factors
        self.factor_names = factor_names
        self.bounds = bounds

    def _axes(self):
        lv = self.levels
        # np.ndim would choke on a ragged list of explicit level values
        if isinstance(lv, (int, np.integer)):
            if self.n_factors is None:
                raise ValueError("an integer `levels` needs n_factors")
            lv = [int(lv)] * int(self.n_factors)
        axes, explicit = [], False
        for item in lv:
            if np.ndim(item) == 0:
                m = int(item)
                if m < 2:
                    raise ValueError("every factor needs at least 2 levels")
                axes.append(np.linspace(-1.0, 1.0, m))
            else:
                vals = np.asarray(item, dtype=float)
                if vals.size < 2:
                    raise ValueError("every factor needs at least 2 levels")
                axes.append(vals)
                explicit = True
        return axes, explicit

    def _generate(self, rng):
        axes, explicit = self._axes()
        pts = _yates(axes)
        props = {"n_runs": len(pts), "levels": [len(a) for a in axes]}
        if explicit:
            props["level_values"] = [a.tolist() for a in axes]
        return Design(pts, factor_names=self.factor_names,
                      space="natural" if explicit else "coded", kind=self.kind,
                      bounds=None if explicit else self.bounds, properties=props)


# ---------------------------------------------------------------- fractional factorial

def _popcount(v):
    return bin(v).count("1")


def _subgroup(words):
    """All non-identity products (XOR of bitmasks) of the generator words."""
    out = []
    for r in range(1, len(words) + 1):
        for combo in itertools.combinations(words, r):
            v = 0
            for w in combo:
                v ^= w
            out.append(v)
    return out


def _wlp(dcs, k):
    lengths = [_popcount(w) for w in dcs]
    return tuple(lengths.count(L) for L in range(1, k + 1))


class FractionalFactorial(DesignGenerator):
    """Two-level regular fractional factorial ``2^(k-p)``.

    Give either explicit ``generators`` -- ``"E=ABC F=BCD"`` (the base factors are the
    letters before the first added one), ``["ABC", "BCD"]`` (one word per added factor over
    the base letters) or pyDOE-style ``"a b c abc"`` -- or
    ``(n_factors, p)``, in which case the generators are **searched**: maximum resolution
    first, then minimum aberration (smallest word-length pattern). With ``p`` omitted and
    ``resolution`` given, the smallest fraction reaching that resolution is chosen.
    ``design_.properties`` carries the ``defining_relation``, ``resolution``,
    ``word_length_pattern`` and ``aliases`` (every main effect and two-factor interaction
    mapped to its aliases of order <= ``max_order``).
    """

    kind = "fractional factorial"
    _EXHAUSTIVE_LIMIT = 200_000

    def __init__(self, n_factors=None, p=None, generators=None, resolution=None, max_order=3,
                 random_state=None):
        self.n_factors = n_factors
        self.p = p
        self.generators = generators
        self.resolution = resolution
        self.max_order = int(max_order)
        self.random_state = random_state

    # --------------------------------------------------------------- generator parsing
    def _parse(self):
        """``(names, columns)``: columns[j] is a bitmask over the *base* factors."""
        gens = self.generators
        tokens = gens.split() if isinstance(gens, str) else list(gens)
        tokens = [str(t).upper() for t in tokens if str(t).strip()]
        if any("=" in t for t in tokens):
            # "D=ABC E=BCD": the base factors are every letter before the first added one
            lhs = [t.split("=")[0] for t in tokens]
            added = [t.split("=")[1] for t in tokens]
            base = list(_LETTERS[:min(_LETTERS.index(c) for c in lhs)])
            return base + lhs, ([1 << i for i in range(len(base))]
                                + [self._word_mask(w, base) for w in added])
        if any(len(t) == 1 for t in tokens):
            # pyDOE style: single letters are base factors, words are generated columns
            base = [t for t in tokens if len(t) == 1]
            names, cols, used = [], [], set(base)
            for t in tokens:
                if len(t) == 1:
                    names.append(t)
                    cols.append(1 << base.index(t))
                else:
                    nxt = next(c for c in _LETTERS if c not in used)
                    used.add(nxt)
                    names.append(nxt)
                    cols.append(self._word_mask(t, base))
            return names, cols
        # list of words for the added factors, over the leading base letters
        letters = sorted({c for w in tokens for c in w}, key=_LETTERS.index)
        b = _LETTERS.index(letters[-1]) + 1
        base = list(_LETTERS[:b])
        added = [_LETTERS[b + j] for j in range(len(tokens))]
        return base + added, [1 << i for i in range(b)] + [self._word_mask(w, base)
                                                           for w in tokens]

    @staticmethod
    def _word_mask(word, base):
        mask = 0
        for c in word:
            if c not in base:
                raise ValueError(f"generator {word!r} uses {c!r}, not a base factor {base}")
            mask ^= 1 << base.index(c)
        if _popcount(mask) < 2:
            raise ValueError(f"generator {word!r} must involve at least two base factors")
        return mask

    # --------------------------------------------------------------- generator search
    @classmethod
    def _search(cls, k, p, rng):
        b = k - p
        if b < 2 or p < 1:
            raise ValueError(f"2^({k}-{p}) is not a valid fraction")
        words = [w for w in range(1, 1 << b) if _popcount(w) >= 2]
        if len(words) < p:
            raise ValueError(f"2^({k}-{p}) needs more generators than {b} base factors allow")
        n_combos = math.comb(len(words), p)
        if n_combos <= cls._EXHAUSTIVE_LIMIT:
            combos = itertools.combinations(words, p)
        else:
            combos = (tuple(sorted(rng.choice(words, p, replace=False).tolist()))
                      for _ in range(20_000))
        best, best_key = None, None
        for combo in combos:
            full = [w | (1 << (b + j)) for j, w in enumerate(combo)]
            dcs = _subgroup(full)
            wlp = _wlp(dcs, k)
            res = min(_popcount(w) for w in dcs)
            key = (-res,) + wlp
            if best_key is None or key < best_key:
                best_key, best = key, combo
        return best

    # ---------------------------------------------------------------------- generate
    def _columns(self, rng):
        if self.generators is not None:
            return self._parse()
        k = self.n_factors
        if k is None:
            raise ValueError("give generators, or n_factors with p (or resolution)")
        k = int(k)
        if self.p is not None:
            p = int(self.p)
        elif self.resolution is not None:
            p = None
            for cand in range(k - 2, 0, -1):
                b = k - cand
                if (1 << b) - b - 1 < cand:
                    continue  # too few base factors to generate that many columns
                gens = self._search(k, cand, rng)
                full = [w | (1 << (k - cand + j)) for j, w in enumerate(gens)]
                if min(_popcount(w) for w in _subgroup(full)) >= int(self.resolution):
                    p = cand
                    break
            if p is None:
                raise ValueError(f"no 2^({k}-p) fraction reaches resolution {self.resolution}")
        else:
            raise ValueError("give p or resolution alongside n_factors")
        b = k - p
        gens = self._search(k, p, rng)
        names = _factor_names(k)
        return names, [1 << i for i in range(b)] + list(gens)

    def _generate(self, rng):
        names, cols = self._columns(rng)
        k = len(cols)
        base_idx = [j for j, c in enumerate(cols) if _popcount(c) == 1]
        b = len(base_idx)
        base = _yates([np.array([-1.0, 1.0])] * b)
        pts = np.empty((len(base), k))
        for j, c in enumerate(cols):
            members = [i for i in range(b) if c >> i & 1]
            pts[:, j] = np.prod(base[:, members], axis=1)
        # defining words over the full factor set: generated factor j times its base word
        base_pos = {i: base_idx[i] for i in range(b)}
        words = []
        for j, c in enumerate(cols):
            if _popcount(c) > 1:
                w = 1 << j
                for i in range(b):
                    if c >> i & 1:
                        w ^= 1 << base_pos[i]
                words.append(w)
        dcs = sorted(set(_subgroup(words)), key=lambda w: (_popcount(w), w)) if words else []
        props = self._structure(dcs, names, k)
        props["generators"] = {names[j]: _effect_label(
            [base_pos[i] for i in range(b) if cols[j] >> i & 1], names)
            for j in range(k) if _popcount(cols[j]) > 1}
        props["n_runs"] = len(pts)
        props["fraction"] = f"2^({k}-{k - b})"
        self._dcs, self._names = dcs, names
        return Design(pts, factor_names=names, space="coded", kind=self.kind, properties=props)

    def _structure(self, dcs, names, k):
        def label(w):
            return _effect_label([i for i in range(k) if w >> i & 1], names)

        props = {"defining_relation": ["I"] + [label(w) for w in dcs]}
        props["resolution"] = min((_popcount(w) for w in dcs), default=None)
        props["word_length_pattern"] = _wlp(dcs, k) if dcs else tuple([0] * k)
        aliases = {}
        effects = [1 << i for i in range(k)] + [(1 << i) | (1 << j)
                                                  for i, j in itertools.combinations(range(k), 2)]
        for e in effects:
            al = sorted({e ^ w for w in dcs if _popcount(e ^ w) <= self.max_order},
                        key=lambda v: (_popcount(v), v))
            aliases[label(e)] = [label(v) for v in al]
        props["aliases"] = aliases
        return props

    def fold_over(self):
        """The full fold-over: the design plus all runs with every sign reversed.

        It cancels every odd-length word of the defining relation, so a resolution-III
        fraction becomes resolution IV.
        """
        if not hasattr(self, "design_"):
            self.generate()
        d = self.design_
        k = d.n_factors
        dcs = [w for w in self._dcs if _popcount(w) % 2 == 0]
        props = self._structure(dcs, self._names, k)
        props["n_runs"] = 2 * d.n_runs
        props["fold_over"] = True
        return Design(np.vstack([d.points, -d.points]), factor_names=list(d.factor_names),
                      space="coded", kind="fold-over fractional factorial", properties=props)


# ---------------------------------------------------------------------- Plackett-Burman

def _paley_prime(q):
    """Cyclic Paley design of order q+1 for prime q = 3 mod 4 (the published PB rows)."""
    squares = {(i * i) % q for i in range(1, q)}
    row = np.array([1.0] + [1.0 if j in squares else -1.0 for j in range(1, q)])
    D = np.array([np.roll(row, s) for s in range(q)])
    return np.column_stack([np.ones(q + 1), np.vstack([D, -np.ones(q)])])


def _paley_prime_power(q):
    """Paley construction I over GF(q): H = I + [[0, 1'], [-1, Q]], Q the Jacobsthal matrix."""
    add, _ = _gf_tables(q)
    neg = _gf_neg(q)
    sq = _gf_squares(q)
    Q = np.empty((q, q))
    for a in range(q):
        for b in range(q):
            d = int(add[a, neg[b]])
            Q[a, b] = 0.0 if d == 0 else (1.0 if d in sq else -1.0)
    S = np.zeros((q + 1, q + 1))
    S[0, 1:] = 1.0
    S[1:, 0] = -1.0
    S[1:, 1:] = Q
    return np.eye(q + 1) + S


def _hadamard(N):
    """A normalized Hadamard matrix of order N (first column all +1), or None."""
    if N == 1:
        return np.ones((1, 1))
    if N == 2:
        return np.array([[1.0, 1.0], [1.0, -1.0]])
    if N % 4:
        return None
    H = None
    if N & (N - 1) == 0:
        H = np.ones((1, 1))
        while len(H) < N:
            H = np.block([[H, H], [H, -H]])
    else:
        q = N - 1
        pm = _is_prime_power(q)
        if pm is not None and q % 4 == 3:
            H = _paley_prime(q) if _is_prime(q) else _paley_prime_power(q)
        else:
            half = _hadamard(N // 2)
            if half is not None:
                H = np.block([[half, half], [half, -half]])
    if H is None:
        return None
    return H * np.sign(H[:, :1])


class Plackett_Burman(DesignGenerator):
    """Plackett-Burman two-level screening design for up to ``N - 1`` factors in ``N`` runs.

    ``N`` is the smallest constructible multiple of 4 above ``n_factors``: powers of two
    (Sylvester), ``q + 1`` for a prime power ``q = 3 (mod 4)`` (Paley; for prime ``q`` this
    is the cyclic construction and reproduces Plackett & Burman's published generator rows
    for N = 12, 20, 24), and doublings of either. Orders such as 36 and 52 have no
    construction here and are skipped to the next constructible N.
    """

    kind = "Plackett-Burman"

    def __init__(self, n_factors):
        self.n_factors = int(n_factors)
        if self.n_factors < 1:
            raise ValueError("n_factors must be >= 1")

    def _generate(self, rng):
        N = 4 * math.ceil((self.n_factors + 1) / 4)
        while True:
            H = _hadamard(N)
            if H is not None:
                break
            N += 4
        pts = H[:, 1:1 + self.n_factors]
        return Design(pts, space="coded", kind=self.kind,
                      properties={"n_runs": N, "hadamard_order": N})


# ------------------------------------------------------------------ central composite

class CCD(DesignGenerator):
    """Central composite design: a two-level cube, ``2k`` axial runs and centre runs.

    ``alpha``: ``"rotatable"`` (``F^(1/4)``, prediction variance constant on spheres),
    ``"orthogonal"`` (the quadratic effects' columns become orthogonal), ``"face"`` (alpha
    = 1, three levels only), ``"inscribed"`` (axial runs at +/-1, cube shrunk by the
    rotatable alpha) or an explicit float. ``fraction=p`` puts a ``2^(k-p)`` fraction in
    the cube instead of the full factorial.
    """

    kind = "central composite"

    def __init__(self, n_factors, alpha="rotatable", center=4, fraction=None):
        self.n_factors = int(n_factors)
        if self.n_factors < 2:
            raise ValueError("a central composite design needs at least 2 factors")
        self.alpha = alpha
        self.center = int(center)
        self.fraction = fraction

    def _generate(self, rng):
        k = self.n_factors
        if self.fraction:
            cube = FractionalFactorial(k, p=int(self.fraction)).generate().points
        else:
            cube = _yates([np.array([-1.0, 1.0])] * k)
        F, n0 = len(cube), self.center
        a_rot = F ** 0.25
        if isinstance(self.alpha, str):
            if self.alpha == "rotatable":
                alpha = a_rot
            elif self.alpha == "orthogonal":
                T = 2 * k + n0
                alpha = ((np.sqrt(F + T) - np.sqrt(F)) ** 2 * F / 4.0) ** 0.25
            elif self.alpha in ("face", "inscribed"):
                alpha = 1.0
            else:
                raise ValueError("alpha must be 'rotatable', 'orthogonal', 'face', "
                                 "'inscribed' or a number")
        else:
            alpha = float(self.alpha)
        if self.alpha == "inscribed":
            cube = cube / a_rot
        axial = np.vstack([s * alpha * np.eye(k)[i] for i in range(k) for s in (-1.0, 1.0)])
        pts = np.vstack([cube, axial, np.zeros((n0, k))])
        return Design(pts, space="coded", kind=self.kind,
                      properties={"alpha": float(alpha), "alpha_kind": str(self.alpha),
                                  "n_factorial": F, "n_axial": 2 * k, "n_center": n0,
                                  "n_runs": len(pts)})


# ------------------------------------------------------------------------ Box-Behnken

# Box & Behnken (1960) block structures for k = 6, 7 (0-based factor indices); k = 3..5
# use all factor pairs.
_BB_BLOCKS = {
    6: [(0, 1, 3), (1, 2, 4), (2, 3, 5), (0, 3, 4), (1, 4, 5), (0, 2, 5)],
    7: [(3, 4, 5), (0, 5, 6), (1, 4, 6), (0, 1, 3), (2, 3, 6), (0, 2, 4), (1, 2, 5)],
}


class BoxBehnken(DesignGenerator):
    """Box-Behnken three-level response-surface design for 3 to 7 factors.

    Each block runs a two-level factorial on its factors with every other factor at its
    centre; no run sits at a corner of the cube. Run counts before centre points are 12,
    24, 40, 48 and 56 for k = 3..7.
    """

    kind = "Box-Behnken"

    def __init__(self, n_factors, center=3):
        self.n_factors = int(n_factors)
        if not 3 <= self.n_factors <= 7:
            raise ValueError("BoxBehnken is tabulated for 3 to 7 factors")
        self.center = int(center)

    def _generate(self, rng):
        k = self.n_factors
        blocks = _BB_BLOCKS.get(k) or list(itertools.combinations(range(k), 2))
        runs = []
        for block in blocks:
            for signs in itertools.product((-1.0, 1.0), repeat=len(block)):
                r = np.zeros(k)
                r[list(block)] = signs
                runs.append(r)
        pts = np.vstack([np.array(runs), np.zeros((self.center, k))])
        return Design(pts, space="coded", kind=self.kind,
                      properties={"n_edge": len(runs), "n_center": self.center,
                                  "n_blocks": len(blocks), "n_runs": len(pts)})


# ------------------------------------------------------------- Latin / Graeco-Latin

def _square_anova(y, factors, names, n):
    """ANOVA for an n x n square with the given (n*n,) factor columns of integer levels."""
    from stochpylib.statistics._common import _f_sf
    from stochpylib.statistics._result import TestResult

    y = np.asarray(y, dtype=float).ravel()
    if y.size != n * n:
        raise ValueError(f"y must have {n * n} observations, one per cell")
    grand = y.mean()
    rows, ss_used = [], 0.0
    for col, name in zip(factors, names):
        means = np.array([y[col == lv].mean() for lv in range(n)])
        ss = float(n * np.sum((means - grand) ** 2))
        ss_used += ss
        rows.append({"source": name, "ss": ss, "df": n - 1})
    ss_total = float(np.sum((y - grand) ** 2))
    df_e = n * n - 1 - len(names) * (n - 1)
    ss_e = ss_total - ss_used
    ms_e = ss_e / df_e if df_e > 0 else None
    for r in rows:
        r["ms"] = r["ss"] / r["df"]
        if ms_e:
            r["F"] = r["ms"] / ms_e
            r["p"] = float(_f_sf(r["F"], r["df"], df_e))
        else:
            r["F"] = r["p"] = None
    rows.append({"source": "Residual", "ss": ss_e, "df": df_e, "ms": ms_e, "F": None,
                 "p": None})
    rows.append({"source": "Total", "ss": ss_total, "df": n * n - 1, "ms": None, "F": None,
                 "p": None})
    trt = rows[2]
    return TestResult(trt["F"], trt["p"], (n - 1, df_e), "no treatment effect",
                      f"{'Graeco-' if len(names) == 4 else ''}Latin-square ANOVA", table=rows)


class _SquareDesign(DesignGenerator):
    def __init__(self, n, randomize=True, random_state=None):
        self.n = int(n)
        self.randomize = bool(randomize)
        self.random_state = random_state

    def _perms(self, rng):
        n = self.n
        if not self.randomize:
            return np.arange(n), np.arange(n)
        return rng.permutation(n), rng.permutation(n)

    def _cells(self, squares, names):
        n = self.n
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        cols = [i.ravel(), j.ravel()] + [s.ravel() for s in squares]
        return Design(np.column_stack(cols).astype(float), factor_names=names,
                      space="natural", kind=self.kind,
                      bounds=[(0.0, n - 1.0)] * len(names), properties={"n": n})

    def analyze(self, y, design=None):
        """ANOVA of the responses ``y`` (one per cell, in the design's run order)."""
        d = design if design is not None else getattr(self, "design_", None)
        if d is None:
            d = self.generate()
        P = np.asarray(d, dtype=float).astype(int)
        return _square_anova(y, [P[:, c] for c in range(P.shape[1])],
                             [n.capitalize() for n in d.factor_names], self.n)


class LatinSquare(_SquareDesign):
    """``n x n`` Latin square: every treatment once per row and once per column.

    Built cyclically (``(i + j) mod n``) and, with ``randomize=True``, randomized by
    independent row, column and symbol permutations. The design's columns are ``row``,
    ``column`` and ``treatment`` (integer levels); :meth:`analyze` runs the three-way
    ANOVA. ``square_`` is the ``(n, n)`` treatment layout.
    """

    kind = "Latin square"

    def _generate(self, rng):
        n = self.n
        if n < 2:
            raise ValueError("a Latin square needs n >= 2")
        L = (np.arange(n)[:, None] + np.arange(n)[None, :]) % n
        r, c = self._perms(rng)
        sym = rng.permutation(n) if self.randomize else np.arange(n)
        L = sym[L[r][:, c]]
        self.square_ = L
        return self._cells([L], ["row", "column", "treatment"])


class GraecoLatin(_SquareDesign):
    """``n x n`` Graeco-Latin square: two superimposed, mutually orthogonal Latin squares.

    Every ordered pair of (Latin, Greek) symbols occurs exactly once. Built over GF(q) for
    prime powers and by Kronecker products of those for composite orders, so every n not
    congruent to 2 mod 4 is covered; n = 2 and 6 are impossible and n = 10, 14, ... are not
    constructed (a ValueError says so). ``latin_``/``greek_`` are the two layouts.
    """

    kind = "Graeco-Latin square"

    def _generate(self, rng):
        A, B = _mols_pair(self.n)
        r, c = self._perms(rng)
        A, B = A[r][:, c], B[r][:, c]
        if self.randomize:
            A = rng.permutation(self.n)[A]
            B = rng.permutation(self.n)[B]
        self.latin_, self.greek_ = A, B
        return self._cells([A, B], ["row", "column", "latin", "greek"])
