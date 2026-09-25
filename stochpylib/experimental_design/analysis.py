"""Analysis of designed experiments: factorial ANOVA and effect estimates, interaction and
(half-)normal-plot data with Lenth's method for unreplicated designs, and global
sensitivity analysis -- Morris screening, standardized/partial-rank regression
coefficients, and variance-based Sobol indices with bootstrap standard errors.

``InteractionPlot`` and ``NormalPlot`` compute the numbers a plot shows (cell means,
effect quantiles, significance thresholds) and a plain-text table; rendering them is the
job of the planned ``viz`` module, so nothing here depends on a plotting library.
"""

import itertools

import numpy as np

from stochpylib.experimental_design._common import (
    _as_2d,
    _factor_names,
    _normalize_bounds,
    _pearson,
    _ranks,
    _rng,
    _spearman,
)

__all__ = ["ANOVA_DOE", "MainEffects", "InteractionPlot", "NormalPlot", "SensitivityIndex",
           "SobolIndex"]


def _design_inputs(design, y, factor_names=None):
    X = _as_2d(np.asarray(design, dtype=float), "design")
    y = np.asarray(y, dtype=float).ravel()
    if y.size != len(X):
        raise ValueError(f"y must have {len(X)} values, one per run")
    names = factor_names or getattr(design, "factor_names", None) or _factor_names(X.shape[1])
    return X, y, list(names)


def _label(members, names):
    parts = [names[i] for i in sorted(members)]
    sep = "" if all(len(p) == 1 for p in parts) else "*"
    return sep.join(parts)


def _parse_terms(model, k, names):
    """Model spec -> list of factor-index tuples (no intercept)."""
    if model == "main":
        return [(i,) for i in range(k)]
    if model == "interaction":
        return [(i,) for i in range(k)] + list(itertools.combinations(range(k), 2))
    if model == "full":
        return [c for r in range(1, k + 1) for c in itertools.combinations(range(k), r)]
    if isinstance(model, str):
        raise ValueError("model must be 'main', 'interaction', 'full' or a list of terms")
    terms = []
    for t in model:
        if isinstance(t, str):
            parts = t.split("*") if "*" in t else (list(t) if t not in names else [t])
            try:
                terms.append(tuple(sorted(names.index(p) for p in parts)))
            except ValueError:
                raise ValueError(f"unknown term {t!r} for factors {names}") from None
        else:
            terms.append(tuple(sorted(int(i) for i in t)))
    return terms


def _two_level_coded(X):
    """X recoded to -1/+1 when every column has exactly two levels, else None."""
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        lv = np.unique(X[:, j])
        if len(lv) != 2:
            return None
        out[:, j] = np.where(X[:, j] == lv[1], 1.0, -1.0)
    return out


def _contrast_effects(Xc, y, terms, names):
    N = len(y)
    effects, ss = {}, {}
    for t in terms:
        col = np.prod(Xc[:, list(t)], axis=1)
        c = float(col @ y)
        lab = _label(t, names)
        effects[lab] = 2.0 * c / N
        ss[lab] = c * c / N
    return effects, ss


def _orthogonal(Xc, terms):
    F = np.column_stack([np.ones(len(Xc))] + [np.prod(Xc[:, list(t)], axis=1) for t in terms])
    G = F.T @ F
    return np.allclose(G - np.diag(np.diag(G)), 0.0, atol=1e-9)


def _dummy_block(col):
    levels = np.unique(col)
    return np.column_stack([(col == lv).astype(float) for lv in levels[1:]])


def _term_block(X, t, cache):
    block = None
    for i in t:
        D = cache.setdefault(i, _dummy_block(X[:, i]))
        block = D if block is None else np.einsum("ni,nj->nij", block, D).reshape(len(X), -1)
    return block


def _rss(cols, y):
    F = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(F, y, rcond=None)
    r = y - F @ beta
    return float(r @ r), int(np.linalg.matrix_rank(F))


# ------------------------------------------------------------------------------ ANOVA

class ANOVA_DOE:
    """ANOVA for a designed experiment.

    ``model``: ``"full"`` (every interaction), ``"interaction"`` (main effects + two-factor
    interactions), ``"main"``, or a list of terms (``["A", "C", "AC"]``, ``"A*C"`` or
    factor-index tuples). Terms left out of the model are pooled into the residual; a
    saturated unreplicated model has no residual degrees of freedom and reports no F tests
    (use :class:`NormalPlot` there). Two-level orthogonal designs use contrast sums of
    squares, ``SS = (col . y)^2 / N``, and report ``effects_`` (``2 (col . y) / N``);
    anything else (multi-level or non-orthogonal) uses Type II sums of squares over
    dummy-coded factors. ``table_`` is a ``statistics.TestResult`` whose statistic is the
    overall model F test.
    """

    def __init__(self, model="full", alpha=0.05):
        self.model = model
        self.alpha = float(alpha)

    def fit(self, design, y, factor_names=None):
        from stochpylib.statistics._common import _f_sf
        from stochpylib.statistics._result import TestResult

        X, y, names = _design_inputs(design, y, factor_names)
        N, k = X.shape
        terms = _parse_terms(self.model, k, names)
        sst = float(np.sum((y - y.mean()) ** 2))
        Xc = _two_level_coded(X)
        rows, self.effects_ = [], {}
        if Xc is not None and _orthogonal(Xc, terms):
            self.effects_, ss = _contrast_effects(Xc, y, terms, names)
            for t in terms:
                lab = _label(t, names)
                rows.append({"source": lab, "ss": ss[lab], "df": 1})
            ss_model = sum(ss.values())
            df_model = len(terms)
            method = "DOE ANOVA (contrast SS)"
        else:
            cache = {}
            blocks = {t: _term_block(X, t, cache) for t in terms}
            ones = [np.ones(N)]
            full_cols = ones + [blocks[t] for t in terms]
            rss_full, rank_full = _rss(full_cols, y)
            for t in terms:
                others = [s for s in terms if s != t and not set(t) <= set(s)]
                base_cols = ones + [blocks[s] for s in others]
                rss_without, r0 = _rss(base_cols, y)
                rss_with, r1 = _rss(base_cols + [blocks[t]], y)
                rows.append({"source": _label(t, names), "ss": rss_without - rss_with,
                             "df": r1 - r0})
            ss_model = sst - rss_full
            df_model = rank_full - 1
            method = "DOE ANOVA (Type II SS)"
        df_res = N - 1 - df_model
        ss_res = sst - ss_model
        ms_res = ss_res / df_res if df_res > 0 else None
        for r in rows:
            r["ms"] = r["ss"] / r["df"] if r["df"] else None
            if ms_res and r["df"]:
                r["F"] = r["ms"] / ms_res
                r["p"] = float(_f_sf(r["F"], r["df"], df_res))
            else:
                r["F"] = r["p"] = None
        rows.append({"source": "Residual", "ss": ss_res, "df": df_res, "ms": ms_res,
                     "F": None, "p": None})
        rows.append({"source": "Total", "ss": sst, "df": N - 1, "ms": None, "F": None,
                     "p": None})
        f_model = (ss_model / df_model) / ms_res if ms_res and df_model else float("nan")
        p_model = float(_f_sf(f_model, df_model, df_res)) if ms_res and df_model else None
        self.table_ = TestResult(f_model, p_model, (df_model, df_res),
                                 "no main or interaction effects", method, table=rows)
        self.residual_df_ = df_res
        self.significant_ = [r["source"] for r in rows[:-2]
                             if r["p"] is not None and r["p"] < self.alpha]
        return self


# ---------------------------------------------------------------------- main effects

class MainEffects:
    """Main-effect estimates: per-level means of every factor and, for two-level factors,
    the effect ``mean(high) - mean(low)``. With replicated runs, ``std_errors_`` holds the
    pure-error standard error ``2 s / sqrt(N)`` of each two-level effect."""

    def fit(self, design, y, factor_names=None):
        X, y, names = _design_inputs(design, y, factor_names)
        N = len(y)
        self.level_means_, self.effects_ = {}, {}
        for j, name in enumerate(names):
            lv = np.unique(X[:, j])
            self.level_means_[name] = {float(v): float(y[X[:, j] == v].mean()) for v in lv}
            if len(lv) == 2:
                self.effects_[name] = float(y[X[:, j] == lv[1]].mean()
                                            - y[X[:, j] == lv[0]].mean())
        _, inv = np.unique(np.round(X, 12), axis=0, return_inverse=True)
        inv = inv.ravel()
        df_pe = N - len(np.unique(inv))
        self.std_errors_ = None
        if df_pe > 0:
            ss_pe = sum(float(np.sum((y[inv == g] - y[inv == g].mean()) ** 2))
                        for g in np.unique(inv))
            s = np.sqrt(ss_pe / df_pe)
            self.std_errors_ = {n: float(2 * s / np.sqrt(N)) for n in self.effects_}
        self.ranking_ = sorted(self.effects_, key=lambda n: -abs(self.effects_[n]))
        return self


# ------------------------------------------------------------------- interaction plot

class InteractionPlot:
    """Interaction-plot data for two factors: the table of cell means, one line per level
    of the second ("trace") factor, and the two-factor interaction effect.

    Parallel lines mean no interaction; :meth:`is_parallel` tests that directly on the
    cell means (every interaction contrast within ``tol``). ``interaction_effect_`` is the
    classical ``AB`` effect for a 2x2 layout (``None`` otherwise).
    """

    def __init__(self, factors=(0, 1)):
        if len(factors) != 2:
            raise ValueError("InteractionPlot takes exactly two factors")
        self.factors = factors

    def fit(self, design, y, factor_names=None):
        X, y, names = _design_inputs(design, y, factor_names)
        ia, ib = [names.index(f) if isinstance(f, str) else int(f) for f in self.factors]
        la, lb = np.unique(X[:, ia]), np.unique(X[:, ib])
        M = np.full((len(la), len(lb)), np.nan)
        for r, a in enumerate(la):
            for c, b in enumerate(lb):
                mask = (X[:, ia] == a) & (X[:, ib] == b)
                if np.any(mask):
                    M[r, c] = y[mask].mean()
        self.factor_names_ = (names[ia], names[ib])
        self.levels_ = (la, lb)
        self.cell_means_ = M
        self.lines_ = {float(b): (la.copy(), M[:, c].copy()) for c, b in enumerate(lb)}
        self.interaction_effect_ = (
            float(((M[1, 1] - M[0, 1]) - (M[1, 0] - M[0, 0])) / 2.0)
            if M.shape == (2, 2) else None)
        return self

    def interaction_contrasts(self):
        M = self.cell_means_
        return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()

    def is_parallel(self, tol=1e-8):
        return bool(np.nanmax(np.abs(self.interaction_contrasts())) <= tol)

    def to_text(self):
        a, b = self.factor_names_
        la, lb = self.levels_
        head = f"{a} \\ {b}".ljust(12) + "".join(f"{v:>12.4g}" for v in lb)
        lines = [head] + [f"{v:<12.4g}" + "".join(f"{m:>12.4f}" for m in row)
                          for v, row in zip(la, self.cell_means_)]
        return "\n".join(lines)


# ------------------------------------------------------------------------ normal plot

class NormalPlot:
    """(Half-)normal probability plot of factorial effects (Daniel 1959) with Lenth's (1989)
    method for unreplicated two-level designs.

    Effects that are only noise fall on a straight line through the origin; active ones
    stand off it. Lenth's pseudo standard error ``PSE = 1.5 median(|e| : |e| < 2.5 s0)``
    (``s0 = 1.5 median|e|``) gives the margin of error ``ME = t_{1-alpha/2, m/3} PSE`` and
    the simultaneous margin ``SME = t_{gamma, m/3} PSE``, ``gamma = (1 + (1-alpha)^(1/m))/2``;
    ``active_`` lists the effects beyond ME. Plotting positions are ``Phi^-1((i-0.5)/m)``
    (half-normal: ``Phi^-1(0.5 + 0.5 (i-0.5)/m)`` on ``|e|``).
    """

    def __init__(self, half=False, alpha=0.05, model="full"):
        self.half = bool(half)
        self.alpha = float(alpha)
        self.model = model

    def fit(self, design, y, factor_names=None):
        from stochpylib.statistics._common import _norm_ppf, _t_ppf

        X, y, names = _design_inputs(design, y, factor_names)
        Xc = _two_level_coded(X)
        if Xc is None:
            raise ValueError("NormalPlot needs a two-level design")
        terms = _parse_terms(self.model, X.shape[1], names)
        if not _orthogonal(Xc, terms):
            raise ValueError("NormalPlot needs orthogonal effect columns")
        self.effects_, _ = _contrast_effects(Xc, y, terms, names)
        labels = list(self.effects_)
        e = np.array([self.effects_[l] for l in labels])
        m = len(e)
        key = np.abs(e) if self.half else e
        order = np.argsort(key, kind="mergesort")
        self.sorted_effects_ = [(labels[i], float(e[i])) for i in order]
        i = np.arange(1, m + 1)
        self.theoretical_quantiles_ = (_norm_ppf(0.5 + 0.5 * (i - 0.5) / m) if self.half
                                       else _norm_ppf((i - 0.5) / m))
        a = np.abs(e)
        s0 = 1.5 * np.median(a)
        self.pse_ = float(1.5 * np.median(a[a < 2.5 * s0]))
        d = m / 3.0
        self.me_ = float(_t_ppf(1.0 - self.alpha / 2.0, d) * self.pse_)
        gamma = (1.0 + (1.0 - self.alpha) ** (1.0 / m)) / 2.0
        self.sme_ = float(_t_ppf(gamma, d) * self.pse_)
        self.active_ = [l for l in labels if abs(self.effects_[l]) > self.me_]
        return self

    def to_text(self):
        lines = [f"{'effect':<10}{'estimate':>12}{'quantile':>12}"]
        for (lab, val), q in zip(self.sorted_effects_, self.theoretical_quantiles_):
            flag = "  *" if lab in self.active_ else ""
            lines.append(f"{lab:<10}{val:>12.4f}{q:>12.4f}{flag}")
        lines.append(f"PSE={self.pse_:.4f}  ME={self.me_:.4f}  SME={self.sme_:.4f}")
        return "\n".join(lines)


# ------------------------------------------------------------------------- sensitivity

def _input_count(bounds, distributions):
    if distributions is not None:
        return len(distributions)
    if bounds is not None:
        return np.shape(bounds)[0] if np.ndim(bounds) == 2 else 1
    raise ValueError("give bounds or distributions for the model inputs")


def _map_inputs(U, bounds, distributions):
    """Unit-cube samples -> model inputs through bounds (affine) or distribution ppfs."""
    if distributions is not None:
        U = np.clip(U, 1e-12, 1 - 1e-12)
        return np.column_stack([np.asarray(d.ppf(U[:, i]), dtype=float)
                                for i, d in enumerate(distributions)])
    b = _normalize_bounds(bounds, U.shape[1])
    return b[:, 0] + U * (b[:, 1] - b[:, 0])


def _eval(model, X):
    y = np.asarray(model(X), dtype=float).ravel()
    if y.size != len(X):
        raise ValueError("model(X) must return one value per row of X")
    return y


class SensitivityIndex:
    """Global sensitivity analysis of a model's output to its inputs.

    ``analyze(model, bounds=None, distributions=None)`` samples the inputs itself;
    ``fit(X, y)`` works on given data (regression/correlation methods only). ``model(X)``
    must be vectorized over the rows of ``X``. Methods:

    - ``"morris"`` -- Morris (1991) elementary effects on an ``n_levels`` grid in the unit
      cube (step ``p / (2(p-1))``), ``n_trajectories`` one-at-a-time trajectories:
      ``mu_``, ``mu_star_`` (Campolongo et al. 2007, the mean absolute effect) and
      ``sigma_`` (nonlinearity/interaction). Effects are per unit of the *scaled* input, so
      a linear term ``b x`` on ``[lo, hi]`` has ``mu_star = |b| (hi - lo)``.
    - ``"src"`` -- standardized regression coefficients (``src_``; ``r2_`` says how much a
      linear model explains).
    - ``"prcc"`` -- partial rank correlation coefficients (monotone, not only linear).
    - ``"correlation"`` -- Pearson and Spearman correlations.

    ``indices_`` collects every array the method produced.
    """

    _METHODS = ("morris", "src", "prcc", "correlation")

    def __init__(self, method="morris", n_trajectories=20, n_levels=4, random_state=None):
        if method not in self._METHODS:
            raise ValueError(f"method must be one of {self._METHODS}")
        self.method = method
        self.n_trajectories = int(n_trajectories)
        self.n_levels = int(n_levels)
        self.random_state = random_state

    def analyze(self, model, bounds=None, distributions=None, n_samples=1000):
        k = _input_count(bounds, distributions)
        rng = _rng(self.random_state)
        if self.method == "morris":
            return self._morris(model, k, bounds, distributions, rng)
        from stochpylib.montecarlo import LatinHypercubeSampling

        U = LatinHypercubeSampling(dim=k, n=int(n_samples), random_state=rng).generate()
        X = _map_inputs(U, bounds, distributions)
        return self.fit(X, _eval(model, X))

    def _morris(self, model, k, bounds, distributions, rng):
        p = self.n_levels
        if p < 2:
            raise ValueError("n_levels must be >= 2")
        delta = p / (2.0 * (p - 1))
        grid = np.arange(p) / (p - 1)
        start_levels = grid[grid <= 1.0 - delta + 1e-12]
        EE = np.empty((self.n_trajectories, k))
        points = []
        for r in range(self.n_trajectories):
            x = rng.choice(start_levels, k)
            signs = rng.choice([-1.0, 1.0], k)
            x = np.where(signs < 0, x + delta, x)
            traj = [x.copy()]
            for i in rng.permutation(k):
                x = x.copy()
                x[i] += signs[i] * delta
                traj.append(x)
            traj = np.array(traj)
            f = _eval(model, _map_inputs(traj, bounds, distributions))
            changed = np.argmax(np.abs(np.diff(traj, axis=0)), axis=1)
            for step, i in enumerate(changed):
                EE[r, i] = (f[step + 1] - f[step]) / (signs[i] * delta)
            points.append(traj)
        self.elementary_effects_ = EE
        self.mu_ = EE.mean(0)
        self.mu_star_ = np.abs(EE).mean(0)
        self.sigma_ = EE.std(0, ddof=1) if self.n_trajectories > 1 else np.zeros(k)
        self.n_evals_ = self.n_trajectories * (k + 1)
        self.indices_ = {"mu": self.mu_, "mu_star": self.mu_star_, "sigma": self.sigma_}
        self.ranking_ = list(np.argsort(-self.mu_star_))
        return self

    def fit(self, X, y):
        if self.method == "morris":
            raise ValueError("Morris screening needs the model: use analyze(model, ...)")
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        if y.size != len(X):
            raise ValueError("X and y must have the same number of rows")
        k = X.shape[1]
        if self.method == "src":
            from stochpylib.statistics import linear_regression

            Z = (X - X.mean(0)) / X.std(0)
            reg = linear_regression(Z, (y - y.mean()) / y.std())
            self.src_ = np.asarray(reg.coef_[1:], dtype=float)
            self.r2_ = float(reg.r2_)
            self.indices_ = {"src": self.src_, "r2": self.r2_}
        elif self.method == "prcc":
            R = np.column_stack([_ranks(X[:, i]) for i in range(k)])
            ry = _ranks(y)
            out = np.empty(k)
            for i in range(k):
                others = np.column_stack([np.ones(len(y))] + [R[:, j] for j in range(k)
                                                              if j != i])
                rx = R[:, i] - others @ np.linalg.lstsq(others, R[:, i], rcond=None)[0]
                rr = ry - others @ np.linalg.lstsq(others, ry, rcond=None)[0]
                out[i] = _pearson(rx, rr)
            self.prcc_ = out
            self.indices_ = {"prcc": out}
        else:
            self.pearson_ = np.array([_pearson(X[:, i], y) for i in range(k)])
            self.spearman_ = np.array([_spearman(X[:, i], y) for i in range(k)])
            self.indices_ = {"pearson": self.pearson_, "spearman": self.spearman_}
        return self


class SobolIndex(SensitivityIndex):
    """Variance-based Sobol sensitivity indices by the Saltelli sampling scheme.

    From two independent ``(N, k)`` input matrices ``A``, ``B`` (a scrambled Sobol sequence
    by default, via :class:`stochpylib.montecarlo.SobolSequence`) and the ``k`` hybrids
    ``AB_i`` (``A`` with column ``i`` from ``B``), ``N (k + 2)`` model runs give the
    first-order indices ``S_i = mean(f_B (f_ABi - f_A)) / V`` (Saltelli et al. 2010) and the
    total-order indices ``ST_i = mean((f_A - f_ABi)^2) / (2V)`` (Jansen 1999).
    ``second_order=True`` adds ``k`` more hybrids ``BA_i`` and the closed second-order
    indices ``S_ij`` (Saltelli 2002). Standard errors come from ``n_bootstrap`` row
    resamples; ``first_order_``/``total_order_`` are lists of
    :class:`stochpylib.montecarlo.MCResult`, so every index carries its confidence
    interval. ``S1_``/``ST_``/``S2_`` are the plain arrays.
    """

    def __init__(self, n_samples=4096, second_order=False, sampler="sobol", n_bootstrap=200,
                 random_state=None):
        if sampler not in ("sobol", "random"):
            raise ValueError("sampler must be 'sobol' or 'random'")
        self.n_samples = int(n_samples)
        self.second_order = bool(second_order)
        self.sampler = sampler
        self.n_bootstrap = int(n_bootstrap)
        self.random_state = random_state
        self.method = "sobol"

    @staticmethod
    def _indices(fA, fB, fAB, fBA):
        V = np.var(np.concatenate([fA, fB]))
        if V <= 0:
            raise ValueError("the model output has zero variance")
        S1 = np.mean(fB[None, :] * (fAB - fA[None, :]), axis=1) / V
        ST = 0.5 * np.mean((fA[None, :] - fAB) ** 2, axis=1) / V
        S2 = None
        if fBA is not None:
            k = len(fAB)
            S2 = np.full((k, k), np.nan)
            f0 = np.mean(fA) * np.mean(fB)
            for i, j in itertools.combinations(range(k), 2):
                Vij = np.mean(fBA[i] * fAB[j]) - f0
                S2[i, j] = S2[j, i] = Vij / V - S1[i] - S1[j]
        return S1, ST, S2

    def analyze(self, model, bounds=None, distributions=None):
        from stochpylib.montecarlo import MCResult, SobolSequence

        k = _input_count(bounds, distributions)
        rng = _rng(self.random_state)
        N = self.n_samples
        if self.sampler == "sobol":
            U = SobolSequence(dim=2 * k, random_state=rng).generate(N)
        else:
            U = rng.random((N, 2 * k))
        A, B = U[:, :k], U[:, k:]
        fA = _eval(model, _map_inputs(A, bounds, distributions))
        fB = _eval(model, _map_inputs(B, bounds, distributions))
        fAB = np.empty((k, N))
        fBA = np.empty((k, N)) if self.second_order else None
        for i in range(k):
            ABi = A.copy()
            ABi[:, i] = B[:, i]
            fAB[i] = _eval(model, _map_inputs(ABi, bounds, distributions))
            if self.second_order:
                BAi = B.copy()
                BAi[:, i] = A[:, i]
                fBA[i] = _eval(model, _map_inputs(BAi, bounds, distributions))
        S1, ST, S2 = self._indices(fA, fB, fAB, fBA)
        boot1 = np.empty((self.n_bootstrap, k))
        bootT = np.empty((self.n_bootstrap, k))
        for b in range(self.n_bootstrap):
            r = rng.integers(0, N, N)
            s1, st, _ = self._indices(fA[r], fB[r], fAB[:, r], None)
            boot1[b], bootT[b] = s1, st
        se1 = boot1.std(0, ddof=1) if self.n_bootstrap > 1 else np.full(k, np.nan)
        seT = bootT.std(0, ddof=1) if self.n_bootstrap > 1 else np.full(k, np.nan)
        n_evals = N * (k + 2) + (N * k if self.second_order else 0)
        self.S1_, self.ST_, self.S2_ = S1, ST, S2
        self.first_order_ = [MCResult(float(S1[i]), float(se1[i]), n_evals,
                                      "sobol first-order (Saltelli 2010)") for i in range(k)]
        self.total_order_ = [MCResult(float(ST[i]), float(seT[i]), n_evals,
                                      "sobol total-order (Jansen 1999)") for i in range(k)]
        self.n_evals_ = n_evals
        self.variance_ = float(np.var(np.concatenate([fA, fB])))
        self.indices_ = {"S1": S1, "ST": ST, "S2": S2}
        return self

    def fit(self, X, y):
        raise ValueError("Sobol indices need the model: use analyze(model, bounds=...)")
