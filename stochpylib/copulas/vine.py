"""Vine copulas: sequential pair-copula constructions.

One recursive machinery backs all three structures. Every *edge* joins two
child nodes that share exactly one child (the proximity condition, expressed
as shared child identity) and stores

- its fitted :class:`PairCopulaConstruction`,
- references to its two children,
- two *exposed* columns ``h(x|y)`` / ``h(y|x)`` — each child's non-shared
  head variable conditioned on everything else — consumed by the next tree
  level, the likelihood and the sampler.

Growth policies:
- ``CVine`` : star around one anchor variable at every level.
- ``DVine`` : path adjacency following a variable order (greedy tau-MST order
  by default).
- ``RVine`` : maximum-spanning-tree selection on |Kendall's tau| among all
  proximity-admissible pairs at each level (Disshmann-style heuristic under
  the simplifying assumption — documented deviation from full search).

Likelihood and simulation replay the stored structure: the likelihood sums
pair log-densities over recomputed fitting columns; the sequential sampler
draws every newly introduced variable as ``h_inv(p | conditioning column)``
in introduction order.
"""

import numpy as np

from stochpylib.copulas._base import BaseCopula
from stochpylib.copulas._utils import kendall_tau_estimate, pseudo_obs
from stochpylib.copulas.pair import PAIR_FAMILIES, PairCopulaConstruction

__all__ = [
    "CVine", "DVine", "RVine", "VineStructureSelect", "VineCopula",
]

_EPS = 1e-12


class _LeafNode:
    is_leaf = True

    def __init__(self, var, column):
        self.var = int(var)
        self.column = column
        self.head = self.var                 # the variable this node carries

    def exposed(self, side="a"):
        return self.column


class _EdgeNode:
    is_leaf = False

    def __init__(self, child_a, child_b, pair, col_a, col_b,
                 head_a=None, head_b=None, side_a="a", side_b="b"):
        """``col_a``/``col_b`` are the fitting columns — ``child_a.exposed(
        side_a)`` and ``child_b.exposed(side_b)`` respectively; ``pair`` was
        fitted on exactly those two columns. ``head_a``/``head_b`` are the
        ORIGINAL variables those columns carry."""
        self.child_a = child_a
        self.child_b = child_b
        self.pair = pair
        self.head_a = child_a.head if head_a is None else int(head_a)
        self.head_b = child_b.head if head_b is None else int(head_b)
        self.side_a = side_a
        self.side_b = side_b
        self.introduced = None               # filled by the builder
        self._col_a = col_a                  # h(col_a | col_b)
        self._col_b = col_b                  # h(col_b | col_a)

    @property
    def heads(self):
        return {self.head_a, self.head_b}

    @property
    def head(self):
        return self.head_a

    def exposed(self, side):
        return self._col_a if side == "a" else self._col_b


def _shared_child(A, B):
    """The single child object shared by nodes A and B, or None."""
    if A.is_leaf or B.is_leaf:
        return None
    for ca in (A.child_a, A.child_b):
        for cb in (B.child_a, B.child_b):
            if ca is cb:
                return ca
    return None


def _tau_mst_order(u):
    """Greedy variable ordering: maximum spanning tree on |tau|, DFS order."""
    d = u.shape[1]
    T = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            T[i, j] = T[j, i] = abs(kendall_tau_estimate(u[:, i], u[:, j]))
    visited = [0]
    rest = set(range(1, d))
    while rest:
        best = None
        for v in visited:
            for w in rest:
                if best is None or T[v, w] > T[best[0], best[1]]:
                    best = (v, w)
        visited.append(best[1])
        rest.discard(best[1])
    # DFS order of the MST path approximation: the greedy insertion order is
    # itself a valid path-like ordering
    return visited


class _VineBase(BaseCopula):
    dimension = "d"
    structure_type = "?"

    def __init__(self, order=None, families=PAIR_FAMILIES,
                 allow_rotations=True):
        super().__init__()
        self.order = list(order) if order is not None else None
        self.families = tuple(families)
        self.allow_rotations = bool(allow_rotations)
        self.levels_ = None
        self.u_obs_ = None
        self.loglik_ = None

    def _require_fit(self):
        if self.levels_ is None:
            raise RuntimeError("fit() must be called first")

    # -- policy hooks -----------------------------------------------------------
    def _leaf_order(self, u):
        if self.order is not None:
            return [int(v) for v in self.order]
        return _tau_mst_order(u)

    def _choose_merges(self, adj, nodes, level_idx):
        """Choose node-index pairs among admissible ``adj`` pairs."""
        raise NotImplementedError

    # -- construction -------------------------------------------------------------
    def _build(self, u):
        order = self._leaf_order(u)
        nodes_meta = [(pos, _LeafNode(var, u[:, var]))
                      for pos, var in enumerate(order)]
        levels = []
        all_edges = []
        while True:
            active = [n for _, n in sorted(nodes_meta, key=lambda t: t[0])]
            if len(active) <= 1:
                break
            adj = [(i, j) for i in range(len(active))
                   for j in range(i + 1, len(active))
                   if self._admissible(active, i, j, len(levels))]
            if not adj:
                break
            merges = self._choose_merges(adj, active, len(levels))
            new_meta = []
            for i, j in merges:
                A, B = active[i], active[j]
                sa = self._away_side(A, B)
                sb = self._away_side(B, A)
                if sa is None or sb is None:
                    continue
                x = np.clip(A.exposed(sa), _EPS, 1.0 - _EPS)
                y = np.clip(B.exposed(sb), _EPS, 1.0 - _EPS)
                pair = PairCopulaConstruction.fit(
                    x, y, families=self.families,
                    allow_rotations=self.allow_rotations)
                def _away_head(node, side):
                    return node.var if node.is_leaf else \
                        (node.head_a if side == "a" else node.head_b)

                edge = _EdgeNode(A, B, pair,
                                 col_a=np.clip(pair.h(x, y), _EPS, 1 - _EPS),
                                 col_b=np.clip(pair.h(y, x), _EPS, 1 - _EPS),
                                 head_a=_away_head(A, sa),
                                 head_b=_away_head(B, sb),
                                 side_a=sa, side_b=sb)
                edge._tau = float(abs(kendall_tau_estimate(x, y)))
                all_edges.append(edge)
                pos = self._pos_of(nodes_meta, active[i])
                new_meta.append((pos, edge))
            if not new_meta:
                raise RuntimeError('vine construction stalled: admissible '
                                   'pairs could not be merged')
            nodes_meta = sorted(new_meta, key=lambda t: t[0])
            levels.append(list(nodes_meta))
        self.levels_ = [[n for _, n in lvl] for lvl in levels]
        self._all_edges = all_edges
        # Introduction plan — R-vine-matrix-style peeling order: starting from
        # one seed variable drawn marginally, every further variable is drawn
        # through an edge whose ENTIRE leaf set minus that variable is already
        # drawn, so its conditioning column is fully observed (exact
        # sequential Rosenblatt). Try every seed; take the first feasible peel.
        d = u.shape[1]

        def leaf_set(node):
            if node.is_leaf:
                return {node.var}
            return leaf_set(node.child_a) | leaf_set(node.child_b)

        leaf_sets = {id(e): leaf_set(e) for e in all_edges}

        def try_seed(seed):
            drawn = {seed}
            plan = [(None, seed)]
            while len(drawn) < d:
                progressed = False
                for level in reversed(self.levels_):
                    for e in level:
                        cands = [h for h in (e.head_a, e.head_b)
                                 if h not in drawn]
                        if len(cands) != 1:
                            continue
                        others = leaf_sets[id(e)] - set(cands)
                        if others and others <= drawn:
                            plan.append((e, cands[0]))
                            drawn.add(cands[0])
                            progressed = True
                            break
                    if progressed:
                        break
                if not progressed:
                    return None
            return plan

        plan = None
        for seed in range(d):
            plan = try_seed(seed)
            if plan is not None:
                break
        if plan is None:
            raise RuntimeError("vine structure cannot be peeled into a "
                               "simulation order")
        self._intro_plan_ = plan

    @staticmethod
    def _pos_of(nodes_meta, node):
        for p, n in nodes_meta:
            if n is node:
                return p
        return 0

    @staticmethod
    def _away_side(A, B):
        docstr = 'Exposed side of A whose head is NOT the child shared with B.'
        del docstr
        if A.is_leaf:
            return 'a'
        s = _shared_child(A, B)
        if s is None:
            return None
        return 'b' if A.child_a is s else 'a'

    def _admissible(self, active, i, j, level_idx):
        A, B = active[i], active[j]
        if level_idx == 0:
            return self._level0_admissible(len(active), i, j)
        return _shared_child(A, B) is not None

    def _level0_admissible(self, count, i, j):
        return True  # overridden per structure

    # ------------------------------------------------------------------ surface
    def fit(self, data):
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[0] < 3:
            raise ValueError("data must be a (n, d) array with n >= 3")
        if self.dimension == "d":
            self.dimension = int(data.shape[1])
        elif data.shape[1] != self.dimension:
            raise ValueError(f"expected {self.dimension} columns")
        u = pseudo_obs(data)
        self.n_obs_ = len(u)
        self.u_obs_ = u
        self._build(u)
        if not self._all_edges:
            raise RuntimeError("vine construction produced no edges")
        self.loglik_ = float(self._total_loglik(u))
        return self

    def _iter_edges(self):
        return iter(self._all_edges)

    def _columns(self, edge, u):
        """Recompute the edge's fitting columns for data ``u`` (fresh cache;
        never reuse caches across different datasets/calls)."""
        cache = {}

        def child_col(node, side):
            if node.is_leaf:
                return u[:, node.var]
            key = (id(node), side)
            if key not in cache:
                xl = child_col(node.child_a, node.side_a)
                yl = child_col(node.child_b, node.side_b)
                cache[(id(node), "a")] = node.pair.h(xl, yl)
                cache[(id(node), "b")] = node.pair.h(yl, xl)
            return cache[key]

        x = child_col(edge.child_a, edge.side_a)
        y = child_col(edge.child_b, edge.side_b)
        return x, y

    def loglik(self, data=None, raw=True):
        self._require_fit()
        if data is None:
            return self.loglik_
        u = pseudo_obs(data) if raw else as_u_matrix(data)
        return self._total_loglik(u)

    def _total_loglik(self, u):
        total = 0.0
        for e in self._all_edges:
            x, y = self._columns(e, u)
            total += e.pair.loglik(x, y)
        return float(total)

    def aic(self, data=None, raw=True):
        ll = self.loglik(data, raw=raw)
        n_params = sum(getattr(e.pair.copula, "_n_params", 1)
                       for e in self._iter_edges())
        return -2.0 * ll + 2.0 * n_params

    def kendall_tau(self):
        self._require_fit()
        u = self.u_obs_
        d = u.shape[1]
        T = np.eye(d)
        for i in range(d):
            for j in range(i + 1, d):
                T[i, j] = T[j, i] = kendall_tau_estimate(u[:, i], u[:, j])
        return T if d > 2 else float(T[0, 1])

    def tail_dependence(self):
        return {"upper": None, "lower": None}

    def summary(self):
        self._require_fit()
        rows = []
        for k, level in enumerate(self.levels_, start=1):
            for e in level:
                rows.append(f"tree {k}: {sorted(e.heads)} | "
                            f"{e.pair.describe()} | |tau|={e._tau:.2f}")
        return "\n".join(rows)

    # ------------------------------------------------------------------ sampling
    def sample(self, n, random_state=None):
        """Sequential Rosenblatt simulation following the deepest-first
        introduction plan; every sibling subtree is fully seeded before its
        conditioning column is used."""
        self._require_fit()
        n = self._validate_sample_n(n)
        rng = np.random.default_rng(random_state)
        d = self.dimension
        sim = np.full((n, d), np.nan)
        cache = {}

        def child_col(node, side):
            if node.is_leaf:
                return sim[:, node.var]
            key = (id(node), side)
            if key not in cache:
                xl = child_col(node.child_a, node.side_a)
                yl = child_col(node.child_b, node.side_b)
                cache[(id(node), "a")] = node.pair.h(xl, yl)
                cache[(id(node), "b")] = node.pair.h(yl, xl)
            return cache[key]

        def realize(node, var, vals):
            """Invert the h-chain from an edge's exposed column down to the
            head variable `var` (the exposed column equals h(head | rest), so
            recover the head by applying pair.h_inv towards the leaves)."""
            if node.is_leaf:
                sim[:, node.var] = vals
                return
            if var == node.head_a:
                my_side, my_child = node.side_a, node.child_a
                other_side, other_child = node.side_b, node.child_b
            else:
                my_side, my_child = node.side_b, node.child_b
                other_side, other_child = node.side_a, node.child_a
            other_vals = child_col(other_child, other_side)
            if np.any(np.isnan(other_vals)):
                raise RuntimeError("vine simulation order violated: "
                                   "conditioning column not fully drawn")
            child_vals = np.clip(node.pair.h_inv(vals, other_vals), _EPS,
                                 1 - _EPS)
            cache.clear()
            realize(my_child, var, child_vals)

        for e, var in self._intro_plan_:
            if e is None:                     # seed variable
                sim[:, var] = rng.random(n)
                continue
            p = rng.random(n)
            realize(e, var, p)
            cache.clear()
        if np.any(np.isnan(sim)):
            raise RuntimeError("vine sampler left unsimulated variables")
        return sim


class DVine(_VineBase):
    """Drawable vine following a variable order (greedy tau-MST order by
    default)::

        dv = DVine().fit(data)
        dv.aic(data); sims = dv.sample(1000, random_state=0); print(dv.summary())
    """

    structure_type = "D"

    def _level0_admissible(self, count, i, j):
        return j == i + 1                       # path adjacency

    def _choose_merges(self, adj, nodes, level_idx):
        if level_idx == 0:
            return [(i, j) for (i, j) in adj]   # consecutive pairs
        # later levels: consecutive nodes in the (order-preserving) list
        return [(i, i + 1) for i in range(len(nodes) - 1)]


class CVine(_VineBase):
    """Canonical vine: star around one anchor variable at every level."""

    structure_type = "C"

    def __init__(self, anchor=None, **kwargs):
        super().__init__(**kwargs)
        self.anchor = int(anchor) if anchor is not None else None

    def _leaf_order(self, u):
        if self.order is not None:
            return [int(v) for v in self.order]
        order = _tau_mst_order(u)
        if self.anchor is not None:
            order.remove(self.anchor)
            order.insert(0, self.anchor)
        return order

    def _level0_admissible(self, count, i, j):
        return i == 0                           # star around the anchor

    def _choose_merges(self, adj, nodes, level_idx):
        return [(i, j) for (i, j) in adj if i == 0]


class RVine(_VineBase):
    """R-vine with Disshmann-style MST selection on |tau|."""

    structure_type = "R"

    def _choose_merges(self, adj, nodes, level_idx):
        weights = []
        for i, j in adj:
            x = nodes[i].exposed("l") if hasattr(nodes[i], "exposed") else None
            x = nodes[i].exposed("a")
            y = nodes[j].exposed("a")
            weights.append(abs(kendall_tau_estimate(x, y)))
        ranked = sorted(zip(adj, weights), key=lambda e: -e[1])
        chosen, parent = [], {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for (i, j), _w in ranked:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
                chosen.append((i, j))
            if len(chosen) == len(nodes) - 1:
                break
        return chosen


def VineStructureSelect(data, types=("CVine", "DVine", "RVine"), **kwargs):
    """Fit several vine structures and return the best by AIC (fluent)."""
    best = None
    for name in types:
        cls = {"CVine": CVine, "DVine": DVine, "RVine": RVine}[name]
        cand = cls(**kwargs)
        try:
            cand.fit(data)
        except (ValueError, RuntimeError):
            continue
        score = cand.aic(data)
        if np.isfinite(score) and (best is None or score < best[0]):
            best = (score, cand)
    if best is None:
        raise RuntimeError("no vine structure could be fitted")
    return best[1]


class VineCopula(BaseCopula):
    """Spec-facing facade::

        vc = VineCopula(type="DVine").fit(data_5d)
        sims = vc.sample(2000, random_state=0)
        vc.aic(data_5d); print(vc.summary())
    """

    dimension = "d"
    _types = {"C": CVine, "D": DVine, "R": RVine,
              "CVine": CVine, "DVine": DVine, "RVine": RVine}

    def __init__(self, type="RVine", order=None, families=PAIR_FAMILIES,
                 allow_rotations=True):
        super().__init__()
        if type not in self._types:
            raise ValueError(f"unknown vine type {type!r}")
        self.type = type
        self.vine_ = self._types[type](order=order, families=families,
                                       allow_rotations=allow_rotations)

    def _require_fit(self):
        self.vine_._require_fit()

    def _estimate(self, u):     # pragma: no cover - fit is overridden
        raise NotImplementedError

    def fit(self, data):
        self.dimension = int(np.asarray(data).shape[1])
        self.vine_.fit(data)
        self.n_obs_ = self.vine_.n_obs_
        return self

    def sample(self, n, random_state=None):
        return self.vine_.sample(n, random_state=random_state)

    def cdf(self, u):
        raise NotImplementedError("vine copula CDF has no closed form; "
                                  "use the likelihood instead")

    def loglik(self, data=None, raw=True):
        return self.vine_.loglik(data, raw=raw)

    def aic(self, data=None, raw=True):
        return self.vine_.aic(data, raw=raw)

    def kendall_tau(self):
        return self.vine_.kendall_tau()

    def summary(self):
        return self.vine_.summary()
