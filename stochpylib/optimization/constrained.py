"""Constrained optimization: quadratic penalty, augmented Lagrangian, Lagrangian
relaxation by dual ascent, an active-set solver for linearly constrained quadratics, and a
primal log-barrier interior-point method.

Every class subclasses :class:`~stochpylib.optimization._base.ConstrainedOptimizer` and
takes constraints in the canonical ``{"type": "eq"|"ineq", "fun": ..., "jac": ...}`` form,
with ``"ineq"`` meaning ``fun(x) >= 0`` is feasible -- the same sign convention
``scipy.optimize.minimize(method="SLSQP")`` uses, so the two are directly comparable.
``result_.extras["violation"]`` and ``["feasible"]`` report how well the returned point
actually satisfies the constraints: none of these methods guarantees exact feasibility, and
silently returning an infeasible point would be the worst failure mode here.
"""

import numpy as np

from stochpylib.optimization._base import ConstrainedOptimizer
from stochpylib.optimization._result import OptimizeResult

__all__ = ["LagrangianRelaxation", "PenaltyMethod", "AugmentedLagrangian", "ActiveSet",
           "InteriorPoint"]


def _default_inner():
    from stochpylib.optimization.second_order import LBFGS

    return LBFGS(max_iter=400)


def _relaxed_log(c, delta):
    """``-log(c)``, continued below ``delta`` by its quadratic extrapolation.

    A plain log barrier is ``+inf`` outside the feasible region, so any finite-difference
    gradient taken near the boundary comes back NaN and poisons the inner optimizer. The
    relaxed barrier (Hauser 2003; Feller & Ebenbauer 2017) is C^2 and finite everywhere,
    agrees with ``-log(c)`` wherever ``c > delta``, and grows quadratically outside -- so an
    infeasible trial point is pushed back rather than producing a NaN.
    """
    c = np.asarray(c, dtype=float)
    safe = np.maximum(c, delta)
    k = (c - 2.0 * delta) / delta
    return np.where(c > delta, -np.log(safe), 0.5 * (k * k - 1.0) - np.log(delta))


class PenaltyMethod(ConstrainedOptimizer):
    """Quadratic penalty method: minimize ``f(x) + (mu/2) * ||violation(x)||^2`` for an
    increasing sequence of ``mu``.

    Simple and robust far from the solution, but the subproblem's condition number grows
    like ``mu``, so the last few outer iterations are numerically hard and the returned
    point is only feasible to ``O(1/mu)``. That inexactness is the reason
    :class:`AugmentedLagrangian` exists; use this one when you want the unconstrained
    subproblem to stay a plain smooth minimization.
    """

    method = "penalty"

    def __init__(self, mu0=10.0, mu_factor=10.0, n_outer=12, ctol=1e-6, inner=None,
                 constraints=(), bounds=None, random_state=None):
        self.mu0 = mu0
        self.mu_factor = mu_factor
        self.n_outer = n_outer
        self.ctol = ctol
        self.inner = inner
        self.constraints = constraints
        self.bounds = bounds
        self.random_state = random_state

    def _minimize(self, obj, x0, rng):
        cons = self._normalized_constraints()
        inner = self.inner if self.inner is not None else _default_inner()
        x = self._project(x0)
        mu = float(self.mu0)
        history = [obj(x)]
        converged, nit = False, 0
        for k in range(1, int(self.n_outer) + 1):
            nit = k

            def penalized(z, _mu=mu):
                eq, ineq = self._constraint_values(cons, z)
                pen = float(np.sum(eq ** 2) + np.sum(np.minimum(0.0, ineq) ** 2))
                return obj(z) + 0.5 * _mu * pen

            x = self._project(inner.minimize(penalized, x).result_.x)
            history.append(obj(x))
            if self._violation(cons, x) <= self.ctol:
                converged = True
                break
            mu *= self.mu_factor
        result = OptimizeResult(x, obj(x), nit=nit, converged=converged, method=self.method,
                                message="feasible" if converged else "penalty budget exhausted",
                                history=history, extras={"mu": mu})
        return self._finish(result, cons, x)


class AugmentedLagrangian(ConstrainedOptimizer):
    """Augmented Lagrangian (method of multipliers, Hestenes 1969 / Powell 1969).

    Minimizes ``f(x) - lambda.c(x) + (mu/2)||c(x)||^2`` and then updates the multipliers by
    ``lambda <- lambda - mu * c(x)``. The multiplier term absorbs the constraint force that
    the pure penalty method has to buy with an ever-growing ``mu``, so this converges to an
    *exactly* feasible point at a finite penalty and is far better conditioned.
    Inequalities use the standard shifted form ``min(0, c - lambda/mu)``.
    """

    method = "augmented lagrangian"

    def __init__(self, mu0=10.0, mu_factor=5.0, n_outer=25, ctol=1e-8, inner=None,
                 constraints=(), bounds=None, random_state=None):
        self.mu0 = mu0
        self.mu_factor = mu_factor
        self.n_outer = n_outer
        self.ctol = ctol
        self.inner = inner
        self.constraints = constraints
        self.bounds = bounds
        self.random_state = random_state

    def _minimize(self, obj, x0, rng):
        cons = self._normalized_constraints()
        inner = self.inner if self.inner is not None else _default_inner()
        x = self._project(x0)
        eq0, ineq0 = self._constraint_values(cons, x)
        lam_eq = np.zeros(eq0.size)
        lam_in = np.zeros(ineq0.size)
        mu = float(self.mu0)
        history = [obj(x)]
        converged, nit = False, 0
        prev_violation = np.inf
        for k in range(1, int(self.n_outer) + 1):
            nit = k

            def augmented(z, _mu=mu, _le=lam_eq, _li=lam_in):
                eq, ineq = self._constraint_values(cons, z)
                out = obj(z)
                if eq.size:
                    out += -float(_le @ eq) + 0.5 * _mu * float(np.sum(eq ** 2))
                if ineq.size:
                    shifted = np.minimum(0.0, ineq - _li / _mu)
                    out += 0.5 * _mu * float(np.sum(shifted ** 2)) - float(np.sum(_li ** 2)) / (2 * _mu)
                return out

            x = self._project(inner.minimize(augmented, x).result_.x)
            eq, ineq = self._constraint_values(cons, x)
            if eq.size:
                lam_eq = lam_eq - mu * eq
            if ineq.size:
                lam_in = np.maximum(0.0, lam_in - mu * ineq)
            history.append(obj(x))
            violation = self._violation(cons, x)
            if violation <= self.ctol:
                converged = True
                break
            # only tighten the penalty when the multipliers alone did not help
            if violation > 0.25 * prev_violation:
                mu *= self.mu_factor
            prev_violation = violation
        result = OptimizeResult(x, obj(x), nit=nit, converged=converged, method=self.method,
                                message="feasible" if converged else "outer budget exhausted",
                                history=history,
                                extras={"mu": mu, "lambda_eq": lam_eq, "lambda_ineq": lam_in})
        return self._finish(result, cons, x)


class LagrangianRelaxation(ConstrainedOptimizer):
    """Lagrangian relaxation with subgradient dual ascent.

    Drops the constraints into the objective as ``L(x, lambda) = f(x) - lambda.c(x)``,
    minimizes over ``x`` for fixed multipliers, then takes a subgradient *ascent* step on
    ``lambda`` using the constraint residual. Unlike the two methods above, the value it
    reports is a genuine **lower bound** on the constrained optimum (weak duality) --
    ``result_.extras["dual_bound"]`` -- which is what makes relaxation useful even when the
    primal point it returns is infeasible. For a convex problem the duality gap closes;
    otherwise ``extras["duality_gap"]`` quantifies what is left.

    The primal point returned is the least-infeasible iterate seen (objective as
    tie-break), the usual primal-recovery heuristic. It is genuinely not guaranteed
    feasible: check ``extras["violation"]``, and prefer :class:`AugmentedLagrangian` when a
    feasible point rather than a bound is what you need. ``ctol`` defaults to ``1e-4``
    rather than the ``1e-8`` the multiplier methods use, because subgradient dual ascent
    converges at ``O(1/k)`` and a tighter default would report non-convergence on problems
    it has in fact solved.
    """

    method = "lagrangian relaxation"

    def __init__(self, step_size=0.1, n_outer=400, step_decay=0.995, ctol=1e-4, inner=None,
                 constraints=(), bounds=None, random_state=None):
        self.step_size = step_size
        self.n_outer = n_outer
        self.step_decay = step_decay
        self.ctol = ctol
        self.inner = inner
        self.constraints = constraints
        self.bounds = bounds
        self.random_state = random_state

    def _minimize(self, obj, x0, rng):
        cons = self._normalized_constraints()
        inner = self.inner if self.inner is not None else _default_inner()
        x = self._project(x0)
        eq0, ineq0 = self._constraint_values(cons, x)
        lam_eq = np.zeros(eq0.size)
        lam_in = np.zeros(ineq0.size)
        # x0 is never a candidate: it minimizes no Lagrangian, and a feasible-but-poor
        # start would otherwise win outright
        best_x, best_key = None, None
        dual_bound = -np.inf
        history = []
        step = float(self.step_size)
        for k in range(1, int(self.n_outer) + 1):

            def lagrangian(z, _le=lam_eq, _li=lam_in):
                eq, ineq = self._constraint_values(cons, z)
                return obj(z) - float(_le @ eq) - float(_li @ ineq)

            x = self._project(inner.minimize(lagrangian, x).result_.x)
            eq, ineq = self._constraint_values(cons, x)
            dual_bound = max(dual_bound, lagrangian(x))
            f_x = obj(x)
            history.append(f_x)
            violation = self._violation(cons, x)
            # standard primal recovery: least-infeasible iterate, objective as tie-break
            key = (max(0.0, violation - self.ctol), f_x)
            if best_key is None or key < best_key:
                best_x, best_key = x.copy(), key
            if violation <= self.ctol:
                break
            # dg/dlambda = -c(x*), so ascending the dual means stepping against the
            # residual; the inequality multipliers stay non-negative by projection
            if eq.size:
                lam_eq = lam_eq - step * eq
            if ineq.size:
                lam_in = np.maximum(0.0, lam_in - step * ineq)
            step *= self.step_decay
        x_final = best_x if best_x is not None else x
        primal = obj(x_final)
        result = OptimizeResult(x_final, primal, nit=int(self.n_outer),
                                converged=self._violation(cons, x_final) <= self.ctol,
                                method=self.method,
                                message="dual ascent finished", history=history,
                                extras={"dual_bound": dual_bound,
                                        "duality_gap": primal - dual_bound,
                                        "lambda_eq": lam_eq, "lambda_ineq": lam_in})
        return self._finish(result, cons, x_final)


class ActiveSet(ConstrainedOptimizer):
    """Primal active-set method for a quadratic objective under linear constraints.

    Solves ``min 0.5 x'Gx + c'x`` subject to ``A_eq x = b_eq`` and ``A_ub x >= b_ub`` by
    repeatedly solving the equality-constrained KKT system on the current working set,
    adding a blocking constraint when a step is cut short and dropping the working
    constraint with the most negative multiplier when none is (Nocedal & Wright Alg. 16.3).
    The objective's ``G``/``c`` are taken from the Hessian and gradient at ``x0``, so a
    non-quadratic objective is solved as its second-order model there -- ``ActiveSet``
    is a QP solver, not a general nonlinear one.
    """

    method = "active set"

    def __init__(self, A_eq=None, b_eq=None, A_ub=None, b_ub=None, max_iter=200, ctol=1e-8,
                 constraints=(), bounds=None, random_state=None):
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.max_iter = max_iter
        self.ctol = ctol
        self.constraints = constraints
        self.bounds = bounds
        self.random_state = random_state

    def _matrices(self, dim):
        A_eq = np.zeros((0, dim)) if self.A_eq is None else np.atleast_2d(np.asarray(self.A_eq, float))
        b_eq = np.zeros(0) if self.b_eq is None else np.atleast_1d(np.asarray(self.b_eq, float))
        A_ub = np.zeros((0, dim)) if self.A_ub is None else np.atleast_2d(np.asarray(self.A_ub, float))
        b_ub = np.zeros(0) if self.b_ub is None else np.atleast_1d(np.asarray(self.b_ub, float))
        return A_eq, b_eq, A_ub, b_ub

    @staticmethod
    def _kkt_step(G, g, A_work):
        """Solve the equality-constrained QP step; returns ``(p, multipliers)``."""
        n = G.shape[0]
        m = A_work.shape[0]
        if m == 0:
            return np.linalg.lstsq(G, -g, rcond=None)[0], np.zeros(0)
        K = np.block([[G, -A_work.T], [A_work, np.zeros((m, m))]])
        rhs = np.concatenate([-g, np.zeros(m)])
        sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        return sol[:n], sol[n:]

    def _minimize(self, obj, x0, rng):
        cons = self._normalized_constraints()
        dim = len(x0)
        G = obj.hess(x0)
        c = obj.grad(x0) - G @ x0
        A_eq, b_eq, A_ub, b_ub = self._matrices(dim)
        x = np.asarray(x0, dtype=float).copy()
        if A_eq.shape[0]:  # start from a point that already satisfies the equalities
            x = x + np.linalg.lstsq(A_eq, b_eq - A_eq @ x, rcond=None)[0]
        working = [i for i in range(A_ub.shape[0]) if abs(A_ub[i] @ x - b_ub[i]) <= self.ctol]
        history = [obj(x)]
        converged, nit = False, 0
        for k in range(1, int(self.max_iter) + 1):
            nit = k
            A_work = np.vstack([A_eq, A_ub[working]]) if (A_eq.shape[0] or working) \
                else np.zeros((0, dim))
            g = G @ x + c
            p, lam = self._kkt_step(G, g, A_work)
            if np.linalg.norm(p) <= self.ctol:
                lam_ub = lam[A_eq.shape[0]:]
                if lam_ub.size == 0 or np.min(lam_ub) >= -self.ctol:
                    converged = True
                    break
                working.pop(int(np.argmin(lam_ub)))
                continue
            # the largest feasible step along p, and the constraint that blocks it
            alpha, blocking = 1.0, None
            for i in range(A_ub.shape[0]):
                if i in working:
                    continue
                denom = A_ub[i] @ p
                if denom < -self.ctol:
                    ratio = (b_ub[i] - A_ub[i] @ x) / denom
                    if ratio < alpha:
                        alpha, blocking = ratio, i
            x = x + alpha * p
            if blocking is not None:
                working.append(blocking)
            history.append(obj(x))
        result = OptimizeResult(x, obj(x), nit=nit, converged=converged, method=self.method,
                                message="KKT satisfied" if converged else "max_iter reached",
                                history=history,
                                extras={"working_set": sorted(working), "G": G, "c": c})
        result.extras["violation"] = self._linear_violation(A_eq, b_eq, A_ub, b_ub, x)
        result.extras["feasible"] = result.extras["violation"] <= max(self.ctol, 1e-8)
        if cons:
            result.extras["violation"] = max(result.extras["violation"],
                                             self._violation(cons, x))
        return result

    @staticmethod
    def _linear_violation(A_eq, b_eq, A_ub, b_ub, x):
        v = 0.0
        if A_eq.shape[0]:
            v = max(v, float(np.max(np.abs(A_eq @ x - b_eq))))
        if A_ub.shape[0]:
            v = max(v, float(np.max(np.maximum(0.0, b_ub - A_ub @ x))))
        return v


class InteriorPoint(ConstrainedOptimizer):
    """Primal log-barrier interior-point method.

    Replaces each inequality ``c(x) >= 0`` by ``-t * log(c(x))`` and minimizes the barrier
    subproblem for a decreasing sequence of ``t``, so the iterates approach the boundary
    along the central path from strictly inside. Equalities are handled by a quadratic
    penalty alongside the barrier. The classical duality-gap bound ``m * t`` (``m`` =
    number of inequalities) gives the stopping rule and is reported as
    ``extras["duality_gap_bound"]``.

    ``x0`` **must be strictly feasible** for the inequalities, as for any primal
    interior-point method. Trial points that leave the region during a subproblem are
    handled by a relaxed barrier (see :func:`_relaxed_log`) rather than by ``+inf``.
    Equality constraints are carried as a fixed quadratic penalty rather than by a
    primal-dual KKT step, so they are only satisfied to ``O(1/mu_eq)``;
    :class:`AugmentedLagrangian` is the better choice when equalities dominate.
    """

    method = "interior point"

    def __init__(self, t0=1.0, t_factor=0.2, n_outer=30, mu_eq=1e6, delta=1e-4, tol=1e-9,
                 ctol=1e-6, inner=None, constraints=(), bounds=None, random_state=None):
        self.t0 = t0
        self.t_factor = t_factor
        self.n_outer = n_outer
        self.mu_eq = mu_eq
        self.delta = delta
        self.tol = tol
        self.ctol = ctol
        self.inner = inner
        self.constraints = constraints
        self.bounds = bounds
        self.random_state = random_state

    def _minimize(self, obj, x0, rng):
        cons = self._normalized_constraints()
        inner = self.inner if self.inner is not None else _default_inner()
        x = self._project(x0)
        _, ineq0 = self._constraint_values(cons, x)
        if ineq0.size and np.min(ineq0) <= 0:
            raise ValueError("InteriorPoint requires a strictly feasible x0 (all ineq > 0)")
        m = max(1, ineq0.size)
        t = float(self.t0)
        history = [obj(x)]
        converged, nit = False, 0
        for k in range(1, int(self.n_outer) + 1):
            nit = k

            def barrier(z, _t=t):
                eq, ineq = self._constraint_values(cons, z)
                out = obj(z)
                if ineq.size:
                    out += _t * float(np.sum(_relaxed_log(ineq, self.delta * _t)))
                if eq.size:
                    out += 0.5 * self.mu_eq * float(np.sum(eq ** 2))
                return out

            x = self._project(inner.minimize(barrier, x).result_.x)
            history.append(obj(x))
            if m * t < self.tol:
                converged = True
                break
            t *= self.t_factor
        result = OptimizeResult(x, obj(x), nit=nit, converged=converged, method=self.method,
                                message="central path converged" if converged else "outer budget exhausted",
                                history=history,
                                extras={"t": t, "duality_gap_bound": m * t})
        return self._finish(result, cons, x)
