"""Second-order and quasi-Newton optimizers: Newton with a modified-Cholesky safeguard,
BFGS and limited-memory BFGS, nonlinear conjugate gradient, a dogleg/Steihaug trust region,
and Levenberg-Marquardt for nonlinear least squares.

All but ``LevenbergMarquardt`` take a scalar objective and match
``scipy.optimize.minimize``'s converged solution (not its iteration path -- line-search
details differ). ``LevenbergMarquardt`` takes a *residual* function and is the counterpart
of ``scipy.optimize.least_squares(method="lm")``.
"""

import numpy as np

from stochpylib.optimization._base import Optimizer
from stochpylib.optimization._common import (
    _backtracking_armijo, _converged, _modified_cholesky_solve, _numeric_jacobian,
    _strong_wolfe,
)
from stochpylib.optimization._result import OptimizeResult

__all__ = ["NewtonMethod", "BFGS", "LBFGS", "ConjugateGradient", "TrustRegion",
           "LevenbergMarquardt"]


class NewtonMethod(Optimizer):
    """Damped Newton's method with a modified-Cholesky safeguard.

    The raw Newton direction ``-H^-1 g`` only descends when the Hessian is positive
    definite; at a saddle or in a region of negative curvature it points uphill. The
    Hessian's diagonal is therefore inflated until the Cholesky factorization succeeds, and
    the resulting direction is accepted through an Armijo line search -- so convergence is
    still quadratic near a strict minimum, but a bad start does not diverge. With no
    analytic ``hess`` the Hessian is a finite difference of the (possibly also
    finite-difference) gradient.
    """

    method = "newton"

    def __init__(self, max_iter=200, xtol=1e-12, ftol=1e-14, gtol=1e-8, line_search=True,
                 track_trajectory=False, random_state=None):
        self.max_iter = max_iter
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.line_search = line_search
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def _minimize(self, obj, x0, rng):
        x = x0
        f = obj(x)
        g = obj.grad(x)
        history, trajectory = [f], ([x.copy()] if self.track_trajectory else [])
        converged, message, nit = False, "max_iter reached", 0
        for t in range(1, int(self.max_iter) + 1):
            nit = t
            if not np.all(np.isfinite(g)):
                message = "non-finite gradient"
                break
            if np.max(np.abs(g)) <= self.gtol:
                converged, message = True, "gradient below gtol"
                break
            d = _modified_cholesky_solve(obj.hess(x), g)
            if self.line_search:
                alpha, x_new, f_new = _backtracking_armijo(obj, x, d, f, g)
                if alpha == 0.0:
                    message = "line search failed"
                    break
            else:
                x_new = x + d
                f_new = obj(x_new)
            g_new = obj.grad(x_new)
            history.append(f_new)
            if trajectory:
                trajectory.append(x_new.copy())
            converged, message = _converged(x_new, x, f_new, f, g_new, self.xtol,
                                            self.ftol, self.gtol)
            x, f, g = x_new, f_new, g_new
            if converged:
                break
        return OptimizeResult(x, f, nit=nit, converged=converged, method=self.method,
                              message=message, jac=g, history=history,
                              trajectory=trajectory)


class BFGS(Optimizer):
    """BFGS quasi-Newton with a strong-Wolfe line search.

    Maintains the *inverse* Hessian approximation directly (Nocedal & Wright eq. 6.17), so
    each step is a matrix-vector product rather than a solve, and exposes it as
    ``result_.hess_inv`` -- at a minimum it is an estimate of the inverse Hessian and
    therefore of an asymptotic covariance matrix, which is what makes this the default
    engine behind the library's MLE fits. Curvature pairs with ``s.y <= 0`` are skipped
    rather than applied, which is what keeps the approximation positive definite.
    """

    method = "bfgs"

    def __init__(self, max_iter=500, xtol=1e-12, ftol=1e-14, gtol=1e-8,
                 track_trajectory=False, random_state=None):
        self.max_iter = max_iter
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def _minimize(self, obj, x0, rng):
        n = len(x0)
        H = np.eye(n)
        x = x0
        f = obj(x)
        g = obj.grad(x)
        history, trajectory = [f], ([x.copy()] if self.track_trajectory else [])
        converged, message, nit = False, "max_iter reached", 0
        identity = np.eye(n)
        for t in range(1, int(self.max_iter) + 1):
            nit = t
            if not np.all(np.isfinite(g)):
                message = "non-finite gradient"
                break
            if np.max(np.abs(g)) <= self.gtol:
                converged, message = True, "gradient below gtol"
                break
            d = -H @ g
            alpha, x_new, f_new, g_new = _strong_wolfe(obj, x, d, f, g)
            if alpha == 0.0:
                message = "line search failed"
                break
            s = x_new - x
            y = g_new - g
            sy = float(np.dot(s, y))
            if sy > 1e-12 * max(1.0, float(np.dot(y, y))):
                rho = 1.0 / sy
                V = identity - rho * np.outer(s, y)
                H = V @ H @ V.T + rho * np.outer(s, s)
            history.append(f_new)
            if trajectory:
                trajectory.append(x_new.copy())
            converged, message = _converged(x_new, x, f_new, f, g_new, self.xtol,
                                            self.ftol, self.gtol)
            x, f, g = x_new, f_new, g_new
            if converged:
                break
        return OptimizeResult(x, f, nit=nit, converged=converged, method=self.method,
                              message=message, jac=g, hess_inv=H, history=history,
                              trajectory=trajectory)


class LBFGS(Optimizer):
    """Limited-memory BFGS: the two-loop recursion over the last ``memory`` curvature pairs.

    Storage and per-iteration cost are ``O(memory * dim)`` instead of BFGS's ``O(dim^2)``,
    so this is the one to reach for above a few hundred parameters. ``result_.hess_inv`` is
    ``None`` -- the whole point is never to form that matrix.
    """

    method = "l-bfgs"

    def __init__(self, memory=10, max_iter=500, xtol=1e-12, ftol=1e-14, gtol=1e-8,
                 track_trajectory=False, random_state=None):
        self.memory = memory
        self.max_iter = max_iter
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    @staticmethod
    def _two_loop(g, pairs):
        q = g.copy()
        alphas = []
        for s, y, rho in reversed(pairs):
            a = rho * float(np.dot(s, q))
            alphas.append(a)
            q = q - a * y
        if pairs:
            s, y, _ = pairs[-1]
            q = q * (float(np.dot(s, y)) / float(np.dot(y, y)))
        for (s, y, rho), a in zip(pairs, reversed(alphas)):
            b = rho * float(np.dot(y, q))
            q = q + (a - b) * s
        return -q

    def _minimize(self, obj, x0, rng):
        x = x0
        f = obj(x)
        g = obj.grad(x)
        pairs = []
        history, trajectory = [f], ([x.copy()] if self.track_trajectory else [])
        converged, message, nit = False, "max_iter reached", 0
        for t in range(1, int(self.max_iter) + 1):
            nit = t
            if not np.all(np.isfinite(g)):
                message = "non-finite gradient"
                break
            if np.max(np.abs(g)) <= self.gtol:
                converged, message = True, "gradient below gtol"
                break
            d = self._two_loop(g, pairs)
            alpha, x_new, f_new, g_new = _strong_wolfe(obj, x, d, f, g)
            if alpha == 0.0:
                message = "line search failed"
                break
            s = x_new - x
            y = g_new - g
            sy = float(np.dot(s, y))
            if sy > 1e-12 * max(1.0, float(np.dot(y, y))):
                pairs.append((s, y, 1.0 / sy))
                if len(pairs) > int(self.memory):
                    pairs.pop(0)
            history.append(f_new)
            if trajectory:
                trajectory.append(x_new.copy())
            converged, message = _converged(x_new, x, f_new, f, g_new, self.xtol,
                                            self.ftol, self.gtol)
            x, f, g = x_new, f_new, g_new
            if converged:
                break
        return OptimizeResult(x, f, nit=nit, converged=converged, method=self.method,
                              message=message, jac=g, history=history,
                              trajectory=trajectory)


class ConjugateGradient(Optimizer):
    """Nonlinear conjugate gradient, Fletcher-Reeves or Polak-Ribiere.

    ``beta="polak-ribiere"`` (the default) uses the PR+ safeguard ``max(beta, 0)``, which
    restarts along steepest descent whenever the update would otherwise lose conjugacy --
    without it PR can cycle. On an exactly quadratic objective with an exact line search the
    method terminates in at most ``dim`` iterations; ``restart_every`` forces a steepest
    descent step periodically to recover that behaviour on nonquadratic problems.
    """

    method = "conjugate gradient"

    def __init__(self, beta="polak-ribiere", restart_every=None, max_iter=500, xtol=1e-12,
                 ftol=1e-14, gtol=1e-8, track_trajectory=False, random_state=None):
        self.beta = beta
        self.restart_every = restart_every
        self.max_iter = max_iter
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def _beta(self, g_new, g, d):
        gg = float(np.dot(g, g))
        if gg <= 0:
            return 0.0
        if self.beta in ("fletcher-reeves", "fr"):
            return float(np.dot(g_new, g_new)) / gg
        if self.beta in ("polak-ribiere", "pr", "pr+"):
            return max(0.0, float(np.dot(g_new, g_new - g)) / gg)
        raise ValueError("beta must be 'fletcher-reeves' or 'polak-ribiere'")

    def _minimize(self, obj, x0, rng):
        x = x0
        f = obj(x)
        g = obj.grad(x)
        d = -g
        restart = int(self.restart_every) if self.restart_every else len(x0)
        history, trajectory = [f], ([x.copy()] if self.track_trajectory else [])
        converged, message, nit = False, "max_iter reached", 0
        for t in range(1, int(self.max_iter) + 1):
            nit = t
            if not np.all(np.isfinite(g)):
                message = "non-finite gradient"
                break
            if np.max(np.abs(g)) <= self.gtol:
                converged, message = True, "gradient below gtol"
                break
            alpha, x_new, f_new, g_new = _strong_wolfe(obj, x, d, f, g, c2=0.1)
            if alpha == 0.0:
                message = "line search failed"
                break
            d = -g_new if t % restart == 0 else -g_new + self._beta(g_new, g, d) * d
            history.append(f_new)
            if trajectory:
                trajectory.append(x_new.copy())
            converged, message = _converged(x_new, x, f_new, f, g_new, self.xtol,
                                            self.ftol, self.gtol)
            x, f, g = x_new, f_new, g_new
            if converged:
                break
        return OptimizeResult(x, f, nit=nit, converged=converged, method=self.method,
                              message=message, jac=g, history=history,
                              trajectory=trajectory)


class TrustRegion(Optimizer):
    """Trust-region Newton with a dogleg or Steihaug-CG subproblem solver.

    Rather than choosing a direction then a step length, each iteration minimizes the
    quadratic model inside a ball of radius ``radius`` and grows or shrinks that ball by the
    agreement ratio ``rho`` between predicted and actual reduction. That makes it robust
    where a line search is not: indefinite Hessians are handled by the subproblem itself.
    ``subproblem="steihaug"`` (the default) is the truncated-CG solver and handles negative
    curvature by running to the trust-region boundary; ``"dogleg"`` requires a positive
    definite Hessian and falls back to Steihaug when it is not.
    """

    method = "trust region"

    def __init__(self, radius=1.0, max_radius=1000.0, eta=0.15, subproblem="steihaug",
                 max_iter=500, xtol=1e-12, ftol=1e-14, gtol=1e-8, track_trajectory=False,
                 random_state=None):
        self.radius = radius
        self.max_radius = max_radius
        self.eta = eta
        self.subproblem = subproblem
        self.max_iter = max_iter
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    @staticmethod
    def _boundary_step(p, d, radius):
        """The positive root of ``||p + tau d|| = radius``."""
        a = float(np.dot(d, d))
        b = 2.0 * float(np.dot(p, d))
        c = float(np.dot(p, p)) - radius ** 2
        disc = max(b * b - 4 * a * c, 0.0)
        tau = (-b + np.sqrt(disc)) / (2 * a) if a > 0 else 0.0
        return p + tau * d

    def _steihaug(self, g, H, radius, tol=1e-10, max_inner=None):
        n = len(g)
        p = np.zeros(n)
        r = g.copy()
        d = -r
        if np.linalg.norm(r) < tol:
            return p
        for _ in range(int(max_inner or 2 * n + 10)):
            dHd = float(d @ H @ d)
            if dHd <= 0:
                return self._boundary_step(p, d, radius)
            alpha = float(np.dot(r, r)) / dHd
            p_next = p + alpha * d
            if np.linalg.norm(p_next) >= radius:
                return self._boundary_step(p, d, radius)
            r_next = r + alpha * (H @ d)
            if np.linalg.norm(r_next) < tol:
                return p_next
            beta = float(np.dot(r_next, r_next)) / float(np.dot(r, r))
            p, r, d = p_next, r_next, -r_next + beta * d
        return p

    def _dogleg(self, g, H, radius):
        try:
            L = np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            return self._steihaug(g, H, radius)
        p_b = -np.linalg.solve(L.T, np.linalg.solve(L, g))
        if np.linalg.norm(p_b) <= radius:
            return p_b
        gHg = float(g @ H @ g)
        if gHg <= 0:
            return -radius * g / np.linalg.norm(g)
        p_u = -(float(np.dot(g, g)) / gHg) * g
        if np.linalg.norm(p_u) >= radius:
            return radius * p_u / np.linalg.norm(p_u)
        return self._boundary_step(p_u, p_b - p_u, radius)

    def _minimize(self, obj, x0, rng):
        x = x0
        f = obj(x)
        g = obj.grad(x)
        radius = float(self.radius)
        history, trajectory = [f], ([x.copy()] if self.track_trajectory else [])
        converged, message, nit = False, "max_iter reached", 0
        for t in range(1, int(self.max_iter) + 1):
            nit = t
            if not np.all(np.isfinite(g)):
                message = "non-finite gradient"
                break
            if np.max(np.abs(g)) <= self.gtol:
                converged, message = True, "gradient below gtol"
                break
            H = obj.hess(x)
            p = (self._dogleg(g, H, radius) if self.subproblem == "dogleg"
                 else self._steihaug(g, H, radius))
            predicted = -(float(np.dot(g, p)) + 0.5 * float(p @ H @ p))
            f_trial = obj(x + p)
            actual = f - f_trial
            rho = actual / predicted if predicted > 0 else -np.inf
            if rho < 0.25:
                radius *= 0.25
            elif rho > 0.75 and abs(np.linalg.norm(p) - radius) < 1e-10:
                radius = min(2.0 * radius, self.max_radius)
            if rho <= self.eta:
                if radius < 1e-14:
                    message = "trust radius collapsed"
                    break
                continue
            x_new, f_new = x + p, f_trial
            g_new = obj.grad(x_new)
            history.append(f_new)
            if trajectory:
                trajectory.append(x_new.copy())
            converged, message = _converged(x_new, x, f_new, f, g_new, self.xtol,
                                            self.ftol, self.gtol)
            x, f, g = x_new, f_new, g_new
            if converged:
                break
        return OptimizeResult(x, f, nit=nit, converged=converged, method=self.method,
                              message=message, jac=g, history=history,
                              trajectory=trajectory,
                              extras={"radius": radius})


class LevenbergMarquardt(Optimizer):
    """Levenberg-Marquardt for nonlinear least squares.

    Takes a **residual** function ``r(x) -> (m,)``, not a scalar objective, and minimizes
    ``0.5 * sum(r^2)``; ``result_.fun`` is that sum-of-squares halved and
    ``result_.extras["residual"]`` the final residual vector. The damping ``lambda``
    interpolates between Gauss-Newton (small ``lambda``, fast near the solution) and
    gradient descent (large ``lambda``, safe far from it), and is adapted by the same
    accept/reject rule as the trust region. Matches
    ``scipy.optimize.least_squares(method="lm")``'s solution; the Jacobian is a
    finite difference of ``r`` unless ``jac`` is given.
    """

    method = "levenberg-marquardt"

    def __init__(self, damping=1e-3, damping_up=10.0, damping_down=0.1, max_iter=200,
                 xtol=1e-12, ftol=1e-14, gtol=1e-10, track_trajectory=False,
                 random_state=None):
        self.damping = damping
        self.damping_up = damping_up
        self.damping_down = damping_down
        self.max_iter = max_iter
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def minimize(self, residual, x0=None, jac=None, args=(), random_state=None):
        """``residual(x) -> (m,)``; ``jac(x) -> (m, dim)`` when the Jacobian is analytic."""
        self._jac_fn = jac
        self._res_fn = residual
        self._args = tuple(args)
        return super().minimize(
            lambda x: 0.5 * float(np.sum(np.asarray(residual(x, *args), dtype=float) ** 2)),
            x0, random_state=random_state)

    def _residual(self, x):
        return np.atleast_1d(np.asarray(self._res_fn(x, *self._args), dtype=float))

    def _jacobian(self, x):
        if self._jac_fn is not None:
            return np.atleast_2d(np.asarray(self._jac_fn(x, *self._args), dtype=float))
        return np.atleast_2d(_numeric_jacobian(self._residual, x))

    def _minimize(self, obj, x0, rng):
        x = x0
        r = self._residual(x)
        f = 0.5 * float(np.dot(r, r))
        lam = float(self.damping)
        history, trajectory = [f], ([x.copy()] if self.track_trajectory else [])
        converged, message, nit = False, "max_iter reached", 0
        J = self._jacobian(x)
        for t in range(1, int(self.max_iter) + 1):
            nit = t
            J = self._jacobian(x)
            g = J.T @ r
            if not np.all(np.isfinite(g)):
                message = "non-finite gradient"
                break
            if np.max(np.abs(g)) <= self.gtol:
                converged, message = True, "gradient below gtol"
                break
            JTJ = J.T @ J
            # Marquardt's scaling: damp each coordinate by its own curvature, so the
            # method is invariant to a rescaling of the parameters.
            scale = np.diag(np.maximum(np.diag(JTJ), 1e-12))
            try:
                step = np.linalg.solve(JTJ + lam * scale, -g)
            except np.linalg.LinAlgError:
                lam *= self.damping_up
                continue
            x_trial = x + step
            r_trial = self._residual(x_trial)
            f_trial = 0.5 * float(np.dot(r_trial, r_trial))
            if f_trial < f:
                converged, message = _converged(x_trial, x, f_trial, f, g, self.xtol,
                                                self.ftol, self.gtol)
                x, r, f = x_trial, r_trial, f_trial
                lam = max(lam * self.damping_down, 1e-14)
                history.append(f)
                if trajectory:
                    trajectory.append(x.copy())
                if converged:
                    break
            else:
                lam *= self.damping_up
                if lam > 1e14:
                    message = "damping diverged"
                    break
        return OptimizeResult(x, f, nit=nit, converged=converged, method=self.method,
                              message=message, jac=J.T @ r, history=history,
                              trajectory=trajectory,
                              extras={"residual": r, "damping": lam, "jacobian": J})
