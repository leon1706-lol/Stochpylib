"""First-order optimizers: plain and stochastic gradient descent plus the adaptive
step-size family (AdaGrad, RMSProp, Adadelta, Adam, NADAM, AMSGrad).

Every class subclasses :class:`_GradientOptimizer`, which owns the descent loop and
delegates the per-iteration parameter update to a ``_update`` hook, so the eight differ
only in the few lines that define them. ``GradientDescent`` uses a line search and is
deterministic; the other seven take a fixed ``learning_rate`` and match the published
update rules (Duchi 2011, Tieleman & Hinton 2012, Zeiler 2012, Kingma & Ba 2015,
Dozat 2016, Reddi 2018) term for term, including bias correction where the paper has it.
"""

import numpy as np

from stochpylib.optimization._base import Optimizer
from stochpylib.optimization._common import (
    _backtracking_armijo, _clip_to_bounds, _converged, _normalize_bounds,
)
from stochpylib.optimization._result import OptimizeResult

__all__ = ["GradientDescent", "StochasticGD", "AdamOptimizer", "AdaGrad", "RMSProp",
           "Adadelta", "NADAM", "AMSGrad"]


class _GradientOptimizer(Optimizer):
    """Shared descent loop for the first-order family.

    ``_update(obj, x, g, state, t)`` returns the new iterate; ``_init_state(dim)`` returns
    the per-algorithm accumulators. Convergence is the shared gradient/step/objective test
    from ``_common._converged``, so all eight stop on the same terms.
    """

    method = "gradient"

    def _init_state(self, dim):
        return {}

    def _update(self, obj, x, g, state, t):
        raise NotImplementedError

    def _gradient_at(self, obj, x, rng):
        return obj.grad(x)

    def _minimize(self, obj, x0, rng):
        bounds = _normalize_bounds(getattr(self, "bounds", None), len(x0))
        x = _clip_to_bounds(x0, bounds)
        state = self._init_state(len(x))
        f = obj(x)
        history = [f]
        trajectory = [x.copy()] if getattr(self, "track_trajectory", False) else []
        converged, message, nit = False, "max_iter reached", 0
        g = obj.grad(x)
        for t in range(1, int(self.max_iter) + 1):
            nit = t
            g = self._gradient_at(obj, x, rng)
            if not np.all(np.isfinite(g)):
                message = "non-finite gradient"
                break
            x_new = _clip_to_bounds(self._update(obj, x, g, state, t), bounds)
            f_new = obj(x_new)
            if not np.isfinite(f_new):  # fixed-step methods do diverge; stop at the last good point
                message = "objective diverged"
                break
            history.append(f_new)
            if trajectory:
                trajectory.append(x_new.copy())
            converged, message = _converged(x_new, x, f_new, f, g, self.xtol, self.ftol,
                                            self.gtol)
            x, f = x_new, f_new
            if converged:
                break
        return OptimizeResult(x, f, nit=nit, converged=converged, method=self.method,
                              message=message, jac=g, history=history,
                              trajectory=trajectory)


class GradientDescent(_GradientOptimizer):
    """Steepest descent with an Armijo backtracking line search.

    The only member of this family that adapts its step length to the objective rather than
    to the gradient history, and therefore the only deterministic one: with
    ``line_search=True`` (the default) ``learning_rate`` is merely the initial trial step.
    Setting ``line_search=False`` gives textbook fixed-step descent, which diverges on
    badly scaled problems -- that is the point of the default.
    """

    method = "gradient descent"

    def __init__(self, learning_rate=1.0, line_search=True, max_iter=1000, xtol=1e-10,
                 ftol=1e-12, gtol=1e-8, bounds=None, track_trajectory=False,
                 random_state=None):
        self.learning_rate = learning_rate
        self.line_search = line_search
        self.max_iter = max_iter
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.bounds = bounds
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def _update(self, obj, x, g, state, t):
        if not self.line_search:
            return x - self.learning_rate * g
        bounds = _normalize_bounds(self.bounds, len(x))
        alpha, x_new, _ = _backtracking_armijo(obj, x, -g, obj(x), g,
                                               alpha0=self.learning_rate, bounds=bounds)
        return x_new if alpha > 0 else x


class StochasticGD(_GradientOptimizer):
    """Mini-batch stochastic gradient descent with an optional decaying step size.

    ``minimize`` is given the full-sample objective; the stochastic part comes from
    ``grad_sample(x, idx)``, a callable returning the gradient of the mean loss over the
    given observation indices, which are drawn without replacement from
    ``range(n_samples)`` each iteration. With ``grad_sample=None`` the full gradient is
    used and this reduces to heavy-ball descent on the same schedule -- the deterministic
    reference the stochastic runs are compared against. Step size is
    ``learning_rate / (1 + decay * t)``; ``nesterov=True`` applies the look-ahead form of
    the momentum term.
    """

    method = "stochastic gradient descent"

    def __init__(self, learning_rate=0.01, batch_size=32, n_samples=None,
                 grad_sample=None, decay=0.0, momentum=0.0, nesterov=False, max_iter=1000,
                 xtol=0.0, ftol=0.0, gtol=1e-8, bounds=None, track_trajectory=False,
                 random_state=None):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.grad_sample = grad_sample
        self.decay = decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.max_iter = max_iter
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.bounds = bounds
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def _init_state(self, dim):
        return {"v": np.zeros(dim)}

    def _gradient_at(self, obj, x, rng):
        if self.grad_sample is None or self.n_samples is None:
            return obj.grad(x)
        n = int(self.n_samples)
        idx = rng.choice(n, size=min(int(self.batch_size), n), replace=False)
        return np.atleast_1d(np.asarray(self.grad_sample(x, idx), dtype=float))

    def _update(self, obj, x, g, state, t):
        lr = self.learning_rate / (1.0 + self.decay * t)
        v = self.momentum * state["v"] - lr * g
        state["v"] = v
        if self.nesterov and self.momentum > 0:
            return x + self.momentum * v - lr * g
        return x + v


class AdaGrad(_GradientOptimizer):
    """AdaGrad (Duchi, Hazan & Singer 2011): per-coordinate step ``lr / sqrt(sum of squared
    gradients)``. The accumulator never decays, so the effective step is monotonically
    non-increasing -- strong on sparse problems, prone to stalling on long runs.
    """

    method = "adagrad"

    def __init__(self, learning_rate=0.5, epsilon=1e-8, max_iter=2000, xtol=0.0,
                 ftol=1e-12, gtol=1e-8, bounds=None, track_trajectory=False,
                 random_state=None):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.bounds = bounds
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def _init_state(self, dim):
        return {"s": np.zeros(dim)}

    def _update(self, obj, x, g, state, t):
        state["s"] += g * g
        return x - self.learning_rate * g / (np.sqrt(state["s"]) + self.epsilon)


class RMSProp(_GradientOptimizer):
    """RMSProp (Tieleman & Hinton 2012): AdaGrad with an exponentially decaying second-moment
    accumulator, so the effective step can recover instead of decaying to zero. ``momentum``
    adds the optional heavy-ball term from the same lecture notes.
    """

    method = "rmsprop"

    def __init__(self, learning_rate=0.01, rho=0.9, momentum=0.0, epsilon=1e-8,
                 max_iter=1000, xtol=0.0, ftol=1e-12, gtol=1e-8, bounds=None,
                 track_trajectory=False, random_state=None):
        self.learning_rate = learning_rate
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.bounds = bounds
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def _init_state(self, dim):
        return {"s": np.zeros(dim), "v": np.zeros(dim)}

    def _update(self, obj, x, g, state, t):
        state["s"] = self.rho * state["s"] + (1.0 - self.rho) * g * g
        step = self.learning_rate * g / (np.sqrt(state["s"]) + self.epsilon)
        state["v"] = self.momentum * state["v"] + step
        return x - state["v"]


class Adadelta(_GradientOptimizer):
    """Adadelta (Zeiler 2012): RMSProp with the step size itself estimated from the running
    RMS of past updates, so the method is unit-consistent and has no learning rate in the
    usual sense (``learning_rate`` is a plain multiplier, 1.0 in the paper).

    The update-RMS accumulator starts at zero, so the first steps are of order
    ``sqrt(epsilon)`` and the method needs far more iterations than Adam on a
    well-conditioned problem. That is the algorithm, not a defect -- raise
    ``learning_rate`` if you want the scale-free property without the slow start.
    """

    method = "adadelta"

    def __init__(self, learning_rate=1.0, rho=0.95, epsilon=1e-6, max_iter=5000, xtol=0.0,
                 ftol=1e-12, gtol=1e-8, bounds=None, track_trajectory=False,
                 random_state=None):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.bounds = bounds
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def _init_state(self, dim):
        return {"s": np.zeros(dim), "du": np.zeros(dim)}

    def _update(self, obj, x, g, state, t):
        state["s"] = self.rho * state["s"] + (1.0 - self.rho) * g * g
        step = np.sqrt(state["du"] + self.epsilon) / np.sqrt(state["s"] + self.epsilon) * g
        state["du"] = self.rho * state["du"] + (1.0 - self.rho) * step * step
        return x - self.learning_rate * step


class AdamOptimizer(_GradientOptimizer):
    """Adam (Kingma & Ba 2015): bias-corrected first and second moment estimates.

    The bias correction ``m_hat = m / (1 - beta1^t)`` matters most in the first few dozen
    iterations, where the zero-initialized moments would otherwise make the step far too
    small; it is included exactly as in the paper, not folded into the learning rate.
    ``amsgrad=True`` switches to the :class:`AMSGrad` max-of-second-moments variant.
    """

    method = "adam"

    def __init__(self, learning_rate=0.05, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 amsgrad=False, max_iter=2000, xtol=0.0, ftol=1e-12, gtol=1e-8,
                 bounds=None, track_trajectory=False, random_state=None):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.max_iter = max_iter
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.bounds = bounds
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def _init_state(self, dim):
        return {"m": np.zeros(dim), "v": np.zeros(dim), "vmax": np.zeros(dim)}

    def _update(self, obj, x, g, state, t):
        state["m"] = self.beta1 * state["m"] + (1.0 - self.beta1) * g
        state["v"] = self.beta2 * state["v"] + (1.0 - self.beta2) * g * g
        m_hat = state["m"] / (1.0 - self.beta1 ** t)
        if self.amsgrad:
            state["vmax"] = np.maximum(state["vmax"], state["v"])
            denom = np.sqrt(state["vmax"] / (1.0 - self.beta2 ** t)) + self.epsilon
        else:
            denom = np.sqrt(state["v"] / (1.0 - self.beta2 ** t)) + self.epsilon
        return x - self.learning_rate * m_hat / denom


class NADAM(_GradientOptimizer):
    """Nadam (Dozat 2016): Adam with Nesterov momentum applied to the bias-corrected first
    moment, using the paper's schedule-decayed product of ``mu_t`` factors rather than a
    plain ``beta1^t``.
    """

    method = "nadam"

    def __init__(self, learning_rate=0.05, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 schedule_decay=0.004, max_iter=2000, xtol=0.0, ftol=1e-12, gtol=1e-8,
                 bounds=None, track_trajectory=False, random_state=None):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.schedule_decay = schedule_decay
        self.max_iter = max_iter
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.bounds = bounds
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def _init_state(self, dim):
        return {"m": np.zeros(dim), "v": np.zeros(dim), "mu_prod": 1.0}

    def _update(self, obj, x, g, state, t):
        mu_t = self.beta1 * (1.0 - 0.5 * 0.96 ** (t * self.schedule_decay))
        mu_next = self.beta1 * (1.0 - 0.5 * 0.96 ** ((t + 1) * self.schedule_decay))
        mu_prod = state["mu_prod"] * mu_t
        state["mu_prod"] = mu_prod
        state["m"] = self.beta1 * state["m"] + (1.0 - self.beta1) * g
        state["v"] = self.beta2 * state["v"] + (1.0 - self.beta2) * g * g
        g_hat = g / (1.0 - mu_prod)
        m_hat = state["m"] / (1.0 - mu_prod * mu_next)
        m_bar = (1.0 - mu_t) * g_hat + mu_next * m_hat
        v_hat = state["v"] / (1.0 - self.beta2 ** t)
        return x - self.learning_rate * m_bar / (np.sqrt(v_hat) + self.epsilon)


class AMSGrad(_GradientOptimizer):
    """AMSGrad (Reddi, Kale & Kumar 2018): Adam with a non-decreasing second-moment
    denominator ``max(v_1..v_t)``, which restores the convergence guarantee Adam loses when
    a rare large gradient is forgotten too quickly. Equivalent to
    ``AdamOptimizer(amsgrad=True)`` and kept separate because the spec names both.
    """

    method = "amsgrad"

    def __init__(self, learning_rate=0.05, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 max_iter=2000, xtol=0.0, ftol=1e-12, gtol=1e-8, bounds=None,
                 track_trajectory=False, random_state=None):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.bounds = bounds
        self.track_trajectory = track_trajectory
        self.random_state = random_state

    def _init_state(self, dim):
        return {"m": np.zeros(dim), "v": np.zeros(dim), "vmax": np.zeros(dim)}

    def _update(self, obj, x, g, state, t):
        state["m"] = self.beta1 * state["m"] + (1.0 - self.beta1) * g
        state["v"] = self.beta2 * state["v"] + (1.0 - self.beta2) * g * g
        state["vmax"] = np.maximum(state["vmax"], state["v"])
        m_hat = state["m"] / (1.0 - self.beta1 ** t)
        denom = np.sqrt(state["vmax"] / (1.0 - self.beta2 ** t)) + self.epsilon
        return x - self.learning_rate * m_hat / denom
