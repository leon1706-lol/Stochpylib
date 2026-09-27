"""Seeding and reproducible streams: the public face of ``stochpylib._rng``.

Every stochastic method in the library already takes ``random_state=`` and resolves
it through ``_rng.as_generator``; this submodule adds the ability to set a
library-wide root seed (``set_seed``) so ``random_state=None`` calls become
reproducible too, plus explicit stream construction (``random_state``,
``spawn_generator``).
"""

import random as _py_random

import numpy as np

from stochpylib import _rng

__all__ = ["Generator", "SeedSequence", "random_state", "set_seed", "spawn_generator"]

#: identity re-exports -- these ARE numpy's own classes, not stochpylib wrappers.
Generator = np.random.Generator
SeedSequence = np.random.SeedSequence


def set_seed(seed=None, *, numpy_global=False, python_random=False):
    """Set (``seed`` given) or clear (``seed=None``) the library-wide root seed.

    After ``set_seed(s)``, every stochpylib call made with ``random_state=None``
    draws an independent, reproducible child stream from the same root -- the
    library never relies on numpy's or Python's *global* random state by
    default. Pass ``numpy_global=True``/``python_random=True`` to additionally
    seed ``numpy.random.seed``/the stdlib ``random`` module, for code outside
    stochpylib that does rely on global state.

    Returns the root ``SeedSequence`` (or ``None`` if clearing).
    """
    root = _rng.set_root(seed)
    if seed is not None:
        if numpy_global:
            np.random.seed(int(np.random.default_rng(seed).integers(0, 2**31 - 1)))
        if python_random:
            _py_random.seed(seed)
    return root


def random_state(seed=None):
    """Resolve ``seed`` to a ``numpy.random.Generator`` the same way every
    stochpylib ``random_state=`` argument is resolved internally."""
    return _rng.as_generator(seed)


def spawn_generator(random_state=None, n=None):
    """One independent child ``Generator`` (``n=None``), or a list of ``n``.

    Streams are drawn via ``SeedSequence.spawn``, so they are statistically
    independent of each other and of the parent stream.
    """
    if n is None:
        return _rng.spawn(random_state, 1)[0]
    return _rng.spawn(random_state, n)
