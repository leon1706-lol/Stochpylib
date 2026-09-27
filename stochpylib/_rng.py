"""Shared RNG resolution -- the single place every module turns ``random_state=``
into a ``numpy.random.Generator``.

Not part of the public API (``stochpylib.utils.random`` is the public face:
``random_state()``/``set_seed()``/``spawn_generator()`` are thin wrappers around
this module). Kept dependency-free (numpy + stdlib only) so every module,
including ``utils`` itself, can import it with no cycle.
"""

import threading

import numpy as np

_LOCK = threading.Lock()
_STATE = {"root": None}


def set_root(seed):
    """Set (or clear, with ``None``) the library-wide root seed.

    Once set, every ``as_generator(None)`` call draws an independent,
    reproducible child stream from this root instead of fresh OS entropy.
    """
    with _LOCK:
        if seed is None:
            _STATE["root"] = None
        elif isinstance(seed, np.random.SeedSequence):
            _STATE["root"] = seed
        else:
            _STATE["root"] = np.random.SeedSequence(seed)
    return _STATE["root"]


def get_root():
    """The current root ``SeedSequence``, or ``None`` if unset."""
    return _STATE["root"]


def as_generator(random_state=None):
    """Resolve any of the accepted ``random_state=`` inputs to a ``Generator``.

    - ``None``: a root seed set via :func:`set_root` spawns a fresh, reproducible
      child; otherwise a fresh-entropy ``default_rng()``.
    - ``Generator``: returned unchanged (identity preserved, so a shared stream
      keeps advancing across calls).
    - ``RandomState``: wrapped over its own bit generator, so it shares state.
    - anything with ``_as_generator()`` (e.g. ``utils.reproducibility.RandomStream``):
      delegates to it.
    - int / sequence of ints / ``SeedSequence`` / ``BitGenerator``: passed straight
      to ``np.random.default_rng`` (numpy's own validation/errors apply).
    """
    if random_state is None:
        with _LOCK:
            root = _STATE["root"]
            if root is not None:
                child = root.spawn(1)[0]
                return np.random.default_rng(child)
        return np.random.default_rng()
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        return np.random.Generator(random_state._bit_generator)
    as_gen = getattr(random_state, "_as_generator", None)
    if callable(as_gen):
        return as_gen()
    return np.random.default_rng(random_state)


def spawn(random_state, n):
    """``n`` independent, reproducible child ``Generator``s.

    A ``Generator`` input spawns from its own bit generator's ``SeedSequence``
    (``Generator.spawn``); anything else is treated as a ``SeedSequence`` seed
    (``None`` spawns from the global root if one is set, else fresh entropy).
    For int/None input this matches
    ``[default_rng(c) for c in SeedSequence(seed).spawn(n)]`` exactly.
    """
    if isinstance(random_state, np.random.Generator):
        return random_state.spawn(n)
    if random_state is None:
        with _LOCK:
            root = _STATE["root"]
        seq = root if root is not None else np.random.SeedSequence()
    elif isinstance(random_state, np.random.SeedSequence):
        seq = random_state
    else:
        seq = np.random.SeedSequence(random_state)
    return [np.random.default_rng(c) for c in seq.spawn(n)]


def legacy_spawn(rng, n):
    """Pre-V0.20.0 child-stream draw (int64 seeds off ``rng``).

    Kept verbatim (not migrated to :func:`spawn`'s ``SeedSequence`` scheme) so
    multi-chain MCMC samplers and population-based optimizers that already
    shipped on this scheme keep producing the exact same streams.
    """
    seeds = rng.integers(0, np.iinfo(np.int64).max, size=n)
    return [np.random.default_rng(int(s)) for s in seeds]
