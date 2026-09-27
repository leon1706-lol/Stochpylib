"""Reproducibility scaffolding: a seeding context manager, checkpointable named
streams, dependency version locking, environment capture, and experiment logging.
"""

import hashlib
import json
import os
import platform
import random as _py_random
import sys
import time
import uuid
from datetime import datetime, timezone

import numpy as np

from stochpylib import _rng
from stochpylib.utils._backends import available_backends

__all__ = ["EnvironmentCapture", "ExperimentLogger", "RandomStream", "Reproducibility",
           "VersionLock"]


class Reproducibility:
    """Context manager: seed the library root for the block's duration and
    restore the prior root (and, if requested, numpy/stdlib global state)
    on exit."""

    def __init__(self, seed, numpy_global=False, python_random=False):
        self.seed = seed
        self.numpy_global = numpy_global
        self.python_random = python_random

    def __enter__(self):
        self._prev_root = _rng.get_root()
        if self.numpy_global:
            self._prev_np_state = np.random.get_state()
        if self.python_random:
            self._prev_py_state = _py_random.getstate()
        from stochpylib.utils.random import set_seed
        set_seed(self.seed, numpy_global=self.numpy_global, python_random=self.python_random)
        return self

    def __exit__(self, *exc):
        with _rng._LOCK:
            _rng._STATE["root"] = self._prev_root  # restore the exact SeedSequence object
        if self.numpy_global:
            np.random.set_state(self._prev_np_state)
        if self.python_random:
            _py_random.setstate(self._prev_py_state)
        return False

    def check(self, fn, n_runs=2):
        """Re-seed before each of ``n_runs`` calls to ``fn()`` and compare
        fingerprints. Returns ``{"reproducible": bool, "fingerprints": [...]}``."""
        prints = []
        for _ in range(n_runs):
            with Reproducibility(self.seed, self.numpy_global, self.python_random):
                prints.append(self.fingerprint(fn()))
        return {"reproducible": len(set(prints)) == 1, "fingerprints": prints}

    @staticmethod
    def fingerprint(obj):
        try:
            arr = np.asarray(obj, dtype=float)
            return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
        except (TypeError, ValueError):
            from stochpylib.utils.io import to_json
            return hashlib.sha256(to_json(obj).encode()).hexdigest()


class RandomStream:
    """A named, checkpointable random stream. ``random_state=stream`` works
    anywhere in the library via ``_as_generator()``."""

    def __init__(self, seed=None, name="stream"):
        self.name = name
        self._seed_seq = seed if isinstance(seed, np.random.SeedSequence) \
            else np.random.SeedSequence(seed)
        self._bit_gen = np.random.PCG64(self._seed_seq)
        self.generator = np.random.Generator(self._bit_gen)

    def _as_generator(self):
        return self.generator

    def spawn(self, n):
        return [RandomStream(seed=s, name=f"{self.name}/{i}")
                for i, s in enumerate(self._seed_seq.spawn(n))]

    def checkpoint(self):
        return json.loads(json.dumps(self._bit_gen.state, default=lambda o: o.tolist()
                                     if hasattr(o, "tolist") else o))

    def restore(self, state):
        self._bit_gen.state = state
        return self

    def reset(self):
        self._bit_gen = np.random.PCG64(self._seed_seq)
        self.generator = np.random.Generator(self._bit_gen)
        return self

    def jumped(self, k=1):
        jumped_bg = self._bit_gen.jumped(k)
        new = RandomStream.__new__(RandomStream)
        new.name = f"{self.name}+jump{k}"
        new._seed_seq = self._seed_seq
        new._bit_gen = jumped_bg
        new.generator = np.random.Generator(jumped_bg)
        return new

    def __repr__(self):
        return f"RandomStream(name={self.name!r})"


class VersionLock:
    """Capture/compare installed package versions against a locked snapshot."""

    def __init__(self, packages=("numpy", "scipy", "stochpylib"), strict=False):
        self.packages = tuple(packages)
        self.strict = strict
        self.versions_ = {}

    def capture(self):
        import importlib.metadata as md
        for pkg in self.packages:
            try:
                self.versions_[pkg] = md.version(pkg)
            except md.PackageNotFoundError:
                self.versions_[pkg] = None
        return self

    def check(self):
        import importlib.metadata as md
        mismatches = []
        for pkg, locked in self.versions_.items():
            try:
                installed = md.version(pkg)
            except md.PackageNotFoundError:
                installed = None
            if installed != locked:
                mismatches.append({"package": pkg, "locked": locked, "installed": installed})
        if self.strict and mismatches:
            raise RuntimeError(f"VersionLock: version mismatch: {mismatches}")
        return mismatches

    def to_dict(self):
        return dict(self.versions_)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            versions = json.load(f)
        vl = cls(packages=tuple(versions))
        vl.versions_ = versions
        return vl

    def to_requirements(self):
        return "\n".join(f"{pkg}=={ver}" for pkg, ver in self.versions_.items() if ver)


class EnvironmentCapture:
    """Snapshot of the running Python/numpy/scipy/stochpylib environment."""

    def __init__(self, include_env=True):
        self.include_env = include_env
        self.info_ = {}

    def capture(self):
        import scipy
        import stochpylib

        info = {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": sys.executable,
            "platform_system": platform.system(),
            "platform_release": platform.release(),
            "platform_machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "stochpylib_version": stochpylib.__version__,
            "available_backends": available_backends(),
            "root_seed_set": _rng.get_root() is not None,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            info["blas_info"] = np.show_config(mode="dicts")
        except Exception:
            info["blas_info"] = None
        if self.include_env:
            info["thread_env"] = {
                k: os.environ.get(k) for k in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "PYTHONHASHSEED")
            }
        self.info_ = info
        return self

    def to_dict(self):
        return dict(self.info_)

    def to_json(self, path=None, indent=2):
        from stochpylib.utils.io import to_json
        return to_json(self.info_, path=path, indent=indent)

    def diff(self, other):
        a, b = self.info_, other.info_
        keys = set(a) | set(b)
        return {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}

    def __repr__(self):
        return f"EnvironmentCapture(n_fields={len(self.info_)})"


def _jsonify(v):
    from stochpylib.montecarlo import MCResult

    if isinstance(v, MCResult):
        return {"estimate": float(v.estimate), "std_error": float(v.std_error)}
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {k: _jsonify(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    return v


class ExperimentLogger:
    """Structured experiment/run logging with one environment snapshot per
    logger instance; runs append as JSON Lines when ``path`` is given."""

    def __init__(self, path=None, name="experiment"):
        self.path = path
        self.name = name
        self.runs_ = []
        self._env = EnvironmentCapture().capture().to_dict()

    class _Run:
        def __init__(self, logger, params, seed):
            self.logger = logger
            self.params = params
            self.seed = seed
            self.metrics = {}

        def __enter__(self):
            self._t0 = time.perf_counter()
            self.run_id = uuid.uuid4().hex[:12]
            return self

        def log(self, **metrics):
            self.metrics.update(metrics)

        def __exit__(self, *exc):
            record = {
                "run_id": self.run_id, "name": self.logger.name,
                "params": _jsonify(self.params), "seed": self.seed,
                "metrics": _jsonify(self.metrics),
                "start": self._t0, "duration_s": time.perf_counter() - self._t0,
            }
            self.logger.runs_.append(record)
            if self.logger.path:
                with open(self.logger.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            return False

    def run(self, params=None, seed=None):
        return ExperimentLogger._Run(self, params or {}, seed)

    @classmethod
    def load(cls, path):
        logger = cls(path=path)
        logger.runs_ = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    logger.runs_.append(json.loads(line))
        return logger

    def to_table(self):
        return list(self.runs_)

    def best(self, metric, mode="min"):
        candidates = [r for r in self.runs_ if metric in r.get("metrics", {})]
        if not candidates:
            return None
        key = (lambda r: r["metrics"][metric]) if mode == "min" else \
            (lambda r: -r["metrics"][metric])
        return min(candidates, key=key)

    def __repr__(self):
        return f"ExperimentLogger(name={self.name!r}, n_runs={len(self.runs_)})"
