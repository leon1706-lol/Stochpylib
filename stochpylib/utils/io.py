"""JSON-safe serialization, pickling, npz archives, nested configuration, logging,
and a universal ``summary()`` dispatcher.

``to_dict``/``from_dict`` recognize stochpylib's own object families (distributions,
GP kernels, the shared result dataclasses, ``Generator``/``SeedSequence``) plus a
user-extensible registry (``Serialization.register``). ``from_dict`` only ever
resolves a ``__type__`` tag inside the ``stochpylib``/``numpy`` namespaces.
"""

import dataclasses
import importlib
import json
import logging
import pickle
from contextlib import contextmanager

import numpy as np

__all__ = ["Configuration", "Logging", "Serialization", "from_dict", "from_json",
           "from_pickle", "summary", "to_dict", "to_json", "to_pickle"]

logging.getLogger("stochpylib").addHandler(logging.NullHandler())

_ALLOWED_TOP_MODULES = ("stochpylib", "numpy")
_CUSTOM_REGISTRY = {}


def _top_module(obj_type):
    return ".".join(obj_type.__module__.split(".")[:2])


def _qualname(obj_type):
    return f"{_top_module(obj_type)}.{obj_type.__name__}"


def to_dict(obj):
    """Recursively convert ``obj`` into a JSON-safe, type-tagged ``dict``/list/
    scalar tree. Raises ``TypeError`` for callables and other unsupported types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, complex):
        return {"__complex__": [obj.real, obj.imag]}
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype), "shape": list(obj.shape)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, set):
        return {"__set__": [to_dict(v) for v in obj]}
    if isinstance(obj, dict):
        return {str(k): to_dict(v) for k, v in obj.items()}

    for cls, (to_fn, _from_fn) in _CUSTOM_REGISTRY.items():
        if isinstance(obj, cls):
            return {"__type__": _qualname(cls), "kind": "custom", "data": to_fn(obj)}

    if isinstance(obj, np.random.Generator):
        bg = obj.bit_generator
        return {"__type__": "numpy.random.Generator", "kind": "generator",
                "bit_generator": type(bg).__name__, "state": to_dict(bg.state)}
    if isinstance(obj, np.random.SeedSequence):
        return {"__type__": "numpy.random.SeedSequence", "kind": "seed_sequence",
                "entropy": obj.entropy if isinstance(obj.entropy, int) else list(obj.entropy),
                "spawn_key": list(obj.spawn_key), "pool_size": obj.pool_size}

    from stochpylib.queueing import QueueResult
    if isinstance(obj, QueueResult):
        return {"__type__": _qualname(type(obj)), "kind": "queue_result", "fields": obj.to_dict()}

    from stochpylib.distributions._base import Distribution
    if isinstance(obj, Distribution):
        import inspect
        names = [p for p in inspect.signature(type(obj).__init__).parameters if p != "self"]
        params = {name: to_dict(np.asarray(getattr(obj, name)) if
                                not np.isscalar(getattr(obj, name)) else getattr(obj, name))
                 for name in names}
        return {"__type__": _qualname(type(obj)), "kind": "distribution", "params": params}

    from stochpylib.gaussian_processes.kernels._base import BaseKernel
    if isinstance(obj, BaseKernel):
        return {"__type__": _qualname(type(obj)), "kind": "kernel",
                "params": to_dict(obj.get_params())}

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        return {"__type__": _qualname(type(obj)), "kind": "dataclass", "fields": fields}

    if type(obj).__module__.startswith("stochpylib"):
        return {"__type__": _qualname(type(obj)), "kind": "object", "fields": to_dict(obj.__dict__)}

    raise TypeError(f"utils.to_dict: unsupported type {type(obj)!r}")


def _resolve_type(qualname):
    top_module = qualname.rsplit(".", 1)[0]
    top = top_module.split(".")[0]
    if top not in _ALLOWED_TOP_MODULES:
        raise ValueError(f"utils.from_dict: type {qualname!r} is outside the allowed "
                         f"{_ALLOWED_TOP_MODULES} namespaces")
    name = qualname.rsplit(".", 1)[1]
    mod = importlib.import_module(top_module)
    return getattr(mod, name)


def from_dict(d):
    """Reconstruct an object from :func:`to_dict`'s tagged tree."""
    if isinstance(d, dict):
        if "__ndarray__" in d:
            return np.array(d["__ndarray__"], dtype=d["dtype"]).reshape(d["shape"])
        if "__complex__" in d:
            re, im = d["__complex__"]
            return complex(re, im)
        if "__set__" in d:
            return {from_dict(v) for v in d["__set__"]}
        if "__type__" in d:
            qualname = d["__type__"]
            kind = d.get("kind")
            if qualname == "numpy.random.Generator":
                bg_cls = getattr(np.random, d["bit_generator"])
                bg = bg_cls()
                bg.state = from_dict(d["state"])
                return np.random.Generator(bg)
            if qualname == "numpy.random.SeedSequence":
                return np.random.SeedSequence(d["entropy"], spawn_key=tuple(d["spawn_key"]),
                                              pool_size=d["pool_size"])
            if kind == "custom":
                # A custom type is whatever the caller registered (opt-in), so it is looked up
                # directly, not through the module-allowlist import in _resolve_type.
                for reg_cls, (_to_fn, from_fn) in _CUSTOM_REGISTRY.items():
                    if _qualname(reg_cls) == qualname:
                        return from_fn(d["data"])
                raise ValueError(f"utils.from_dict: no Serialization.register(...) "
                                 f"entry for {qualname!r}")
            cls = _resolve_type(qualname)
            if kind == "queue_result":
                fields = dict(d["fields"])
                extras = {k: v for k, v in fields.items()
                         if k not in ("L", "Lq", "W", "Wq", "rho")}
                return cls(fields["L"], fields["Lq"], fields["W"], fields["Wq"],
                          fields["rho"], **extras)
            if kind == "distribution":
                params = {k: from_dict(v) for k, v in d["params"].items()}
                return cls(**params)
            if kind == "kernel":
                return cls(**from_dict(d["params"]))
            if kind == "dataclass":
                fields = {k: from_dict(v) for k, v in d["fields"].items()}
                return cls(**fields)
            if kind == "object":
                obj = cls.__new__(cls)
                obj.__dict__.update(from_dict(d["fields"]))
                return obj
            raise ValueError(f"utils.from_dict: unknown kind {kind!r} for {qualname!r}")
        return {k: from_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [from_dict(v) for v in d]
    return d


def to_json(obj, path=None, indent=2):
    text = json.dumps(to_dict(obj), indent=indent)
    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def from_json(text_or_path):
    try:
        return from_dict(json.loads(text_or_path))
    except (json.JSONDecodeError, ValueError):
        with open(text_or_path, encoding="utf-8") as f:
            return from_dict(json.load(f))


def to_pickle(obj, path, protocol=pickle.HIGHEST_PROTOCOL):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)
    return path


def from_pickle(path):
    """Load a pickle written by :func:`to_pickle`. Pickle can execute arbitrary
    code on load -- only unpickle files you trust."""
    with open(path, "rb") as f:
        return pickle.load(f)


class Serialization:
    """Format-dispatching (de)serialization: ``json``/``pickle``/``npz``, inferred
    from the path extension when ``format=None``."""

    def __init__(self, format=None):
        self.format = format

    def _resolve(self, path):
        if self.format is not None:
            return self.format
        ext = str(path).rsplit(".", 1)[-1].lower()
        return {"json": "json", "pkl": "pickle", "pickle": "pickle", "npz": "npz"}.get(ext, "json")

    def dumps(self, obj):
        fmt = self.format or "json"
        if fmt == "json":
            return to_json(obj)
        if fmt == "pickle":
            return pickle.dumps(obj)
        raise ValueError(f"dumps() supports 'json'/'pickle', not {fmt!r}")

    def loads(self, data):
        fmt = self.format or ("pickle" if isinstance(data, bytes) else "json")
        if fmt == "json":
            return from_json(data)
        if fmt == "pickle":
            return pickle.loads(data)
        raise ValueError(f"loads() supports 'json'/'pickle', not {fmt!r}")

    def save(self, obj, path):
        fmt = self._resolve(path)
        if fmt == "json":
            to_json(obj, path=path)
        elif fmt == "pickle":
            to_pickle(obj, path)
        elif fmt == "npz":
            d = to_dict(obj)
            np.savez(path, __meta__=json.dumps(d))
        else:
            raise ValueError(f"unknown format {fmt!r}")
        return path

    def load(self, path):
        fmt = self._resolve(path)
        if fmt == "json":
            return from_json(path)
        if fmt == "pickle":
            return from_pickle(path)
        if fmt == "npz":
            with np.load(path, allow_pickle=True) as z:
                return from_dict(json.loads(str(z["__meta__"])))
        raise ValueError(f"unknown format {fmt!r}")

    @classmethod
    def register(cls, target_cls, to_fn, from_fn):
        """Register a custom (de)serializer for ``target_cls``."""
        _CUSTOM_REGISTRY[target_cls] = (to_fn, from_fn)


class Configuration:
    """Nested config with dot-path access (``cfg.get("a.b")``, ``cfg["a.b"]``,
    ``cfg.a.b``), env overrides, validation and freezing."""

    def __init__(self, defaults=None, **kw):
        object.__setattr__(self, "_data", {})
        object.__setattr__(self, "_frozen", False)
        if defaults:
            self.update(defaults)
        if kw:
            self.update(kw)

    def _split(self, key):
        return key.split(".")

    def get(self, key, default=None):
        node = self._data
        for part in self._split(key):
            if isinstance(node, Configuration):
                node = node._data
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, key, value):
        if self._frozen:
            raise TypeError("Configuration is frozen")
        parts = self._split(key)
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def update(self, d):
        if self._frozen:
            raise TypeError("Configuration is frozen")
        d = d._data if isinstance(d, Configuration) else d
        self._deep_merge(self._data, d)

    @staticmethod
    def _deep_merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                Configuration._deep_merge(dst[k], v)
            else:
                dst[k] = v

    def __getitem__(self, key):
        node = self.get(key, _MISSING)
        if node is _MISSING:
            raise KeyError(key)
        return Configuration(node) if isinstance(node, dict) else node

    def __setitem__(self, key, value):
        self.set(key, value)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        data = object.__getattribute__(self, "_data")
        if name not in data:
            raise AttributeError(name)
        v = data[name]
        return Configuration(v) if isinstance(v, dict) else v

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self.set(name, value)

    def to_dict(self):
        return json.loads(json.dumps(self._data))

    def to_json(self, path=None, indent=2):
        text = json.dumps(self._data, indent=indent)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def from_env(self, prefix="STOCHPYLIB_"):
        import os
        for env_key, env_val in os.environ.items():
            if not env_key.startswith(prefix):
                continue
            dotted = env_key[len(prefix):].lower().replace("__", ".")
            try:
                value = json.loads(env_val)
            except json.JSONDecodeError:
                value = env_val
            self.set(dotted, value)
        return self

    def validate(self, schema):
        errors = []
        for key, spec in schema.items():
            value = self.get(key, _MISSING)
            if value is _MISSING:
                errors.append(f"{key}: missing")
                continue
            typ, pred = (spec, None) if not isinstance(spec, tuple) else spec
            if typ is not None and not isinstance(value, typ):
                errors.append(f"{key}: expected {typ}, got {type(value)}")
            elif pred is not None and not pred(value):
                errors.append(f"{key}: failed validation predicate")
        return errors

    def freeze(self):
        object.__setattr__(self, "_frozen", True)
        return self

    def __repr__(self):
        return f"Configuration({self._data!r})"

    def __eq__(self, other):
        return isinstance(other, Configuration) and self._data == other._data


_MISSING = object()


class Logging:
    """Facade over the stdlib ``"stochpylib"`` logger."""

    _configured = False

    @staticmethod
    def get_logger(name="stochpylib"):
        return logging.getLogger(name)

    @classmethod
    def configure(cls, level="INFO", fmt=None, stream=None, file=None):
        logger = logging.getLogger("stochpylib")
        for h in list(logger.handlers):
            if getattr(h, "_stochpylib_managed", False):
                logger.removeHandler(h)
        fmt = fmt or "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        handler = logging.StreamHandler(stream) if file is None else logging.FileHandler(file)
        handler.setFormatter(logging.Formatter(fmt))
        handler._stochpylib_managed = True
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        cls._configured = True
        return logger

    @staticmethod
    def set_level(level):
        logging.getLogger("stochpylib").setLevel(level)

    @staticmethod
    def disable():
        logging.getLogger("stochpylib").setLevel(logging.CRITICAL + 1)

    @staticmethod
    @contextmanager
    def capture(level="DEBUG"):
        logger = logging.getLogger("stochpylib")
        records = []

        class _ListHandler(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _ListHandler()
        prev_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(level)
        try:
            yield records
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev_level)


def summary(obj, file=None):
    """Universal summary dispatcher: uses ``obj.summary()`` when present,
    otherwise formats known shapes (dict/list-of-dict/MCResult/TestResult/
    ndarray/Distribution/FitResult) into a readable table."""
    text = _summary_text(obj)
    if file is not None:
        print(text, file=file)
    return text


def _summary_text(obj):
    if hasattr(obj, "summary") and callable(obj.summary):
        result = obj.summary()
        if isinstance(result, str):
            return result

    from stochpylib.montecarlo import MCResult
    if isinstance(obj, MCResult):
        lo, hi = obj.confidence_interval()
        return f"{obj.estimate:.6g} +/- {obj.std_error:.4g}  [95% CI: {lo:.6g}, {hi:.6g}]"

    from stochpylib.statistics import TestResult
    if isinstance(obj, TestResult):
        return repr(obj)

    from stochpylib.utils._results import FitResult
    if isinstance(obj, FitResult):
        return obj.summary()

    from stochpylib.distributions._base import Distribution
    if isinstance(obj, Distribution):
        try:
            return (f"{type(obj).__name__}: mean={obj.mean():.6g}, var={obj.var():.6g}, "
                   f"skewness={obj.skewness():.4g}, kurtosis={obj.kurtosis():.4g}")
        except Exception:
            return repr(obj)

    if isinstance(obj, np.ndarray):
        from stochpylib.statistics import describe
        return summary(describe(obj))

    if isinstance(obj, dict):
        lines = [f"{k}: {v}" for k, v in obj.items()]
        return "\n".join(lines)

    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        keys = list(obj[0].keys())
        header = "  ".join(f"{k:>12}" for k in keys)
        rows = ["  ".join(f"{str(r.get(k, '')):>12}" for k in keys) for r in obj]
        return "\n".join([header] + rows)

    return str(obj)
