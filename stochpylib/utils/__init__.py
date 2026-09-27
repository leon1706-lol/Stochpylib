"""Infrastructure & utilities: seeding/reproducible streams, benchmarking/profiling/
parallel/GPU/JIT backends, reproducibility scaffolding, distribution fitting & data
cleaning, serialization, and interop with numpy/scipy/pandas/torch/jax.

Optional heavy dependencies (pandas/torch/jax/numba/cupy) are lazily imported and
confined to ``utils/_backends.py`` -- ``import stochpylib.utils`` never requires any
of them, mirroring ``viz``'s optional-matplotlib-backend pattern.

``FitResult``/``OutlierResult``/``from_pickle`` are documented extras beyond the 38
spec names (see ``Modules/utils.md``).
"""

from stochpylib.utils.random import Generator, SeedSequence, random_state, set_seed, spawn_generator
from stochpylib.utils.performance import (
    Benchmark, GPUBackend, JIT_compile, MemoryPool, ParallelSimulation, Profiler,
    VectorizedOps,
)
from stochpylib.utils.reproducibility import (
    EnvironmentCapture, ExperimentLogger, RandomStream, Reproducibility, VersionLock,
)
from stochpylib.utils.data import (
    DataValidation, ecdf, fit, goodness_of_fit, missing_imputation, moment_matching,
    outlier_detection,
)
from stochpylib.utils.io import (
    Configuration, Logging, Serialization, from_dict, from_json, from_pickle, summary,
    to_dict, to_json, to_pickle,
)
from stochpylib.utils.compat import (
    jax_interface, numpy_interface, pandas_interface, scipy_interface, torch_interface,
)
from stochpylib.utils._results import FitResult, OutlierResult

__all__ = [
    "Benchmark",
    "Configuration",
    "DataValidation",
    "EnvironmentCapture",
    "ExperimentLogger",
    "FitResult",
    "GPUBackend",
    "Generator",
    "JIT_compile",
    "Logging",
    "MemoryPool",
    "OutlierResult",
    "ParallelSimulation",
    "Profiler",
    "RandomStream",
    "Reproducibility",
    "Serialization",
    "SeedSequence",
    "VectorizedOps",
    "VersionLock",
    "ecdf",
    "fit",
    "from_dict",
    "from_json",
    "from_pickle",
    "goodness_of_fit",
    "jax_interface",
    "missing_imputation",
    "moment_matching",
    "numpy_interface",
    "outlier_detection",
    "pandas_interface",
    "random_state",
    "scipy_interface",
    "set_seed",
    "spawn_generator",
    "summary",
    "to_dict",
    "to_json",
    "to_pickle",
    "torch_interface",
]

assert len(__all__) == len(set(__all__))
