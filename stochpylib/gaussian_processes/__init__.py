"""Gaussian-process models & kernels."""

from stochpylib.gaussian_processes._utils import cholesky_with_jitter
from stochpylib.gaussian_processes.kernels import (
    ArcCosineKernel, BaseKernel, LinearKernel, MaternKernel,
    NeuralNetworkKernel, NonStationaryKernel, PeriodicKernel, PolynomialKernel,
    RBFKernel, RationalQuadraticKernel, SpectralMixtureKernel, StationaryKernel,
    WhiteNoiseKernel,
)
from stochpylib.gaussian_processes.kernel_ops import (
    KernelComposition, KernelPower, KernelProduct, KernelSum,
    NonStationaryKernelOp, StationaryKernelOp, kernel_grad, kernel_matrix,
)
from stochpylib.gaussian_processes.models import (
    ExactInference, GaussianProcess, GPClassification, GPRegression,
    GPTimeSeriesModel,
)
from stochpylib.gaussian_processes.inference import (
    ExpectationPropagation, LaplacePropagation, VariationalInference,
)
from stochpylib.gaussian_processes.sparse import (
    FITC, InducingPointGP, SparseGaussianProcess, SparseVFE, VFE,
)
from stochpylib.gaussian_processes.deep_gp import DeepGP
from stochpylib.gaussian_processes.hyperparams import (
    ARD, MarginalLikelihood, cross_validate_gp, optimize_hyperparams,
)

__all__ = [
    # kernels
    "BaseKernel", "StationaryKernel", "NonStationaryKernel",
    "RBFKernel", "MaternKernel", "PeriodicKernel", "LinearKernel",
    "PolynomialKernel", "RationalQuadraticKernel", "WhiteNoiseKernel",
    "SpectralMixtureKernel", "NeuralNetworkKernel", "ArcCosineKernel",
    # kernel ops
    "KernelSum", "KernelProduct", "KernelPower", "KernelComposition",
    "StationaryKernelOp", "NonStationaryKernelOp",
    "kernel_matrix", "kernel_grad",
    # models
    "GaussianProcess", "GPRegression", "ExactInference", "GPTimeSeriesModel",
    "GPClassification",
    # inference engines
    "LaplacePropagation", "ExpectationPropagation", "VariationalInference",
    # sparse
    "FITC", "VFE", "SparseVFE", "InducingPointGP", "SparseGaussianProcess",
    # deep
    "DeepGP",
    # hyperparams
    "ARD", "MarginalLikelihood", "optimize_hyperparams", "cross_validate_gp",
    # utils
    "cholesky_with_jitter",
]
