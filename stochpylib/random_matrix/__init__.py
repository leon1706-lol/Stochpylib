"""Random matrix theory: classical ensembles (GOE/GUE/GSE, Wigner, Wishart, CUE,
Ginibre-type), limiting spectral laws (semicircle, Marchenko-Pastur, Tracy-Widom) and
tridiagonal beta-ensembles, Haar-distributed rotations, and spectral statistics
(spacings, level repulsion, empirical spectra, edge fluctuations) -- 23 spec names
across four submodules, natively on numpy/scipy.
"""

from stochpylib.random_matrix.empirical_spectra import (
    BetaEnsemble,
    JacobiEnsemble,
    MarchenkoPastur,
    TracyWidomDistribution,
    WignerSemicircle,
)
from stochpylib.random_matrix.ensembles import (
    CUE,
    GOE,
    GSE,
    GUE,
    CircularLaw,
    InverseWishart,
    MatrixEnsemble,
    MuresanMatrix,
    WignerMatrix,
    WishartMatrix,
)
from stochpylib.random_matrix.random_rotations import (
    HaarMeasure,
    RandomOrthogonalMatrix,
    RandomSymplectic,
    RandomUnitaryMatrix,
)
from stochpylib.random_matrix.statistics import (
    BulkSpectrum,
    EigenvalueDistribution,
    EigenvalueSpacing,
    LargestEigenvalue,
    LevelRepulsion,
    SpectralEdge,
)

__all__ = [
    "BetaEnsemble",
    "BulkSpectrum",
    "CUE",
    "CircularLaw",
    "EigenvalueDistribution",
    "EigenvalueSpacing",
    "GOE",
    "GSE",
    "GUE",
    "HaarMeasure",
    "InverseWishart",
    "JacobiEnsemble",
    "LargestEigenvalue",
    "LevelRepulsion",
    "MarchenkoPastur",
    "MatrixEnsemble",
    "MuresanMatrix",
    "RandomOrthogonalMatrix",
    "RandomSymplectic",
    "RandomUnitaryMatrix",
    "SpectralEdge",
    "TracyWidomDistribution",
    "WignerMatrix",
    "WignerSemicircle",
    "WishartMatrix",
]
