"""Information-theoretic measures: entropy family, divergences, mutual
information, channel capacity, transfer entropy and coding theory."""

from stochpylib.information_theory._base import (
    _joint_table, _normalise, _safe_log2, _validate_probs,
)
from stochpylib.information_theory.entropy import (
    ConditionalEntropy,
    CrossEntropy,
    DifferentialEntropy,
    Entropy,
    JointEntropy,
    MaxEntropy,
    RenyiEntropy,
    TsallisEntropy,
)
from stochpylib.information_theory.divergences import (
    AlphaDivergence,
    ChiSquaredDivergence,
    HellingerDistance,
    JensenShannonDivergence,
    KLDivergence,
    RelativeEntropy,
    TotalVariation,
    WassersteinDistance,
)
from stochpylib.information_theory.mutual_info import (
    ConditionalMutualInfo,
    InteractionInformation,
    MutualInformation,
    MultiInformation,
    NormalizedMutualInformation,
    VariationOfInformation,
)
from stochpylib.information_theory.channels import (
    ChannelCapacity,
    DirectedInformation,
    InformationGain,
    SymbolicTransferEntropy,
    TransferEntropy,
)
from stochpylib.information_theory.coding import (
    AEP,
    HuffmanCode,
    ShannonLimit,
    TypicalSet,
)

__all__ = [
    # entropy
    "Entropy", "JointEntropy", "ConditionalEntropy", "CrossEntropy",
    "TsallisEntropy", "RenyiEntropy", "DifferentialEntropy", "MaxEntropy",
    # divergences
    "KLDivergence", "RelativeEntropy", "JensenShannonDivergence",
    "WassersteinDistance", "HellingerDistance", "TotalVariation",
    "ChiSquaredDivergence", "AlphaDivergence",
    # mutual information
    "MutualInformation", "NormalizedMutualInformation",
    "VariationOfInformation", "ConditionalMutualInfo",
    "InteractionInformation", "MultiInformation",
    # channels
    "ChannelCapacity", "InformationGain", "TransferEntropy",
    "DirectedInformation", "SymbolicTransferEntropy",
    # coding
    "ShannonLimit", "HuffmanCode", "TypicalSet", "AEP",
]
