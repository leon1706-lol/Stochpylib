"""Monte Carlo simulation & variance reduction.

Public names (spec: ``Modules/montecarlo.md``) are re-exported at module level per
ARCHITECTURE.md::

    from stochpylib.montecarlo import SobolSequence, AntitheticVariates, pi_estimation
"""

from stochpylib.montecarlo.quasi_random import (
    DigitalNetBase2,
    FaureSequence,
    HaltonSequence,
    LowDiscrepancy,
    NiederreiterSequence,
    SobolSequence,
)

# spec-facing alias: the general engine is exposed under its spec name
DigitalNet = DigitalNetBase2

from stochpylib.montecarlo.simulation import (  # noqa: E402
    crude_mc,
    importance_sampling,
    quasi_montecarlo,
    rejection_sampling,
    simulate,
    stratified_sampling,
)
from stochpylib.montecarlo.variance_reduction import (  # noqa: E402
    AntitheticVariates,
    ConditionedMC,
    ControlVariates,
    LatinHypercubeSampling,
    OrthogonalSampling,
    RejectionControl,
    StratifiedSampling,
)
from stochpylib.montecarlo.applications import (  # noqa: E402
    MonteCarloIntegration,
    option_pricing_mc,
    pi_estimation,
    reliability_mc,
    risk_analysis,
    sensitivity_analysis,
)
from stochpylib.montecarlo._result import MCResult  # noqa: E402

__all__ = [
    # quasi_random
    "SobolSequence",
    "HaltonSequence",
    "FaureSequence",
    "NiederreiterSequence",
    "DigitalNet",
    "DigitalNetBase2",
    "LowDiscrepancy",
    # simulation
    "simulate",
    "crude_mc",
    "importance_sampling",
    "rejection_sampling",
    "stratified_sampling",
    "quasi_montecarlo",
    # variance reduction
    "AntitheticVariates",
    "ControlVariates",
    "StratifiedSampling",
    "LatinHypercubeSampling",
    "OrthogonalSampling",
    "ConditionedMC",
    "RejectionControl",
    # applications
    "MonteCarloIntegration",
    "pi_estimation",
    "option_pricing_mc",
    "risk_analysis",
    "reliability_mc",
    "sensitivity_analysis",
    # shared result type
    "MCResult",
]
