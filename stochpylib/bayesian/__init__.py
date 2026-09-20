"""Bayesian inference framework: priors/likelihoods/posteriors, ten conjugate families
with predictives and evidence, Bayesian models (linear/logistic regression, naive Bayes,
hierarchical normal, finite mixtures, discrete Bayesian networks, Dirichlet-process
mixtures), model-selection criteria (AIC/BIC/DIC/WAIC/PSIS-LOO/TIC, Bayes factors), and
posterior approximations (Laplace, EP, variational, importance sampling) -- 25 spec names
across four submodules, natively on numpy/scipy. MCMC/SMC/variational inference delegate
to :mod:`stochpylib.advanced_mcmc` rather than duplicating it; moment matching for
expectation propagation delegates to :mod:`stochpylib.numerical_methods`.
"""

from stochpylib.bayesian.core import (
    ConjugateFamily,
    Likelihood,
    Prior,
    bayes_update,
    conjugate_prior,
    evidence,
    likelihood,
    posterior,
    posterior_predictive,
    prior,
)
from stochpylib.bayesian.computation import (
    EP_Posterior,
    ImportanceSamplingPosterior,
    LaplacePosterior,
    MFVariational,
)
from stochpylib.bayesian.selection import (
    AIC,
    BIC,
    DIC,
    LOO_CV,
    TICfit,
    WAIC,
    bayes_factor,
)
from stochpylib.bayesian.models import (
    BayesianLinear,
    BayesianLogistic,
    BayesianNetwork,
    DirichletProcess,
    HierarchicalModel,
    MixtureModel,
    NaiveBayes,
)
from stochpylib.bayesian._result import (
    EmpiricalPredictive,
    ICResult,
    Posterior,
    PosteriorApproximation,
)

__all__ = [
    "AIC",
    "BIC",
    "BayesianLinear",
    "BayesianLogistic",
    "BayesianNetwork",
    "ConjugateFamily",
    "DIC",
    "DirichletProcess",
    "EP_Posterior",
    "EmpiricalPredictive",
    "HierarchicalModel",
    "ICResult",
    "ImportanceSamplingPosterior",
    "LOO_CV",
    "LaplacePosterior",
    "Likelihood",
    "MFVariational",
    "MixtureModel",
    "NaiveBayes",
    "Posterior",
    "PosteriorApproximation",
    "Prior",
    "TICfit",
    "WAIC",
    "bayes_factor",
    "bayes_update",
    "conjugate_prior",
    "evidence",
    "likelihood",
    "posterior",
    "posterior_predictive",
    "prior",
]
