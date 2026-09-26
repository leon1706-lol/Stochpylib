"""Statistical visualization, native on this library's own numbers.

Every ``plot_*()`` computes its data by reusing the rest of stochpylib (distributions,
timeseries, survival, gaussian_processes, spatial_statistics, advanced_mcmc, statistics,
copulas, random_matrix, ...) rather than re-deriving anything, then returns a
:class:`Figure` -- a small scene graph, not a matplotlib object. ``Figure.to_svg()`` /
``.save("*.svg")`` render from scratch with no dependency beyond numpy/scipy;
``Figure.to_matplotlib()`` / ``.save("*.png"/"*.pdf")`` are available only if matplotlib
is installed (optional, lazily imported, never a runtime dependency of this library).

``Figure`` and ``Axes`` are documented extras beyond the 35 spec names (the scene graph
every plot function returns/composes into).
"""

from stochpylib.viz._figure import Axes, Figure
from stochpylib.viz.distributions import (
    plot_cdf, plot_hazard, plot_histogram, plot_kde, plot_pdf, plot_pmf, plot_ppplot,
    plot_qqplot, plot_survival,
)
from stochpylib.viz.processes import (
    plot_acf, plot_pacf, plot_periodogram, plot_process, plot_spectrogram, plot_trajectory,
    plot_wavelet,
)
from stochpylib.viz.diagnostics import (
    funnel_plot, influence_plot, leverage_plot, pair_plot, posterior_plot, residual_plot,
    trace_plot,
)
from stochpylib.viz.multivariate import (
    plot_biplot, plot_copula, plot_correlation, plot_dendrogram, plot_heatmap,
    plot_scatter_matrix,
)
from stochpylib.viz.special import (
    plot_brownian, plot_eigenvalues, plot_gp, plot_markov_chain, plot_survival_km,
    plot_variogram,
)

__all__ = [
    "Axes",
    "Figure",
    "funnel_plot",
    "influence_plot",
    "leverage_plot",
    "pair_plot",
    "plot_acf",
    "plot_biplot",
    "plot_brownian",
    "plot_cdf",
    "plot_copula",
    "plot_correlation",
    "plot_dendrogram",
    "plot_eigenvalues",
    "plot_gp",
    "plot_hazard",
    "plot_heatmap",
    "plot_histogram",
    "plot_kde",
    "plot_markov_chain",
    "plot_pacf",
    "plot_pdf",
    "plot_periodogram",
    "plot_pmf",
    "plot_ppplot",
    "plot_process",
    "plot_qqplot",
    "plot_scatter_matrix",
    "plot_spectrogram",
    "plot_survival",
    "plot_survival_km",
    "plot_trajectory",
    "plot_variogram",
    "plot_wavelet",
    "posterior_plot",
    "residual_plot",
    "trace_plot",
]

assert len(__all__) == len(set(__all__))
