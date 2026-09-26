"""Real-matplotlib backend tests -- NOT auto-collected by the main suite.

``pyproject.toml``'s ``python_files = ["tests.py", "e2e.py"]`` means this file never
joins ``pytest tests/`` (keeping the 2 permanently-skipped tests the only skips the docs
suite counts); the ``viz-matplotlib`` CI job runs it by explicit path instead, after
installing matplotlib. Everything in ``tests.py``/``e2e.py`` already passes with no
matplotlib installed at all -- this file only adds coverage for the optional
``Figure.to_matplotlib()``/``.save("*.png"/"*.pdf")`` path with the real library.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from stochpylib.viz import (  # noqa: E402
    Figure, plot_acf, plot_heatmap, plot_histogram, plot_markov_chain, plot_pdf,
    trace_plot,
)

_RNG = np.random.default_rng(0)


class TestMatplotlibRealBackend:
    def test_to_matplotlib_returns_real_figure(self):
        from stochpylib.distributions import Normal

        fig = plot_pdf(Normal(0, 1))
        mfig = fig.to_matplotlib()
        assert type(mfig).__module__.startswith("matplotlib")
        assert len(mfig.axes) == 1

    def test_line_data_matches_fig_data(self):
        from stochpylib.distributions import Normal

        fig = plot_pdf(Normal(0, 1))
        mfig = fig.to_matplotlib()
        xdata, ydata = mfig.axes[0].lines[0].get_data()
        assert np.allclose(xdata, fig.data["x"])
        assert np.allclose(ydata, fig.data["pdf"])

    def test_multi_panel_axes_count(self):
        fig = trace_plot(_RNG.standard_normal((4, 100, 3)))
        mfig = fig.to_matplotlib()
        assert len(mfig.axes) == 3

    def test_heatmap_creates_pcolormesh_and_colorbar(self):
        fig = plot_heatmap(_RNG.random((4, 5)))
        mfig = fig.to_matplotlib()
        ax = mfig.axes[0]
        assert len(ax.collections) >= 1
        # a colorbar is a second axes attached to the figure
        assert len(mfig.axes) >= 2

    def test_markov_chain_self_loops_are_circle_patches(self):
        P = np.array([[0.9, 0.1], [0.3, 0.7]])
        fig = plot_markov_chain(P)
        mfig = fig.to_matplotlib()
        assert len(mfig.axes[0].patches) >= 2  # 2 node circles + self-loop rings

    def test_categorical_ticks_carry_over(self):
        fig = plot_heatmap(_RNG.random((3, 3)), row_labels=["a", "b", "c"],
                           col_labels=["x", "y", "z"])
        mfig = fig.to_matplotlib()
        labels = [t.get_text() for t in mfig.axes[0].get_xticklabels()]
        assert labels == ["x", "y", "z"]

    def test_log_scale_carries_over(self):
        x = _RNG.standard_normal(300)
        fig = plot_acf(x)
        fig.ax.yscale = "log"
        mfig = fig.to_matplotlib()
        assert mfig.axes[0].get_yscale() == "log"

    def test_save_png(self, tmp_path):
        fig = plot_histogram(_RNG.standard_normal(200))
        path = tmp_path / "hist.png"
        fig.save(str(path))
        content = path.read_bytes()
        assert content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_save_pdf(self, tmp_path):
        fig = plot_histogram(_RNG.standard_normal(200))
        path = tmp_path / "hist.pdf"
        fig.save(str(path))
        assert path.read_bytes()[:5] == b"%PDF-"

    def test_legend_created_when_labels_present(self):
        fig = Figure()
        fig.ax.line([0, 1], [0, 1], label="a")
        fig.ax.line([0, 1], [1, 0], label="b")
        fig.ax.legend = "best"
        mfig = fig.to_matplotlib()
        assert mfig.axes[0].get_legend() is not None

    def test_show_with_agg_backend_does_not_raise(self):
        from stochpylib.distributions import Normal

        f = plot_pdf(Normal(0, 1))
        f.show()  # Agg has no GUI event loop; .show() must not raise
