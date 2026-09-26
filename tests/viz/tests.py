"""Oracle suite for :mod:`stochpylib.viz`.

Every ``plot_*()`` computes its numbers by reusing the rest of stochpylib; this suite
checks those numbers (``fig.data``), not pixels, against independent oracles: scipy.stats
(distributions), statsmodels (ACF/PACF/OLS influence), scipy.cluster.hierarchy
(dendrogram), lifelines (Kaplan-Meier), numpy/scipy linear algebra (PCA/eigenvalues), and
direct brute-force formulas elsewhere. The rendering layer itself (native SVG, always
available; matplotlib, optional) is checked structurally: every figure must parse as XML
and render deterministically. All randomness is seeded; statistical assertions are set at
>= 3 standard errors.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy import cluster, stats

from stochpylib import viz
from stochpylib.viz import (
    Axes, Figure, funnel_plot, influence_plot, leverage_plot, pair_plot, plot_acf,
    plot_biplot, plot_brownian, plot_cdf, plot_copula, plot_correlation, plot_dendrogram,
    plot_eigenvalues, plot_gp, plot_hazard, plot_heatmap, plot_histogram, plot_kde,
    plot_markov_chain, plot_pacf, plot_pdf, plot_periodogram, plot_pmf, plot_ppplot,
    plot_process, plot_qqplot, plot_scatter_matrix, plot_spectrogram, plot_survival,
    plot_survival_km, plot_trajectory, plot_variogram, plot_wavelet, posterior_plot,
    residual_plot, trace_plot,
)
from stochpylib.viz._colors import colormap, palette, to_hex
from stochpylib.viz._common import (
    _durbin_levinson, _extract_chains, _extract_samples, _filliben_pp, _leverage_stats,
)
from stochpylib.viz._figure import Band, Line, Scatter, Step
from stochpylib.viz._ticks import fmt, log_ticks, nice_ticks

_RNG = np.random.default_rng(0)


def _parse(fig):
    """Parse a Figure's SVG and return the ElementTree root (fails loudly if malformed)."""
    return ET.fromstring(fig.to_svg())


# ============================================================================== ticks


class TestTicks:
    @pytest.mark.parametrize("lo,hi", [(0, 97), (-3.2, 3.2), (0.001, 0.004), (-500, 1200)])
    def test_nice_ticks_cover_range(self, lo, hi):
        ticks = nice_ticks(lo, hi)
        assert ticks.min() <= lo and ticks.max() >= hi

    def test_nice_ticks_are_nice_numbers(self):
        ticks = nice_ticks(0, 97)
        step = ticks[1] - ticks[0]
        mantissa = step / 10 ** np.floor(np.log10(step))
        assert np.isclose(mantissa, 1.0) or np.isclose(mantissa, 2.0) or np.isclose(mantissa, 5.0)

    def test_nice_ticks_monotone(self):
        ticks = nice_ticks(-12.3, 44.7)
        assert np.all(np.diff(ticks) > 0)

    def test_nice_ticks_degenerate_range(self):
        ticks = nice_ticks(5.0, 5.0)
        assert ticks.min() < 5.0 < ticks.max()

    def test_nice_ticks_negative_only(self):
        ticks = nice_ticks(-10, -2)
        assert ticks.min() <= -10 and ticks.max() >= -2

    def test_log_ticks_are_decades(self):
        ticks = log_ticks(0.5, 200)
        ratios = ticks[1:] / ticks[:-1]
        assert np.allclose(ratios, 10.0)

    def test_log_ticks_cover_range(self):
        ticks = log_ticks(3.0, 3000.0)
        assert ticks.min() <= 3.0 and ticks.max() >= 3000.0

    @pytest.mark.parametrize("v", [0.0, -0.0])
    def test_fmt_never_prints_negative_zero(self, v):
        assert fmt(v) == "0"

    def test_fmt_tiny_nonzero_values_are_not_rounded_to_zero(self):
        # a tiny but genuinely nonzero value must stay visibly nonzero -- only exact
        # (positive or negative) zero collapses to "0"
        assert fmt(-1e-12) == "-1e-12"
        assert fmt(1e-12) == "1e-12"

    def test_fmt_compact(self):
        assert fmt(1234567) == "1.235e+06"
        assert fmt(0.000123) == "0.000123"
        assert fmt(3.0) == "3"


# ============================================================================== colors


class TestColors:
    def test_palette_cycles(self):
        assert palette(0) == palette(10)  # 10-color cycle

    def test_palette_returns_valid_hex(self):
        for i in range(12):
            c = palette(i)
            assert c.startswith("#") and len(c) == 7
            int(c[1:], 16)

    def test_viridis_endpoints_are_dark_then_bright(self):
        cm = colormap("viridis")
        lo = tuple(int(cm(0.0)[i : i + 2], 16) for i in (1, 3, 5))
        hi = tuple(int(cm(1.0)[i : i + 2], 16) for i in (1, 3, 5))
        assert sum(lo) < sum(hi)  # dark purple -> bright yellow

    def test_rdbu_center_is_near_white(self):
        c = colormap("RdBu")(0.5)
        rgb = [int(c[i : i + 2], 16) for i in (1, 3, 5)]
        assert all(v > 230 for v in rgb)

    def test_rdbu_endpoints_are_blue_and_red(self):
        cm = colormap("RdBu")
        blue = cm(0.0)
        red = cm(1.0)
        assert int(blue[5:7], 16) > int(blue[1:3], 16)  # more blue than red
        assert int(red[1:3], 16) > int(red[5:7], 16)  # more red than blue

    def test_unknown_colormap_raises(self):
        with pytest.raises(ValueError):
            colormap("not-a-colormap")

    def test_to_hex_clips(self):
        assert to_hex((2.0, -1.0, 0.5)) == "#FF0080"


# ============================================================================== figure model


class TestFigureModel:
    def test_line_extent_and_padding(self):
        fig = Figure()
        fig.ax.line([0, 1, 2], [0, 5, 0])
        (xlo, xhi), (ylo, yhi) = fig.ax.data_limits()
        assert xlo < 0 and xhi > 2
        assert ylo < 0 and yhi > 5  # 5% padding both sides

    def test_nan_splits_line_into_runs(self):
        # two >=2-point runs either side of the NaN (a length-1 run alone can't draw a
        # segment and is correctly dropped, so this needs 3 points on each side)
        fig = Figure()
        fig.ax.line([0, 1, 2, 3, 4, 5], [0.0, 0.5, np.nan, 1.0, 1.5, 2.0])
        svg = fig.to_svg()
        assert svg.count("<polyline") == 2  # split around the NaN

    def test_ax_composition_returns_owning_figure(self):
        fig = Figure(nrows=1, ncols=2)
        target_ax = fig.axes[1]
        out = plot_histogram(_RNG.standard_normal(50), ax=target_ax)
        assert out is fig
        assert len(target_ax.artists) > 0
        assert len(fig.axes[0].artists) == 0

    def test_multi_panel_grid_indexing(self):
        fig = Figure(nrows=2, ncols=3)
        assert len(fig.axes) == 6
        assert fig.ax is fig.axes[0]

    def test_repr_reports_counts(self):
        fig = Figure()
        fig.ax.line([0, 1], [0, 1])
        fig.ax.scatter([0], [0])
        assert "artists=2" in repr(fig)

    def test_heatmap_default_edges(self):
        fig = Figure()
        art = fig.ax.heatmap(np.ones((3, 4)))
        assert art.x_edges.shape == (5,) and art.y_edges.shape == (4,)

    def test_bad_segments_shape_raises(self):
        fig = Figure()
        with pytest.raises(ValueError):
            fig.ax.segments(np.zeros((3, 2)))

    def test_mismatched_xy_raises(self):
        fig = Figure()
        with pytest.raises(ValueError):
            fig.ax.line([1, 2, 3], [1, 2])

    def test_line_artist_defaults(self):
        art = Line(np.array([0.0]), np.array([0.0]))
        assert art.kind == "line" and art.width == 1.5 and art.color is None

    def test_scatter_step_band_artist_kinds(self):
        assert Scatter(np.array([0.0]), np.array([0.0])).kind == "scatter"
        assert Step(np.array([0.0]), np.array([0.0])).kind == "step"
        assert Band(np.array([0.0]), np.array([0.0]), np.array([1.0])).kind == "band"

    def test_extract_chains_normalizes_shapes(self):
        assert _extract_chains(_RNG.standard_normal(50)).shape == (1, 50, 1)
        assert _extract_chains(_RNG.standard_normal((3, 50))).shape == (3, 50, 1)
        assert _extract_chains(_RNG.standard_normal((3, 50, 2))).shape == (3, 50, 2)

    def test_extract_samples_treats_2d_as_samples_table(self):
        arr = _RNG.standard_normal((50, 2))
        assert _extract_samples(arr).shape == (50, 2)
        assert _extract_samples(_RNG.standard_normal(50)).shape == (50, 1)


# ============================================================================== SVG rendering


class TestSVG:
    def _figures(self):
        rng = np.random.default_rng(1)
        from stochpylib.distributions import Gamma, Normal, Poisson

        data = rng.gamma(3, 2, 200)
        return {
            "pdf": plot_pdf(Normal(0, 1)),
            "pmf": plot_pmf(Poisson(4.0)),
            "histogram": plot_histogram(data),
            "heatmap": plot_heatmap(rng.random((4, 5))),
            "process": plot_process(rng.standard_normal((10, 30)).cumsum(axis=1)),
            "brownian": plot_brownian(n_paths=3, n_steps=20, random_state=0),
        }

    @pytest.mark.parametrize("name", ["pdf", "pmf", "histogram", "heatmap", "process",
                                      "brownian"])
    def test_parses_as_xml(self, name):
        fig = self._figures()[name]
        _parse(fig)

    def test_deterministic_rendering(self):
        from stochpylib.distributions import Normal

        fig1 = plot_pdf(Normal(0, 1))
        fig2 = plot_pdf(Normal(0, 1))
        assert fig1.to_svg() == fig2.to_svg()

    def test_histogram_rect_count_matches_bins(self):
        fig = plot_histogram(_RNG.standard_normal(300), bins=10)
        n_bins = len(fig.data["heights"])
        svg = fig.to_svg()
        # 3 fixed rects per figure: white background, the clip-path rect, the axes frame
        assert svg.count("<rect") - 3 == n_bins

    def test_heatmap_rect_count_matches_cells(self):
        Z = _RNG.random((3, 4))
        fig = plot_heatmap(Z)
        svg = fig.to_svg()
        # as above, plus one more rect for the colorbar gradient plot_heatmap always adds
        assert svg.count("<rect") - 4 == Z.size

    def test_km_step_is_non_increasing(self):
        durations = _RNG.exponential(5, 40)
        events = np.ones(40)
        fig = plot_survival_km(durations, events)
        s = fig.data["KM"]["survival"]
        assert np.all(np.diff(s) <= 1e-12)

    def test_escapes_special_characters_in_title(self):
        fig = plot_histogram(_RNG.standard_normal(20), title="a<b&c")
        root = _parse(fig)
        texts = [el.text for el in root.iter() if el.tag.endswith("text")]
        assert "a<b&c" in texts

    def test_save_svg_roundtrip(self, tmp_path):
        fig = plot_histogram(_RNG.standard_normal(20))
        path = fig.save(str(tmp_path / "x.svg"))
        content = open(path, encoding="utf-8").read()
        assert content == fig.to_svg()

    def test_points_inside_clip_rectangle(self):
        from stochpylib.distributions import Normal

        fig = plot_pdf(Normal(0, 1))
        root = _parse(fig)
        # ElementTree namespaces every tag ("{http://www.w3.org/2000/svg}clipPath"), so a
        # plain "clipPath/rect" XPath never matches -- walk and match by local name instead
        clip_path_el = next(el for el in root.iter() if el.tag.endswith("clipPath"))
        clip_rect = next(child for child in clip_path_el if child.tag.endswith("rect"))
        x0, y0 = float(clip_rect.get("x")), float(clip_rect.get("y"))
        w, h = float(clip_rect.get("width")), float(clip_rect.get("height"))
        for poly in root.iter():
            if not poly.tag.endswith("polyline"):
                continue
            pts = poly.get("points").split()
            for p in pts:
                px, py = map(float, p.split(","))
                assert x0 - 1 <= px <= x0 + w + 1
                assert y0 - 1 <= py <= y0 + h + 1

    def test_log_scale_axis_renders(self):
        freqs, power = np.geomspace(1, 100, 20), np.geomspace(1, 100, 20)
        fig = Figure()
        fig.ax.line(freqs, power)
        fig.ax.set(yscale="log")
        _parse(fig)


# ============================================================================== matplotlib backend


class _FakeMplAxes:
    """Records every call instead of drawing, so the mapping can be checked with no real
    matplotlib installed."""

    def __init__(self):
        self.calls = []
        self.figure = self

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return None

        return record

    def add_collection(self, coll):
        self.calls.append(("add_collection", (coll,), {}))

    def add_patch(self, patch):
        self.calls.append(("add_patch", (patch,), {}))

    def colorbar(self, *a, **k):
        pass


class TestMatplotlibBackend:
    def test_line_maps_to_plot(self):
        from stochpylib.viz._mpl import draw_axes

        fig = Figure()
        fig.ax.line([0, 1], [0, 1], label="l")
        fake = _FakeMplAxes()
        draw_axes(fake, fig.ax)
        names = [c[0] for c in fake.calls]
        assert "plot" in names

    def test_scatter_bar_band_stem_map(self):
        from stochpylib.viz._mpl import draw_axes

        fig = Figure()
        fig.ax.scatter([0, 1], [0, 1])
        fig.ax.bar([0, 1], [1, 2], width=0.5)
        fig.ax.band([0, 1], [0, 0], [1, 1])
        fig.ax.stem([0, 1], [1, 2])
        fake = _FakeMplAxes()
        draw_axes(fake, fig.ax)
        names = {c[0] for c in fake.calls}
        assert {"scatter", "bar", "fill_between", "stem"} <= names

    def test_heatmap_maps_to_pcolormesh(self):
        from stochpylib.viz._mpl import draw_axes

        fig = Figure()
        fig.ax.heatmap(np.ones((3, 3)))
        fake = _FakeMplAxes()
        draw_axes(fake, fig.ax)
        assert "pcolormesh" in {c[0] for c in fake.calls}

    def test_arrow_maps_to_annotate(self):
        from stochpylib.viz._mpl import draw_axes

        fig = Figure()
        fig.ax.arrow(0, 0, 1, 1, curved=0.2)
        fake = _FakeMplAxes()
        draw_axes(fake, fig.ax)
        assert "annotate" in {c[0] for c in fake.calls}

    def test_missing_matplotlib_raises_clear_error(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("simulated: matplotlib not installed")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        fig = plot_histogram(_RNG.standard_normal(20))
        with pytest.raises(ImportError, match="matplotlib is optional"):
            fig.to_matplotlib()
        # the native path must still work with matplotlib "absent"
        assert fig.to_svg()

    def test_to_matplotlib_real_backend_if_available(self):
        pytest.importorskip("matplotlib")
        fig = plot_histogram(_RNG.standard_normal(30))
        mfig = fig.to_matplotlib()
        assert type(mfig).__name__ == "Figure"


# ============================================================================== distributions


class TestDistributionsSubmodule:
    def test_plot_pdf_matches_scipy(self):
        from stochpylib.distributions import Gamma

        g = Gamma(shape=3.0, scale=2.0)
        fig = plot_pdf(g)
        oracle = stats.gamma.pdf(fig.data["x"], a=3.0, scale=2.0)
        assert np.allclose(fig.data["pdf"], oracle, rtol=1e-8)

    def test_plot_pdf_discrete_delegates_to_pmf(self):
        from stochpylib.distributions import Poisson

        fig = plot_pdf(Poisson(4.0))
        assert "pmf" in fig.data

    def test_plot_pdf_overlay_multiple(self):
        from stochpylib.distributions import Gamma, Normal

        fig = plot_pdf([Normal(0, 1), Gamma(3, 2)])
        assert len(fig.ax.artists) == 2

    def test_plot_pmf_matches_scipy(self):
        from stochpylib.distributions import Poisson

        fig = plot_pmf(Poisson(4.0))
        oracle = stats.poisson.pmf(fig.data["k"], mu=4.0)
        assert np.allclose(fig.data["pmf"], oracle, rtol=1e-8)
        assert fig.data["pmf"].sum() >= 0.99

    def test_plot_pmf_rejects_continuous(self):
        from stochpylib.distributions import Normal

        with pytest.raises(ValueError):
            plot_pmf(Normal(0, 1))

    def test_plot_cdf_distribution_matches_scipy(self):
        from stochpylib.distributions import Gamma

        g = Gamma(shape=2.0, scale=1.5)
        fig = plot_cdf(g)
        oracle = stats.gamma.cdf(fig.data["x"], a=2.0, scale=1.5)
        assert np.allclose(fig.data["cdf"], oracle, rtol=1e-8)

    def test_plot_cdf_empirical_matches_direct_ecdf(self):
        data = _RNG.standard_normal(80)
        fig = plot_cdf(data, confidence=0.95)
        expected = np.searchsorted(np.sort(data), fig.data["x"], side="right") / len(data)
        assert np.allclose(fig.data["cdf"], expected)
        assert np.all(fig.data["upper"] >= fig.data["cdf"])
        assert np.all(fig.data["lower"] <= fig.data["cdf"])

    def test_plot_survival_matches_one_minus_cdf(self):
        from stochpylib.distributions import Gamma

        g = Gamma(shape=3.0, scale=2.0)
        fig = plot_survival(g)
        assert np.allclose(fig.data["survival"], 1.0 - g.cdf(fig.data["x"]))

    def test_plot_hazard_exponential_is_constant(self):
        from stochpylib.distributions import Exponential

        lam = 0.7
        e = Exponential(rate=lam)
        fig = plot_hazard(e)
        assert np.allclose(fig.data["hazard"], lam, rtol=1e-6)

    def test_plot_hazard_cumulative_is_neg_log_survival(self):
        from stochpylib.distributions import Gamma

        g = Gamma(shape=2.0, scale=1.0)
        fig = plot_hazard(g, cumulative=True)
        assert np.allclose(fig.data["hazard"], -np.log(1.0 - g.cdf(fig.data["x"])), rtol=1e-6)

    def test_plot_qqplot_matches_scipy_probplot(self):
        # dist=None fits Normal(mean, std) and reports theoretical quantiles already
        # scaled to the sample's units (mean + std * z), not scipy probplot's raw
        # standard-normal osm -- rescale scipy's osm the same way before comparing.
        data = _RNG.standard_normal(50)
        (osm, osr), _ = stats.probplot(data, dist="norm", fit=True)
        fig = plot_qqplot(data)
        assert np.allclose(np.sort(fig.data["sample"]), osr)
        mean, std = data.mean(), data.std(ddof=1)
        assert np.allclose(fig.data["theoretical"], mean + std * osm)

    def test_plot_qqplot_good_fit_has_high_r(self):
        from stochpylib.distributions import Gamma

        g = Gamma(shape=3.0, scale=2.0)
        data = g.rvs(1000, random_state=1)
        fig = plot_qqplot(data, dist=g)
        assert fig.data["r"] > 0.98

    def test_plot_ppplot_45_line_for_correct_distribution(self):
        from stochpylib.distributions import Normal

        data = _RNG.standard_normal(500)
        fig = plot_ppplot(data, Normal(0, 1))
        assert np.corrcoef(fig.data["empirical"], fig.data["theoretical"])[0, 1] > 0.99

    def test_plot_histogram_density_integrates_to_one(self):
        fig = plot_histogram(_RNG.standard_normal(500))
        widths = np.diff(fig.data["edges"])
        assert np.sum(fig.data["heights"] * widths) == pytest.approx(1.0, abs=1e-8)

    def test_plot_histogram_matches_numpy(self):
        data = _RNG.standard_normal(200)
        fig = plot_histogram(data, bins=15)
        heights, edges = np.histogram(data, bins=15, density=True)
        assert np.allclose(fig.data["heights"], heights)
        assert np.allclose(fig.data["edges"], edges)

    def test_plot_kde_matches_scipy_gaussian_kde(self):
        data = _RNG.standard_normal(300)
        fig = plot_kde(data)
        h = fig.data["bandwidth"]
        oracle = stats.gaussian_kde(data, bw_method=h / data.std(ddof=1))
        got = oracle(fig.data["x"])
        assert np.max(np.abs(got - fig.data["density"])) < 1e-6

    def test_plot_kde_integrates_near_one(self):
        data = _RNG.standard_normal(300)
        fig = plot_kde(data)
        x, d = fig.data["x"], fig.data["density"]
        assert np.trapezoid(d, x) == pytest.approx(1.0, abs=0.02)

    def test_filliben_matches_scipy_exactly(self):
        n = 20
        pp = _filliben_pp(n)
        theoretical = stats.norm.ppf(pp)
        osm, _ = stats.probplot(_RNG.standard_normal(n), dist="norm", fit=False)
        assert np.allclose(np.sort(theoretical), osm)

    def test_empty_data_raises(self):
        with pytest.raises(ValueError):
            plot_histogram([])

    def test_nan_data_raises(self):
        with pytest.raises(ValueError):
            plot_kde([1.0, 2.0, np.nan])


# ============================================================================== processes


class TestProcessesSubmodule:
    def _ar1(self, n=400, phi=0.6, seed=0):
        rng = np.random.default_rng(seed)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + rng.standard_normal()
        return x

    def test_plot_acf_matches_statsmodels(self):
        pytest.importorskip("statsmodels")
        import statsmodels.api as sm

        x = self._ar1()
        fig = plot_acf(x, nlags=20)
        oracle = sm.tsa.stattools.acf(x, nlags=20, fft=True)
        assert np.allclose(fig.data["acf"], oracle, atol=1e-10)

    def test_plot_acf_band_matches_statsmodels_confint(self):
        pytest.importorskip("statsmodels")
        import statsmodels.api as sm

        x = self._ar1()
        fig = plot_acf(x, nlags=20, alpha=0.05)
        acf, confint = sm.tsa.stattools.acf(x, nlags=20, fft=True, alpha=0.05)
        half_width = (confint[:, 1] - confint[:, 0]) / 2.0
        assert np.allclose(fig.data["band"], half_width, atol=1e-8)

    def test_plot_acf_recovers_ar1_phi(self):
        x = self._ar1(n=2000, phi=0.6)
        fig = plot_acf(x, nlags=5)
        se = 1.0 / np.sqrt(2000)
        assert abs(fig.data["acf"][1] - 0.6) < 3 * se

    def test_plot_pacf_matches_statsmodels(self):
        pytest.importorskip("statsmodels")
        import statsmodels.api as sm

        x = self._ar1()
        fig = plot_pacf(x, nlags=20)
        oracle = sm.tsa.stattools.pacf(x, nlags=20, method="ldb")
        assert np.allclose(fig.data["pacf"], oracle, atol=1e-8)

    def test_plot_pacf_ar1_cuts_off_after_lag_1(self):
        x = self._ar1(n=2000, phi=0.6)
        fig = plot_pacf(x, nlags=10)
        assert abs(fig.data["pacf"][1] - 0.6) < 0.05
        assert abs(fig.data["pacf"][2]) < 3 * fig.data["band"][0]

    def test_durbin_levinson_matches_statsmodels(self):
        pytest.importorskip("statsmodels")
        import statsmodels.api as sm

        x = self._ar1()
        acov = np.array([np.sum((x[k:] - x.mean()) * (x[: len(x) - k] - x.mean()))
                        for k in range(21)]) / len(x)
        acf = acov / acov[0]
        mine = _durbin_levinson(acf)
        oracle = sm.tsa.stattools.pacf(x, nlags=20, method="ldb")
        assert np.allclose(mine, oracle, atol=1e-8)

    def test_plot_periodogram_matches_timeseries_function(self):
        from stochpylib.timeseries import Periodogram

        x = _RNG.standard_normal(256)
        fig = plot_periodogram(x)
        freqs, power = Periodogram(x)
        assert np.allclose(fig.data["freqs"], freqs)
        assert np.allclose(fig.data["power"], power)

    def test_plot_periodogram_finds_sinusoid_peak(self):
        t = np.arange(400)
        sig = np.sin(2 * np.pi * 5 * t / 400) + 0.2 * _RNG.standard_normal(400)
        fig = plot_periodogram(sig, fs=400)
        peak = fig.data["freqs"][1:][np.argmax(fig.data["power"][1:])]
        assert peak == pytest.approx(5.0, abs=1.0)

    def test_plot_spectrogram_shape(self):
        x = _RNG.standard_normal(300)
        fig = plot_spectrogram(x, window_len=64, hop=16)
        assert fig.data["magnitude"].shape[1] == len(fig.data["times"])
        assert fig.data["magnitude"].shape[0] == len(fig.data["freqs"])

    def test_plot_wavelet_cwt_peak_scale_matches_theory(self):
        n = 512
        t = np.arange(n)
        period = 40
        sig = np.sin(2 * np.pi * t / period)
        fig = plot_wavelet(sig)
        power = fig.data["power"]
        peak_scale = fig.data["scales"][np.argmax(power.mean(axis=1))]
        expected = 6.0 * period / (2 * np.pi)
        assert peak_scale == pytest.approx(expected, rel=0.15)

    def test_plot_wavelet_dwt_levels(self):
        sig = _RNG.standard_normal(512)
        fig = plot_wavelet(sig, kind="dwt")
        assert fig.nrows == len(fig.data["details"]) + 1

    def test_plot_wavelet_bad_kind_raises(self):
        with pytest.raises(ValueError):
            plot_wavelet(_RNG.standard_normal(64), kind="bogus")

    def test_plot_process_mean_matches_numpy(self):
        paths = _RNG.standard_normal((30, 50)).cumsum(axis=1)
        fig = plot_process(paths)
        assert np.allclose(fig.data["mean"], paths.mean(axis=0))

    def test_plot_process_tuple_input(self):
        t = np.linspace(0, 1, 20)
        paths = _RNG.standard_normal((5, 20))
        fig = plot_process((t, paths))
        assert np.allclose(fig.data["t"], t)

    def test_plot_trajectory_2d(self):
        path = np.column_stack([np.cos(np.linspace(0, 2 * np.pi, 50)),
                                np.sin(np.linspace(0, 2 * np.pi, 50))])
        fig = plot_trajectory(path)
        assert np.allclose(fig.data["x"], path[:, 0])

    def test_plot_trajectory_lag_phase_portrait(self):
        x = self._ar1()
        fig = plot_trajectory(x, lag=1)
        assert np.allclose(fig.data["x"], x[:-1])
        assert np.allclose(fig.data["y"], x[1:])

    def test_plot_trajectory_needs_lag_or_2d(self):
        with pytest.raises(ValueError):
            plot_trajectory(_RNG.standard_normal(10))


# ============================================================================== diagnostics


class TestDiagnosticsSubmodule:
    def _hmc_samples(self, seed=0, n_samples=400, n_warmup=200, n_chains=4, corr=0.0):
        from stochpylib.advanced_mcmc import HamiltonianMonteCarlo

        def logp(theta):
            return -0.5 * np.sum(theta ** 2) - corr * theta[0] * theta[1]

        sampler = HamiltonianMonteCarlo(logp, n_samples=n_samples, n_warmup=n_warmup,
                                        n_chains=n_chains, step_size=0.5, n_leapfrog=10)
        sampler.sample(np.zeros(2), random_state=seed)
        return sampler

    def test_trace_plot_rhat_ess_match_advanced_mcmc(self):
        from stochpylib.advanced_mcmc import ESS, Rhat

        sampler = self._hmc_samples()
        fig = trace_plot(sampler)
        assert np.allclose(fig.data["rhat"], Rhat(sampler.get_chains()))
        assert np.allclose(fig.data["ess"], ESS(sampler.get_chains()))

    def test_trace_plot_rhat_near_one_for_converged_chains(self):
        sampler = self._hmc_samples(n_samples=800, n_warmup=400)
        fig = trace_plot(sampler)
        assert np.all(fig.data["rhat"] < 1.05)

    def test_trace_plot_accepts_raw_chains_array(self):
        chains = _RNG.standard_normal((4, 100, 2))
        fig = trace_plot(chains)
        assert fig.data["chains"].shape == (4, 100, 2)

    def test_posterior_plot_mean_near_zero_for_standard_normal(self):
        sampler = self._hmc_samples(n_samples=1000, n_warmup=400)
        fig = posterior_plot(sampler, hdi_prob=0.94)
        se = 1.0 / np.sqrt(4000)
        assert np.all(np.abs(fig.data["mean"]) < 5 * se)

    def test_posterior_plot_hdi_matches_brute_force(self):
        x = _RNG.standard_normal(5000)
        from stochpylib.bayesian._result import Posterior

        lo, hi = Posterior._hpd_from_samples(x, 0.94)
        fig = posterior_plot(x[:, None])
        assert fig.data["hdi"][0] == (lo, hi)

    def test_pair_plot_corr_matches_numpy(self):
        # An exact-recomputation check (does pair_plot report np.corrcoef of whatever it
        # actually plotted?), not a statistical convergence one -- a small chain suffices.
        # max_points must exceed the sample count, or pair_plot's default thinning
        # (max_points=2000) would compute corr on a random subset instead.
        sampler = self._hmc_samples(corr=-0.3, n_samples=150, n_warmup=100, n_chains=2)
        s = sampler.get_samples()
        fig = pair_plot(sampler, max_points=len(s))
        assert np.allclose(fig.data["corr"], np.corrcoef(s, rowvar=False))

    def test_pair_plot_needs_two_params(self):
        with pytest.raises(ValueError):
            pair_plot(_RNG.standard_normal((100, 1)))

    def _ols_fixture(self, n=60, seed=3):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 2))
        y = 1 + 2 * X[:, 0] - X[:, 1] + rng.standard_normal(n) * 0.5
        from stochpylib.statistics import linear_regression

        return X, y, linear_regression(X, y)

    def test_residual_plot_fitted_matches_model(self):
        X, y, res = self._ols_fixture()
        fig = residual_plot(res)
        assert np.allclose(fig.data["fitted"], res.fitted_)
        assert np.allclose(fig.data["residuals"], res.resid_)

    def test_residual_plot_qq_delegates(self):
        X, y, res = self._ols_fixture()
        fig = residual_plot(res, kind="qq")
        assert "sample" in fig.data

    def test_residual_plot_scale_location(self):
        X, y, res = self._ols_fixture()
        fig = residual_plot(res, kind="scale-location")
        assert np.all(fig.data["smooth"] >= 0)

    def test_residual_plot_bad_kind_raises(self):
        X, y, res = self._ols_fixture()
        with pytest.raises(ValueError):
            residual_plot(res, kind="bogus")

    def test_leverage_stats_match_statsmodels(self):
        pytest.importorskip("statsmodels")
        import statsmodels.api as sm

        X, y, res = self._ols_fixture(n=80)
        Xd = sm.add_constant(X)
        model = sm.OLS(y, Xd).fit()
        infl = model.get_influence()
        mine = _leverage_stats(X, model.resid, fit_intercept=True)
        assert np.allclose(mine["hat"], infl.hat_matrix_diag)
        assert np.allclose(mine["resid_internal"], infl.resid_studentized_internal)
        assert np.allclose(mine["resid_external"], infl.resid_studentized_external)
        assert np.allclose(mine["cooks"], infl.cooks_distance[0])

    def test_leverage_plot_flags_outlier(self):
        X, y, _ = self._ols_fixture(n=60)
        from stochpylib.statistics import linear_regression

        X2 = np.vstack([X, [5, 5]])
        y2 = np.append(y, 30)
        res2 = linear_regression(X2, y2)
        fig = leverage_plot(X2, res2)
        assert np.argmax(fig.data["cooks"]) == len(X2) - 1

    def test_influence_plot_labels_top_n(self):
        X, y, _ = self._ols_fixture(n=60)
        from stochpylib.statistics import linear_regression

        X2 = np.vstack([X, [5, 5]])
        y2 = np.append(y, 30)
        res2 = linear_regression(X2, y2)
        fig = influence_plot(X2, res2, n_labels=2)
        assert fig.data["cooks"].max() > 1.0

    def test_funnel_plot_pooled_is_inverse_variance_mean(self):
        effects = _RNG.normal(0.5, 0.2, 15)
        se = np.abs(_RNG.normal(0.3, 0.1, 15)) + 0.05
        fig = funnel_plot(effects, se)
        w = 1.0 / se ** 2
        expected = np.sum(w * effects) / np.sum(w)
        assert fig.data["pooled"] == pytest.approx(expected)

    def test_funnel_plot_egger_matches_ols(self):
        pytest.importorskip("statsmodels")
        import statsmodels.api as sm

        effects = _RNG.normal(0.5, 0.2, 20)
        se = np.abs(_RNG.normal(0.3, 0.1, 20)) + 0.05
        fig = funnel_plot(effects, se)
        precision = 1.0 / se
        snd = effects / se
        model = sm.OLS(snd, sm.add_constant(precision)).fit()
        assert fig.data["egger_intercept"] == pytest.approx(model.params[0])
        assert fig.data["egger_pvalue"] == pytest.approx(model.pvalues[0], rel=1e-6)

    def test_funnel_plot_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            funnel_plot([1, 2, 3], [1, 2])


# ============================================================================== multivariate


class TestMultivariateSubmodule:
    def test_plot_heatmap_data_roundtrip(self):
        Z = _RNG.random((4, 5))
        fig = plot_heatmap(Z)
        assert np.array_equal(fig.data["Z"], Z)

    def test_plot_correlation_matches_numpy_pearson(self):
        X = _RNG.standard_normal((100, 3))
        fig = plot_correlation(X)
        assert np.allclose(fig.data["corr"], np.corrcoef(X, rowvar=False))

    def test_plot_correlation_matches_scipy_spearman(self):
        X = _RNG.standard_normal((100, 3))
        fig = plot_correlation(X, method="spearman")
        oracle = stats.spearmanr(X).statistic
        assert np.allclose(fig.data["corr"], oracle, atol=1e-8)

    def test_plot_correlation_is_corr_passthrough(self):
        C = np.array([[1.0, 0.5], [0.5, 1.0]])
        fig = plot_correlation(C, is_corr=True)
        assert np.array_equal(fig.data["corr"], C)

    def test_plot_copula_scatter_kendall_tau_matches_theory(self):
        from stochpylib.copulas import ClaytonCopula

        theta = 3.0
        c = ClaytonCopula(theta=theta)
        fig = plot_copula(c, n=4000, random_state=0)
        u = fig.data["u"]
        tau = stats.kendalltau(u[:, 0], u[:, 1]).statistic
        expected = theta / (theta + 2.0)
        assert abs(tau - expected) < 0.05

    def test_plot_copula_density_integrates_near_one(self):
        from stochpylib.copulas import GaussianCopula

        mean = [0.0, 0.0]
        cov = [[1.0, 0.5], [0.5, 1.0]]
        data = _RNG.multivariate_normal(mean, cov, size=2000)
        c = GaussianCopula().fit(data)
        fig = plot_copula(c, kind="density", n_grid=80)
        du = 0.98 / 79
        assert np.sum(fig.data["density"]) * du * du == pytest.approx(1.0, abs=0.1)

    def test_plot_copula_raw_data_pseudo_observations(self):
        data = _RNG.standard_normal((50, 2))
        fig = plot_copula(data)
        assert np.all((fig.data["u"] > 0) & (fig.data["u"] < 1))

    def test_plot_scatter_matrix_shape(self):
        X = _RNG.standard_normal((60, 3))
        fig = plot_scatter_matrix(X)
        assert fig.data["corr"].shape == (3, 3)
        assert len(fig.axes) == 9

    def test_plot_biplot_matches_svd_variance_ratio(self):
        X = _RNG.standard_normal((100, 4)) @ np.diag([2.0, 1.0, 0.5, 0.2])
        Xc = X - X.mean(axis=0)
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)
        var = s ** 2 / (len(X) - 1)
        ratio = var / var.sum()
        fig = plot_biplot(X)
        assert fig.data["explained"] == pytest.approx(ratio[:2], rel=1e-6)

    def test_plot_dendrogram_matches_scipy_leaf_order(self):
        X = np.vstack([_RNG.standard_normal((5, 2)), _RNG.standard_normal((5, 2)) + 10])
        fig = plot_dendrogram(X)
        Z_scipy = cluster.hierarchy.linkage(X, method="ward")
        assert np.allclose(sorted(fig.data["Z"][:, 2]), sorted(Z_scipy[:, 2]))
        dendro = cluster.hierarchy.dendrogram(Z_scipy, no_plot=True)
        assert list(fig.data["leaf_order"]) == dendro["leaves"]

    def test_plot_dendrogram_from_precomputed_Z(self):
        X = _RNG.standard_normal((8, 2))
        Z = cluster.hierarchy.linkage(X, method="ward")
        fig = plot_dendrogram(Z)
        assert np.allclose(fig.data["Z"], Z)


# ============================================================================== special


class TestSpecialSubmodule:
    def test_plot_markov_chain_stationary_matches_power_iteration(self):
        P = np.array([[0.9, 0.1], [0.3, 0.7]])
        fig = plot_markov_chain(P)
        expected = np.array([0.75, 0.25])
        assert np.allclose(fig.data["stationary"], expected, atol=1e-6)
        Pn = np.linalg.matrix_power(P, 200)
        assert np.allclose(Pn[0], fig.data["stationary"], atol=1e-6)

    def test_plot_markov_chain_bad_rows_raise(self):
        with pytest.raises(ValueError):
            plot_markov_chain(np.array([[0.5, 0.6], [0.5, 0.5]]))

    def test_plot_markov_chain_self_loops_drawn(self):
        P = np.array([[0.9, 0.1], [0.3, 0.7]])
        fig = plot_markov_chain(P)
        svg = fig.to_svg()
        assert svg.count("<circle") >= 2 + 2  # 2 nodes + >=2 self-loop rings

    def test_plot_brownian_terminal_moments(self):
        fig = plot_brownian(n_paths=4000, n_steps=100, sigma=1.5, mu=0.2, T=2.0,
                            random_state=1)
        terminal = fig.data["paths"][:, -1]
        se_mean = 1.5 * np.sqrt(2.0) / np.sqrt(4000)
        assert abs(terminal.mean() - 0.4) < 4 * se_mean
        assert abs(terminal.var() - 1.5 ** 2 * 2.0) < 0.3 * 1.5 ** 2 * 2.0

    def test_plot_brownian_gbm_positive(self):
        fig = plot_brownian(n_paths=20, n_steps=50, geometric=True, random_state=2)
        assert np.all(fig.data["paths"] > 0)

    def test_plot_gp_band_matches_formula(self):
        from stochpylib.gaussian_processes import GPRegression, RBFKernel

        X = _RNG.uniform(0, 10, 15)[:, None]
        y = np.sin(X[:, 0]) + _RNG.standard_normal(15) * 0.1
        gp = GPRegression(kernel=RBFKernel(length_scale=1.5), noise=0.05).fit(X, y)
        fig = plot_gp(gp, level=0.95)
        mean, std = gp.predict(fig.data["x"][:, None], return_std=True)
        z = stats.norm.ppf(0.975)
        assert np.allclose(fig.data["upper"] - fig.data["mean"], z * std, atol=1e-8)
        assert np.allclose(fig.data["mean"], mean, atol=1e-8)

    def test_plot_gp_samples_mean_reverts_to_posterior(self):
        from stochpylib.gaussian_processes import GPRegression, RBFKernel

        X = _RNG.uniform(0, 10, 15)[:, None]
        y = np.sin(X[:, 0]) + _RNG.standard_normal(15) * 0.1
        gp = GPRegression(kernel=RBFKernel(length_scale=1.5), noise=0.05).fit(X, y)
        fig = plot_gp(gp, n_samples=500, random_state=0)
        assert np.max(np.abs(fig.data["samples"].mean(axis=1) - fig.data["mean"])) < 0.3

    def test_plot_survival_km_matches_lifelines(self):
        pytest.importorskip("lifelines")
        from lifelines import KaplanMeierFitter

        durations = _RNG.exponential(10, 60)
        events = (_RNG.random(60) > 0.3).astype(float)
        fig = plot_survival_km(durations, events)
        kmf = KaplanMeierFitter().fit(durations, events)
        t = fig.data["KM"]["time"]
        s = fig.data["KM"]["survival"]
        oracle = kmf.survival_function_at_times(t).values
        assert np.max(np.abs(s - oracle)) < 1e-9

    def test_plot_survival_km_groups_runs_logrank(self):
        durations = _RNG.exponential(10, 60)
        events = (_RNG.random(60) > 0.3).astype(float)
        groups = np.where(_RNG.random(60) > 0.5, "A", "B")
        fig = plot_survival_km(durations, events, groups=groups)
        assert set(fig.data) == {"A", "B"}
        assert "log-rank p=" in fig.ax.title

    def test_plot_variogram_experimental_matches_brute_force(self):
        from stochpylib.spatial_statistics import ExperimentalVariogram

        coords = _RNG.uniform(0, 10, size=(60, 2))
        values = _RNG.standard_normal(60)
        fig = plot_variogram(None, coords=coords, values=values, bins=8)
        ev = ExperimentalVariogram(bins=8).fit(coords, values)
        assert np.allclose(fig.data["lags"], ev.lags_)
        assert np.allclose(fig.data["gamma"], ev.gamma_)

    def test_plot_variogram_with_fitted_model(self):
        from stochpylib.spatial_statistics import ExperimentalVariogram, VariogramFitting

        coords = _RNG.uniform(0, 10, size=(150, 2))
        values = _RNG.standard_normal(150)
        ev = ExperimentalVariogram(bins=10).fit(coords, values)
        vf = VariogramFitting(model="exponential").fit(ev)
        fig = plot_variogram(ev, model=vf)
        assert np.allclose(fig.data["model_gamma"],
                           [vf.variogram_(h) for h in fig.data["model_h"]])

    def test_plot_eigenvalues_goe_integrates_to_one(self):
        from stochpylib.random_matrix import GOE

        goe = GOE(n=150)
        fig = plot_eigenvalues(goe, random_state=1, bins=40)
        widths = np.diff(fig.data["centers"])[0]
        assert np.sum(fig.data["empirical"]) * widths == pytest.approx(1.0, rel=0.05)

    def test_plot_eigenvalues_matches_ensemble_histogram_vs_density(self):
        from stochpylib.random_matrix import GOE

        goe = GOE(n=100)
        eig = goe.eigenvalues(random_state=2)
        fig = plot_eigenvalues(goe, random_state=2, bins=30)
        centers, empirical, theoretical = goe.limit_law().histogram_vs_density(
            goe.normalize(eig), bins=30)
        assert np.allclose(fig.data["centers"], centers)
        assert np.allclose(fig.data["theoretical"], theoretical)

    def test_plot_eigenvalues_complex_scatter(self):
        from stochpylib.random_matrix import MuresanMatrix

        m = MuresanMatrix(n=50)
        eig = m.eigenvalues(random_state=3)
        fig = plot_eigenvalues(eig)
        assert "eigenvalues" in fig.data

    def test_plot_eigenvalues_no_law_no_ensemble_raises(self):
        with pytest.raises(ValueError):
            plot_eigenvalues(_RNG.standard_normal(50))


# ============================================================================== cross-module


class TestCrossModule:
    def test_interaction_plot_to_figure(self):
        from stochpylib.experimental_design import FullFactorial, InteractionPlot

        design = FullFactorial(n_factors=3).generate().points
        y = (design[:, 0] * 2 - design[:, 1] * 1.5 + design[:, 0] * design[:, 1]
             + 0.01 * _RNG.standard_normal(len(design)))
        ip = InteractionPlot(factors=(0, 1)).fit(design, y)
        fig = ip.to_figure()
        _parse(fig)
        assert len(fig.ax.artists) == len(ip.lines_)

    def test_normal_plot_to_figure_flags_active_effects(self):
        from stochpylib.experimental_design import FullFactorial, NormalPlot

        design = FullFactorial(n_factors=3).generate().points
        y = (design[:, 0] * 2 - design[:, 1] * 1.5 + design[:, 2] * 0.3
             + design[:, 0] * design[:, 1] * 1.0 + 0.05 * _RNG.standard_normal(len(design)))
        npt = NormalPlot(half=True).fit(design, y)
        fig = npt.to_figure()
        _parse(fig)
        assert set(npt.active_) >= {"A", "B"}

    def test_plot_variogram_renders_spatial_function(self):
        from stochpylib.spatial_statistics import PoissonPointProcess, RipleyK

        # intensity is per unit AREA: keep the point count small since RipleyK's pairwise
        # distances are O(n^2), repeated once per Monte Carlo simulation.
        window = [(0.0, 10.0), (0.0, 10.0)]
        pts = PoissonPointProcess(intensity=2.0, window=window).sample(random_state=3)
        sf = RipleyK(pts, window=window, n_simulations=10, random_state=4)
        fig = plot_variogram(sf)
        assert np.allclose(fig.data["estimate"], sf.estimate)


# ============================================================================== hygiene


class TestHygiene:
    def test_public_surface_is_unique(self):
        assert len(viz.__all__) == len(set(viz.__all__)) == 37

    def test_spec_names_are_subset_of_all(self):
        import json
        import pathlib

        spec = json.loads((pathlib.Path(__file__).parents[1] / "library" /
                           "_spec_names.json").read_text())
        assert set(spec["viz"]) <= set(viz.__all__)

    def test_every_plot_function_takes_ax_and_returns_figure(self):
        import inspect

        for name in viz.__all__:
            if name in ("Figure", "Axes"):
                continue
            fn = getattr(viz, name)
            sig = inspect.signature(fn)
            assert "ax" in sig.parameters, name

    def test_library_code_never_imports_scipy_stats(self):
        import ast
        import pathlib

        pkg_dir = pathlib.Path(viz.__file__).parent
        seen = 0
        for path in pkg_dir.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith("scipy.stats"), path
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("scipy.stats"), path
            seen += 1
        assert seen >= 8

    def test_matplotlib_only_imported_inside_functions_of_mpl_module(self):
        import ast
        import pathlib

        pkg_dir = pathlib.Path(viz.__file__).parent
        for path in pkg_dir.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                is_mpl_import = (
                    (isinstance(node, ast.ImportFrom) and node.module
                     and node.module.startswith("matplotlib"))
                    or (isinstance(node, ast.Import)
                        and any(a.name.startswith("matplotlib") for a in node.names))
                )
                if not is_mpl_import:
                    continue
                assert path.name == "_mpl.py", f"matplotlib imported outside _mpl.py: {path}"

    def test_no_module_level_matplotlib_import_in_mpl_py(self):
        import ast
        import pathlib

        pkg_dir = pathlib.Path(viz.__file__).parent
        tree = ast.parse((pkg_dir / "_mpl.py").read_text(encoding="utf-8"))
        for node in tree.body:  # only top-level statements
            if isinstance(node, ast.ImportFrom) and node.module and \
                    node.module.startswith("matplotlib"):
                pytest.fail("matplotlib imported at module level in _mpl.py")
            if isinstance(node, ast.Import) and any(
                    a.name.startswith("matplotlib") for a in node.names):
                pytest.fail("matplotlib imported at module level in _mpl.py")

    def test_base_classes_and_extras_present(self):
        assert issubclass(type(Figure()), Figure)
        assert issubclass(Axes, Axes)

    def test_quickstart_example_runs(self):
        from stochpylib.distributions import Gamma

        g = Gamma(shape=3.0, scale=2.0)
        data = g.rvs(500, random_state=1)
        fig = plot_histogram(data, dist=g, kde=True)
        svg = fig.to_svg()
        assert svg.startswith("<svg")
        assert "</svg>" in svg
