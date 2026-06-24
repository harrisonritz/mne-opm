"""Tests for auto_ica.py — unified per-IC scoring, PCA-whitened GESD, figures.

These tests focus on the computational methods that can be validated with
synthetic data, without needing real BIDS or ICA files.
"""

from __future__ import annotations

from types import SimpleNamespace

import mne
import numpy as np
import pytest

from custom.preprocessing.bad_ICs import BadICAnalysis, PCAGesdResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ica_cfg():
    """Config for auto ICA tests.

    Defaults to diagnostic-only scores so the common tests stay deterministic;
    individual tests override ``_ica_metrics`` to exercise targeted scores.
    """
    return SimpleNamespace(
        _auto_ica=True,
        spatial_filter="ica",
        ch_types=["mag"],
        deriv_root="/tmp/deriv",
        subjects=["001"],
        sessions=["01"],
        task="restingstate",
        # Single unified selection list (diagnostics only here).
        _ica_metrics=list(BadICAnalysis.AVAILABLE_ICA_METRICS),
        _auto_ica_overlay=False,  # skip figure PNGs in unit tests
    )


@pytest.fixture()
def synthetic_ica_and_raw():
    """Create a synthetic mag-only ICA + Raw pair for testing diagnostics.

    Returns (ica, raw) where ica has been fit on raw.
    """
    rng = np.random.RandomState(42)
    n_ch = 20
    sfreq = 300.0
    n_times = int(sfreq * 5)  # 5 seconds

    info = mne.create_info(
        [f"MEG{i:03d}" for i in range(n_ch)],
        sfreq, ["mag"] * n_ch,
    )
    data = rng.randn(n_ch, n_times) * 1e-13
    raw = mne.io.RawArray(data, info)

    ica = mne.preprocessing.ICA(
        n_components=10, method="fastica", random_state=42, max_iter=100
    )
    ica.fit(raw)

    return ica, raw


@pytest.fixture()
def synthetic_ica_and_raw_full():
    """Synthetic Raw with mag + ref_meg + EOG channels, ICA fit on mag.

    Exercises the targeted-score paths (EOG / ECG / reference).
    """
    rng = np.random.RandomState(7)
    n_mag, n_ref, n_eog = 20, 4, 3
    sfreq = 300.0
    n_times = int(sfreq * 6)

    ch_names = (
        [f"MEG{i:03d}" for i in range(n_mag)]
        + [f"REF{i:03d}" for i in range(n_ref)]
        + [f"eye_nmf{i + 1}" for i in range(n_eog)]
    )
    ch_types = ["mag"] * n_mag + ["ref_meg"] * n_ref + ["eog"] * n_eog
    info = mne.create_info(ch_names, sfreq, ch_types)
    data = rng.randn(len(ch_names), n_times) * 1e-13
    raw = mne.io.RawArray(data, info)
    with raw.info._unlock():
        raw.info["line_freq"] = 60.0

    ica = mne.preprocessing.ICA(
        n_components=10, method="fastica", random_state=0, max_iter=200
    )
    ica.fit(raw, picks="mag")

    return ica, raw


# ---------------------------------------------------------------------------
# is_enabled
# ---------------------------------------------------------------------------

class TestAutoICAIsEnabled:
    def test_enabled_when_both_set(self, ica_cfg):
        assert BadICAnalysis(ica_cfg).is_enabled() is True

    def test_disabled_without_flag(self, ica_cfg):
        ica_cfg._auto_ica = False
        assert BadICAnalysis(ica_cfg).is_enabled() is False

    def test_disabled_without_spatial_filter(self, ica_cfg):
        ica_cfg.spatial_filter = None
        assert BadICAnalysis(ica_cfg).is_enabled() is False

    def test_disabled_with_wrong_spatial_filter(self, ica_cfg):
        ica_cfg.spatial_filter = "ssp"
        assert BadICAnalysis(ica_cfg).is_enabled() is False


# ---------------------------------------------------------------------------
# _ica_component_diagnostics
# ---------------------------------------------------------------------------

class TestICAComponentDiagnostics:
    """Test that diagnostic metrics are computed correctly."""

    def test_returns_expected_keys(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        analysis = BadICAnalysis(ica_cfg)
        diag = analysis._ica_component_diagnostics(ica, raw)

        expected_keys = {
            "sensor_var",
            "mean_abs_gradient",
            "hf_ratio",
            "line_ratio",
            "source_kurtosis",
            "autocorr_1lag",
            "spectral_slope",
            "spatial_kurtosis",
            "spectral_deriv_kurtosis",
            "spectral_resid_kurtosis",
        }
        assert set(diag.keys()) == expected_keys

    def test_metric_shapes(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        analysis = BadICAnalysis(ica_cfg)
        diag = analysis._ica_component_diagnostics(ica, raw)

        n_comps = ica.n_components_
        for key, vals in diag.items():
            assert len(vals) == n_comps, (
                f"Metric '{key}' has length {len(vals)}, expected {n_comps}"
            )

    def test_sensor_var_positive(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        analysis = BadICAnalysis(ica_cfg)
        diag = analysis._ica_component_diagnostics(ica, raw)
        assert np.all(diag["sensor_var"] >= 0)

    def test_autocorr_in_range(self, ica_cfg, synthetic_ica_and_raw):
        """Autocorrelation should be in [-1, 1]."""
        ica, raw = synthetic_ica_and_raw
        analysis = BadICAnalysis(ica_cfg)
        diag = analysis._ica_component_diagnostics(ica, raw)
        assert np.all(diag["autocorr_1lag"] >= -1.0)
        assert np.all(diag["autocorr_1lag"] <= 1.0)

    def test_hf_ratio_positive(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        analysis = BadICAnalysis(ica_cfg)
        diag = analysis._ica_component_diagnostics(ica, raw)
        assert np.all(diag["hf_ratio"] > 0)

    def test_line_ratio_in_unit_range(self, ica_cfg, synthetic_ica_and_raw):
        """line_ratio is a power fraction, so it must lie in [0, 1]."""
        ica, raw = synthetic_ica_and_raw
        analysis = BadICAnalysis(ica_cfg)
        diag = analysis._ica_component_diagnostics(ica, raw)
        assert np.all(diag["line_ratio"] >= 0.0)
        assert np.all(diag["line_ratio"] <= 1.0)


# ---------------------------------------------------------------------------
# _prepare_metrics_for_gesd
# ---------------------------------------------------------------------------

class TestPrepareMetricsForGESD:
    """Test metric transformation and direction labeling."""

    @pytest.fixture()
    def sample_diagnostics(self):
        """Create sample diagnostics dictionary."""
        rng = np.random.RandomState(42)
        n = 15  # number of components
        return {
            "sensor_var": np.abs(rng.randn(n)) * 1e-25,
            "mean_abs_gradient": np.abs(rng.randn(n)) * 1e-13,
            "hf_ratio": np.abs(rng.randn(n)) * 1e-12,
            "line_ratio": np.abs(rng.randn(n)) * 1e-2,
            "source_kurtosis": rng.randn(n) * 3,
            "autocorr_1lag": np.clip(rng.randn(n) * 0.3 + 0.5, -0.99, 0.99),
            "spectral_slope": rng.randn(n) * 0.5 - 1.5,
            "spatial_kurtosis": rng.randn(n) * 2,
            "spectral_deriv_kurtosis": rng.randn(n) * 2,
            "spectral_resid_kurtosis": rng.randn(n) * 2,
        }

    def test_returns_list_of_tuples(self, ica_cfg, sample_diagnostics):
        analysis = BadICAnalysis(ica_cfg)
        metrics = analysis._prepare_metrics_for_gesd(sample_diagnostics)
        assert isinstance(metrics, list)
        for item in metrics:
            assert len(item) == 3  # (name, values, direction)
            name, vals, side = item
            assert isinstance(name, str)
            assert isinstance(vals, np.ndarray)
            assert side in (-1, 0, 1)

    def test_nine_metrics_produced(self, ica_cfg, sample_diagnostics):
        """With no explicit names, all diagnostic metrics are produced."""
        analysis = BadICAnalysis(ica_cfg)
        metrics = analysis._prepare_metrics_for_gesd(sample_diagnostics)
        assert len(metrics) == 9

    def test_metric_names(self, ica_cfg, sample_diagnostics):
        analysis = BadICAnalysis(ica_cfg)
        metrics = analysis._prepare_metrics_for_gesd(sample_diagnostics)
        names = [m[0] for m in metrics]
        for expected in BadICAnalysis.AVAILABLE_ICA_METRICS:
            assert expected in names

    def test_directions(self, ica_cfg, sample_diagnostics):
        analysis = BadICAnalysis(ica_cfg)
        metrics = analysis._prepare_metrics_for_gesd(sample_diagnostics)
        direction_map = {name: side for name, _, side in metrics}

        assert direction_map["log_hf_ratio"] == 1
        assert direction_map["log_line_ratio"] == 1
        assert direction_map["temporal_kurtosis_sqrt"] == 1
        assert direction_map["autocorr_fisher_z"] == -1  # low = bad
        assert direction_map["spectral_slope"] == 1
        assert direction_map["spatial_kurtosis_sqrt"] == 1
        assert direction_map["spectral_deriv_kurtosis_sqrt"] == 1
        assert direction_map["spectral_resid_kurtosis_sqrt"] == 1
        assert direction_map["log_mean_abs_gradient"] == 1

    def test_no_nans_in_output(self, ica_cfg, sample_diagnostics):
        analysis = BadICAnalysis(ica_cfg)
        metrics = analysis._prepare_metrics_for_gesd(sample_diagnostics)
        for name, vals, _ in metrics:
            assert not np.any(np.isnan(vals)), f"NaN found in {name}"

    def test_names_param_overrides_selection(self, ica_cfg, sample_diagnostics):
        """Passing ``names`` selects exactly those diagnostic metrics."""
        analysis = BadICAnalysis(ica_cfg)
        metrics = analysis._prepare_metrics_for_gesd(
            sample_diagnostics, names={"log_hf_ratio", "spectral_slope"}
        )
        names = {m[0] for m in metrics}
        assert names == {"log_hf_ratio", "spectral_slope"}

    def test_unknown_metric_name_raises(self, ica_cfg, sample_diagnostics):
        analysis = BadICAnalysis(ica_cfg)
        with pytest.raises(ValueError):
            analysis._prepare_metrics_for_gesd(
                sample_diagnostics, names={"not_a_metric"}
            )


# ---------------------------------------------------------------------------
# _resolve_gesd_scores  (single unified _ica_metrics list)
# ---------------------------------------------------------------------------

class TestResolveGesdScores:
    def test_explicit_list_used_verbatim(self, ica_cfg):
        ica_cfg._ica_metrics = ["log_hf_ratio", "eog", "reference"]
        sel = BadICAnalysis(ica_cfg)._resolve_gesd_scores()
        assert sel == {"log_hf_ratio", "eog", "reference"}

    def test_unknown_token_raises(self, ica_cfg):
        ica_cfg._ica_metrics = ["not_a_score"]
        with pytest.raises(ValueError):
            BadICAnalysis(ica_cfg)._resolve_gesd_scores()

    def test_none_selects_all_available(self, ica_cfg):
        ica_cfg._ica_metrics = None
        sel = BadICAnalysis(ica_cfg)._resolve_gesd_scores()
        assert sel == set(BadICAnalysis.AVAILABLE_ICA_SCORES)

    def test_empty_list_selects_nothing(self, ica_cfg):
        ica_cfg._ica_metrics = []
        assert BadICAnalysis(ica_cfg)._resolve_gesd_scores() == set()

    def test_targeted_tokens_allowed(self, ica_cfg):
        ica_cfg._ica_metrics = ["eog", "ecg", "reference", "corrmap_eog"]
        sel = BadICAnalysis(ica_cfg)._resolve_gesd_scores()
        assert sel == {"eog", "ecg", "reference", "corrmap_eog"}


# ---------------------------------------------------------------------------
# Score-reduction / sanitization helpers
# ---------------------------------------------------------------------------

class TestScoreHelpers:
    def test_reduce_single_array_takes_abs(self):
        scores = np.array([-2.0, 1.0, -0.5])
        out = BadICAnalysis._reduce_multichannel_scores(scores)
        np.testing.assert_allclose(out, [2.0, 1.0, 0.5])

    def test_reduce_multichannel_max_abs(self):
        # list of 2 arrays (2 channels) x 3 components
        scores = [np.array([-3.0, 0.1, 0.2]), np.array([1.0, -2.0, 0.0])]
        out = BadICAnalysis._reduce_multichannel_scores(scores)
        np.testing.assert_allclose(out, [3.0, 2.0, 0.2])

    def test_pearson_cols_self_is_one(self, synthetic_ica_and_raw):
        ica, _ = synthetic_ica_and_raw
        maps = ica.get_components()
        corr = BadICAnalysis._pearson_cols(maps, maps[:, 0])
        assert corr[0] == pytest.approx(1.0, abs=1e-6)
        assert abs(corr[0]) == pytest.approx(np.abs(corr).max())


# ---------------------------------------------------------------------------
# Targeted score methods
# ---------------------------------------------------------------------------

class TestTargetedScores:
    def test_score_eog_none_without_eog_channel(
        self, ica_cfg, synthetic_ica_and_raw
    ):
        """mag-only raw has no EOG channel -> graceful None."""
        ica, raw = synthetic_ica_and_raw
        out = BadICAnalysis(ica_cfg)._score_eog(ica, raw)
        assert out is None

    def test_score_eog_returns_vector_with_eog(
        self, ica_cfg, synthetic_ica_and_raw_full
    ):
        ica, raw = synthetic_ica_and_raw_full
        out = BadICAnalysis(ica_cfg)._score_eog(ica, raw)
        assert out is not None
        assert out.shape == (ica.n_components_,)
        assert np.isfinite(out).all()
        assert np.all(out >= 0)  # |correlation|

    def test_score_ecg_no_raise(self, ica_cfg, synthetic_ica_and_raw_full):
        """ECG is synthesized from mags; returns a finite vector or None."""
        ica, raw = synthetic_ica_and_raw_full
        out = BadICAnalysis(ica_cfg)._score_ecg(ica, raw)
        assert out is None or (
            out.shape == (ica.n_components_,) and np.isfinite(out).all()
        )

    def test_score_reference_no_mutation(
        self, ica_cfg, synthetic_ica_and_raw_full
    ):
        ica, raw = synthetic_ica_and_raw_full
        n_before = len(raw.ch_names)
        out = BadICAnalysis(ica_cfg)._score_reference(ica, raw)
        # raw must not gain REF_* channels
        assert len(raw.ch_names) == n_before
        assert not any(c.startswith("REF_") for c in raw.ch_names)
        assert out is None or (
            out.shape == (ica.n_components_,) and np.isfinite(out).all()
        )

    def test_score_reference_none_without_ref(
        self, ica_cfg, synthetic_ica_and_raw
    ):
        """mag-only raw has no ref_meg -> graceful None."""
        ica, raw = synthetic_ica_and_raw
        out = BadICAnalysis(ica_cfg)._score_reference(ica, raw)
        assert out is None


# ---------------------------------------------------------------------------
# _score_corrmap (direct template correlation)
# ---------------------------------------------------------------------------

class TestScoreCorrmap:
    def test_none_when_no_template_dir(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        assert BadICAnalysis(ica_cfg)._score_corrmap(ica, raw) is None

    def test_none_when_dir_missing(self, ica_cfg, synthetic_ica_and_raw, tmp_path):
        ica, raw = synthetic_ica_and_raw
        ica_cfg._corrmap_template_dir = str(tmp_path / "nope")
        assert BadICAnalysis(ica_cfg)._score_corrmap(ica, raw) is None

    def test_none_when_template_files_missing(
        self, ica_cfg, synthetic_ica_and_raw, tmp_path
    ):
        ica, raw = synthetic_ica_and_raw
        ica_cfg._corrmap_template_dir = str(tmp_path)  # dir exists, no .npy
        ica_cfg._n_eog_templates = 3
        ica_cfg._n_ecg_templates = 0
        assert BadICAnalysis(ica_cfg)._score_corrmap(ica, raw) is None

    def test_detects_self_topography(
        self, ica_cfg, synthetic_ica_and_raw, tmp_path
    ):
        """Template == component 0's topography -> score[0] ~ 1.0."""
        ica, raw = synthetic_ica_and_raw
        comps = ica.get_components()
        target = 0
        np.save(str(tmp_path / "eog_channel_names.npy"), np.array(ica.ch_names))
        np.save(str(tmp_path / "eog_templates.npy"), comps[:, target][:, None])

        ica_cfg._corrmap_template_dir = str(tmp_path)
        ica_cfg._n_eog_templates = 1
        ica_cfg._n_ecg_templates = 0

        out = BadICAnalysis(ica_cfg)._score_corrmap(ica, raw)
        assert out is not None and "corrmap_eog" in out
        s = out["corrmap_eog"]
        assert s.shape == (ica.n_components_,)
        assert s[target] == pytest.approx(1.0, abs=1e-6)
        assert s[target] == pytest.approx(s.max())

    def test_channel_subset_alignment(
        self, ica_cfg, synthetic_ica_and_raw, tmp_path
    ):
        """Reference with extra channels aligns to the ICA's channel subset."""
        ica, raw = synthetic_ica_and_raw
        comps = ica.get_components()
        target_topo = comps[:, 0]
        extra = [f"EXTRA{i:03d}" for i in range(5)]
        ref_channels = np.array(list(ica.ch_names) + extra)
        rng = np.random.RandomState(1)
        full_template = np.concatenate([target_topo, rng.randn(5) * 0.01])
        np.save(str(tmp_path / "eog_channel_names.npy"), ref_channels)
        np.save(str(tmp_path / "eog_templates.npy"), full_template[:, None])

        ica_cfg._corrmap_template_dir = str(tmp_path)
        ica_cfg._n_eog_templates = 1
        ica_cfg._n_ecg_templates = 0

        out = BadICAnalysis(ica_cfg)._score_corrmap(ica, raw)
        assert out is not None
        assert out["corrmap_eog"][0] == pytest.approx(1.0, abs=1e-6)

    def test_n_templates_limits_columns(
        self, ica_cfg, synthetic_ica_and_raw, tmp_path
    ):
        """_n_eog_templates=1 uses only the first template column."""
        ica, raw = synthetic_ica_and_raw
        comps = ica.get_components()
        np.save(str(tmp_path / "eog_channel_names.npy"), np.array(ica.ch_names))
        np.save(
            str(tmp_path / "eog_templates.npy"),
            np.column_stack([comps[:, 0], comps[:, 1]]),
        )
        ica_cfg._corrmap_template_dir = str(tmp_path)
        ica_cfg._n_eog_templates = 1
        ica_cfg._n_ecg_templates = 0

        out = BadICAnalysis(ica_cfg)._score_corrmap(ica, raw)
        s = out["corrmap_eog"]
        assert s[0] == pytest.approx(1.0, abs=1e-6)  # from column 0
        assert s[1] < 0.99  # column 1 not used


# ---------------------------------------------------------------------------
# _compute_ic_scores
# ---------------------------------------------------------------------------

class TestComputeICScores:
    def test_filters_to_selected(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        ica_cfg._ica_metrics = ["log_hf_ratio", "spectral_slope"]
        specs = BadICAnalysis(ica_cfg)._compute_ic_scores(ica, raw)
        names = {s.name for s in specs}
        assert names == {"log_hf_ratio", "spectral_slope"}
        for s in specs:
            assert s.values.shape == (ica.n_components_,)
            assert np.isfinite(s.values).all()

    def test_empty_when_nothing_selected(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        ica_cfg._ica_metrics = []
        assert BadICAnalysis(ica_cfg)._compute_ic_scores(ica, raw) == []

    def test_targeted_scores_fisher_z_transformed(
        self, ica_cfg, synthetic_ica_and_raw
    ):
        """Targeted correlation scores are atanh-transformed, side=+1."""
        from custom.preprocessing.pca_gesd import fisher_z

        ica, raw = synthetic_ica_and_raw
        ica_cfg._ica_metrics = ["eog"]
        analysis = BadICAnalysis(ica_cfg)
        corr = np.linspace(0.1, 0.9, ica.n_components_)
        analysis._score_eog = lambda *a, **k: corr
        specs = analysis._compute_ic_scores(ica, raw)
        eog = next(s for s in specs if s.name == "eog")
        np.testing.assert_allclose(eog.values, fisher_z(corr))
        assert eog.side == 1


# ---------------------------------------------------------------------------
# _run_unified_gesd
# ---------------------------------------------------------------------------

class TestRunUnifiedGesd:
    def _specs(self, analysis, ica, raw):
        return analysis._compute_ic_scores(ica, raw)

    def test_does_not_crash(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        analysis = BadICAnalysis(ica_cfg)
        specs = self._specs(analysis, ica, raw)
        ica2, gesd = analysis._run_unified_gesd(ica, raw, specs)
        assert isinstance(ica2, mne.preprocessing.ICA)
        assert isinstance(gesd, PCAGesdResult)
        assert gesd.n_pcs >= 1

    def test_exclude_sorted_unique_valid(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        analysis = BadICAnalysis(ica_cfg)
        specs = self._specs(analysis, ica, raw)
        ica2, _ = analysis._run_unified_gesd(ica, raw, specs)
        assert ica2.exclude == sorted(set(ica2.exclude))
        for idx in ica2.exclude:
            assert 0 <= idx < ica.n_components_

    def test_single_score_one_pc(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        ica_cfg._ica_metrics = ["log_hf_ratio"]
        analysis = BadICAnalysis(ica_cfg)
        specs = self._specs(analysis, ica, raw)
        assert len(specs) == 1
        ica2, gesd = analysis._run_unified_gesd(ica, raw, specs)
        assert gesd.n_pcs == 1
        assert gesd.alpha_per_pc == pytest.approx(gesd.alpha)
        for idx in ica2.exclude:
            assert 0 <= idx < ica.n_components_

    def test_skips_when_too_few_components(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        ica.exclude = list(range(ica.n_components_ - 3))
        n_before = len(ica.exclude)
        analysis = BadICAnalysis(ica_cfg)
        specs = self._specs(analysis, ica, raw)
        ica2, gesd = analysis._run_unified_gesd(ica, raw, specs)
        assert len(ica2.exclude) == n_before
        assert gesd.n_pcs == 0

    def test_empty_specs_no_op(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        analysis = BadICAnalysis(ica_cfg)
        ica2, gesd = analysis._run_unified_gesd(ica, raw, [])
        assert gesd.n_pcs == 0
        assert ica2.exclude == []

    def test_result_shapes(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        analysis = BadICAnalysis(ica_cfg)
        specs = self._specs(analysis, ica, raw)
        _, gesd = analysis._run_unified_gesd(ica, raw, specs)
        k = len(specs)
        n = ica.n_components_
        assert gesd.M.shape == (k, n)
        assert gesd.M_std.shape == (k, n)
        assert gesd.loadings.shape == (k, gesd.n_pcs)
        assert gesd.eigenscores.shape == (gesd.n_pcs, n)
        assert len(gesd.per_pc_flagged) == gesd.n_pcs
        assert gesd.flagged.shape == (n,)


# ---------------------------------------------------------------------------
# _build_components_tsv
# ---------------------------------------------------------------------------

class TestBuildComponentsTSV:
    def test_columns(self, ica_cfg, synthetic_ica_and_raw, tmp_path):
        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        ica_cfg.deriv_root = str(tmp_path)
        ica_cfg._auto_ica_overlay = False
        analysis = BadICAnalysis(ica_cfg)
        analysis._auto_ica(ica, raw)
        df = analysis._build_components_tsv(ica)

        cols = set(df.columns)
        assert {
            "component",
            "type",
            "description",
            "status",
            "status_description",
            "method_gesd",
            "method_pipeline_icalabel",
        } <= cols
        assert any(c.startswith("score_") for c in cols)
        assert any(c.startswith("gesd_score_PC") for c in cols)
        assert "gesd_pc_loadings" in cols
        assert "gesd_var_explained" in cols
        # Removed legacy columns
        assert "method_reference" not in cols
        assert "method_corrmap_eog" not in cols
        assert "method_corrmap_ecg" not in cols
        assert "method_pipeline_ecg" not in cols
        assert "method_pipeline_eog" not in cols
        assert not any(c.startswith("metric_") for c in cols)
        assert len(df) == ica.n_components_


# ---------------------------------------------------------------------------
# _auto_ica integration
# ---------------------------------------------------------------------------

class TestAutoICAIntegration:
    def test_runs_and_sorts(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        analysis = BadICAnalysis(ica_cfg)
        result = analysis._auto_ica(ica, raw)
        assert isinstance(result, mne.preprocessing.ICA)
        assert result.exclude == sorted(set(result.exclude))

    def test_no_scores_no_exclude(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        ica_cfg._ica_metrics = []
        analysis = BadICAnalysis(ica_cfg)
        result = analysis._auto_ica(ica, raw)
        assert result.exclude == []


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

class TestMakeOverlayEvoked:
    def test_builds_mag_evoked_from_raw(
        self, ica_cfg, synthetic_ica_and_raw, tmp_path
    ):
        _, raw = synthetic_ica_and_raw
        ica_cfg.deriv_root = str(tmp_path)  # no icafit file -> fallback
        ev = BadICAnalysis(ica_cfg)._make_overlay_evoked(raw)
        assert isinstance(ev, mne.Evoked)
        assert set(ev.get_channel_types()) == {"mag"}


class TestICAOverlay:
    """Tests for the report-style ICA overlay PNGs written during _auto_ica."""

    def test_overlay_basepath_in_ica_folder(self, ica_cfg, tmp_path):
        ica_cfg.deriv_root = str(tmp_path)
        out_dir, basename = BadICAnalysis(ica_cfg)._overlay_basepath()
        assert out_dir.name == "ICA"
        assert out_dir.parent.name == "meg"
        assert "sub-001" in str(out_dir) and "ses-01" in str(out_dir)
        assert basename == "sub-001_ses-01_task-restingstate_proc-ica"

    def test_overlay_disabled_writes_nothing(
        self, ica_cfg, synthetic_ica_and_raw, tmp_path
    ):
        ica, raw = synthetic_ica_and_raw
        ica_cfg.deriv_root = str(tmp_path)
        ica_cfg._auto_ica_overlay = False
        analysis = BadICAnalysis(ica_cfg)
        analysis._save_ica_overlay(ica, None, 0, "pre-custom")
        assert list(tmp_path.rglob("*.png")) == []

    def test_overlay_writes_named_png(
        self, ica_cfg, synthetic_ica_and_raw, tmp_path
    ):
        import matplotlib
        matplotlib.use("Agg")

        ica, raw = synthetic_ica_and_raw
        ica.exclude = [0]
        ica_cfg.deriv_root = str(tmp_path)
        ica_cfg._auto_ica_overlay = True
        analysis = BadICAnalysis(ica_cfg)
        evoked = analysis._make_overlay_evoked(raw)
        analysis._save_ica_overlay(ica, evoked, 1, "gesd-PC1")

        expected = (
            tmp_path / "sub-001" / "ses-01" / "meg" / "ICA"
            / "sub-001_ses-01_task-restingstate_proc-ica_icaOverlay_01_gesd-PC1.png"
        )
        assert expected.exists()

    def test_overlay_steps_during_auto_ica(
        self, ica_cfg, synthetic_ica_and_raw, tmp_path
    ):
        import matplotlib
        matplotlib.use("Agg")

        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        ica_cfg.deriv_root = str(tmp_path)
        ica_cfg._auto_ica_overlay = True
        analysis = BadICAnalysis(ica_cfg)
        analysis._auto_ica(ica, raw)

        pngs = sorted(p.name for p in tmp_path.rglob("*icaOverlay*.png"))
        assert any("_00_pre-custom.png" in p for p in pngs)
        assert any("gesd-PC" in p for p in pngs)
        assert any("final.png" in p for p in pngs)
        # all overlays land in the meg/ICA folder
        for p in tmp_path.rglob("*icaOverlay*.png"):
            assert p.parent.name == "ICA"


# ---------------------------------------------------------------------------
# Diagnostic figures
# ---------------------------------------------------------------------------

class TestGesdFigures:
    def test_figures_written(self, ica_cfg, synthetic_ica_and_raw, tmp_path):
        import matplotlib
        matplotlib.use("Agg")

        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        ica_cfg.deriv_root = str(tmp_path)
        ica_cfg._auto_ica_overlay = True
        analysis = BadICAnalysis(ica_cfg)
        analysis._auto_ica(ica, raw)

        names = sorted(p.name for p in tmp_path.rglob("*.png"))
        assert any("gesdLoadings" in n for n in names)
        assert any("gesdEigenscores" in n for n in names)
        assert any("gesdScree" in n for n in names)
        assert any("gesdOutliers" in n for n in names)

    def test_figure_failure_does_not_abort(
        self, ica_cfg, synthetic_ica_and_raw, tmp_path, monkeypatch
    ):
        import matplotlib
        matplotlib.use("Agg")

        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        ica_cfg.deriv_root = str(tmp_path)
        ica_cfg._auto_ica_overlay = True
        analysis = BadICAnalysis(ica_cfg)

        def _boom(_result):
            raise RuntimeError("boom")

        # A single failing figure builder must not abort labelling.
        monkeypatch.setattr(
            "custom.preprocessing.pca_gesd._fig_scree", _boom
        )
        result = analysis._auto_ica(ica, raw)
        assert isinstance(result, mne.preprocessing.ICA)
