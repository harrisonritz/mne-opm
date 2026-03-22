"""Tests for auto_ica.py — ICA component diagnostics, metrics, and GESD logic.

These tests focus on the computational methods that can be validated with
synthetic data, without needing real BIDS or ICA files.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest
from scipy.stats import kurtosis

from custom.preprocessing.auto_ica import AutoICAAnalysis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ica_cfg():
    """Config for auto ICA tests."""
    return SimpleNamespace(
        _auto_ica=True,
        spatial_filter="ica",
        ch_types=["mag"],
        deriv_root="/tmp/deriv",
        subjects=["001"],
        sessions=["01"],
        task="restingstate",
        ref_bads=False,  # disable ref method by default
        gesd_bads=True,
    )


@pytest.fixture()
def synthetic_ica_and_raw():
    """Create a synthetic ICA + Raw pair for testing diagnostics.

    Returns (ica, raw) where ica has been fit on raw.
    """
    rng = np.random.RandomState(42)
    n_ch = 20
    sfreq = 300.0
    n_times = int(sfreq * 5)  # 5 seconds

    # Create raw
    info = mne.create_info(
        [f"MEG{i:03d}" for i in range(n_ch)],
        sfreq, ["mag"] * n_ch,
    )
    data = rng.randn(n_ch, n_times) * 1e-13
    raw = mne.io.RawArray(data, info)

    # Fit ICA
    ica = mne.preprocessing.ICA(
        n_components=10, method="fastica", random_state=42, max_iter=100
    )
    ica.fit(raw)

    return ica, raw


# ---------------------------------------------------------------------------
# is_enabled
# ---------------------------------------------------------------------------

class TestAutoICAIsEnabled:
    def test_enabled_when_both_set(self, ica_cfg):
        assert AutoICAAnalysis(ica_cfg).is_enabled() is True

    def test_disabled_without_flag(self, ica_cfg):
        ica_cfg._auto_ica = False
        assert AutoICAAnalysis(ica_cfg).is_enabled() is False

    def test_disabled_without_spatial_filter(self, ica_cfg):
        ica_cfg.spatial_filter = None
        assert AutoICAAnalysis(ica_cfg).is_enabled() is False

    def test_disabled_with_wrong_spatial_filter(self, ica_cfg):
        ica_cfg.spatial_filter = "ssp"
        assert AutoICAAnalysis(ica_cfg).is_enabled() is False


# ---------------------------------------------------------------------------
# _ica_component_diagnostics
# ---------------------------------------------------------------------------

class TestICAComponentDiagnostics:
    """Test that diagnostic metrics are computed correctly."""

    def test_returns_expected_keys(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        analysis = AutoICAAnalysis(ica_cfg)
        diag = analysis._ica_component_diagnostics(ica, raw)

        expected_keys = {
            "sensor_var",
            "mean_abs_gradient",
            "hf_ratio",
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
        analysis = AutoICAAnalysis(ica_cfg)
        diag = analysis._ica_component_diagnostics(ica, raw)

        n_comps = ica.n_components_
        for key, vals in diag.items():
            assert len(vals) == n_comps, (
                f"Metric '{key}' has length {len(vals)}, expected {n_comps}"
            )

    def test_sensor_var_positive(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        analysis = AutoICAAnalysis(ica_cfg)
        diag = analysis._ica_component_diagnostics(ica, raw)
        assert np.all(diag["sensor_var"] >= 0)

    def test_autocorr_in_range(self, ica_cfg, synthetic_ica_and_raw):
        """Autocorrelation should be in [-1, 1]."""
        ica, raw = synthetic_ica_and_raw
        analysis = AutoICAAnalysis(ica_cfg)
        diag = analysis._ica_component_diagnostics(ica, raw)
        assert np.all(diag["autocorr_1lag"] >= -1.0)
        assert np.all(diag["autocorr_1lag"] <= 1.0)

    def test_hf_ratio_positive(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        analysis = AutoICAAnalysis(ica_cfg)
        diag = analysis._ica_component_diagnostics(ica, raw)
        assert np.all(diag["hf_ratio"] > 0)

    def test_spectral_deriv_kurtosis_higher_for_narrow_band(self, ica_cfg):
        """Narrow-band artifact should produce higher spectral derivative kurtosis.

        A synthetic component with a boxcar power spectrum (sharp onset/offset at
        a narrow band) should yield higher d(log_psd)/df kurtosis than a broadband
        component with a smooth 1/f-like spectrum.
        """
        rng = np.random.RandomState(0)
        n_ch = 20
        sfreq = 300.0
        n_times = int(sfreq * 10)  # 10 s for stable spectral estimates

        # --- Broadband 1/f-like component ---
        # Filter white noise through a 1/f filter in frequency domain
        freqs_template = np.fft.rfftfreq(n_times, 1 / sfreq)
        freqs_template[0] = 1.0  # avoid divide-by-zero at DC
        amplitude = 1.0 / freqs_template  # 1/f amplitude
        phase = rng.uniform(0, 2 * np.pi, size=len(freqs_template))
        spectrum = amplitude * np.exp(1j * phase)
        broadband = np.fft.irfft(spectrum, n=n_times)  # (n_times,)

        # --- Narrow-band artifact: sinusoid at 20 Hz + low-amplitude broadband ---
        t = np.arange(n_times) / sfreq
        narrowband = np.sin(2 * np.pi * 20 * t) + 0.05 * rng.randn(n_times)

        # Package as two-component ICA source matrix (n_components, n_times)
        # We bypass fitting ICA by mocking the source extraction
        sources = np.vstack([broadband, narrowband])
        sources = sources / sources.std(axis=1, keepdims=True)  # normalize

        # Build a minimal Raw + ICA mock so _ica_component_diagnostics can run
        info = mne.create_info(
            [f"MEG{i:03d}" for i in range(n_ch)], sfreq, ["mag"] * n_ch
        )
        data = rng.randn(n_ch, n_times) * 1e-13
        raw = mne.io.RawArray(data, info)

        ica = mne.preprocessing.ICA(n_components=2, method="fastica", random_state=0, max_iter=200)
        ica.fit(raw)

        # Monkey-patch get_sources to return our controlled sources
        mock_src_obj = MagicMock()
        mock_src_obj.get_data.return_value = sources
        ica.get_sources = MagicMock(return_value=mock_src_obj)

        analysis = AutoICAAnalysis(ica_cfg)
        diag = analysis._ica_component_diagnostics(ica, raw)

        deriv_kurt = diag["spectral_deriv_kurtosis"]
        resid_kurt = diag["spectral_resid_kurtosis"]

        # Narrow-band component (index 1) should have higher kurtosis on both metrics
        assert deriv_kurt[1] > deriv_kurt[0], (
            f"Expected narrow-band deriv kurtosis ({deriv_kurt[1]:.2f}) > "
            f"broadband ({deriv_kurt[0]:.2f})"
        )
        assert resid_kurt[1] > resid_kurt[0], (
            f"Expected narrow-band resid kurtosis ({resid_kurt[1]:.2f}) > "
            f"broadband ({resid_kurt[0]:.2f})"
        )


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
            "source_kurtosis": rng.randn(n) * 3,
            "autocorr_1lag": np.clip(rng.randn(n) * 0.3 + 0.5, -0.99, 0.99),
            "spectral_slope": rng.randn(n) * 0.5 - 1.5,
            "spatial_kurtosis": rng.randn(n) * 2,
            "spectral_deriv_kurtosis": rng.randn(n) * 2,
            "spectral_resid_kurtosis": rng.randn(n) * 2,
        }

    def test_returns_list_of_tuples(self, ica_cfg, sample_diagnostics):
        analysis = AutoICAAnalysis(ica_cfg)
        metrics = analysis._prepare_metrics_for_gesd(sample_diagnostics)
        assert isinstance(metrics, list)
        for item in metrics:
            assert len(item) == 3  # (name, values, direction)
            name, vals, side = item
            assert isinstance(name, str)
            assert isinstance(vals, np.ndarray)
            assert side in (-1, 0, 1)

    def test_seven_metrics_produced(self, ica_cfg, sample_diagnostics):
        analysis = AutoICAAnalysis(ica_cfg)
        metrics = analysis._prepare_metrics_for_gesd(sample_diagnostics)
        assert len(metrics) == 8

    def test_metric_names(self, ica_cfg, sample_diagnostics):
        analysis = AutoICAAnalysis(ica_cfg)
        metrics = analysis._prepare_metrics_for_gesd(sample_diagnostics)
        names = [m[0] for m in metrics]
        assert "log_hf_ratio" in names
        assert "temporal_kurtosis_sqrt" in names
        assert "autocorr_fisher_z" in names
        assert "spectral_slope" in names
        assert "spatial_kurtosis_sqrt" in names
        assert "spectral_deriv_kurtosis_sqrt" in names
        assert "spectral_resid_kurtosis_sqrt" in names
        assert "log_mean_abs_gradient" in names

    def test_directions(self, ica_cfg, sample_diagnostics):
        analysis = AutoICAAnalysis(ica_cfg)
        metrics = analysis._prepare_metrics_for_gesd(sample_diagnostics)
        direction_map = {name: side for name, _, side in metrics}

        assert direction_map["log_hf_ratio"] == 1  # high = bad
        assert direction_map["temporal_kurtosis_sqrt"] == 1
        assert direction_map["autocorr_fisher_z"] == -1  # low = bad
        assert direction_map["spectral_slope"] == 1
        assert direction_map["spatial_kurtosis_sqrt"] == 1
        assert direction_map["spectral_deriv_kurtosis_sqrt"] == 1  # high = bad (sharp transitions)
        assert direction_map["spectral_resid_kurtosis_sqrt"] == 1  # high = bad (clustered deviation)
        assert direction_map["log_mean_abs_gradient"] == 1  # high = temporally rough

    def test_no_nans_in_output(self, ica_cfg, sample_diagnostics):
        analysis = AutoICAAnalysis(ica_cfg)
        metrics = analysis._prepare_metrics_for_gesd(sample_diagnostics)
        for name, vals, _ in metrics:
            assert not np.any(np.isnan(vals)), f"NaN found in {name}"

    def test_fisher_z_handles_extreme_autocorr(self, ica_cfg):
        """Autocorrelation near +/-1 should be clipped before arctanh."""
        analysis = AutoICAAnalysis(ica_cfg)
        diagnostics = {
            "sensor_var": np.ones(5),
            "mean_abs_gradient": np.ones(5),
            "hf_ratio": np.ones(5),
            "source_kurtosis": np.zeros(5),
            "autocorr_1lag": np.array([0.999, -0.999, 1.0, -1.0, 0.5]),
            "spectral_slope": np.zeros(5),
            "spatial_kurtosis": np.zeros(5),
            "spectral_deriv_kurtosis": np.zeros(5),
            "spectral_resid_kurtosis": np.zeros(5),
        }
        metrics = analysis._prepare_metrics_for_gesd(diagnostics)
        fisher_z_vals = next(v for n, v, _ in metrics if n == "autocorr_fisher_z")
        assert not np.any(np.isinf(fisher_z_vals)), "Fisher z should not be infinite"


# ---------------------------------------------------------------------------
# _label_by_gesd_new (PCA-whitened GESD)
# ---------------------------------------------------------------------------

class TestLabelByGESDNew:
    """Test the PCA-whitened GESD method with synthetic ICA."""

    def test_does_not_crash(self, ica_cfg, synthetic_ica_and_raw):
        """Run the full pipeline on synthetic data without crashing."""
        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        analysis = AutoICAAnalysis(ica_cfg)
        result = analysis._label_by_gesd(ica, raw)
        assert isinstance(result, mne.preprocessing.ICA)

    def test_exclude_list_is_sorted_unique(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        analysis = AutoICAAnalysis(ica_cfg)
        result = analysis._label_by_gesd(ica, raw)
        assert result.exclude == sorted(set(result.exclude))

    def test_exclude_indices_valid(self, ica_cfg, synthetic_ica_and_raw):
        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        analysis = AutoICAAnalysis(ica_cfg)
        result = analysis._label_by_gesd(ica, raw)
        for idx in result.exclude:
            assert 0 <= idx < ica.n_components_

    def test_skips_when_too_few_components(self, ica_cfg, synthetic_ica_and_raw):
        """Should skip GESD when < 5 components remain."""
        ica, raw = synthetic_ica_and_raw
        # Exclude almost all components
        ica.exclude = list(range(ica.n_components_ - 3))
        n_excluded_before = len(ica.exclude)
        analysis = AutoICAAnalysis(ica_cfg)
        result = analysis._label_by_gesd(ica, raw)
        # Should not add any more (too few remaining)
        assert len(result.exclude) == n_excluded_before


# ---------------------------------------------------------------------------
# _label_by_corrmap (spatial template matching)
# ---------------------------------------------------------------------------

class TestLabelByCorrmap:
    """Tests for _label_by_corrmap template matching via corrmap."""

    # ---- graceful-skip cases ------------------------------------------------

    def test_skips_when_no_template_dir_set(self, ica_cfg, synthetic_ica_and_raw):
        """Returns ICA unchanged when _corrmap_template_dir is not configured."""
        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        # _corrmap_template_dir deliberately NOT set
        analysis = AutoICAAnalysis(ica_cfg)
        result = analysis._label_by_corrmap(ica, raw)
        assert result.exclude == []

    def test_skips_when_dir_missing(self, ica_cfg, synthetic_ica_and_raw, tmp_path):
        """Returns ICA unchanged when the template directory does not exist."""
        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        ica_cfg._corrmap_template_dir = str(tmp_path / "nonexistent")
        analysis = AutoICAAnalysis(ica_cfg)
        result = analysis._label_by_corrmap(ica, raw)
        assert result.exclude == []

    def test_skips_when_channel_names_missing(
        self, ica_cfg, synthetic_ica_and_raw, tmp_path
    ):
        """Returns ICA unchanged when {type}_channel_names.npy is absent."""
        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        ica_cfg._corrmap_template_dir = str(tmp_path)  # dir exists but no .npy
        analysis = AutoICAAnalysis(ica_cfg)
        result = analysis._label_by_corrmap(ica, raw)
        assert result.exclude == []

    def test_skips_when_no_templates_file(
        self, ica_cfg, synthetic_ica_and_raw, tmp_path
    ):
        """Skips gracefully when {type}_templates.npy is absent."""
        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        np.save(str(tmp_path / "eog_channel_names.npy"), np.array(ica.ch_names))
        ica_cfg._corrmap_template_dir = str(tmp_path)
        ica_cfg._n_eog_templates = 3
        ica_cfg._n_ecg_templates = 0
        analysis = AutoICAAnalysis(ica_cfg)
        result = analysis._label_by_corrmap(ica, raw)
        assert result.exclude == []

    def test_skips_when_both_types_disabled(
        self, ica_cfg, synthetic_ica_and_raw, tmp_path
    ):
        """Returns ICA unchanged when _n_eog_templates and _n_ecg_templates are both 0."""
        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        ica_cfg._corrmap_template_dir = str(tmp_path)
        ica_cfg._n_eog_templates = 0
        ica_cfg._n_ecg_templates = 0
        analysis = AutoICAAnalysis(ica_cfg)
        result = analysis._label_by_corrmap(ica, raw)
        assert result.exclude == []

    # ---- detection tests ----------------------------------------------------

    def test_detects_eog_matching_component(
        self, ica_cfg, synthetic_ica_and_raw, tmp_path
    ):
        """A template built from an ICA component's own topography must be found."""
        ica, raw = synthetic_ica_and_raw
        components = ica.get_components()  # (n_channels, n_components)
        target_idx = 0
        target_topo = components[:, target_idx]

        np.save(str(tmp_path / "eog_channel_names.npy"), np.array(ica.ch_names))
        np.save(str(tmp_path / "eog_templates.npy"), target_topo[:, np.newaxis])

        ica.exclude = []
        ica_cfg._corrmap_template_dir = str(tmp_path)
        ica_cfg._n_eog_templates = 1
        ica_cfg._n_ecg_templates = 0
        ica_cfg._corrmap_threshold = 0.9  # high threshold: only near-exact match

        analysis = AutoICAAnalysis(ica_cfg)
        result = analysis._label_by_corrmap(ica, raw)

        assert target_idx in result.exclude, (
            f"Expected component {target_idx} in exclude, got {result.exclude}"
        )
        assert target_idx in result.labels_.get("eog", [])

    def test_detects_ecg_matching_component(
        self, ica_cfg, synthetic_ica_and_raw, tmp_path
    ):
        """ECG template matching works the same way as EOG."""
        ica, raw = synthetic_ica_and_raw
        components = ica.get_components()
        target_idx = 1
        target_topo = components[:, target_idx]

        np.save(str(tmp_path / "ecg_channel_names.npy"), np.array(ica.ch_names))
        np.save(str(tmp_path / "ecg_templates.npy"), target_topo[:, np.newaxis])

        ica.exclude = []
        ica_cfg._corrmap_template_dir = str(tmp_path)
        ica_cfg._n_eog_templates = 0
        ica_cfg._n_ecg_templates = 1
        ica_cfg._corrmap_threshold = 0.9

        analysis = AutoICAAnalysis(ica_cfg)
        result = analysis._label_by_corrmap(ica, raw)

        assert target_idx in result.exclude
        assert target_idx in result.labels_.get("ecg", [])

    def test_labels_and_exclude_are_consistent(
        self, ica_cfg, synthetic_ica_and_raw, tmp_path
    ):
        """Every component in ica.labels_['eog'] must also appear in ica.exclude."""
        ica, raw = synthetic_ica_and_raw
        components = ica.get_components()
        np.save(str(tmp_path / "eog_channel_names.npy"), np.array(ica.ch_names))
        np.save(str(tmp_path / "eog_templates.npy"), components[:, 0:1])

        ica.exclude = []
        ica_cfg._corrmap_template_dir = str(tmp_path)
        ica_cfg._n_eog_templates = 1
        ica_cfg._n_ecg_templates = 0
        ica_cfg._corrmap_threshold = 0.9

        analysis = AutoICAAnalysis(ica_cfg)
        result = analysis._label_by_corrmap(ica, raw)

        for idx in result.labels_.get("eog", []):
            assert idx in result.exclude

    def test_multiple_templates_accumulate(
        self, ica_cfg, synthetic_ica_and_raw, tmp_path
    ):
        """Matches from multiple templates of the same type are unioned, not overwritten."""
        ica, raw = synthetic_ica_and_raw
        components = ica.get_components()
        np.save(str(tmp_path / "eog_channel_names.npy"), np.array(ica.ch_names))
        # Two template columns targeting different components
        np.save(
            str(tmp_path / "eog_templates.npy"),
            np.column_stack([components[:, 0], components[:, 1]]),
        )

        ica.exclude = []
        ica_cfg._corrmap_template_dir = str(tmp_path)
        ica_cfg._n_eog_templates = 2
        ica_cfg._n_ecg_templates = 0
        ica_cfg._corrmap_threshold = 0.9

        analysis = AutoICAAnalysis(ica_cfg)
        result = analysis._label_by_corrmap(ica, raw)

        # Both targets should be matched
        assert 0 in result.exclude
        assert 1 in result.exclude

    # ---- channel alignment --------------------------------------------------

    def test_channel_subset_alignment(
        self, ica_cfg, synthetic_ica_and_raw, tmp_path
    ):
        """Reference with extra channels aligns correctly to the ICA's subset."""
        ica, raw = synthetic_ica_and_raw
        components = ica.get_components()
        target_topo = components[:, 0]

        # Reference = ICA channels + 5 phantom channels not in the ICA
        ica_channels = list(ica.ch_names)
        extra_channels = [f"EXTRA{i:03d}" for i in range(5)]
        ref_channels = np.array(ica_channels + extra_channels)

        # Template has values for all reference channels; extras are noise
        rng = np.random.RandomState(42)
        full_template = np.concatenate([target_topo, rng.randn(5) * 0.01])

        np.save(str(tmp_path / "eog_channel_names.npy"), ref_channels)
        np.save(str(tmp_path / "eog_templates.npy"), full_template[:, np.newaxis])

        ica.exclude = []
        ica_cfg._corrmap_template_dir = str(tmp_path)
        ica_cfg._n_eog_templates = 1
        ica_cfg._n_ecg_templates = 0
        ica_cfg._corrmap_threshold = 0.9

        analysis = AutoICAAnalysis(ica_cfg)
        result = analysis._label_by_corrmap(ica, raw)

        # Should not crash, and should still identify the target component
        assert isinstance(result, mne.preprocessing.ICA)
        assert 0 in result.exclude

    def test_n_templates_limits_columns_used(
        self, ica_cfg, synthetic_ica_and_raw, tmp_path
    ):
        """_n_eog_templates=1 should only use the first template column."""
        ica, raw = synthetic_ica_and_raw
        components = ica.get_components()
        np.save(str(tmp_path / "eog_channel_names.npy"), np.array(ica.ch_names))
        # Column 0 targets component 0 (will be used)
        # Column 1 targets component 1 (should be ignored when _n_eog_templates=1)
        np.save(
            str(tmp_path / "eog_templates.npy"),
            np.column_stack([components[:, 0], components[:, 1]]),
        )

        ica.exclude = []
        ica_cfg._corrmap_template_dir = str(tmp_path)
        ica_cfg._n_eog_templates = 1  # only use first column
        ica_cfg._n_ecg_templates = 0
        ica_cfg._corrmap_threshold = 0.9

        analysis = AutoICAAnalysis(ica_cfg)
        result = analysis._label_by_corrmap(ica, raw)

        assert 0 in result.exclude        # component from column 0
        assert 1 not in result.exclude    # component from column 1 (ignored)

    # ---- integration --------------------------------------------------------

    def test__corrmap_bads_wired_into_auto_ica(
        self, ica_cfg, synthetic_ica_and_raw, tmp_path
    ):
        """_corrmap_bads=True calls _label_by_corrmap inside _auto_ica."""
        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        ica_cfg.ref_bads = False
        ica_cfg.gesd_bads = False
        ica_cfg._corrmap_bads = True
        # Point at a dir with no reference_channels.npy → graceful skip
        ica_cfg._corrmap_template_dir = str(tmp_path)

        analysis = AutoICAAnalysis(ica_cfg)
        result = analysis._auto_ica(ica, raw)

        # Should complete without error (skips due to missing reference file)
        assert isinstance(result, mne.preprocessing.ICA)
        assert result.exclude == []


# ---------------------------------------------------------------------------
# _auto_ica integration
# ---------------------------------------------------------------------------

class TestAutoICAIntegration:
    """Test the full _auto_ica method."""

    def test_gesd_only(self, ica_cfg, synthetic_ica_and_raw):
        """With ref_bads=False, only GESD should run."""
        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        ica_cfg.ref_bads = False
        ica_cfg.gesd_bads = True
        analysis = AutoICAAnalysis(ica_cfg)
        result = analysis._auto_ica(ica, raw)
        assert isinstance(result, mne.preprocessing.ICA)
        # Exclude list should be sorted and deduplicated
        assert result.exclude == sorted(set(result.exclude))

    def test_no_methods_enabled(self, ica_cfg, synthetic_ica_and_raw):
        """With both methods disabled, no components should be excluded."""
        ica, raw = synthetic_ica_and_raw
        ica.exclude = []
        ica_cfg.ref_bads = False
        ica_cfg.gesd_bads = False
        analysis = AutoICAAnalysis(ica_cfg)
        result = analysis._auto_ica(ica, raw)
        assert result.exclude == []
