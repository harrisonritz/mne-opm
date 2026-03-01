"""Tests for run_decoding.py -- multivariate decoding functions.

These tests exercise the MultivariateNoiseNormalizer, _prep_contrast,
pipeline factories, decoding functions, save functions, and main()
flow *without* requiring actual BIDS data files.  Where possible,
synthetic MNE objects stand in for the real thing.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pandas as pd
import pytest

from custom.transformers import MultivariateNoiseNormalizer
from custom.run_decoding import (
    get_data_rank,
    _prep_contrast,
    _make_time_clf,
    _make_epoch_clf,
    run_subject_time_decoding,
    run_subject_temporal_gen,
    run_subject_epoch_decoding,
    run_subject_cross_decoding,
    save_time_results,
    save_epoch_results,
    save_tg_results,
    save_cross_time_results,
    save_cross_epoch_results,
    save_cross_tg_results,
    save_fig,
    plot_subject_time_patterns,
    process_subject,
    main,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def rng():
    """Deterministic NumPy random state."""
    return np.random.RandomState(42)


@pytest.fixture()
def decoder_cfg(tmp_path):
    """Standard decoder config namespace."""
    return SimpleNamespace(
        _run_decoding=True,
        _decoder_scoring="roc_auc",
        _decoder_n_jobs_inner=1,
        _decoder_decim=1,
        _decoder_run_temporal_gen=False,
        _decoder_baseline=None,
        _decoder_chance=0.5,
        _decoder_save_formats=["png"],
        _decoder_group_column="run",
        _decoder_contrasts=[
            {
                "name": "task",
                "conditions": ['task1 == "read"', 'task1 == "listen"'],
            },
        ],
        _decoder_cross_contrasts=[],
        subjects=["007"],
        sessions=["01"],
        task="TSX",
        deriv_root=str(tmp_path / "derivatives"),
        datatype="meg",
        data_type="meg",
    )


@pytest.fixture()
def binary_epochs(rng):
    """Create synthetic epochs with binary metadata for decoding tests.

    20 epochs, 8 MEG channels, 50 time points, 2 conditions, 2 runs.
    Condition 1 (first 10 epochs) has added signal in channels 0-3.
    """
    n_epochs = 20
    n_channels = 8
    n_times = 50
    sfreq = 100.0

    info = mne.create_info(
        ch_names=[f"MEG{i:03d}" for i in range(n_channels)],
        sfreq=sfreq,
        ch_types=["mag"] * n_channels,
    )

    data = rng.randn(n_epochs, n_channels, n_times) * 1e-13
    # Add signal difference between conditions
    data[:10, :4, 20:35] += 5e-13

    events = np.column_stack([
        np.arange(n_epochs) * n_times,
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int),
    ])

    epochs = mne.EpochsArray(data, info, events=events, event_id={"trial": 1})

    metadata = pd.DataFrame({
        "task1": ["read"] * 10 + ["listen"] * 10,
        "run": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2] * 2,
        "resp": ["m"] * 5 + ["z"] * 5 + ["m"] * 5 + ["z"] * 5,
    })
    epochs.metadata = metadata

    return epochs


@pytest.fixture()
def simple_contrast():
    """A standard binary contrast dict."""
    return {
        "name": "task",
        "conditions": ['task1 == "read"', 'task1 == "listen"'],
    }


# ---------------------------------------------------------------------------
# MultivariateNoiseNormalizer
# ---------------------------------------------------------------------------

class TestMultivariateNoiseNormalizer:
    """Test the whitening transformer."""

    def test_fit_transform_2d(self, rng):
        """2-D input: (n_samples, n_features)."""
        X = rng.randn(50, 8) * 1e-13
        y = np.array([0] * 25 + [1] * 25)
        mnn = MultivariateNoiseNormalizer()
        X_out = mnn.fit_transform(X, y)
        assert X_out.shape == X.shape
        assert hasattr(mnn, "whitening_")
        assert hasattr(mnn, "coloring_")

    def test_fit_transform_3d(self, rng):
        """3-D input: (n_epochs, n_channels, n_times)."""
        X = rng.randn(20, 8, 50) * 1e-13
        y = np.array([0] * 10 + [1] * 10)
        mnn = MultivariateNoiseNormalizer()
        X_out = mnn.fit_transform(X, y)
        assert X_out.shape == X.shape

    def test_inverse_transform_2d(self, rng):
        """inverse_transform should approximately recover original data."""
        X = rng.randn(50, 8) * 1e-13
        y = np.array([0] * 25 + [1] * 25)
        mnn = MultivariateNoiseNormalizer()
        X_w = mnn.fit_transform(X, y)
        X_back = mnn.inverse_transform(X_w)
        np.testing.assert_allclose(X_back, X, atol=1e-20)

    def test_inverse_transform_3d(self, rng):
        """inverse_transform should approximately recover 3-D data."""
        X = rng.randn(20, 8, 50) * 1e-13
        y = np.array([0] * 10 + [1] * 10)
        mnn = MultivariateNoiseNormalizer()
        X_w = mnn.fit_transform(X, y)
        X_back = mnn.inverse_transform(X_w)
        np.testing.assert_allclose(X_back, X, atol=1e-20)

    def test_fit_without_y(self, rng):
        """Should work without labels (no per-class centering)."""
        X = rng.randn(50, 8) * 1e-13
        mnn = MultivariateNoiseNormalizer()
        X_out = mnn.fit_transform(X)
        assert X_out.shape == X.shape

    def test_whitening_coloring_shapes(self, rng):
        """Whitening and coloring matrices should be square (n_features)."""
        X = rng.randn(50, 8)
        mnn = MultivariateNoiseNormalizer().fit(X)
        assert mnn.whitening_.shape == (8, 8)
        assert mnn.coloring_.shape == (8, 8)

    def test_3d_whitening_shape(self, rng):
        """3-D input should produce channel-space matrices."""
        X = rng.randn(20, 6, 30)
        y = np.array([0] * 10 + [1] * 10)
        mnn = MultivariateNoiseNormalizer().fit(X, y)
        assert mnn.whitening_.shape == (6, 6)
        assert mnn.coloring_.shape == (6, 6)

    def test_pickle_roundtrip(self, rng):
        """Fitted transformer must survive pickle (joblib/loky workers)."""
        import pickle

        X = rng.randn(40, 8) * 1e-13
        y = np.array([0] * 20 + [1] * 20)
        mnn = MultivariateNoiseNormalizer().fit(X, y)

        data = pickle.dumps(mnn)
        mnn2 = pickle.loads(data)

        np.testing.assert_array_equal(mnn2.whitening_, mnn.whitening_)
        np.testing.assert_array_equal(mnn2.coloring_, mnn.coloring_)
        np.testing.assert_array_equal(mnn2.transform(X), mnn.transform(X))


# ---------------------------------------------------------------------------
# get_data_rank
# ---------------------------------------------------------------------------

class TestGetDataRank:
    """Test data rank estimation."""

    def test_returns_positive_int(self, binary_epochs):
        rank = get_data_rank(binary_epochs)
        assert isinstance(rank, (int, np.integer))
        assert rank > 0

    def test_rank_leq_n_channels(self, binary_epochs):
        rank = get_data_rank(binary_epochs)
        assert rank <= len(binary_epochs.ch_names)


# ---------------------------------------------------------------------------
# _prep_contrast
# ---------------------------------------------------------------------------

class TestPrepContrast:
    """Test epoch subsetting and equalization."""

    def test_returns_correct_fields(self, binary_epochs, simple_contrast):
        result = _prep_contrast(binary_epochs, simple_contrast)
        assert result is not None
        ep_all, X, y, times, groups, n1, n2 = result
        assert X.ndim == 3
        assert len(y) == n1 + n2
        assert set(y) == {0.0, 1.0}
        assert len(times) == X.shape[2]

    def test_equalization(self, binary_epochs, simple_contrast):
        """After equalization, both classes should have equal counts."""
        result = _prep_contrast(binary_epochs, simple_contrast)
        _, _, y, _, _, n1, n2 = result
        assert n1 == n2

    def test_empty_condition_returns_none(self, binary_epochs):
        """If a query matches zero epochs, return None."""
        contrast = {
            "name": "empty",
            "conditions": ['task1 == "nonexistent"', 'task1 == "listen"'],
        }
        result = _prep_contrast(binary_epochs, contrast)
        assert result is None

    def test_decimation(self, binary_epochs, simple_contrast):
        """Decimation should reduce the number of time points."""
        result_full = _prep_contrast(binary_epochs, simple_contrast, decim=1)
        result_decim = _prep_contrast(binary_epochs, simple_contrast, decim=2)
        _, X_full, _, _, _, _, _ = result_full
        _, X_decim, _, _, _, _, _ = result_decim
        assert X_decim.shape[2] < X_full.shape[2]

    def test_custom_group_column(self, binary_epochs, simple_contrast):
        """group_column should use the specified metadata column."""
        result = _prep_contrast(
            binary_epochs, simple_contrast, group_column="run"
        )
        _, _, _, _, groups, _, _ = result
        assert len(groups) > 0
        assert set(groups).issubset({1, 2})

    def test_groups_match_metadata(self, binary_epochs, simple_contrast):
        """Groups should come from the metadata."""
        result = _prep_contrast(binary_epochs, simple_contrast)
        ep_all, _, _, _, groups, _, _ = result
        np.testing.assert_array_equal(
            groups, ep_all.metadata["run"].values
        )


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------

class TestPipelineFactories:
    """Test that pipeline factories return valid sklearn pipelines."""

    def test_make_time_clf_returns_pipeline(self):
        clf = _make_time_clf(n_components=5)
        assert hasattr(clf, "fit")
        assert hasattr(clf, "predict")
        assert len(clf.steps) == 3

    def test_make_epoch_clf_returns_pipeline(self):
        clf = _make_epoch_clf(n_components=0.99)
        assert hasattr(clf, "fit")
        assert hasattr(clf, "predict")
        assert len(clf.steps) == 4

    def test_time_clf_pca_components(self):
        """PCA n_components should match the value passed in."""
        clf = _make_time_clf(n_components=7)
        pca_step = clf.steps[1][1]
        assert pca_step.n_components == 7

    def test_epoch_clf_pca_variance(self):
        """Epoch PCA should use the variance threshold passed in."""
        clf = _make_epoch_clf(n_components=0.99)
        pca_step = clf.steps[2][1]
        assert pca_step.n_components == 0.99

    def test_time_clf_first_step_is_mnn(self):
        clf = _make_time_clf(n_components=5)
        assert isinstance(clf.steps[0][1], MultivariateNoiseNormalizer)

    def test_epoch_clf_first_step_is_mnn(self):
        clf = _make_epoch_clf(n_components=0.99)
        assert isinstance(clf.steps[0][1], MultivariateNoiseNormalizer)


# ---------------------------------------------------------------------------
# Decoding functions (with mocked CV)
# ---------------------------------------------------------------------------

class TestTimeDecoding:
    """Test run_subject_time_decoding."""

    def test_returns_expected_keys(self, binary_epochs, simple_contrast):
        """Result dict should have all expected keys."""
        n_times = len(binary_epochs.times)
        n_ch = len(binary_epochs.ch_names)

        with patch("custom.run_decoding.cross_val_multiscore") as mock_cv, \
             patch("custom.run_decoding.get_coef") as mock_coef:
            mock_cv.return_value = np.random.rand(2, n_times)
            mock_coef.return_value = np.random.rand(n_times, n_ch)
            result = run_subject_time_decoding(
                binary_epochs, simple_contrast, scoring="roc_auc"
            )

        assert result is not None
        for key in ("times", "scores", "cv_scores", "patterns", "filters",
                     "info", "n_cond1", "n_cond2"):
            assert key in result

    def test_returns_none_for_empty(self, binary_epochs):
        contrast = {
            "name": "empty",
            "conditions": ['task1 == "nonexistent"', 'task1 == "listen"'],
        }
        result = run_subject_time_decoding(binary_epochs, contrast)
        assert result is None

    def test_scores_shape(self, binary_epochs, simple_contrast):
        """Mean scores should be 1-D with length n_times."""
        n_times = len(binary_epochs.times)

        with patch("custom.run_decoding.cross_val_multiscore") as mock_cv, \
             patch("custom.run_decoding.get_coef") as mock_coef:
            mock_cv.return_value = np.random.rand(2, n_times)
            mock_coef.return_value = np.random.rand(n_times, 8)
            result = run_subject_time_decoding(
                binary_epochs, simple_contrast
            )

        assert result["scores"].ndim == 1


class TestTemporalGen:
    """Test run_subject_temporal_gen."""

    def test_returns_expected_keys(self, binary_epochs, simple_contrast):
        n_times = len(binary_epochs.times)

        with patch("custom.run_decoding.cross_val_multiscore") as mock_cv:
            mock_cv.return_value = np.random.rand(2, n_times, n_times)
            result = run_subject_temporal_gen(
                binary_epochs, simple_contrast, scoring="roc_auc"
            )

        assert result is not None
        for key in ("times", "cv_scores", "scores_mean", "n_cond1", "n_cond2"):
            assert key in result

    def test_scores_mean_shape(self, binary_epochs, simple_contrast):
        """scores_mean should be (n_times, n_times)."""
        n_times = len(binary_epochs.times)

        with patch("custom.run_decoding.cross_val_multiscore") as mock_cv:
            mock_cv.return_value = np.random.rand(2, n_times, n_times)
            result = run_subject_temporal_gen(
                binary_epochs, simple_contrast
            )

        assert result["scores_mean"].shape == (n_times, n_times)

    def test_returns_none_for_empty(self, binary_epochs):
        contrast = {
            "name": "empty",
            "conditions": ['task1 == "nonexistent"', 'task1 == "listen"'],
        }
        result = run_subject_temporal_gen(binary_epochs, contrast)
        assert result is None


class TestEpochDecoding:
    """Test run_subject_epoch_decoding."""

    def test_returns_expected_keys(self, binary_epochs, simple_contrast):
        with patch("custom.run_decoding.cross_val_score") as mock_cv:
            mock_cv.return_value = np.array([0.6, 0.7])
            result = run_subject_epoch_decoding(
                binary_epochs, simple_contrast, scoring="roc_auc"
            )

        assert result is not None
        assert "score" in result
        assert "cv_scores" in result
        assert isinstance(result["score"], (float, np.floating))

    def test_returns_none_for_empty(self, binary_epochs):
        contrast = {
            "name": "empty",
            "conditions": ['task1 == "nonexistent"', 'task1 == "listen"'],
        }
        result = run_subject_epoch_decoding(binary_epochs, contrast)
        assert result is None

    def test_cv_scores_array(self, binary_epochs, simple_contrast):
        """cv_scores should be a 1-D array."""
        with patch("custom.run_decoding.cross_val_score") as mock_cv:
            mock_cv.return_value = np.array([0.5, 0.6, 0.7])
            result = run_subject_epoch_decoding(
                binary_epochs, simple_contrast
            )

        assert result["cv_scores"].ndim == 1


class TestCrossDecoding:
    """Test run_subject_cross_decoding."""

    def test_time_analysis(self, binary_epochs):
        cross_contrast = {
            "name": "test_cross",
            "train": {
                "name": "task_train",
                "conditions": ['task1 == "read"', 'task1 == "listen"'],
            },
            "test": {
                "name": "resp_test",
                "conditions": ['resp == "m"', 'resp == "z"'],
            },
            "analyses": ["time"],
        }
        with patch("custom.run_decoding.SlidingEstimator") as MockSliding, \
             patch("custom.run_decoding.get_coef") as mock_coef:
            mock_se = MagicMock()
            mock_se.score.return_value = np.random.rand(50)
            MockSliding.return_value = mock_se
            mock_coef.return_value = np.random.rand(50, 8)
            result = run_subject_cross_decoding(
                binary_epochs, cross_contrast,
            )

        assert result is not None
        assert "time" in result
        assert "name" in result

    def test_empty_analyses_returns_none(self, binary_epochs):
        cross_contrast = {
            "name": "test_cross",
            "train": {"name": "a",
                       "conditions": ['task1 == "read"', 'task1 == "listen"']},
            "test": {"name": "b",
                      "conditions": ['resp == "m"', 'resp == "z"']},
            "analyses": [],
        }
        result = run_subject_cross_decoding(binary_epochs, cross_contrast)
        assert result is None

    def test_epoch_analysis(self, binary_epochs):
        cross_contrast = {
            "name": "test_cross",
            "train": {
                "name": "task_train",
                "conditions": ['task1 == "read"', 'task1 == "listen"'],
            },
            "test": {
                "name": "resp_test",
                "conditions": ['resp == "m"', 'resp == "z"'],
            },
            "analyses": ["epoch"],
        }
        with patch("custom.run_decoding._make_epoch_clf") as mock_factory:
            mock_clf = MagicMock()
            mock_clf.decision_function.return_value = np.random.rand(10)
            mock_factory.return_value = mock_clf
            with patch("custom.run_decoding.roc_auc_score", return_value=0.65):
                result = run_subject_cross_decoding(
                    binary_epochs, cross_contrast,
                )

        assert result is not None
        assert "epoch" in result
        assert "score" in result["epoch"]

    def test_tg_analysis(self, binary_epochs):
        cross_contrast = {
            "name": "test_cross",
            "train": {
                "name": "task_train",
                "conditions": ['task1 == "read"', 'task1 == "listen"'],
            },
            "test": {
                "name": "resp_test",
                "conditions": ['resp == "m"', 'resp == "z"'],
            },
            "analyses": ["tg"],
        }
        with patch("custom.run_decoding.GeneralizingEstimator") as MockTG:
            mock_tg = MagicMock()
            mock_tg.score.return_value = np.random.rand(50, 50)
            MockTG.return_value = mock_tg
            result = run_subject_cross_decoding(
                binary_epochs, cross_contrast,
            )

        assert result is not None
        assert "tg" in result
        assert "scores_mean" in result["tg"]


# ---------------------------------------------------------------------------
# Save functions
# ---------------------------------------------------------------------------

class TestSaveFunctions:
    """Test BIDS path construction and save calls."""

    def _make_mock_bids_path(self, tmp_path):
        """Create a mock BIDSPath with working copy().update().fpath."""
        bids_path = MagicMock()

        def make_update_chain(fpath):
            mock_copy = MagicMock()
            mock_copy.update.return_value.fpath = fpath
            return mock_copy

        bids_path.copy.side_effect = lambda: make_update_chain(
            tmp_path / "output_file"
        )
        return bids_path

    def test_save_time_results(self, tmp_path):
        bids_path = self._make_mock_bids_path(tmp_path)
        result = {
            "times": np.linspace(-0.2, 1.0, 50),
            "scores": np.random.rand(50),
            "patterns": np.random.rand(50, 8),
            "filters": np.random.rand(50, 8),
            "info": mne.create_info(
                [f"MEG{i:03d}" for i in range(8)], 100.0, ["mag"] * 8
            ),
        }
        contrast = {"name": "task", "conditions": ["c1", "c2"]}

        with patch("custom.run_decoding.sanitize_cond_name",
                    return_value="task"):
            save_time_results(bids_path, contrast, result, "roc_auc", tmp_path)

        # Should call copy() twice: once for TSV, once for NPZ
        assert bids_path.copy.call_count == 2

    def test_save_epoch_results(self, tmp_path):
        bids_path = self._make_mock_bids_path(tmp_path)
        result = {"score": 0.75}
        contrast = {"name": "task", "conditions": ["c1", "c2"]}

        with patch("custom.run_decoding.sanitize_cond_name",
                    return_value="task"):
            save_epoch_results(bids_path, contrast, result, "roc_auc", tmp_path)

        assert bids_path.copy.call_count == 1

    def test_save_tg_results(self, tmp_path):
        bids_path = self._make_mock_bids_path(tmp_path)
        result = {
            "cv_scores": np.random.rand(2, 50, 50),
            "scores_mean": np.random.rand(50, 50),
            "times": np.linspace(-0.2, 1.0, 50),
        }
        contrast = {"name": "task", "conditions": ["c1", "c2"]}

        with patch("custom.run_decoding.sanitize_cond_name",
                    return_value="task"):
            save_tg_results(bids_path, contrast, result, tmp_path)

        assert bids_path.copy.call_count == 1

    def test_save_cross_time_results(self, tmp_path):
        bids_path = self._make_mock_bids_path(tmp_path)
        cross_contrast = {
            "name": "cross_test",
            "train": {"conditions": ["tc1", "tc2"]},
            "test": {"conditions": ["xc1", "xc2"]},
        }
        result = {
            "times": np.linspace(-0.2, 1.0, 50),
            "scores": np.random.rand(50),
            "patterns": np.random.rand(50, 8),
            "filters": np.random.rand(50, 8),
            "info": mne.create_info(
                [f"MEG{i:03d}" for i in range(8)], 100.0, ["mag"] * 8
            ),
        }

        with patch("custom.run_decoding.sanitize_cond_name",
                    return_value="crosstest"):
            save_cross_time_results(
                bids_path, cross_contrast, result, "roc_auc", tmp_path
            )

        # TSV + NPZ = 2 calls
        assert bids_path.copy.call_count == 2

    def test_save_cross_epoch_results(self, tmp_path):
        bids_path = self._make_mock_bids_path(tmp_path)
        cross_contrast = {
            "name": "cross_test",
            "train": {"conditions": ["tc1", "tc2"]},
            "test": {"conditions": ["xc1", "xc2"]},
        }
        result = {"score": 0.72}

        with patch("custom.run_decoding.sanitize_cond_name",
                    return_value="crosstest"):
            save_cross_epoch_results(
                bids_path, cross_contrast, result, "roc_auc", tmp_path
            )

        assert bids_path.copy.call_count == 1

    def test_save_cross_tg_results(self, tmp_path):
        bids_path = self._make_mock_bids_path(tmp_path)
        cross_contrast = {"name": "cross_test"}
        cc_res = {"train_name": "a", "test_name": "b"}
        result = {
            "scores_mean": np.random.rand(50, 50),
            "times": np.linspace(-0.2, 1.0, 50),
        }

        with patch("custom.run_decoding.sanitize_cond_name",
                    return_value="crosstest"):
            save_cross_tg_results(
                bids_path, cross_contrast, cc_res, result, tmp_path
            )

        assert bids_path.copy.call_count == 1


# ---------------------------------------------------------------------------
# save_fig
# ---------------------------------------------------------------------------

class TestSaveFig:
    """Test figure saving utility."""

    def test_saves_in_requested_formats(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        save_fig(fig, tmp_path / "test_fig", ["png"])
        assert (tmp_path / "test_fig.png").exists()
        plt.close(fig)

    def test_saves_multiple_formats(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        save_fig(fig, tmp_path / "test_fig", ["png", "pdf"])
        assert (tmp_path / "test_fig.png").exists()
        assert (tmp_path / "test_fig.pdf").exists()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plotting functions (smoke tests)
# ---------------------------------------------------------------------------

class TestPlotFunctions:
    """Smoke tests for plot functions -- verify they don't raise."""

    def test_plot_time_ribbon_no_raise(self, tmp_path):
        from custom.run_decoding import plot_subject_time_ribbon
        time_results = {
            "task": {
                "times": np.linspace(-0.2, 1.0, 50),
                "cv_scores": np.random.rand(3, 50),
            },
        }
        plot_subject_time_ribbon(
            time_results, "sub-007", tmp_path, ["png"], chance=0.5
        )

    def test_plot_epoch_bar_no_raise(self, tmp_path):
        from custom.run_decoding import plot_subject_epoch_bar
        epoch_results = {
            "task": {"cv_scores": np.array([0.6, 0.7, 0.8])},
        }
        plot_subject_epoch_bar(
            epoch_results, "sub-007", tmp_path, ["png"], chance=0.5
        )

    def test_plot_tg_heatmap_no_raise(self, tmp_path):
        from custom.run_decoding import plot_subject_tg_heatmap
        tg_results = {
            "task": {
                "times": np.linspace(-0.2, 1.0, 20),
                "scores_mean": np.random.rand(20, 20),
            },
        }
        plot_subject_tg_heatmap(
            tg_results, "sub-007", tmp_path, ["png"]
        )

    def test_plot_cross_ribbon_no_raise(self, tmp_path):
        from custom.run_decoding import plot_subject_cross_ribbon
        cross_results = {
            "cross_task": {
                "times": np.linspace(-0.2, 1.0, 50),
                "scores": np.random.rand(50),
                "train_name": "a",
                "test_name": "b",
            },
        }
        plot_subject_cross_ribbon(
            cross_results, "sub-007", tmp_path, ["png"], chance=0.5
        )

    def test_plot_cross_epoch_bar_no_raise(self, tmp_path):
        from custom.run_decoding import plot_subject_cross_epoch_bar
        cross_results = {
            "cross_task": {"score": 0.65},
        }
        plot_subject_cross_epoch_bar(
            cross_results, "sub-007", tmp_path, ["png"], chance=0.5
        )

    def test_plot_cross_tg_heatmap_no_raise(self, tmp_path):
        from custom.run_decoding import plot_subject_cross_tg_heatmap
        cross_results = {
            "cross_task": {
                "times": np.linspace(-0.2, 1.0, 20),
                "scores_mean": np.random.rand(20, 20),
                "train_name": "a",
                "test_name": "b",
            },
        }
        plot_subject_cross_tg_heatmap(
            cross_results, "sub-007", tmp_path, ["png"]
        )

    def test_plot_time_patterns_no_raise(self, tmp_path, capsys):
        """Smoke test: function should not raise even when plot_joint fails.

        Synthetic info objects have no channel positions, so plot_joint will
        fail gracefully (caught exception + WARNING printed).  The important
        thing is that the function does not propagate the exception.
        """
        n_channels = 8
        n_times = 20
        sfreq = 100.0
        info = mne.create_info(
            ch_names=[f"MEG{i:03d}" for i in range(n_channels)],
            sfreq=sfreq,
            ch_types=["mag"] * n_channels,
        )
        results = {
            "task": {
                "patterns": np.random.rand(n_channels, n_times) * 1e-13,
                "info": info,
                "times": np.linspace(-0.1, 0.1, n_times),
            },
        }
        # Must not raise regardless of whether plot_joint succeeds
        plot_subject_time_patterns(
            results, "sub-007", tmp_path, ["png"], pattern_times=None
        )

    def test_plot_time_patterns_missing_keys_no_raise(self, tmp_path):
        """Results without patterns/info keys should be silently skipped."""
        results = {"task": {"times": np.linspace(-0.1, 0.1, 20)}}
        plot_subject_time_patterns(results, "sub-007", tmp_path, ["png"])

    def test_empty_results_no_raise(self, tmp_path):
        """Empty result dicts should not raise."""
        from custom.run_decoding import (
            plot_subject_time_ribbon,
            plot_subject_epoch_bar,
            plot_subject_tg_heatmap,
        )
        plot_subject_time_ribbon({}, "sub-007", tmp_path, ["png"])
        plot_subject_epoch_bar({}, "sub-007", tmp_path, ["png"])
        plot_subject_tg_heatmap({}, "sub-007", tmp_path, ["png"])


# ---------------------------------------------------------------------------
# main() flow
# ---------------------------------------------------------------------------

class TestMain:
    """Test config loading and master switch."""

    def test_disabled_decoding_exits_early(self, capsys):
        with patch("custom.run_decoding._import_config") as mock_import, \
             patch("custom.run_decoding._update_config_from_path"), \
             patch("custom.run_decoding.parse_args") as mock_args:
            mock_args.return_value = SimpleNamespace(config="/fake/config.py")
            cfg = SimpleNamespace(_run_decoding=False)
            mock_import.return_value = cfg
            main()

        output = capsys.readouterr().out
        assert "Decoding disabled" in output

    def test_enabled_decoding_calls_process(self):
        with patch("custom.run_decoding._import_config") as mock_import, \
             patch("custom.run_decoding._update_config_from_path"), \
             patch("custom.run_decoding.parse_args") as mock_args, \
             patch("custom.run_decoding.process_subject") as mock_proc:
            mock_args.return_value = SimpleNamespace(config="/fake/config.py")
            cfg = SimpleNamespace(_run_decoding=True)
            mock_import.return_value = cfg
            main()

        mock_proc.assert_called_once()

    def test_missing_run_decoding_exits_early(self, capsys):
        """If _run_decoding is not set at all, should exit early."""
        with patch("custom.run_decoding._import_config") as mock_import, \
             patch("custom.run_decoding._update_config_from_path"), \
             patch("custom.run_decoding.parse_args") as mock_args:
            mock_args.return_value = SimpleNamespace(config="/fake/config.py")
            cfg = SimpleNamespace()  # no _run_decoding attribute
            mock_import.return_value = cfg
            main()

        output = capsys.readouterr().out
        assert "Decoding disabled" in output


# needed for plot tests
import matplotlib.pyplot as plt
