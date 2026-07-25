"""Tests for run_beamformer.py — LCMV beamformer functions.

These tests exercise parameter validation, data flow, and computation logic
*without* requiring actual BIDS data or forward models.  Where possible,
synthetic MNE objects stand in for the real thing.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from custom.run_beamformer import (
    compute_lcmv_filters,
    resolve_per_space,
    resolve_rank,
    run_beamformer_timecourse,
    run_beamformer_power,
    save_beamformer_results,
    surface_orient_forward,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def beamformer_cfg():
    """Standard beamformer config namespace."""
    return SimpleNamespace(
        _beamformer_reg=0.05,
        _beamformer_pick_ori="max-power",
        _beamformer_weight_norm="unit-noise-gain",
        _beamformer_depth=None,
        _beamformer_rank=None,
        _beamformer_surf_ori=True,
        _beamformer_cov_method="empirical",
        _reduce_rank=False,
        _beamformer_save_filters=False,
        _beamformer_add_to_report=False,
        _beamformer_power_tmin=0.0,
        _beamformer_power_tmax=0.5,
        conditions=["stim"],
        contrasts=[],
        ch_types=["mag"],
        n_jobs=1,
        subjects=["001"],
        sessions=["01"],
        task="task",
        deriv_root="/tmp/deriv",
        datatype="meg",
    )


# ---------------------------------------------------------------------------
# compute_lcmv_filters — parameter validation
# ---------------------------------------------------------------------------


class TestComputeLcmvFiltersValidation:
    """Test that invalid parameters raise appropriate errors."""

    def test_invalid_pick_ori_raises(self, beamformer_cfg):
        beamformer_cfg._beamformer_pick_ori = "invalid_orientation"
        with pytest.raises(ValueError, match="Invalid _beamformer_pick_ori"):
            compute_lcmv_filters(
                forward=MagicMock(),
                data_cov=MagicMock(),
                noise_cov=MagicMock(),
                info=MagicMock(),
                cfg=beamformer_cfg,
            )

    def test_invalid_weight_norm_raises(self, beamformer_cfg):
        beamformer_cfg._beamformer_weight_norm = "bad_norm"
        with pytest.raises(ValueError, match="Invalid _beamformer_weight_norm"):
            compute_lcmv_filters(
                forward=MagicMock(),
                data_cov=MagicMock(),
                noise_cov=MagicMock(),
                info=MagicMock(),
                cfg=beamformer_cfg,
            )

    @pytest.mark.parametrize(
        "ori",
        ["max-power", "vector", None],
    )
    def test_valid_pick_ori_accepted(self, beamformer_cfg, ori):
        """These orientations should not raise during validation."""
        beamformer_cfg._beamformer_pick_ori = ori
        # We can't run the full make_lcmv without real data, so just
        # verify the validation passes before the actual computation.
        # Patch make_lcmv to avoid the heavy computation.
        with patch("custom.run_beamformer.make_lcmv") as mock_lcmv:
            mock_lcmv.return_value = {"mock": "filters"}
            info = mne.create_info(["MEG001"], 300.0, ["mag"])
            result = compute_lcmv_filters(
                forward=MagicMock(),
                data_cov=MagicMock(),
                noise_cov=MagicMock(),
                info=info,
                cfg=beamformer_cfg,
            )
            mock_lcmv.assert_called_once()

    @pytest.mark.parametrize(
        "norm",
        ["unit-noise-gain", "nai", "unit-noise-gain-invariant", None],
    )
    def test_valid_weight_norm_accepted(self, beamformer_cfg, norm):
        beamformer_cfg._beamformer_weight_norm = norm
        with patch("custom.run_beamformer.make_lcmv") as mock_lcmv:
            mock_lcmv.return_value = {"mock": "filters"}
            info = mne.create_info(["MEG001"], 300.0, ["mag"])
            compute_lcmv_filters(
                forward=MagicMock(),
                data_cov=MagicMock(),
                noise_cov=MagicMock(),
                info=info,
                cfg=beamformer_cfg,
            )
            mock_lcmv.assert_called_once()

    def test_none_noise_cov_passed_through(self, beamformer_cfg):
        """noise_cov=None must reach make_lcmv untouched.

        Building an ad-hoc covariance here instead would make MNE treat it as a
        real one (allow_mismatch=False), and because an ad-hoc covariance is
        isotropic its eigenvectors are the canonical channel basis — so rank
        truncation in _get_ch_whitener would zero out whole channels.
        """
        info = mne.create_info(["MEG001"], 300.0, ["mag"])
        with (
            patch("custom.run_beamformer.make_lcmv") as mock_lcmv,
            patch("custom.run_beamformer.mne.make_ad_hoc_cov") as mock_adhoc,
        ):
            mock_lcmv.return_value = {"mock": "filters"}
            compute_lcmv_filters(
                forward=MagicMock(),
                data_cov=MagicMock(),
                noise_cov=None,
                info=info,
                cfg=beamformer_cfg,
            )
            mock_adhoc.assert_not_called()
            assert mock_lcmv.call_args.kwargs["noise_cov"] is None


# ---------------------------------------------------------------------------
# Per-source-space settings
# ---------------------------------------------------------------------------


class TestResolvePerSpace:
    """Orientation settings may be global or keyed by source space."""

    def test_plain_value_applies_to_every_space(self):
        assert resolve_per_space("max-power", "volume") == "max-power"
        assert resolve_per_space("max-power", "surface") == "max-power"

    def test_dict_selects_by_space(self):
        value = {"volume": "vector", "surface": "max-power"}
        assert resolve_per_space(value, "volume") == "vector"
        assert resolve_per_space(value, "surface") == "max-power"

    def test_dict_falls_back_to_default(self):
        assert resolve_per_space({"volume": "vector"}, "surface", "max-power") == (
            "max-power"
        )

    def test_dict_without_entry_or_default_raises(self):
        with pytest.raises(ValueError, match="No entry for source space"):
            resolve_per_space({"volume": "vector"}, "surface")

    def test_none_is_a_valid_plain_value(self):
        """None means 'fixed orientation', not 'unset' — it must pass through."""
        assert resolve_per_space(None, "surface", "max-power") is None

    @pytest.mark.parametrize(
        "space,expected_ori,expected_norm",
        [
            ("volume", "vector", "unit-noise-gain-invariant"),
            ("surface", "max-power", "nai"),
        ],
    )
    def test_compute_lcmv_filters_uses_per_space_entry(
        self, beamformer_cfg, space, expected_ori, expected_norm
    ):
        beamformer_cfg._beamformer_pick_ori = {
            "volume": "vector",
            "surface": "max-power",
        }
        beamformer_cfg._beamformer_weight_norm = {
            "volume": "unit-noise-gain-invariant",
            "surface": "nai",
        }
        info = mne.create_info(["MEG001"], 300.0, ["mag"])
        with patch("custom.run_beamformer.make_lcmv") as mock_lcmv:
            mock_lcmv.return_value = {"mock": "filters"}
            compute_lcmv_filters(
                forward=MagicMock(),
                data_cov=MagicMock(),
                noise_cov=MagicMock(),
                info=info,
                cfg=beamformer_cfg,
                source_space=space,
            )
        kwargs = mock_lcmv.call_args.kwargs
        assert kwargs["pick_ori"] == expected_ori
        assert kwargs["weight_norm"] == expected_norm


# ---------------------------------------------------------------------------
# Orientation / sign convention
# ---------------------------------------------------------------------------


class TestSurfaceOrientForward:
    """The forward must be surface-oriented so max-power signs follow the normal.

    read_forward_solution returns surf_ori=False, and MNE only converts for
    constrained (loose < 1) inverses — never for a beamformer.  Without this
    conversion the sign is resolved against head +Z instead of the cortical
    normal.
    """

    def test_converts_when_not_surf_ori(self, beamformer_cfg):
        fwd = {"surf_ori": False}
        with patch("custom.run_beamformer.mne.convert_forward_solution") as mock_conv:
            mock_conv.return_value = {"surf_ori": True}
            out = surface_orient_forward(fwd, beamformer_cfg)
        mock_conv.assert_called_once()
        kwargs = mock_conv.call_args.kwargs
        assert kwargs["surf_ori"] is True
        assert kwargs["force_fixed"] is False
        assert kwargs["use_cps"] is True
        assert out["surf_ori"] is True

    def test_noop_when_already_surf_ori(self, beamformer_cfg):
        fwd = {"surf_ori": True}
        with patch("custom.run_beamformer.mne.convert_forward_solution") as mock_conv:
            out = surface_orient_forward(fwd, beamformer_cfg)
        mock_conv.assert_not_called()
        assert out is fwd

    def test_can_be_disabled(self, beamformer_cfg):
        beamformer_cfg._beamformer_surf_ori = False
        fwd = {"surf_ori": False}
        with patch("custom.run_beamformer.mne.convert_forward_solution") as mock_conv:
            out = surface_orient_forward(fwd, beamformer_cfg)
        mock_conv.assert_not_called()
        assert out is fwd


# ---------------------------------------------------------------------------
# Rank resolution
# ---------------------------------------------------------------------------


def _rank_side_effect(info_rank, data_rank):
    """Stand in for mne.compute_rank: rank='info' vs a tolerance-based call."""

    def _compute_rank(inst, rank=None, tol=None, tol_kind=None, **kwargs):
        return dict(info_rank) if rank == "info" else dict(data_rank)

    return _compute_rank


class TestResolveRank:
    """`"data"` is the minimum of the info, data and noise ranks.

    Neither ingredient is reliable alone: `rank="info"` misses what ICA removed,
    and a data-driven SVD misses the SSS deficiency unless its tolerance is set
    well above the float32 floor that FIF round-tripping leaves behind.
    """

    def test_uses_explicit_relative_tolerance_not_auto(self, beamformer_cfg):
        """tol='auto' is ~1e-13 relative and cannot see SSS/ICA-nulled directions."""
        beamformer_cfg._beamformer_rank = "data"
        with patch("custom.run_beamformer.mne.compute_rank") as mock_rank:
            mock_rank.side_effect = _rank_side_effect({"mag": 120}, {"mag": 110})
            resolve_rank(beamformer_cfg, MagicMock(), None, None)

        tol_calls = [c for c in mock_rank.call_args_list if "tol" in c.kwargs]
        assert tol_calls, "expected a tolerance-based compute_rank call"
        assert tol_calls[0].kwargs["tol"] == 1e-6
        assert tol_calls[0].kwargs["tol_kind"] == "relative"

    def test_tolerance_is_configurable(self, beamformer_cfg):
        beamformer_cfg._beamformer_rank = "data"
        beamformer_cfg._beamformer_rank_tol = 1e-5
        with patch("custom.run_beamformer.mne.compute_rank") as mock_rank:
            mock_rank.side_effect = _rank_side_effect({"mag": 120}, {"mag": 110})
            resolve_rank(beamformer_cfg, MagicMock(), None, None)
        tol_calls = [c for c in mock_rank.call_args_list if "tol" in c.kwargs]
        assert tol_calls[0].kwargs["tol"] == 1e-5

    def test_info_rank_caps_an_inflated_data_estimate(self, beamformer_cfg):
        """The real-world case: tol-based SVD still overshoots the SSS basis."""
        beamformer_cfg._beamformer_rank = "data"
        with patch("custom.run_beamformer.mne.compute_rank") as mock_rank:
            mock_rank.side_effect = _rank_side_effect({"mag": 120}, {"mag": 186})
            rank = resolve_rank(beamformer_cfg, MagicMock(), None, None)
        assert rank == {"mag": 120}

    def test_warns_when_data_estimate_does_not_beat_info(self, beamformer_cfg, capsys):
        beamformer_cfg._beamformer_rank = "data"
        with patch("custom.run_beamformer.mne.compute_rank") as mock_rank:
            mock_rank.side_effect = _rank_side_effect({"mag": 120}, {"mag": 186})
            resolve_rank(beamformer_cfg, MagicMock(), None, None)
        assert "not detecting the rank lost" in capsys.readouterr().out

    def test_data_rank_wins_when_ica_cut_deeper(self, beamformer_cfg):
        beamformer_cfg._beamformer_rank = "data"
        with patch("custom.run_beamformer.mne.compute_rank") as mock_rank:
            mock_rank.side_effect = _rank_side_effect({"mag": 120}, {"mag": 104})
            rank = resolve_rank(beamformer_cfg, MagicMock(), None, None)
        assert rank == {"mag": 104}

    def test_noise_rank_included_in_the_minimum(self, beamformer_cfg):
        beamformer_cfg._beamformer_rank = "data"
        with patch("custom.run_beamformer.mne.compute_rank") as mock_rank:
            mock_rank.side_effect = _rank_side_effect({"mag": 120}, {"mag": 110})
            rank = resolve_rank(
                beamformer_cfg, MagicMock(), MagicMock(), {"mag": 75}
            )
        assert rank == {"mag": 75}

    def test_survives_missing_info_rank(self, beamformer_cfg):
        """No proc_history (e.g. HFC-only) — fall back to the data estimate."""
        beamformer_cfg._beamformer_rank = "data"

        def _compute_rank(inst, rank=None, tol=None, tol_kind=None, **kwargs):
            if rank == "info":
                raise ValueError("no SSS info")
            return {"mag": 96}

        with patch("custom.run_beamformer.mne.compute_rank") as mock_rank:
            mock_rank.side_effect = _compute_rank
            out = resolve_rank(beamformer_cfg, MagicMock(), None, None)
        assert out == {"mag": 96}

    def test_empty_room_uses_stored_rank(self, beamformer_cfg):
        """Previously a NameError: rank was only bound on the ad-hoc branch."""
        beamformer_cfg._beamformer_rank = "empty_room"
        rank = resolve_rank(beamformer_cfg, MagicMock(), MagicMock(), {"mag": 71})
        assert rank == {"mag": 71}

    def test_empty_room_without_noise_rank_raises(self, beamformer_cfg):
        beamformer_cfg._beamformer_rank = "empty_room"
        with pytest.raises(ValueError, match="requires a noise covariance"):
            resolve_rank(beamformer_cfg, MagicMock(), None, None)

    @pytest.mark.parametrize("setting", ["info", None, {"mag": 64}])
    def test_other_settings_pass_through(self, beamformer_cfg, setting):
        beamformer_cfg._beamformer_rank = setting
        assert resolve_rank(beamformer_cfg, MagicMock(), None, None) == setting

    def test_vector_with_unit_noise_gain_warns(self, beamformer_cfg, capsys):
        """Suboptimal combination should print a warning."""
        beamformer_cfg._beamformer_pick_ori = "vector"
        beamformer_cfg._beamformer_weight_norm = "unit-noise-gain"
        info = mne.create_info(["MEG001"], 300.0, ["mag"])
        with patch("custom.run_beamformer.make_lcmv") as mock_lcmv:
            mock_lcmv.return_value = {}
            compute_lcmv_filters(
                forward=MagicMock(),
                data_cov=MagicMock(),
                noise_cov=MagicMock(),
                info=info,
                cfg=beamformer_cfg,
            )
        output = capsys.readouterr().out
        assert "WARNING" in output
        assert "unit-noise-gain-invariant" in output


# ---------------------------------------------------------------------------
# run_beamformer_timecourse
# ---------------------------------------------------------------------------


class TestRunBeamformerTimecourse:
    """Test time-locked beamformer analysis data flow."""

    def test_skips_missing_condition(self, beamformer_cfg, capsys):
        """Conditions not in epochs should be skipped with a warning."""
        # Create minimal epochs with just 'stim' condition
        info = mne.create_info(["MEG001"], 300.0, ["mag"])
        epochs = mne.EpochsArray(
            np.random.randn(3, 1, 100) * 1e-13,
            info,
            events=np.array([[0, 0, 1], [100, 0, 1], [200, 0, 1]]),
            event_id={"stim": 1},
        )
        beamformer_cfg.conditions = ["stim", "nonexistent"]
        beamformer_cfg.contrasts = []

        with patch("custom.run_beamformer._all_conditions") as mock_conds:
            mock_conds.return_value = ["stim", "nonexistent"]

            with patch("custom.run_beamformer.apply_lcmv") as mock_apply:
                mock_stc = MagicMock()
                mock_stc.data = np.zeros((10, 100))
                mock_apply.return_value = mock_stc

                stcs = run_beamformer_timecourse(epochs, MagicMock(), beamformer_cfg)

        assert "stim" in stcs
        assert "nonexistent" not in stcs
        output = capsys.readouterr().out
        assert "WARNING" in output

    def test_returns_dict(self, beamformer_cfg):
        """Result should be a dictionary of condition->STC mappings."""
        info = mne.create_info(["MEG001"], 300.0, ["mag"])
        epochs = mne.EpochsArray(
            np.random.randn(3, 1, 100) * 1e-13,
            info,
            events=np.array([[0, 0, 1], [100, 0, 1], [200, 0, 1]]),
            event_id={"stim": 1},
        )
        beamformer_cfg.conditions = ["stim"]
        beamformer_cfg.contrasts = []

        with patch("custom.run_beamformer._all_conditions") as mock_conds:
            mock_conds.return_value = ["stim"]
            with patch("custom.run_beamformer.apply_lcmv") as mock_apply:
                mock_stc = MagicMock()
                mock_stc.data = np.zeros((5, 100))
                mock_apply.return_value = mock_stc

                stcs = run_beamformer_timecourse(epochs, MagicMock(), beamformer_cfg)

        assert isinstance(stcs, dict)
        assert "stim" in stcs


# ---------------------------------------------------------------------------
# run_beamformer_power
# ---------------------------------------------------------------------------


class TestRunBeamformerPower:
    """Test power beamformer analysis data flow."""

    def test_skips_missing_condition(self, beamformer_cfg, capsys):
        info = mne.create_info(["MEG001"], 300.0, ["mag"])
        epochs = mne.EpochsArray(
            np.random.randn(3, 1, 100) * 1e-13,
            info,
            events=np.array([[0, 0, 1], [100, 0, 1], [200, 0, 1]]),
            event_id={"stim": 1},
        )
        beamformer_cfg.conditions = ["stim", "nonexistent"]
        beamformer_cfg.contrasts = []

        with (
            patch("custom.run_beamformer.apply_lcmv_cov") as mock_apply,
            patch("custom.run_beamformer.mne.compute_covariance") as mock_cov,
        ):
            mock_stc = MagicMock()
            mock_stc.data = np.zeros((5, 1))
            mock_stc.copy.return_value = MagicMock(data=np.zeros((5, 1)))
            mock_apply.return_value = mock_stc
            mock_cov.return_value = MagicMock()

            stcs = run_beamformer_power(epochs, MagicMock(), beamformer_cfg)

        # 'stim' should succeed, 'nonexistent' should be skipped
        assert "stim" in stcs
        output = capsys.readouterr().out
        assert "WARNING" in output


# ---------------------------------------------------------------------------
# save_beamformer_results
# ---------------------------------------------------------------------------


class TestSaveBeamformerResults:
    """Test result saving logic."""

    def test_save_creates_files_for_conditions(self, beamformer_cfg, tmp_path):
        beamformer_cfg.deriv_root = str(tmp_path)

        mock_stc = MagicMock()
        stcs = {"stim": mock_stc}

        with patch("custom.run_beamformer.sanitize_cond_name") as mock_san:
            mock_san.return_value = "stim"
            out = save_beamformer_results(beamformer_cfg, MagicMock(), stcs, "time")

        assert "stim" in out
        mock_stc.save.assert_called_once()
