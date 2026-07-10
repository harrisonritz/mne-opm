"""Tests for coreg_diagnostics.py — coregistration diagnostic functions.

Exercises argument validation, data flow, JSON shape and per-section logic
without requiring real BIDS data or a working off-screen GL stack.  Heavy
MNE I/O (``mne.viz.plot_bem``, ``mne.viz.plot_alignment``, ``stc.plot``,
``mne.read_forward_solution``, ``mne_bids.get_head_mri_trans``, ...) is
patched.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from custom.coreg_diagnostics import (
    _available_surfaces,
    _diag_paths,
    _inventory_freesurfer,
    _save_fig,
    _signed_volume,
    build_json_report,
    load_diagnostic_data,
    run_alignment_diagnostics,
    run_bem_diagnostics,
    run_headpoint_distance_diagnostic,
    run_sensitivity_diagnostics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def coreg_cfg(tmp_path):
    """Standard coreg-diagnostics config namespace."""
    return SimpleNamespace(
        _run_coreg_diagnostics=True,
        _coreg_diag_output_formats=["png"],
        _coreg_diag_dpi=72,
        _coreg_diag_figsize=(4, 4),
        _coreg_diag_run_bem=True,
        _coreg_diag_run_alignment=True,
        _coreg_diag_run_headpoint=True,
        _coreg_diag_run_sensitivity=True,
        _coreg_diag_alignment_views=["frontal"],
        _coreg_diag_sensitivity_modes=["free"],
        _coreg_diag_make_gif=False,
        _coreg_diag_use_nilearn=False,
        _version="test",
        subjects=["001"],
        sessions=["01"],
        task="task",
        bids_root=str(tmp_path / "bids"),
        deriv_root=str(tmp_path / "deriv"),
        ch_types=["mag"],
        n_jobs=1,
        datatype="meg",
    )


@pytest.fixture()
def fake_fs_subject(tmp_path):
    """Build a minimal FreeSurfer subject directory tree on disk."""
    subjects_dir = tmp_path / "subjects"
    fs = subjects_dir / "sub-001_ses-01"
    (fs / "bem").mkdir(parents=True)
    (fs / "mri").mkdir()
    (fs / "surf").mkdir()
    # Touch a few of the expected files so the inventory finds them.
    (fs / "mri" / "T1.mgz").write_bytes(b"x")
    (fs / "surf" / "lh.pial").write_bytes(b"x")
    (fs / "surf" / "rh.pial").write_bytes(b"x")
    return str(subjects_dir), "sub-001_ses-01"


# ---------------------------------------------------------------------------
# _save_fig — format and kind dispatch
# ---------------------------------------------------------------------------


class TestSaveFig:
    def test_mpl_writes_each_format(self, coreg_cfg, tmp_path):
        coreg_cfg._coreg_diag_output_formats = ["png", "pdf"]
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        written = _save_fig(fig, "stem", coreg_cfg, tmp_path, kind="mpl")
        assert (tmp_path / "stem.png").exists()
        assert (tmp_path / "stem.pdf").exists()
        assert len(written) == 2

    def test_pyvista_screenshot_called(self, coreg_cfg, tmp_path):
        fake = MagicMock()
        _save_fig(fake, "stem", coreg_cfg, tmp_path, kind="pyvista")
        fake.plotter.screenshot.assert_called_once()
        fake.plotter.close.assert_called_once()

    def test_brain_save_image_called(self, coreg_cfg, tmp_path):
        fake = MagicMock()
        _save_fig(fake, "stem", coreg_cfg, tmp_path, kind="brain")
        fake.save_image.assert_called_once()
        fake.close.assert_called_once()

    def test_unknown_kind_raises(self, coreg_cfg, tmp_path):
        with pytest.raises(ValueError, match="Unknown figure kind"):
            _save_fig(MagicMock(), "x", coreg_cfg, tmp_path, kind="bogus")

    def test_pyvista_warns_on_vector_format(self, coreg_cfg, tmp_path):
        coreg_cfg._coreg_diag_output_formats = ["pdf"]
        fake = MagicMock()
        with pytest.warns(UserWarning, match="3D figure"):
            _save_fig(fake, "stem", coreg_cfg, tmp_path, kind="pyvista")

    def test_pyvista_save_failure_writes_error_file(self, coreg_cfg, tmp_path):
        fake = MagicMock()
        fake.plotter.screenshot.side_effect = RuntimeError("no GL")
        _save_fig(fake, "stem", coreg_cfg, tmp_path, kind="pyvista")
        err = tmp_path / "stem.error.txt"
        assert err.exists()
        assert "no GL" in err.read_text()


# ---------------------------------------------------------------------------
# _diag_paths — output folder + basename
# ---------------------------------------------------------------------------


class TestDiagPaths:
    def test_paths_constructed_with_bids_subfolder(self, coreg_cfg):
        paths = _diag_paths(coreg_cfg)
        assert paths["out_dir"].name == "coreg_diagnostics"
        assert paths["out_dir"].exists()
        assert paths["basename"] == "sub-001_ses-01_task-task"
        assert paths["subject_clean"] == "001"
        assert paths["session_clean"] == "01"

    def test_basename_uses_task(self, coreg_cfg):
        coreg_cfg.task = "MMN"
        paths = _diag_paths(coreg_cfg)
        assert paths["basename"].endswith("_task-MMN")


# ---------------------------------------------------------------------------
# _inventory_freesurfer + _signed_volume
# ---------------------------------------------------------------------------


class TestInventory:
    def test_inventory_reports_existing_and_missing(self, fake_fs_subject):
        subjects_dir, fs_subject = fake_fs_subject
        inv = _inventory_freesurfer(subjects_dir, fs_subject)
        assert inv["mri/T1.mgz"]["exists"] is True
        assert inv["surf/lh.pial"]["exists"] is True
        assert inv["bem/inner_skull.surf"]["exists"] is False
        # Glob keys present even when no matches.
        assert inv["bem_solution_glob"]["exists"] is False
        assert inv["bem_solution_glob"]["matches"] == []


class TestSignedVolume:
    def test_unit_cube(self):
        # Triangulate a unit cube — divergence theorem should give volume == 1.
        verts = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=float,
        )
        tris = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],  # bottom
                [4, 5, 6],
                [4, 6, 7],  # top
                [0, 1, 5],
                [0, 5, 4],  # front
                [2, 3, 7],
                [2, 7, 6],  # back
                [1, 2, 6],
                [1, 6, 5],  # right
                [3, 0, 4],
                [3, 4, 7],  # left
            ]
        )
        vol = _signed_volume(verts, tris)
        assert abs(vol - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# run_bem_diagnostics
# ---------------------------------------------------------------------------


class TestRunBemDiagnostics:
    def test_renders_per_orientation(self, coreg_cfg, fake_fs_subject, tmp_path):
        subjects_dir, fs_subject = fake_fs_subject
        with (
            patch("custom.coreg_diagnostics.mne.viz.plot_bem") as mock_plot,
            patch("custom.coreg_diagnostics._surface_metrics", return_value={}),
        ):
            mock_plot.side_effect = lambda **kw: plt.figure()
            out = run_bem_diagnostics(
                coreg_cfg, fs_subject, subjects_dir, tmp_path, "stem"
            )
        assert mock_plot.call_count == 3
        assert len(out["plot_bem"]) == 3
        assert "inventory" in out
        assert out["nilearn_overlay"] is None  # disabled in fixture

    def test_failure_one_orientation_continues(
        self, coreg_cfg, fake_fs_subject, tmp_path
    ):
        subjects_dir, fs_subject = fake_fs_subject

        def _maybe_fail(**kw):
            if kw["orientation"] == "sagittal":
                raise RuntimeError("boom")
            return plt.figure()

        with (
            patch("custom.coreg_diagnostics.mne.viz.plot_bem", side_effect=_maybe_fail),
            patch("custom.coreg_diagnostics._surface_metrics", return_value={}),
        ):
            out = run_bem_diagnostics(
                coreg_cfg, fs_subject, subjects_dir, tmp_path, "stem"
            )
        # Two orientations succeeded, one failed.
        assert len(out["plot_bem"]) == 2


# ---------------------------------------------------------------------------
# run_alignment_diagnostics
# ---------------------------------------------------------------------------


class TestRunAlignmentDiagnostics:
    def test_screenshot_per_view(self, coreg_cfg, fake_fs_subject, tmp_path):
        subjects_dir, fs_subject = fake_fs_subject
        coreg_cfg._coreg_diag_alignment_views = ["frontal", "lateral_left"]

        fake_fig = MagicMock()
        with (
            patch(
                "custom.coreg_diagnostics.mne.viz.plot_alignment",
                return_value=fake_fig,
            ),
            patch("custom.coreg_diagnostics.mne.viz.set_3d_view") as mock_view,
        ):
            out = run_alignment_diagnostics(
                coreg_cfg,
                MagicMock(),
                MagicMock(),
                fs_subject,
                subjects_dir,
                tmp_path,
                "stem",
            )
        assert mock_view.call_count == 2
        assert fake_fig.plotter.screenshot.call_count == 2
        assert len(out["plot_alignment"]) == 2

    def test_unknown_view_skipped(self, coreg_cfg, fake_fs_subject, tmp_path):
        subjects_dir, fs_subject = fake_fs_subject
        coreg_cfg._coreg_diag_alignment_views = ["frontal", "totally_made_up"]

        fake_fig = MagicMock()
        with (
            patch(
                "custom.coreg_diagnostics.mne.viz.plot_alignment",
                return_value=fake_fig,
            ),
            patch("custom.coreg_diagnostics.mne.viz.set_3d_view"),
        ):
            out = run_alignment_diagnostics(
                coreg_cfg,
                MagicMock(),
                MagicMock(),
                fs_subject,
                subjects_dir,
                tmp_path,
                "stem",
            )
        assert len(out["plot_alignment"]) == 1

    def test_plot_alignment_failure_returns_error(
        self, coreg_cfg, fake_fs_subject, tmp_path
    ):
        subjects_dir, fs_subject = fake_fs_subject
        with patch(
            "custom.coreg_diagnostics.mne.viz.plot_alignment",
            side_effect=RuntimeError("no display"),
        ):
            out = run_alignment_diagnostics(
                coreg_cfg,
                MagicMock(),
                MagicMock(),
                fs_subject,
                subjects_dir,
                tmp_path,
                "stem",
            )
        assert "error" in out


class TestAvailableSurfaces:
    def test_filters_to_existing_files(self, fake_fs_subject):
        subjects_dir, fs_subject = fake_fs_subject
        # Only lh/rh.pial exist → expect "brain" but not "head-dense" /
        # "inner_skull".
        surfaces = _available_surfaces(subjects_dir, fs_subject)
        assert "brain" in surfaces
        assert "head-dense" not in surfaces
        assert "inner_skull" not in surfaces


# ---------------------------------------------------------------------------
# run_headpoint_distance_diagnostic
# ---------------------------------------------------------------------------


class TestHeadpointDistance:
    def test_skip_no_trans(self, coreg_cfg, fake_fs_subject, tmp_path):
        subjects_dir, fs_subject = fake_fs_subject
        out = run_headpoint_distance_diagnostic(
            coreg_cfg,
            MagicMock(get=lambda k: []),
            trans=None,
            fs_subject=fs_subject,
            fs_subjects_dir=subjects_dir,
            out_dir=tmp_path,
            basename="stem",
        )
        assert out == {"skipped": "no trans"}

    def test_skip_no_dig(self, coreg_cfg, fake_fs_subject, tmp_path):
        subjects_dir, fs_subject = fake_fs_subject
        info = MagicMock()
        info.get = lambda k: [] if k == "dig" else None
        out = run_headpoint_distance_diagnostic(
            coreg_cfg,
            info,
            trans=MagicMock(),
            fs_subject=fs_subject,
            fs_subjects_dir=subjects_dir,
            out_dir=tmp_path,
            basename="stem",
        )
        assert out == {"skipped": "no dig points in info"}

    def test_stats_and_image(self, coreg_cfg, fake_fs_subject, tmp_path):
        subjects_dir, fs_subject = fake_fs_subject
        info = MagicMock()
        info.get = lambda k: ["dig1"] if k == "dig" else None
        fake_coreg = MagicMock()
        fake_coreg.compute_dig_mri_distances.return_value = np.array(
            [0.001, 0.002, 0.003]
        )
        with patch(
            "custom.coreg_diagnostics.mne.coreg.Coregistration",
            return_value=fake_coreg,
        ):
            out = run_headpoint_distance_diagnostic(
                coreg_cfg,
                info,
                trans=MagicMock(),
                fs_subject=fs_subject,
                fs_subjects_dir=subjects_dir,
                out_dir=tmp_path,
                basename="stem",
            )
        assert out["n_points"] == 3
        assert abs(out["mean_mm"] - 2.0) < 1e-6
        assert abs(out["max_mm"] - 3.0) < 1e-6
        assert len(out["image"]) == 1
        assert Path(out["image"][0]).exists()


# ---------------------------------------------------------------------------
# run_sensitivity_diagnostics
# ---------------------------------------------------------------------------


class TestRunSensitivityDiagnostics:
    def test_skip_no_forward(self, coreg_cfg, fake_fs_subject, tmp_path):
        subjects_dir, fs_subject = fake_fs_subject
        out = run_sensitivity_diagnostics(
            coreg_cfg,
            forward=None,
            fs_subject=fs_subject,
            fs_subjects_dir=subjects_dir,
            out_dir=tmp_path,
            basename="stem",
        )
        assert out == {"skipped": "no forward"}

    def test_per_mode_and_view(self, coreg_cfg, fake_fs_subject, tmp_path):
        subjects_dir, fs_subject = fake_fs_subject
        coreg_cfg._coreg_diag_sensitivity_modes = ["free", "radiality"]
        fake_stc = MagicMock()
        fake_stc.plot.return_value = MagicMock()
        with patch(
            "custom.coreg_diagnostics.mne.sensitivity_map", return_value=fake_stc
        ):
            out = run_sensitivity_diagnostics(
                coreg_cfg,
                forward=MagicMock(),
                fs_subject=fs_subject,
                fs_subjects_dir=subjects_dir,
                out_dir=tmp_path,
                basename="stem",
            )
        # mag × {free, radiality} × 4 hemi/view combinations = 8 plots.
        assert "mag_free" in out
        assert "mag_radiality" in out
        assert len(out["mag_free"]["images"]) == 4
        assert len(out["mag_radiality"]["images"]) == 4
        assert fake_stc.plot.call_count == 8
        assert fake_stc.save.call_count == 2

    def test_sensitivity_map_failure_recorded(
        self, coreg_cfg, fake_fs_subject, tmp_path
    ):
        subjects_dir, fs_subject = fake_fs_subject
        with patch(
            "custom.coreg_diagnostics.mne.sensitivity_map",
            side_effect=RuntimeError("singular"),
        ):
            out = run_sensitivity_diagnostics(
                coreg_cfg,
                forward=MagicMock(),
                fs_subject=fs_subject,
                fs_subjects_dir=subjects_dir,
                out_dir=tmp_path,
                basename="stem",
            )
        assert out["mag_free"]["error"] == "singular"


# ---------------------------------------------------------------------------
# load_diagnostic_data
# ---------------------------------------------------------------------------


class TestLoadDiagnosticData:
    def _common_patches(self, **overrides):
        defaults = dict(
            read_info_return=MagicMock(),
            read_fwd_return=MagicMock(),
            head_mri_trans_return=MagicMock(),
            fs_subject="sub-001_ses-01",
            fs_subjects_dir="/fs",
        )
        defaults.update(overrides)
        return defaults

    def test_happy_path(self, coreg_cfg):
        with (
            patch("custom.coreg_diagnostics.mne.io.read_info") as mock_info,
            patch("custom.coreg_diagnostics.mne.read_forward_solution") as mock_fwd,
            patch("custom.coreg_diagnostics.get_head_mri_trans") as mock_trans,
            patch(
                "custom.coreg_diagnostics.get_fs_subject", return_value="sub-001_ses-01"
            ),
            patch("custom.coreg_diagnostics.get_fs_subjects_dir", return_value="/fs"),
            patch("custom.coreg_diagnostics.Path.exists", return_value=True),
        ):
            mock_info.return_value = MagicMock()
            mock_fwd.return_value = MagicMock(name="fwd")
            mock_trans.return_value = MagicMock(name="trans")
            data = load_diagnostic_data(coreg_cfg)
        assert data["info"] is mock_info.return_value
        assert data["forward"] is mock_fwd.return_value
        assert data["trans"] is mock_trans.return_value
        assert data["fs_subject"] == "sub-001_ses-01"
        assert data["fs_subjects_dir"] == "/fs"

    def test_missing_epochs_raises(self, coreg_cfg):
        # Path.exists returns False for everything → epochs file missing.
        with patch("custom.coreg_diagnostics.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Clean epochs"):
                load_diagnostic_data(coreg_cfg)

    def test_trans_failure_sets_none(self, coreg_cfg):
        with (
            patch("custom.coreg_diagnostics.mne.io.read_info"),
            patch("custom.coreg_diagnostics.mne.read_forward_solution") as mock_fwd,
            patch(
                "custom.coreg_diagnostics.get_head_mri_trans",
                side_effect=RuntimeError("no landmarks"),
            ),
            patch(
                "custom.coreg_diagnostics.get_fs_subject", return_value="sub-001_ses-01"
            ),
            patch("custom.coreg_diagnostics.get_fs_subjects_dir", return_value="/fs"),
            patch("custom.coreg_diagnostics.Path.exists", return_value=True),
        ):
            mock_fwd.return_value = MagicMock()
            data = load_diagnostic_data(coreg_cfg)
        assert data["trans"] is None

    def test_missing_forward_triggers_compute(self, coreg_cfg):
        # epochs exists, fwd does not → _compute_forward should run.
        existing = {"epo.fif"}

        def fake_exists(self):
            # Path.exists is called on many internal Path objects; only the
            # epochs path should be reported as existing here.
            return any(token in str(self) for token in existing)

        with (
            patch("custom.coreg_diagnostics.mne.io.read_info"),
            patch(
                "custom.coreg_diagnostics.get_head_mri_trans",
                return_value=MagicMock(name="trans"),
            ),
            patch(
                "custom.coreg_diagnostics.get_fs_subject", return_value="sub-001_ses-01"
            ),
            patch("custom.coreg_diagnostics.get_fs_subjects_dir", return_value="/fs"),
            patch(
                "custom.coreg_diagnostics._compute_forward",
                return_value=MagicMock(name="fwd"),
            ) as mock_compute,
            patch.object(Path, "exists", autospec=True, side_effect=fake_exists),
        ):
            data = load_diagnostic_data(coreg_cfg)
        mock_compute.assert_called_once()
        assert data["forward"] is mock_compute.return_value


# ---------------------------------------------------------------------------
# build_json_report
# ---------------------------------------------------------------------------


class TestBuildJsonReport:
    def test_round_trip(self, coreg_cfg, tmp_path):
        paths = {
            "out_dir": tmp_path,
            "subject_clean": "001",
            "session_clean": "01",
            "basename": "sub-001_ses-01_task-task",
        }
        results = {
            "bem": {
                "plot_bem": [Path(tmp_path / "x.png")],
                "metrics": {"v": np.float64(1.5)},
            },
            "headpoint": {"skipped": "no trans"},
        }
        out = build_json_report(coreg_cfg, results, paths)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["subject"] == "001"
        assert loaded["session"] == "01"
        assert loaded["task"] == "task"
        assert loaded["results"]["headpoint"] == {"skipped": "no trans"}
        # numpy + Path objects must be coerced.
        assert loaded["results"]["bem"]["metrics"]["v"] == 1.5
        assert loaded["results"]["bem"]["plot_bem"][0].endswith("x.png")


# ---------------------------------------------------------------------------
# main — early exit
# ---------------------------------------------------------------------------


class TestMainEarlyExit:
    def test_disabled_does_nothing(self, coreg_cfg, tmp_path):
        config_path = tmp_path / "config.py"
        config_path.write_text("# stub\n")

        cfg = SimpleNamespace(**vars(coreg_cfg))
        cfg._run_coreg_diagnostics = False

        with (
            patch("custom.coreg_diagnostics._import_config", return_value=cfg),
            patch("custom.coreg_diagnostics._update_config_from_path"),
            patch(
                "custom.coreg_diagnostics.parse_args",
                return_value=SimpleNamespace(config=str(config_path)),
            ),
            patch("custom.coreg_diagnostics.load_diagnostic_data") as mock_load,
            patch("custom.coreg_diagnostics._setup_3d_backend"),
        ):
            from custom.coreg_diagnostics import main

            main()
        mock_load.assert_not_called()
