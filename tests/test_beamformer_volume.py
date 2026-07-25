"""Tests for the volume-source-space beamformer path in run_beamformer.py.

Covers ``build_volume_forward`` (cache hit/miss, BEM found vs built), the
volume branch of ``load_beamformer_data``, the ``+vol`` naming in
``save_beamformer_results``, and the nilearn figure path in ``add_to_report``.

Follows the repo's mock-only convention: no real forward solutions, source
spaces, or BEM surfaces are ever built — the MNE constructors are patched at
``custom.run_beamformer.*``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from custom.run_beamformer import (
    _find_bem_solution,
    add_to_report,
    build_volume_forward,
    load_beamformer_data,
    resolve_source_spaces,
    save_beamformer_results,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def vol_cfg(tmp_path):
    """Beamformer config with volume-source-space options."""
    deriv = tmp_path / "derivatives"
    deriv.mkdir()
    return SimpleNamespace(
        subjects=["001"],
        sessions=["01"],
        task="restingstate",
        bids_root=str(tmp_path / "bids"),
        deriv_root=str(deriv),
        datatype="meg",
        n_jobs=1,
        noise_cov="ad-hoc",
        conditions=["stim_a", "stim_b"],
        contrasts=[],
        ch_types=["mag"],
        use_template_mri=None,
        mindist=5,
        _run_beamformer=True,
        _beamformer_reg=0.05,
        _beamformer_pick_ori="max-power",
        _beamformer_weight_norm="unit-noise-gain",
        _beamformer_depth=None,
        _beamformer_rank=None,
        _beamformer_save_filters=False,
        _beamformer_add_to_report=False,
        _beamformer_power_tmin=0.0,
        _beamformer_power_tmax=0.5,
        # volume options
        _beamformer_source_space="volume",
        _beamformer_volume_pos=5.0,
        _beamformer_volume_mindist=5.0,
        _beamformer_volume_bem_conductivity=(0.3,),
        _beamformer_volume_bem_ico=4,
        _beamformer_volume_cache=True,
    )


def _patch_common(stack_extra=None):
    """Common patches: fs helpers + BEM conductivity tag."""
    return dict(
        get_fs_subject=patch(
            "custom.run_beamformer.get_fs_subject", return_value="sub-001_ses-01"
        ),
        get_fs_subjects_dir=patch(
            "custom.run_beamformer.get_fs_subjects_dir",
            return_value="/fake/subjects_dir",
        ),
        bem_tag=patch(
            "custom.run_beamformer._get_bem_conductivity",
            return_value=((0.3,), "5120"),
        ),
    )


# ---------------------------------------------------------------------------
# resolve_source_spaces
# ---------------------------------------------------------------------------


class TestResolveSourceSpaces:
    def test_default_is_surface(self):
        assert resolve_source_spaces(SimpleNamespace()) == ["surface"]

    def test_single_string(self):
        cfg = SimpleNamespace(_beamformer_source_space="volume")
        assert resolve_source_spaces(cfg) == ["volume"]

    def test_list_runs_both_in_order(self):
        cfg = SimpleNamespace(_beamformer_source_space=["volume", "surface"])
        assert resolve_source_spaces(cfg) == ["volume", "surface"]

    def test_tuple_accepted(self):
        cfg = SimpleNamespace(_beamformer_source_space=("surface", "volume"))
        assert resolve_source_spaces(cfg) == ["surface", "volume"]

    def test_duplicates_collapsed(self):
        cfg = SimpleNamespace(_beamformer_source_space=["volume", "volume", "surface"])
        assert resolve_source_spaces(cfg) == ["volume", "surface"]

    def test_invalid_entry_raises(self):
        cfg = SimpleNamespace(_beamformer_source_space=["surface", "banana"])
        with pytest.raises(ValueError, match="_beamformer_source_space"):
            resolve_source_spaces(cfg)

    def test_empty_list_raises(self):
        cfg = SimpleNamespace(_beamformer_source_space=[])
        with pytest.raises(ValueError, match="must not be empty"):
            resolve_source_spaces(cfg)


# ---------------------------------------------------------------------------
# _find_bem_solution
# ---------------------------------------------------------------------------


class TestFindBemSolution:
    def test_missing_dir_returns_none(self, tmp_path):
        assert _find_bem_solution(str(tmp_path), "sub-001_ses-01", "5120") is None

    def test_prefers_tagged_file(self, tmp_path):
        fs_subject = "sub-001_ses-01"
        bem_dir = tmp_path / fs_subject / "bem"
        bem_dir.mkdir(parents=True)
        tagged = bem_dir / f"{fs_subject}-5120-bem-sol.fif"
        tagged.touch()
        (bem_dir / f"{fs_subject}-other-bem-sol.fif").touch()
        found = _find_bem_solution(str(tmp_path), fs_subject, "5120")
        assert found == tagged

    def test_falls_back_to_glob(self, tmp_path):
        fs_subject = "sub-001_ses-01"
        bem_dir = tmp_path / fs_subject / "bem"
        bem_dir.mkdir(parents=True)
        other = bem_dir / f"{fs_subject}-1234-bem-sol.fif"
        other.touch()
        found = _find_bem_solution(str(tmp_path), fs_subject, "5120")
        assert found == other


# ---------------------------------------------------------------------------
# build_volume_forward
# ---------------------------------------------------------------------------


class TestBuildVolumeForward:
    def test_cache_hit_returns_without_building(self, vol_cfg):
        """When a cached volume forward exists, it is read and returned as-is."""
        # Create the cached file at the path build_volume_forward computes.
        from mne_bids import BIDSPath

        vol_fwd = BIDSPath(
            subject="001",
            session="01",
            task="restingstate",
            root=vol_cfg.deriv_root,
            datatype="meg",
            acquisition="vol",
            suffix="fwd",
            extension=".fif",
            check=False,
        )
        vol_fwd.fpath.parent.mkdir(parents=True, exist_ok=True)
        vol_fwd.fpath.touch()

        info = mne.create_info(["MEG001"], 300.0, ["mag"])
        sentinel = MagicMock(spec=mne.Forward)

        with (
            patch("custom.run_beamformer.get_fs_subject", return_value="sub-001_ses-01"),
            patch(
                "custom.run_beamformer.get_fs_subjects_dir",
                return_value="/fake/subjects_dir",
            ),
            patch(
                "custom.run_beamformer.mne.read_forward_solution",
                return_value=sentinel,
            ) as mock_read,
            patch("custom.run_beamformer.mne.setup_volume_source_space") as mock_vol,
            patch("custom.run_beamformer.mne.make_forward_solution") as mock_fwd,
        ):
            result = build_volume_forward(vol_cfg, info)

        assert result is sentinel
        mock_read.assert_called_once()
        mock_vol.assert_not_called()
        mock_fwd.assert_not_called()

    def test_cache_miss_builds_and_writes(self, vol_cfg):
        """Cache miss + BEM on disk -> build volume src + forward, then cache it."""
        info = mne.create_info(["MEG001"], 300.0, ["mag"])
        fake_bem = MagicMock(name="bem")
        fake_src = MagicMock(name="src")
        fake_fwd = MagicMock(spec=mne.Forward)

        with (
            patch("custom.run_beamformer.get_fs_subject", return_value="sub-001_ses-01"),
            patch(
                "custom.run_beamformer.get_fs_subjects_dir",
                return_value="/fake/subjects_dir",
            ),
            patch(
                "custom.run_beamformer._get_bem_conductivity",
                return_value=((0.3,), "5120"),
            ),
            patch(
                "custom.run_beamformer._find_bem_solution",
                return_value=Path("/fake/bem-sol.fif"),
            ),
            patch(
                "custom.run_beamformer.mne.read_bem_solution", return_value=fake_bem
            ) as mock_read_bem,
            patch("custom.run_beamformer.mne.make_bem_model") as mock_make_model,
            patch(
                "custom.run_beamformer.get_head_mri_trans", return_value="TRANS"
            ) as mock_trans,
            patch(
                "custom.run_beamformer.mne.setup_volume_source_space",
                return_value=fake_src,
            ) as mock_vol,
            patch(
                "custom.run_beamformer.mne.make_forward_solution", return_value=fake_fwd
            ) as mock_fwd,
            patch("custom.run_beamformer.mne.write_forward_solution") as mock_write,
        ):
            result = build_volume_forward(vol_cfg, info)

        assert result is fake_fwd
        mock_read_bem.assert_called_once()  # BEM loaded from disk
        mock_make_model.assert_not_called()  # not rebuilt
        # volume source space built with the configured grid spacing
        assert mock_vol.call_args.kwargs["pos"] == 5.0
        assert mock_vol.call_args.kwargs["bem"] is fake_bem
        # forward built on the volume src and cached
        assert mock_fwd.call_args.kwargs["src"] is fake_src
        mock_write.assert_called_once()

    def test_cache_miss_builds_bem_when_absent(self, vol_cfg):
        """When no BEM is on disk, one is built via make_bem_model/solution."""
        info = mne.create_info(["MEG001"], 300.0, ["mag"])
        with (
            patch("custom.run_beamformer.get_fs_subject", return_value="sub-001_ses-01"),
            patch(
                "custom.run_beamformer.get_fs_subjects_dir",
                return_value="/fake/subjects_dir",
            ),
            patch(
                "custom.run_beamformer._get_bem_conductivity",
                return_value=((0.3,), "5120"),
            ),
            patch("custom.run_beamformer._find_bem_solution", return_value=None),
            patch(
                "custom.run_beamformer.mne.make_bem_model", return_value="MODEL"
            ) as mock_model,
            patch(
                "custom.run_beamformer.mne.make_bem_solution", return_value="BEM"
            ) as mock_sol,
            patch("custom.run_beamformer.get_head_mri_trans", return_value="TRANS"),
            patch(
                "custom.run_beamformer.mne.setup_volume_source_space",
                return_value=MagicMock(),
            ),
            patch(
                "custom.run_beamformer.mne.make_forward_solution",
                return_value=MagicMock(spec=mne.Forward),
            ),
            patch("custom.run_beamformer.mne.write_forward_solution"),
        ):
            build_volume_forward(vol_cfg, info)

        mock_model.assert_called_once()
        assert mock_model.call_args.kwargs["ico"] == 4
        assert mock_model.call_args.kwargs["conductivity"] == (0.3,)
        mock_sol.assert_called_once_with("MODEL")

    def test_no_cache_does_not_write(self, vol_cfg):
        """With caching disabled, the forward is built but never written."""
        vol_cfg._beamformer_volume_cache = False
        info = mne.create_info(["MEG001"], 300.0, ["mag"])
        with (
            patch("custom.run_beamformer.get_fs_subject", return_value="sub-001_ses-01"),
            patch(
                "custom.run_beamformer.get_fs_subjects_dir",
                return_value="/fake/subjects_dir",
            ),
            patch(
                "custom.run_beamformer._get_bem_conductivity",
                return_value=((0.3,), "5120"),
            ),
            patch(
                "custom.run_beamformer._find_bem_solution",
                return_value=Path("/fake/bem-sol.fif"),
            ),
            patch("custom.run_beamformer.mne.read_bem_solution", return_value="BEM"),
            patch("custom.run_beamformer.get_head_mri_trans", return_value="TRANS"),
            patch(
                "custom.run_beamformer.mne.setup_volume_source_space",
                return_value=MagicMock(),
            ),
            patch(
                "custom.run_beamformer.mne.make_forward_solution",
                return_value=MagicMock(spec=mne.Forward),
            ),
            patch("custom.run_beamformer.mne.write_forward_solution") as mock_write,
        ):
            build_volume_forward(vol_cfg, info)
        mock_write.assert_not_called()

    def test_bem_conductivity_shim_supplies_fs_subject(self, vol_cfg):
        """Regression: the real _get_bem_conductivity runs against a raw config
        that has no ``fs_subject``. build_volume_forward must feed it a shim
        carrying fs_subject/use_template_mri/ch_types, so it neither raises
        AttributeError nor mis-tags the BEM lookup (MEG-only -> "5120")."""
        info = mne.create_info(["MEG001"], 300.0, ["mag"])
        # MagicMock so we can inspect the positional args passed to it; return a
        # fake path so the "BEM on disk" branch is taken.
        mock_find = MagicMock(return_value=Path("/fake/bem-sol.fif"))
        with (
            patch("custom.run_beamformer.get_fs_subject", return_value="sub-001_ses-01"),
            patch(
                "custom.run_beamformer.get_fs_subjects_dir",
                return_value="/fake/subjects_dir",
            ),
            # _get_bem_conductivity is deliberately NOT patched here.
            patch("custom.run_beamformer._find_bem_solution", mock_find),
            patch("custom.run_beamformer.mne.read_bem_solution", return_value="BEM"),
            patch("custom.run_beamformer.get_head_mri_trans", return_value="TRANS"),
            patch(
                "custom.run_beamformer.mne.setup_volume_source_space",
                return_value=MagicMock(),
            ),
            patch(
                "custom.run_beamformer.mne.make_forward_solution",
                return_value=MagicMock(spec=mne.Forward),
            ),
            patch("custom.run_beamformer.mne.write_forward_solution"),
        ):
            build_volume_forward(vol_cfg, info)

        # _find_bem_solution(fs_subjects_dir, fs_subject, tag) — the third arg is
        # the conductivity tag the real helper produced from the shim.
        mock_find.assert_called_once()
        assert mock_find.call_args.args[2] == "5120"


# ---------------------------------------------------------------------------
# load_beamformer_data — volume branch
# ---------------------------------------------------------------------------


class TestLoadBeamformerDataVolume:
    def test_volume_mode_builds_forward(self, vol_cfg):
        """Volume mode loads epochs/info then delegates to build_volume_forward,
        without requiring a surface fwd file."""
        deriv = Path(vol_cfg.deriv_root)
        meg_dir = deriv / "sub-001" / "ses-01" / "meg"
        meg_dir.mkdir(parents=True)
        epo_path = meg_dir / "sub-001_ses-01_task-restingstate_proc-clean_epo.fif"
        epo_path.touch()

        info = mne.create_info(["MEG001"], 300.0, ["mag"])
        fake_fwd = MagicMock(spec=mne.Forward)
        fake_fwd.__getitem__ = lambda self, key: [1] if key == "src" else None

        mock_epochs = MagicMock(spec=mne.Epochs)
        mock_epochs.__len__ = lambda self: 10
        mock_epochs.ch_names = ["MEG001"]
        mock_epochs.info = info

        with (
            patch("custom.run_beamformer.mne.read_epochs", return_value=mock_epochs),
            patch("custom.run_beamformer.mne.io.read_info", return_value=info),
            patch(
                "custom.run_beamformer.build_volume_forward", return_value=fake_fwd
            ) as mock_build,
        ):
            data = load_beamformer_data(vol_cfg)

        mock_build.assert_called_once()
        # called with (cfg, info)
        assert mock_build.call_args.args[0] is vol_cfg
        assert data["forward"] is fake_fwd
        assert data["noise_cov"] is None  # ad-hoc -> MNE builds its own

    def test_invalid_source_space_raises(self, vol_cfg):
        vol_cfg._beamformer_source_space = "banana"
        with pytest.raises(ValueError, match="_beamformer_source_space"):
            load_beamformer_data(vol_cfg)

    def test_list_builds_both_forwards(self, vol_cfg):
        """A list runs both: surface loaded from disk, volume built; both kept."""
        vol_cfg._beamformer_source_space = ["surface", "volume"]
        deriv = Path(vol_cfg.deriv_root)
        meg_dir = deriv / "sub-001" / "ses-01" / "meg"
        meg_dir.mkdir(parents=True)
        fwd_path = meg_dir / "sub-001_ses-01_task-restingstate_fwd.fif"
        epo_path = meg_dir / "sub-001_ses-01_task-restingstate_proc-clean_epo.fif"
        fwd_path.touch()
        epo_path.touch()

        info = mne.create_info(["MEG001"], 300.0, ["mag"])
        surf_fwd = MagicMock(spec=mne.Forward)
        surf_fwd.__getitem__ = lambda self, k: [1, 2] if k == "src" else None
        vol_fwd = MagicMock(spec=mne.Forward)
        vol_fwd.__getitem__ = lambda self, k: [1] if k == "src" else None
        mock_epochs = MagicMock(spec=mne.Epochs)
        mock_epochs.__len__ = lambda self: 10
        mock_epochs.ch_names = ["MEG001"]
        mock_epochs.info = info

        with (
            patch(
                "custom.run_beamformer.mne.read_forward_solution",
                return_value=surf_fwd,
            ),
            patch("custom.run_beamformer.mne.read_epochs", return_value=mock_epochs),
            patch("custom.run_beamformer.mne.io.read_info", return_value=info),
            patch(
                "custom.run_beamformer.build_volume_forward", return_value=vol_fwd
            ) as mock_build,
        ):
            data = load_beamformer_data(vol_cfg)

        assert set(data["forwards"]) == {"surface", "volume"}
        assert data["forwards"]["surface"] is surf_fwd
        assert data["forwards"]["volume"] is vol_fwd
        assert data["forward"] is surf_fwd  # first entry in the list
        mock_build.assert_called_once()


# ---------------------------------------------------------------------------
# save_beamformer_results — volume naming
# ---------------------------------------------------------------------------


class TestSaveBeamformerVolumeNaming:
    def test_volume_time_suffix(self, vol_cfg):
        stc = MagicMock()
        stcs = {"stim_a": stc}
        out = save_beamformer_results(
            vol_cfg, filters=MagicMock(), stcs=stcs, analysis_type="time"
        )
        saved_path = str(stc.save.call_args.args[0])
        assert "+lcmv+vol" in saved_path
        assert "+hemi" not in saved_path
        assert "stim_a" in out

    def test_volume_power_suffix(self, vol_cfg):
        stc = MagicMock()
        save_beamformer_results(
            vol_cfg, filters=MagicMock(), stcs={"stim_a": stc}, analysis_type="power"
        )
        saved_path = str(stc.save.call_args.args[0])
        assert "+lcmv-power+vol" in saved_path

    def test_surface_still_uses_hemi(self, vol_cfg):
        """Sanity check: surface mode keeps the +hemi token."""
        vol_cfg._beamformer_source_space = "surface"
        stc = MagicMock()
        save_beamformer_results(
            vol_cfg, filters=MagicMock(), stcs={"stim_a": stc}, analysis_type="time"
        )
        assert "+lcmv+hemi" in str(stc.save.call_args.args[0])

    def test_explicit_source_space_param_overrides_cfg(self, vol_cfg):
        """cfg says volume, but an explicit source_space='surface' wins."""
        stc = MagicMock()
        save_beamformer_results(
            vol_cfg,
            filters=MagicMock(),
            stcs={"stim_a": stc},
            analysis_type="time",
            source_space="surface",
        )
        assert "+lcmv+hemi" in str(stc.save.call_args.args[0])

    def test_volume_filter_gets_acq_vol_tag(self, vol_cfg):
        """Volume filters are tagged acq-vol so a combined run doesn't collide."""
        vol_cfg._beamformer_save_filters = True
        filt = MagicMock()
        save_beamformer_results(
            vol_cfg, filters=filt, stcs={}, analysis_type="time", source_space="volume"
        )
        saved = str(filt.save.call_args.args[0])
        assert "acq-vol" in saved and "lcmv" in saved

    def test_surface_filter_has_no_acq_vol_tag(self, vol_cfg):
        vol_cfg._beamformer_save_filters = True
        filt = MagicMock()
        save_beamformer_results(
            vol_cfg, filters=filt, stcs={}, analysis_type="time", source_space="surface"
        )
        saved = str(filt.save.call_args.args[0])
        assert "acq-vol" not in saved and "lcmv" in saved


# ---------------------------------------------------------------------------
# add_to_report — volume figure path
# ---------------------------------------------------------------------------


class TestAddToReportVolume:
    def _report_cm(self, mock_report):
        cm = MagicMock()
        cm.__enter__ = lambda self: mock_report
        cm.__exit__ = lambda self, *a: False
        return cm

    def test_volume_uses_add_figure_not_add_stc(self, vol_cfg):
        vol_cfg._beamformer_add_to_report = True
        vol_cfg.exec_params = MagicMock()
        vol_cfg.report_stc_n_time_points = 51

        mock_report = MagicMock()
        mock_stc = MagicMock()
        mock_stc.data = np.random.randn(4, 10)
        mock_stc.times = np.linspace(0, 1, 10)

        stcs = {"stim_a": Path("/fake/sub-001_stim_a+lcmv+vol")}

        with (
            patch("custom.run_beamformer.get_fs_subject", return_value="sub-001_ses-01"),
            patch(
                "custom.run_beamformer.get_fs_subjects_dir",
                return_value="/fake/subjects_dir",
            ),
            patch(
                "custom.run_beamformer._open_report",
                return_value=self._report_cm(mock_report),
            ),
            patch(
                "custom.run_beamformer.mne.read_source_estimate",
                return_value=mock_stc,
            ),
            patch("custom.run_beamformer._sanitize_cond_tag", side_effect=lambda c: c),
        ):
            add_to_report(vol_cfg, stcs, analysis_type="time", src=MagicMock())

        mock_report.add_figure.assert_called_once()
        mock_report.add_stc.assert_not_called()
        mock_stc.plot.assert_called_once()
        # src threaded through to the volume plot
        assert "src" in mock_stc.plot.call_args.kwargs

    def test_source_space_param_selects_surface_path(self, vol_cfg):
        """cfg default is volume, but source_space='surface' uses add_stc and a
        space-namespaced title."""
        vol_cfg._beamformer_add_to_report = True
        vol_cfg.exec_params = MagicMock()
        vol_cfg.report_stc_n_time_points = 51
        mock_report = MagicMock()
        stcs = {"stim_a": Path("/fake/sub-001_stim_a+lcmv+hemi")}

        with (
            patch("custom.run_beamformer.get_fs_subject", return_value="sub-001_ses-01"),
            patch(
                "custom.run_beamformer.get_fs_subjects_dir",
                return_value="/fake/subjects_dir",
            ),
            patch(
                "custom.run_beamformer._open_report",
                return_value=self._report_cm(mock_report),
            ),
            patch("custom.run_beamformer._sanitize_cond_tag", side_effect=lambda c: c),
        ):
            add_to_report(
                vol_cfg, stcs, analysis_type="time", src=MagicMock(),
                source_space="surface",
            )

        mock_report.add_stc.assert_called_once()
        mock_report.add_figure.assert_not_called()
        assert "surface" in mock_report.add_stc.call_args.kwargs["title"]

    def test_volume_without_src_skips(self, vol_cfg, capsys):
        vol_cfg._beamformer_add_to_report = True
        vol_cfg.exec_params = MagicMock()
        vol_cfg.report_stc_n_time_points = 51
        mock_report = MagicMock()
        stcs = {"stim_a": Path("/fake/sub-001_stim_a+lcmv+vol")}

        with (
            patch("custom.run_beamformer.get_fs_subject", return_value="sub-001_ses-01"),
            patch(
                "custom.run_beamformer.get_fs_subjects_dir",
                return_value="/fake/subjects_dir",
            ),
            patch(
                "custom.run_beamformer._open_report",
                return_value=self._report_cm(mock_report),
            ),
            patch("custom.run_beamformer._sanitize_cond_tag", side_effect=lambda c: c),
        ):
            add_to_report(vol_cfg, stcs, analysis_type="time", src=None)

        mock_report.add_figure.assert_not_called()
        assert "no source space" in capsys.readouterr().out
