"""Tests for the osl-ephys pipeline stage runners and CLI dispatch."""

from __future__ import annotations

import pickle
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from custom.osl import collate as collate_stage
from custom.osl import preproc as preproc_stage
from custom.osl import source as source_stage
from custom.osl._config import load_config


CONFIG = """
pipeline:
  subject: "007"
  task: TSX
  bids_root: {bids_root}
  outdir: {outdir}
  source_backend: {backend}
preproc:
  - filter: {{l_freq: 0.5, h_freq: 32}}
  - events_from_annotations: {{}}
source_recon:
  rhino:
    - compute_surfaces: {{include_nose: false}}
  freesurfer:
    - fs_coregister: {{}}
    - fs_forward_model: {{gridstep: 8}}
"""


@pytest.fixture
def cfg(tmp_path):
    """A config whose BIDS root and outdir exist under tmp_path."""
    bids_root = tmp_path / "bids"
    outdir = tmp_path / "derivatives" / "osl"
    (bids_root / "sub-007" / "ses-01" / "meg").mkdir(parents=True)

    path = tmp_path / "cfg.yaml"
    path.write_text(
        textwrap.dedent(
            CONFIG.format(bids_root=bids_root, outdir=outdir, backend="rhino")
        )
    )
    return load_config(path, env={})


def touch_input(cfg):
    """Create the BIDS raw file the preproc stage expects."""
    from custom.osl._paths import resolve_paths

    path = resolve_paths(cfg.pipeline).input_fif
    open(path, "w").close()
    return path


def touch_preproc_output(cfg):
    """Create the epochs file the source stage expects."""
    from custom.osl._paths import resolve_paths

    paths = resolve_paths(cfg.pipeline)
    paths.subject_dir.mkdir(parents=True, exist_ok=True)
    paths.source_input_fif.touch()
    return paths.source_input_fif


# ---------------------------------------------------------------------------
# preproc stage
# ---------------------------------------------------------------------------


class TestPreprocStage:
    def test_raises_when_the_bids_input_is_missing(self, cfg):
        with pytest.raises(FileNotFoundError, match="BIDS raw file not found"):
            preproc_stage.run(cfg)

    def test_passes_the_custom_functions_to_osl(self, cfg, monkeypatch):
        touch_input(cfg)
        captured = {}

        def fake_run_proc_chain(config, infile, **kwargs):
            captured["config"] = config
            captured["infile"] = infile
            captured.update(kwargs)
            return {"raw": object(), "events": None, "event_id": None}

        monkeypatch.setattr(
            "osl_ephys.preprocessing.run_proc_chain", fake_run_proc_chain
        )
        cfg.pipeline.gen_report = False

        assert preproc_stage.run(cfg) is True
        assert [f.__name__ for f in captured["extra_funcs"]] == [
            "events_from_annotations",
            "ica_autoreject_safe",
            "ica_kurtosisreject",
        ]

    def test_suppresses_osl_report_generation(self, cfg, monkeypatch):
        # The group page build inside run_proc_chain is a shared-file write and
        # sits inside its try/except; the collate stage does it instead.
        touch_input(cfg)
        captured = {}

        def fake_run_proc_chain(config, infile, **kwargs):
            captured.update(kwargs)
            return {"raw": object()}

        monkeypatch.setattr(
            "osl_ephys.preprocessing.run_proc_chain", fake_run_proc_chain
        )
        cfg.pipeline.gen_report = False
        preproc_stage.run(cfg)

        assert captured["gen_report"] is False
        assert captured["ret_dataset"] is True

    def test_reports_failure_when_osl_returns_an_empty_dataset(
        self, cfg, monkeypatch
    ):
        touch_input(cfg)
        monkeypatch.setattr(
            "osl_ephys.preprocessing.run_proc_chain",
            lambda *a, **k: {},
        )
        assert preproc_stage.run(cfg) is False

    def test_writes_report_data_when_enabled(self, cfg, monkeypatch):
        touch_input(cfg)
        captured = {}

        monkeypatch.setattr(
            "osl_ephys.preprocessing.run_proc_chain",
            lambda *a, **k: {"raw": "RAW", "ica": None, "events": None},
        )

        def fake_gen_html_data(raw, outdir, **kwargs):
            captured["outdir"] = outdir
            captured["run_id"] = kwargs.get("run_id")
            captured["logsdir"] = kwargs.get("logsdir")

        monkeypatch.setattr("osl_ephys.report.gen_html_data", fake_gen_html_data)

        cfg.pipeline.gen_report = True
        assert preproc_stage.run(cfg) is True
        assert captured["run_id"] == "sub-007_ses-01"
        assert captured["outdir"].as_posix().endswith(
            "preproc_report/sub-007_ses-01"
        )

    def test_report_data_arguments_match_what_osl_can_consume(
        self, cfg, monkeypatch
    ):
        # gen_html_data indexes outdir with ``/`` (so it must be a Path) and
        # appends '.log'/'.error.log' to a string logsdir (so it must be the
        # log base, not the logs directory).
        touch_input(cfg)
        captured = {}

        monkeypatch.setattr(
            "osl_ephys.preprocessing.run_proc_chain",
            lambda *a, **k: {"raw": "RAW", "ica": None, "events": None},
        )

        def fake_gen_html_data(raw, outdir, **kwargs):
            captured["outdir"] = outdir
            captured["logsdir"] = kwargs.get("logsdir")

        monkeypatch.setattr("osl_ephys.report.gen_html_data", fake_gen_html_data)

        cfg.pipeline.gen_report = True
        preproc_stage.run(cfg)

        assert isinstance(captured["outdir"], Path)
        assert isinstance(captured["logsdir"], str)
        assert captured["logsdir"].endswith("logs/sub-007_ses-01_preproc")


class TestReportPlotGuard:
    """``_skip_failing_report_plots`` keeps one bad figure from losing a run."""

    def test_a_failing_plot_is_skipped_and_the_rest_still_run(self, monkeypatch):
        from osl_ephys.report import preproc_report

        def boom(*a, **k):
            raise ValueError("electrodes have overlapping positions")

        monkeypatch.setattr(preproc_report, "plot_freqbands", boom)
        monkeypatch.setattr(preproc_report, "plot_rawdata", lambda *a, **k: "raw.png")

        with preproc_stage._skip_failing_report_plots():
            assert preproc_report.plot_freqbands(None) is None
            assert preproc_report.plot_rawdata(None) == "raw.png"

    def test_a_failing_plot_spectra_keeps_its_two_return_values(self, monkeypatch):
        # gen_html_data unpacks plot_spectra into two names.
        from osl_ephys.report import preproc_report

        def boom(*a, **k):
            raise RuntimeError("nope")

        monkeypatch.setattr(preproc_report, "plot_spectra", boom)

        with preproc_stage._skip_failing_report_plots():
            full, zoom = preproc_report.plot_spectra(None)

        assert (full, zoom) == (None, None)

    def test_the_originals_are_restored_afterwards(self, monkeypatch):
        from osl_ephys.report import preproc_report

        original = preproc_report.plot_sensors
        with preproc_stage._skip_failing_report_plots():
            assert preproc_report.plot_sensors is not original
        assert preproc_report.plot_sensors is original

    def test_the_originals_are_restored_when_the_block_raises(self):
        from osl_ephys.report import preproc_report

        original = preproc_report.plot_sensors
        with pytest.raises(KeyError):
            with preproc_stage._skip_failing_report_plots():
                raise KeyError("gen_html_data blew up outside a plot")
        assert preproc_report.plot_sensors is original


# ---------------------------------------------------------------------------
# source stage
# ---------------------------------------------------------------------------


class TestSourceStage:
    def test_raises_when_the_preprocessed_input_is_missing(self, cfg):
        with pytest.raises(FileNotFoundError, match="Preprocessed input not found"):
            source_stage.run(cfg)

    def test_rhino_backend_uses_the_fsl_surface_method(self, cfg, monkeypatch):
        touch_preproc_output(cfg)
        cfg.pipeline.smri = "/anat/T1.nii.gz"
        cfg.pipeline.gen_report = False
        captured = {}

        def fake_run_src_chain(config, **kwargs):
            captured["config"] = config
            captured.update(kwargs)
            return True

        monkeypatch.setattr("osl_ephys.source_recon.run_src_chain", fake_run_src_chain)

        assert source_stage.run(cfg) is True
        assert captured["surface_extraction_method"] == "fsl"
        assert captured["extra_funcs"] is None
        assert captured["gen_report"] is False

    def test_passes_epochs_as_the_epoch_file(self, cfg, monkeypatch):
        touch_preproc_output(cfg)
        cfg.pipeline.smri = "/anat/T1.nii.gz"
        cfg.pipeline.gen_report = False
        captured = {}

        monkeypatch.setattr(
            "osl_ephys.source_recon.run_src_chain",
            lambda config, **kwargs: captured.update(kwargs) or True,
        )
        source_stage.run(cfg)

        assert captured["preproc_file"] is None
        assert captured["epoch_file"].endswith("_epo.fif")

    def test_passes_continuous_data_as_the_preproc_file(self, cfg, monkeypatch):
        cfg.pipeline.source_input = "raw"
        touch_preproc_output(cfg)
        cfg.pipeline.smri = "/anat/T1.nii.gz"
        cfg.pipeline.gen_report = False
        captured = {}

        monkeypatch.setattr(
            "osl_ephys.source_recon.run_src_chain",
            lambda config, **kwargs: captured.update(kwargs) or True,
        )
        source_stage.run(cfg)

        assert captured["epoch_file"] is None
        assert captured["preproc_file"].endswith("_preproc-raw.fif")

    def test_freesurfer_backend_supplies_the_custom_wrappers(
        self, cfg, monkeypatch
    ):
        cfg.pipeline.source_backend = "freesurfer"
        cfg.pipeline.freesurfer_subjects_dir = "/fs"
        cfg.pipeline.gen_report = False
        touch_preproc_output(cfg)
        monkeypatch.setenv("FREESURFER_HOME", "/opt/freesurfer")
        captured = {}

        monkeypatch.setattr(
            "osl_ephys.source_recon.run_src_chain",
            lambda config, **kwargs: captured.update(
                {"config": config, **kwargs}
            )
            or True,
        )

        assert source_stage.run(cfg) is True
        assert captured["surface_extraction_method"] == "freesurfer"
        assert [f.__name__ for f in captured["extra_funcs"]] == [
            "fs_coregister",
            "fs_forward_model",
            "fs_beamform_and_parcellate",
        ]

    def test_raises_when_coregistration_needs_an_missing_smri(self, cfg):
        touch_preproc_output(cfg)
        cfg.pipeline.smri = None
        with pytest.raises(FileNotFoundError, match="No T1w image"):
            source_stage.run(cfg)

    def test_reports_failure_when_osl_returns_false(self, cfg, monkeypatch):
        touch_preproc_output(cfg)
        cfg.pipeline.smri = "/anat/T1.nii.gz"
        monkeypatch.setattr(
            "osl_ephys.source_recon.run_src_chain", lambda *a, **k: False
        )
        assert source_stage.run(cfg) is False


class TestInjectFreesurferDefaults:
    def _paths(self):
        return SimpleNamespace(
            freesurfer_subjects_dir="/fs/subjects",
            trans="/fs/subjects/sub-007/bem/sub-007-trans.fif",
        )

    def test_injects_the_resolved_paths(self):
        config = {
            "source_recon": [
                {"fs_coregister": {}},
                {"fs_forward_model": {"gridstep": 8}},
            ]
        }
        result = source_stage._inject_fs_defaults(config, self._paths())

        coreg = result["source_recon"][0]["fs_coregister"]
        forward = result["source_recon"][1]["fs_forward_model"]
        assert coreg["subjects_dir"] == "/fs/subjects"
        assert forward["trans"].endswith("sub-007-trans.fif")
        assert forward["gridstep"] == 8

    def test_explicit_values_win(self):
        config = {
            "source_recon": [{"fs_forward_model": {"subjects_dir": "/custom"}}]
        }
        result = source_stage._inject_fs_defaults(config, self._paths())
        assert result["source_recon"][0]["fs_forward_model"]["subjects_dir"] == (
            "/custom"
        )

    def test_leaves_other_steps_alone(self):
        config = {"source_recon": [{"compute_surfaces": {"include_nose": False}}]}
        result = source_stage._inject_fs_defaults(config, self._paths())
        assert result["source_recon"][0]["compute_surfaces"] == {
            "include_nose": False
        }

    def test_does_not_mutate_the_input(self):
        config = {"source_recon": [{"fs_coregister": {}}]}
        source_stage._inject_fs_defaults(config, self._paths())
        assert config["source_recon"][0]["fs_coregister"] == {}

    def test_handles_a_step_with_null_options(self):
        config = {"source_recon": [{"fs_coregister": None}]}
        result = source_stage._inject_fs_defaults(config, self._paths())
        assert result["source_recon"][0]["fs_coregister"]["subjects_dir"] == (
            "/fs/subjects"
        )


class TestEnsureFreesurferEnv:
    def test_mirrors_freesurfer_home(self, monkeypatch):
        monkeypatch.delenv("FREESURFERDIR", raising=False)
        monkeypatch.setenv("FREESURFER_HOME", "/opt/fs")
        paths = SimpleNamespace(freesurfer_subjects_dir="/fs/subjects")

        source_stage._ensure_freesurfer_env(paths)

        import os

        assert os.environ["FREESURFERDIR"] == "/opt/fs"
        assert os.environ["SUBJECTS_DIR"] == "/fs/subjects"

    def test_raises_with_a_clear_message_when_unset(self, monkeypatch):
        monkeypatch.delenv("FREESURFERDIR", raising=False)
        monkeypatch.delenv("FREESURFER_HOME", raising=False)
        paths = SimpleNamespace(freesurfer_subjects_dir="/fs")

        with pytest.raises(ValueError, match="FREESURFER_HOME"):
            source_stage._ensure_freesurfer_env(paths)


# ---------------------------------------------------------------------------
# collate stage
# ---------------------------------------------------------------------------


class TestCollateStage:
    def test_skips_report_directories_with_no_subject_data(self, cfg):
        assert collate_stage.run(cfg) is False

    def test_builds_the_reports_that_have_data(self, cfg, monkeypatch):
        from custom.osl._paths import resolve_paths

        paths = resolve_paths(cfg.pipeline)
        subject_report = paths.preproc_reportdir / "sub-007_ses-01"
        subject_report.mkdir(parents=True)
        with open(subject_report / "data.pkl", "wb") as f:
            pickle.dump({"fif_id": "sub-007_ses-01"}, f)

        built = []
        monkeypatch.setattr(
            "osl_ephys.report.gen_html_page", lambda d: built.append(("page", d))
        )
        monkeypatch.setattr(
            "osl_ephys.report.gen_html_summary",
            lambda d: built.append(("summary", d)),
        )

        assert collate_stage.run(cfg) is True
        assert [kind for kind, _ in built] == ["page", "summary"]

    def test_a_failure_in_one_report_does_not_stop_the_other(
        self, cfg, monkeypatch
    ):
        from custom.osl._paths import resolve_paths

        paths = resolve_paths(cfg.pipeline)
        for reportdir in (paths.preproc_reportdir, paths.src_reportdir):
            subject = reportdir / "sub-007_ses-01"
            subject.mkdir(parents=True)
            with open(subject / "data.pkl", "wb") as f:
                pickle.dump({}, f)

        def boom(_):
            raise RuntimeError("render failed")

        built = []
        monkeypatch.setattr("osl_ephys.report.gen_html_page", boom)
        monkeypatch.setattr("osl_ephys.report.gen_html_summary", boom)
        monkeypatch.setattr(
            "osl_ephys.report.src_report.gen_html_page",
            lambda d: built.append(d),
        )
        monkeypatch.setattr(
            "osl_ephys.report.src_report.gen_html_summary",
            lambda d: built.append(d),
        )

        assert collate_stage.run(cfg) is True
        assert len(built) == 2

    def test_has_subject_data_requires_a_data_pkl(self, tmp_path):
        assert collate_stage._has_subject_data(tmp_path / "missing") is False

        reportdir = tmp_path / "report"
        (reportdir / "sub-007").mkdir(parents=True)
        assert collate_stage._has_subject_data(reportdir) is False

        (reportdir / "sub-007" / "data.pkl").touch()
        assert collate_stage._has_subject_data(reportdir) is True


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


class TestCli:
    @pytest.fixture(autouse=True)
    def _import_cli(self):
        import custom.run_osl as run_osl

        self.run_osl = run_osl

    def test_every_stage_is_dispatchable(self, cfg, monkeypatch):
        called = []
        for name, module in [
            ("preproc", preproc_stage),
            ("source", source_stage),
            ("collate", collate_stage),
        ]:
            monkeypatch.setattr(
                module, "run", lambda c, n=name: called.append(n) or True
            )
        from custom.osl import validate as validate_stage

        monkeypatch.setattr(
            validate_stage, "run", lambda c: called.append("validate") or True
        )

        for stage in ("preproc", "source", "collate", "validate"):
            assert self.run_osl.run_stage(stage, cfg) is True

        assert called == ["preproc", "source", "collate", "validate"]

    def test_all_runs_preproc_then_source(self, cfg, monkeypatch):
        called = []
        monkeypatch.setattr(
            preproc_stage, "run", lambda c: called.append("preproc") or True
        )
        monkeypatch.setattr(
            source_stage, "run", lambda c: called.append("source") or True
        )

        assert self.run_osl.run_stage("all", cfg) is True
        assert called == ["preproc", "source"]

    def test_all_stops_when_preprocessing_fails(self, cfg, monkeypatch):
        called = []
        monkeypatch.setattr(preproc_stage, "run", lambda c: False)
        monkeypatch.setattr(
            source_stage, "run", lambda c: called.append("source") or True
        )

        assert self.run_osl.run_stage("all", cfg) is False
        assert called == []

    def test_rejects_an_unknown_stage(self, cfg):
        with pytest.raises(ValueError, match="Unknown stage"):
            self.run_osl.run_stage("beamform", cfg)

    def test_subject_override(self, tmp_path, monkeypatch):
        path = tmp_path / "cfg.yaml"
        path.write_text(
            textwrap.dedent(
                CONFIG.format(
                    bids_root=tmp_path, outdir=tmp_path / "out", backend="rhino"
                )
            )
        )
        seen = {}
        monkeypatch.setattr(
            self.run_osl, "run_stage", lambda s, c: seen.update(
                {"subject": c.pipeline.subject}
            )
            or True,
        )

        exit_code = self.run_osl.main(
            ["--stage", "validate", "--config", str(path), "--subject", "011"]
        )
        assert exit_code == 0
        assert seen["subject"] == "011"

    def test_returns_nonzero_when_a_stage_fails(self, tmp_path, monkeypatch):
        path = tmp_path / "cfg.yaml"
        path.write_text(
            textwrap.dedent(
                CONFIG.format(
                    bids_root=tmp_path, outdir=tmp_path / "out", backend="rhino"
                )
            )
        )
        monkeypatch.setattr(self.run_osl, "run_stage", lambda s, c: False)
        assert (
            self.run_osl.main(["--stage", "preproc", "--config", str(path)]) == 1
        )

    def test_returns_nonzero_for_a_missing_config(self, tmp_path):
        assert (
            self.run_osl.main(
                ["--stage", "preproc", "--config", str(tmp_path / "nope.yaml")]
            )
            == 1
        )
