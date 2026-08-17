"""Tests for the osl-ephys pipeline config loader and path resolution."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from custom.osl._config import (
    PIPELINE_DEFAULTS,
    expand_env,
    load_config,
    preproc_config,
    source_config,
)
from custom.osl._paths import find_smri, resolve_paths


MINIMAL_CONFIG = """
pipeline:
  subject: "007"
  task: TSX
  bids_root: /data/bids
  outdir: /data/derivatives/osl
preproc:
  - filter: {l_freq: 0.5, h_freq: 32}
source_recon:
  - compute_surfaces: {include_nose: false}
"""


def write_config(tmp_path: Path, text: str, name: str = "cfg.yaml") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(text))
    return path


# ---------------------------------------------------------------------------
# expand_env
# ---------------------------------------------------------------------------


class TestExpandEnv:
    def test_expands_a_plain_reference(self):
        assert expand_env("${HOME}/data", {"HOME": "/root"}) == "/root/data"

    def test_expands_several_references_in_one_string(self):
        env = {"A": "x", "B": "y"}
        assert expand_env("${A}/${B}", env) == "x/y"

    def test_uses_the_default_when_unset(self):
        assert expand_env("${MISSING:-fallback}", {}) == "fallback"

    def test_uses_the_default_when_set_but_empty(self):
        assert expand_env("${EMPTY:-fallback}", {"EMPTY": ""}) == "fallback"

    def test_raises_when_unset_and_no_default(self):
        with pytest.raises(KeyError, match="MISSING"):
            expand_env("${MISSING}", {})

    def test_recurses_into_dicts_and_lists(self):
        value = {"a": ["${X}", {"b": "${X}"}]}
        assert expand_env(value, {"X": "1"}) == {"a": ["1", {"b": "1"}]}

    def test_leaves_non_strings_alone(self):
        assert expand_env(5, {}) == 5
        assert expand_env(None, {}) is None
        assert expand_env(True, {}) is True


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_a_minimal_config(self, tmp_path):
        cfg = load_config(write_config(tmp_path, MINIMAL_CONFIG), env={})
        assert cfg.pipeline.subject == "007"
        assert cfg.pipeline.task == "TSX"
        assert cfg.preproc == [{"filter": {"l_freq": 0.5, "h_freq": 32}}]

    def test_fills_in_defaults(self, tmp_path):
        cfg = load_config(write_config(tmp_path, MINIMAL_CONFIG), env={})
        assert cfg.pipeline.session == PIPELINE_DEFAULTS["session"]
        assert cfg.pipeline.source_backend == "rhino"
        assert cfg.pipeline.overwrite is True
        # Every documented key is present, so getattr never surprises callers.
        for key in PIPELINE_DEFAULTS:
            assert hasattr(cfg.pipeline, key)

    def test_expands_environment_variables(self, tmp_path):
        config = """
        pipeline:
          subject: "${SUBJECT}"
          task: "${EXPERIMENT}"
          bids_root: "${BIDS_DIR}"
          outdir: "${BIDS_DIR}/derivatives/osl"
        preproc:
          - filter: {}
        """
        env = {"SUBJECT": "011", "EXPERIMENT": "TSX", "BIDS_DIR": "/d/bids"}
        cfg = load_config(write_config(tmp_path, config), env=env)
        assert cfg.pipeline.subject == "011"
        assert cfg.pipeline.outdir == "/d/bids/derivatives/osl"

    def test_substitutes_identity_placeholders(self, tmp_path):
        config = """
        pipeline:
          analysis: trial
          subject: "007"
          session: "02"
          task: TSX
          bids_root: /data/bids
          outdir: "/data/derivatives/osl-{analysis}"
          subject_label: "sub-{subject}_ses-{session}"
        preproc:
          - filter: {}
        """
        cfg = load_config(write_config(tmp_path, config), env={})
        assert cfg.pipeline.outdir == "/data/derivatives/osl-trial"
        assert cfg.pipeline.subject_label == "sub-007_ses-02"

    def test_leaves_unknown_placeholders_intact(self, tmp_path):
        # osl-ephys does its own {run_id} templating; we must not eat it.
        config = """
        pipeline:
          subject: "007"
          task: TSX
          bids_root: /data/bids
          outdir: "/data/{run_id}"
        preproc:
          - filter: {}
        """
        cfg = load_config(write_config(tmp_path, config), env={})
        assert cfg.pipeline.outdir == "/data/{run_id}"

    def test_rejects_a_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml", env={})

    def test_rejects_an_empty_file(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            load_config(write_config(tmp_path, ""), env={})

    def test_rejects_an_unknown_top_level_section(self, tmp_path):
        config = MINIMAL_CONFIG + "\ntypo_section:\n  - a\n"
        with pytest.raises(ValueError, match="Unrecognised top-level section"):
            load_config(write_config(tmp_path, config), env={})

    def test_rejects_an_unknown_pipeline_key(self, tmp_path):
        config = """
        pipeline:
          subject: "007"
          task: TSX
          bids_root: /data/bids
          outdir: /out
          sorce_backend: rhino
        preproc:
          - filter: {}
        """
        with pytest.raises(ValueError, match="Unrecognised pipeline key"):
            load_config(write_config(tmp_path, config), env={})

    def test_rejects_an_invalid_backend(self, tmp_path):
        config = MINIMAL_CONFIG.replace(
            "outdir: /data/derivatives/osl",
            "outdir: /data/derivatives/osl\n  source_backend: rihno",
        )
        with pytest.raises(ValueError, match="source_backend"):
            load_config(write_config(tmp_path, config), env={})

    def test_rejects_an_invalid_source_input(self, tmp_path):
        config = MINIMAL_CONFIG.replace(
            "outdir: /data/derivatives/osl",
            "outdir: /data/derivatives/osl\n  source_input: evoked",
        )
        with pytest.raises(ValueError, match="source_input"):
            load_config(write_config(tmp_path, config), env={})

    def test_rejects_a_backend_map_with_an_unknown_key(self, tmp_path):
        config = """
        pipeline:
          subject: "007"
          task: TSX
          bids_root: /data/bids
          outdir: /out
        source_recon:
          rhino:
            - compute_surfaces: {}
          rihno:
            - compute_surfaces: {}
        """
        with pytest.raises(ValueError, match="unrecognised key"):
            load_config(write_config(tmp_path, config), env={})


# ---------------------------------------------------------------------------
# preproc_config / source_config
# ---------------------------------------------------------------------------


class TestStageConfigs:
    def test_preproc_config_shape(self, tmp_path):
        config = MINIMAL_CONFIG.replace(
            "preproc:", "meta:\n  event_codes:\n    trial: 1\npreproc:"
        )
        cfg = load_config(write_config(tmp_path, config), env={})
        result = preproc_config(cfg)
        assert set(result) == {"meta", "preproc"}
        assert result["meta"]["event_codes"] == {"trial": 1}

    def test_preproc_config_supplies_empty_meta(self, tmp_path):
        cfg = load_config(write_config(tmp_path, MINIMAL_CONFIG), env={})
        assert preproc_config(cfg)["meta"] == {}

    def test_preproc_config_raises_without_a_preproc_section(self, tmp_path):
        config = """
        pipeline:
          subject: "007"
          task: TSX
          bids_root: /data/bids
          outdir: /out
        source_recon:
          - compute_surfaces: {}
        """
        cfg = load_config(write_config(tmp_path, config), env={})
        with pytest.raises(ValueError, match="No 'preproc' section"):
            preproc_config(cfg)

    def test_source_config_accepts_a_plain_list(self, tmp_path):
        cfg = load_config(write_config(tmp_path, MINIMAL_CONFIG), env={})
        assert source_config(cfg) == {
            "source_recon": [{"compute_surfaces": {"include_nose": False}}]
        }

    def test_source_config_selects_the_backend(self, tmp_path):
        config = """
        pipeline:
          subject: "007"
          task: TSX
          bids_root: /data/bids
          outdir: /out
          source_backend: freesurfer
        source_recon:
          rhino:
            - compute_surfaces: {}
          freesurfer:
            - fs_forward_model: {gridstep: 8}
        """
        cfg = load_config(write_config(tmp_path, config), env={})
        assert source_config(cfg) == {
            "source_recon": [{"fs_forward_model": {"gridstep": 8}}]
        }

        cfg.pipeline.source_backend = "rhino"
        assert source_config(cfg) == {"source_recon": [{"compute_surfaces": {}}]}

    def test_source_config_raises_when_the_backend_is_absent(self, tmp_path):
        config = """
        pipeline:
          subject: "007"
          task: TSX
          bids_root: /data/bids
          outdir: /out
          source_backend: freesurfer
        source_recon:
          rhino:
            - compute_surfaces: {}
        """
        cfg = load_config(write_config(tmp_path, config), env={})
        with pytest.raises(ValueError, match="no 'freesurfer' entry"):
            source_config(cfg)

    def test_source_config_raises_without_a_source_section(self, tmp_path):
        config = """
        pipeline:
          subject: "007"
          task: TSX
          bids_root: /data/bids
          outdir: /out
        preproc:
          - filter: {}
        """
        cfg = load_config(write_config(tmp_path, config), env={})
        with pytest.raises(ValueError, match="No 'source_recon' section"):
            source_config(cfg)


# ---------------------------------------------------------------------------
# resolve_paths
# ---------------------------------------------------------------------------


class TestResolvePaths:
    def test_resolves_bids_input_and_derivative_outputs(self, tmp_path):
        cfg = load_config(write_config(tmp_path, MINIMAL_CONFIG), env={})
        paths = resolve_paths(cfg.pipeline)

        assert paths.subject_label == "sub-007_ses-01"
        assert paths.input_fif.endswith(
            "sub-007/ses-01/meg/sub-007_ses-01_task-TSX_run-01_meg.fif"
        )
        assert str(paths.preproc_fif).endswith(
            "osl/sub-007_ses-01/sub-007_ses-01_preproc-raw.fif"
        )
        assert str(paths.epochs_fif).endswith(
            "osl/sub-007_ses-01/sub-007_ses-01_epo.fif"
        )
        assert str(paths.logsdir).endswith("osl/logs")
        assert str(paths.preproc_reportdir).endswith("osl/preproc_report")
        assert str(paths.src_reportdir).endswith("osl/src_report")

    def test_source_input_follows_the_config(self, tmp_path):
        cfg = load_config(write_config(tmp_path, MINIMAL_CONFIG), env={})

        cfg.pipeline.source_input = "epochs"
        assert resolve_paths(cfg.pipeline).source_input_fif.name.endswith("_epo.fif")

        cfg.pipeline.source_input = "raw"
        assert resolve_paths(cfg.pipeline).source_input_fif.name.endswith(
            "_preproc-raw.fif"
        )

    def test_requires_the_essential_pipeline_keys(self, tmp_path):
        cfg = load_config(write_config(tmp_path, MINIMAL_CONFIG), env={})
        cfg.pipeline.outdir = None
        with pytest.raises(ValueError, match="outdir"):
            resolve_paths(cfg.pipeline)

    def test_freesurfer_backend_requires_a_subjects_dir(self, tmp_path):
        cfg = load_config(write_config(tmp_path, MINIMAL_CONFIG), env={})
        cfg.pipeline.source_backend = "freesurfer"
        with pytest.raises(ValueError, match="freesurfer_subjects_dir"):
            resolve_paths(cfg.pipeline)

    def test_freesurfer_backend_defaults_the_trans_path(self, tmp_path):
        cfg = load_config(write_config(tmp_path, MINIMAL_CONFIG), env={})
        cfg.pipeline.source_backend = "freesurfer"
        cfg.pipeline.freesurfer_subjects_dir = "/fs/subjects"
        paths = resolve_paths(cfg.pipeline)
        assert paths.trans == (
            "/fs/subjects/sub-007_ses-01/bem/sub-007_ses-01-trans.fif"
        )

    def test_an_explicit_trans_wins(self, tmp_path):
        cfg = load_config(write_config(tmp_path, MINIMAL_CONFIG), env={})
        cfg.pipeline.source_backend = "freesurfer"
        cfg.pipeline.freesurfer_subjects_dir = "/fs/subjects"
        cfg.pipeline.trans = "/custom/my-trans.fif"
        assert resolve_paths(cfg.pipeline).trans == "/custom/my-trans.fif"

    def test_no_run_entity_when_run_is_null(self, tmp_path):
        cfg = load_config(write_config(tmp_path, MINIMAL_CONFIG), env={})
        cfg.pipeline.run = None
        assert "run-" not in resolve_paths(cfg.pipeline).input_fif


class TestFindSmri:
    def test_finds_the_conventional_path(self, tmp_path):
        anat = tmp_path / "sub-007" / "ses-01" / "anat"
        anat.mkdir(parents=True)
        expected = anat / "sub-007_ses-01_T1w.nii.gz"
        expected.touch()
        assert find_smri(tmp_path, "007", "01") == str(expected)

    def test_falls_back_to_an_entity_bearing_name(self, tmp_path):
        anat = tmp_path / "sub-007" / "ses-01" / "anat"
        anat.mkdir(parents=True)
        expected = anat / "sub-007_ses-01_acq-mprage_T1w.nii.gz"
        expected.touch()
        assert find_smri(tmp_path, "007", "01") == str(expected)

    def test_returns_none_when_absent(self, tmp_path):
        assert find_smri(tmp_path, "007", "01") is None

    def test_config_smri_takes_priority(self, tmp_path):
        anat = tmp_path / "sub-007" / "ses-01" / "anat"
        anat.mkdir(parents=True)
        (anat / "sub-007_ses-01_T1w.nii.gz").touch()

        cfg = load_config(write_config(tmp_path, MINIMAL_CONFIG), env={})
        cfg.pipeline.bids_root = str(tmp_path)
        cfg.pipeline.smri = "/elsewhere/T1.nii.gz"
        assert resolve_paths(cfg.pipeline).smri == "/elsewhere/T1.nii.gz"
