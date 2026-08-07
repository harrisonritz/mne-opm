"""Tests for osl-ephys pipeline config validation."""

from __future__ import annotations

import textwrap

from custom.osl._config import load_config
from custom.osl.validate import validate_config


BASE = """
pipeline:
  subject: "007"
  task: TSX
  bids_root: /data/bids
  outdir: /out
  source_backend: {backend}
preproc:
{preproc}
source_recon:
{source_recon}
"""


def make_cfg(tmp_path, preproc, source_recon, backend="rhino"):
    text = BASE.format(
        backend=backend,
        preproc=textwrap.indent(textwrap.dedent(preproc).strip(), "  "),
        source_recon=textwrap.indent(textwrap.dedent(source_recon).strip(), "  "),
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(text)
    return load_config(path, env={})


VALID_SOURCE = """
- extract_polhemus_from_info: {}
- compute_surfaces: {include_nose: false}
"""


class TestPreprocSteps:
    def test_accepts_osl_and_mne_wrappers_and_raw_methods(self, tmp_path):
        cfg = make_cfg(
            tmp_path,
            """
            - filter: {l_freq: 0.5, h_freq: 32}
            - notch_filter: {freqs: [60]}
            - bad_segments: {segment_len: 500, picks: mag}
            - bad_channels: {picks: mag}
            - ica_raw: {picks: mag, n_components: 20}
            - ica_autoreject: {apply: true}
            - resample: {sfreq: 250}
            """,
            VALID_SOURCE,
        )
        errors, _ = validate_config(cfg)
        assert errors == []

    def test_accepts_the_custom_events_step(self, tmp_path):
        cfg = make_cfg(
            tmp_path, "- events_from_annotations: {}", VALID_SOURCE
        )
        errors, _ = validate_config(cfg)
        assert errors == []

    def test_catches_a_misspelled_step(self, tmp_path):
        cfg = make_cfg(tmp_path, "- filtre: {l_freq: 1}", VALID_SOURCE)
        errors, _ = validate_config(cfg)
        assert any("filtre" in e for e in errors)

    def test_resolves_epochs_methods_against_the_epochs_target(self, tmp_path):
        # drop_bad is a method on Epochs, not Raw, so it only resolves when the
        # step declares target: epochs.
        cfg = make_cfg(
            tmp_path,
            "- drop_bad: {target: epochs, reject: {mag: 4.0e-12}}",
            VALID_SOURCE,
        )
        errors, _ = validate_config(cfg)
        assert errors == []


class TestSourceSteps:
    def test_accepts_a_valid_rhino_chain(self, tmp_path):
        cfg = make_cfg(
            tmp_path,
            "- filter: {}",
            """
            - extract_polhemus_from_info: {}
            - compute_surfaces: {include_nose: false}
            - coregister: {use_nose: false, use_headshape: true}
            - forward_model: {model: Single Layer, gridstep: 8}
            - beamform_and_parcellate:
                chantypes: mag
                rank: {mag: 40}
                parcellation_file: Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
                method: spatial_basis
                orthogonalisation: symmetric
            """,
        )
        errors, _ = validate_config(cfg)
        assert errors == []

    def test_catches_the_extract_fiducials_alias(self, tmp_path):
        # osl-ephys' own tutorials use this name, but the alias is declared as
        # (*args, **kwargs) and run_src_chain's argument check rejects it.
        cfg = make_cfg(tmp_path, "- filter: {}", "- extract_fiducials_from_fif: {}")
        errors, _ = validate_config(cfg)
        assert len(errors) == 1
        assert "extract_fiducials_from_fif" in errors[0]
        assert "needs to be passed" in errors[0]

    def test_catches_a_missing_required_argument(self, tmp_path):
        cfg = make_cfg(
            tmp_path,
            "- filter: {}",
            """
            - beamform_and_parcellate:
                chantypes: mag
                rank: {mag: 40}
            """,
        )
        errors, _ = validate_config(cfg)
        assert any("parcellation_file" in e for e in errors)
        assert any("method" in e for e in errors)

    def test_catches_an_unknown_option(self, tmp_path):
        cfg = make_cfg(
            tmp_path, "- filter: {}", "- compute_surfaces: {include_nse: false}"
        )
        errors, _ = validate_config(cfg)
        assert any("include_nse" in e for e in errors)

    def test_catches_an_unknown_step(self, tmp_path):
        cfg = make_cfg(tmp_path, "- filter: {}", "- beamform_everything: {}")
        errors, _ = validate_config(cfg)
        assert any("beamform_everything" in e for e in errors)

    def test_custom_steps_only_resolve_on_the_freesurfer_backend(self, tmp_path):
        steps = """
        - fs_coregister: {}
        - fs_forward_model: {gridstep: 8}
        """
        rhino = make_cfg(tmp_path, "- filter: {}", steps, backend="rhino")
        assert any("fs_coregister" in e for e in validate_config(rhino)[0])

        freesurfer = make_cfg(
            tmp_path, "- filter: {}", steps, backend="freesurfer"
        )
        freesurfer.pipeline.freesurfer_subjects_dir = "/fs"
        assert validate_config(freesurfer)[0] == []

    def test_accepts_a_valid_freesurfer_chain(self, tmp_path):
        cfg = make_cfg(
            tmp_path,
            "- filter: {}",
            """
            - fs_coregister: {}
            - fs_forward_model: {gridstep: 8, mindist: 5.0}
            - fs_beamform_and_parcellate:
                chantypes: mag
                rank: {mag: 40}
                parcellation_file: Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
                method: spatial_basis
                orthogonalisation: symmetric
            """,
            backend="freesurfer",
        )
        cfg.pipeline.freesurfer_subjects_dir = "/fs"
        errors, _ = validate_config(cfg)
        assert errors == []

    def test_selects_the_step_list_for_the_active_backend(self, tmp_path):
        text = """
        pipeline:
          subject: "007"
          task: TSX
          bids_root: /data/bids
          outdir: /out
          source_backend: freesurfer
          freesurfer_subjects_dir: /fs
        source_recon:
          rhino:
            - beamform_everything: {}
          freesurfer:
            - fs_coregister: {}
        """
        path = tmp_path / "cfg.yaml"
        path.write_text(textwrap.dedent(text))
        cfg = load_config(path, env={})

        # The broken rhino list must not be checked while freesurfer is active.
        errors, _ = validate_config(cfg)
        assert errors == []


class TestWarnings:
    def test_warns_about_inputs_that_do_not_exist(self, tmp_path):
        cfg = make_cfg(tmp_path, "- filter: {}", VALID_SOURCE)
        _, warnings = validate_config(cfg)
        assert any("BIDS raw file does not exist" in w for w in warnings)
        assert any("no T1w image" in w for w in warnings)

    def test_warns_about_a_missing_trans_on_the_freesurfer_backend(self, tmp_path):
        cfg = make_cfg(tmp_path, "- filter: {}", VALID_SOURCE, backend="freesurfer")
        cfg.pipeline.freesurfer_subjects_dir = "/fs"
        _, warnings = validate_config(cfg)
        assert any("coregistration transform" in w for w in warnings)

    def test_warns_about_a_missing_section(self, tmp_path):
        text = """
        pipeline:
          subject: "007"
          task: TSX
          bids_root: /data/bids
          outdir: /out
        preproc:
          - filter: {}
        """
        path = tmp_path / "cfg.yaml"
        path.write_text(textwrap.dedent(text))
        cfg = load_config(path, env={})

        errors, warnings = validate_config(cfg)
        assert errors == []
        assert any("no 'source_recon' section" in w for w in warnings)

    def test_missing_inputs_are_warnings_not_errors(self, tmp_path):
        # A config is routinely validated before the data exists.
        cfg = make_cfg(tmp_path, "- filter: {}", VALID_SOURCE)
        errors, warnings = validate_config(cfg)
        assert errors == []
        assert warnings

    def test_unresolvable_paths_are_reported_as_errors(self, tmp_path):
        cfg = make_cfg(tmp_path, "- filter: {}", VALID_SOURCE)
        cfg.pipeline.outdir = None
        errors, _ = validate_config(cfg)
        assert any("outdir" in e for e in errors)
