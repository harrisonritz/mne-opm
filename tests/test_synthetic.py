"""Tests for the synthetic dataset generator and the committed subject.

Two groups:

* generator tests, which build small phantoms in ``tmp_path`` and check the
  pieces (geometry, schedule, trigger round trip, sensor array) behave;
* integrity tests for the dataset committed under ``synthetic/``, which is what
  development and agent sessions actually run the pipeline against.  These skip
  cleanly if the dataset is not present.

The end-to-end generation test is slow (it builds a BEM and a forward solution)
and is gated behind ``MNE_OPM_RUN_SLOW_TESTS=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mne
import numpy as np
import pytest

from custom.synthetic import (
    build_head_model,
    build_info,
    build_schedule,
    default_sources,
)
from custom.synthetic._geometry import fibonacci_directions, icosphere, tangent_basis
from custom.synthetic.anatomy import write_freesurfer_subject
from custom.synthetic.events import TRIGGER_DESC, trigger_annotations, trigger_waveforms
from custom.synthetic.sensors import EYE_CHANNELS, TRIGGER_CHANNELS


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "synthetic" / "datasets" / "synth"
BIDS_ROOT = DATASET_ROOT / "bids"
SUBJECTS_DIR = BIDS_ROOT / "derivatives" / "freesurfer" / "subjects"
FS_SUBJECT = "sub-001_ses-01"

requires_dataset = pytest.mark.skipif(
    not (BIDS_ROOT / "ground_truth.json").exists(),
    reason="synthetic dataset not present; run src/custom/make_synthetic.py",
)
slow = pytest.mark.skipif(
    not os.environ.get("MNE_OPM_RUN_SLOW_TESTS"),
    reason="set MNE_OPM_RUN_SLOW_TESTS=1 to run",
)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class TestGeometry:
    """Primitives the phantom is built from."""

    @pytest.mark.parametrize(
        "grade,n_vertices,n_tris", [(4, 2562, 5120), (5, 10242, 20480)]
    )
    def test_icosphere_counts(self, grade, n_vertices, n_tris):
        rr, tris = icosphere(grade)
        assert rr.shape == (n_vertices, 3)
        assert tris.shape == (n_tris, 3)

    def test_icosphere_is_unit_norm(self):
        rr, _ = icosphere(4)
        np.testing.assert_allclose(np.linalg.norm(rr, axis=1), 1.0, atol=1e-12)

    def test_tangent_basis_is_orthonormal(self):
        for normal in ([0, 0, 1.0], [1.0, 0, 0], [0.3, -0.5, 0.8]):
            ex, ey = tangent_basis(normal)
            nrm = np.asarray(normal, float)
            nrm /= np.linalg.norm(nrm)
            for vec in (ex, ey):
                assert abs(np.linalg.norm(vec) - 1.0) < 1e-12
                assert abs(vec @ nrm) < 1e-12
            assert abs(ex @ ey) < 1e-12

    def test_fibonacci_directions_are_unit_norm(self):
        dirs = fibonacci_directions(64)
        np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-12)


class TestHeadModel:
    """Phantom geometry and the head coordinate frame it implies."""

    def test_shells_are_nested(self):
        head = build_head_model(seed=0)
        assert np.all(head.inner_skull_axes < head.outer_skull_axes)
        assert np.all(head.outer_skull_axes < head.scalp_axes)

    def test_cortex_inside_inner_skull(self):
        head = build_head_model(seed=0)
        inner, _ = head.shell("inner_skull")
        for hemi in ("lh", "rh"):
            pial = head.cortex(hemi)["pial"] - head.center
            radius = np.sum((pial / head.inner_skull_axes) ** 2, axis=1)
            assert radius.max() < 1.0, f"{hemi} pial pokes through the inner skull"
        assert len(inner) == 2562

    def test_fiducials_lie_on_the_scalp(self):
        head = build_head_model(seed=0)
        for point in head.fiducials.values():
            radius = np.sum(((point - head.center) / head.scalp_axes) ** 2)
            assert abs(radius - 1.0) < 1e-9

    def test_mri_head_transform_is_rigid_and_not_a_translation(self):
        head = build_head_model(seed=0)
        rotation = np.asarray(head.mri_head_t["trans"])[:3, :3]
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(rotation) - 1.0) < 1e-10
        # A pure translation would make coregistration code paths trivial.
        assert not np.allclose(rotation, np.eye(3), atol=1e-3)

    def test_jitter_changes_geometry_but_stays_plausible(self):
        base = build_head_model(seed=0, jitter=0.0)
        jittered = build_head_model(seed=3, jitter=0.06)
        assert not np.allclose(base.scalp_axes, jittered.scalp_axes)
        ratio = jittered.scalp_axes / base.scalp_axes
        assert np.all(ratio > 0.8) and np.all(ratio < 1.2)

    def test_is_deterministic(self):
        a, b = build_head_model(seed=7, jitter=0.06), build_head_model(seed=7, jitter=0.06)
        np.testing.assert_array_equal(a.scalp_axes, b.scalp_axes)


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------


class TestSensorArray:
    """The Cerca-style triaxial OPM array."""

    @pytest.fixture(scope="class")
    def built(self):
        head = build_head_model(seed=0)
        info, helmet = build_info(head, 200.0, n_slots=16, seed=0)
        return head, info, helmet

    def test_channel_counts(self, built):
        _, info, _ = built
        assert len(mne.pick_types(info, meg=True)) == 16 * 3
        assert len(mne.pick_types(info, stim=True)) == len(TRIGGER_CHANNELS)
        for name in EYE_CHANNELS:
            assert name in info["ch_names"]

    def test_slot_labels_are_unique(self, built):
        _, _, helmet = built
        assert len(set(helmet["labels"])) == len(helmet["labels"])

    def test_channel_names_follow_cerca_convention(self, built):
        _, info, _ = built
        for pick in mne.pick_types(info, meg=True):
            slot, sensor, axis = info["ch_names"][pick].split()
            assert axis in ("X", "Y", "Z")
            assert slot and sensor

    def test_each_slot_has_three_orthogonal_axes(self, built):
        _, info, _ = built
        picks = mne.pick_types(info, meg=True)
        for start in range(0, len(picks), 3):
            axes = np.array([info["chs"][p]["loc"][9:12] for p in picks[start : start + 3]])
            np.testing.assert_allclose(axes @ axes.T, np.eye(3), atol=1e-9)

    def test_coil_type_is_opm(self, built):
        from mne.io.constants import FIFF

        _, info, _ = built
        for pick in mne.pick_types(info, meg=True):
            assert info["chs"][pick]["coil_type"] == FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2
            assert info["chs"][pick]["coord_frame"] == FIFF.FIFFV_COORD_DEVICE

    def test_sensors_sit_outside_the_scalp(self, built):
        head, info, helmet = built
        radius = np.sum(((helmet["positions"] - head.center) / head.scalp_axes) ** 2, axis=1)
        assert np.all(radius > 1.0)

    def test_digitisation_has_fiducials_and_head_points(self, built):
        _, info, _ = built
        kinds = [point["kind"] for point in info["dig"]]
        assert kinds.count(mne.io.constants.FIFF.FIFFV_POINT_CARDINAL) == 3
        assert kinds.count(mne.io.constants.FIFF.FIFFV_POINT_EXTRA) > 50

    def test_hfc_projections_can_be_computed(self, built):
        _, info, _ = built
        meg = mne.pick_info(info, mne.pick_types(info, meg=True))
        projs = mne.preprocessing.compute_proj_hfc(meg, order=2)
        assert len(projs) == 8  # (order + 1)^2 - 1


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class TestSchedule:
    """Trial timing, trigger encoding, and the metadata that must match it."""

    @pytest.fixture(scope="class")
    def schedule(self):
        return build_schedule(60.0, seed=0)

    def test_metadata_has_one_row_per_trial(self, schedule):
        n_stim = sum(1 for code in schedule.codes if code in (4, 8))
        assert len(schedule.metadata) == n_stim

    def test_events_are_ordered_and_in_range(self, schedule):
        assert np.all(np.diff(schedule.onsets) >= 0)
        assert schedule.onsets.min() >= 0
        assert schedule.onsets.max() < schedule.duration

    def test_all_codes_are_known(self, schedule):
        assert set(schedule.codes) <= set(TRIGGER_DESC)

    def test_contains_a_break_long_enough_to_detect(self, schedule):
        # find_breaks defaults to a 6 s minimum in the shipped config.
        assert np.diff(schedule.onsets).max() > 6.0

    def test_some_trials_are_unanswered(self, schedule):
        assert 0 < (schedule.metadata["responded"] == 0).sum() < len(schedule.metadata)

    def test_trigger_round_trip(self, schedule):
        """Encoding to eight lines and decoding back must be lossless.

        This is exactly what format_bids.convert_triggers does, so a
        regression here would silently change every event in the dataset.
        """
        sfreq = 200.0
        n_times = int(schedule.duration * sfreq)
        data = trigger_waveforms(schedule, sfreq, n_times)
        assert data.shape == (8, n_times)

        binary = (data > 2.0).astype(int)
        combined = (binary * (2 ** np.arange(8))[:, None]).sum(axis=0)
        rising = np.flatnonzero(np.diff(combined) > 0) + 1
        np.testing.assert_array_equal(combined[rising], schedule.codes)
        np.testing.assert_allclose(rising / sfreq, schedule.onsets, atol=1.0 / sfreq)

    def test_trigger_annotations_match_the_lines(self, schedule):
        sfreq = 200.0
        data = trigger_waveforms(schedule, sfreq, int(schedule.duration * sfreq))
        annot = trigger_annotations(data, sfreq)
        assert len(annot) == len(schedule.onsets)
        assert set(annot.description) <= set(TRIGGER_CHANNELS)


# ---------------------------------------------------------------------------
# FreeSurfer subject
# ---------------------------------------------------------------------------


class TestFreeSurferWriter:
    """A phantom recon MNE's source machinery accepts."""

    @pytest.fixture(scope="class")
    def written(self, tmp_path_factory):
        subjects_dir = tmp_path_factory.mktemp("fs")
        head = build_head_model(seed=0)
        write_freesurfer_subject(head, subjects_dir, "sub-999_ses-01")
        return subjects_dir, head

    @pytest.mark.parametrize(
        "relpath",
        [
            "mri/T1.mgz",
            "mri/transforms/talairach.xfm",
            "surf/lh.white",
            "surf/rh.white",
            "surf/lh.sphere",
            "surf/lh.sphere.reg",
            "surf/lh.pial",
            "surf/lh.curv",
            "bem/inner_skull.surf",
            "bem/outer_skull.surf",
            "bem/outer_skin.surf",
            "bem/sub-999_ses-01-head.fif",
            "bem/sub-999_ses-01-fiducials.fif",
        ],
    )
    def test_required_file_exists(self, written, relpath):
        subjects_dir, _ = written
        assert (subjects_dir / "sub-999_ses-01" / relpath).exists()

    def test_t1_is_conformed(self, written):
        import nibabel as nib

        subjects_dir, _ = written
        img = nib.load(subjects_dir / "sub-999_ses-01" / "mri" / "T1.mgz")
        assert img.shape == (256, 256, 256)
        np.testing.assert_allclose(img.header.get_zooms()[:3], (1.0, 1.0, 1.0))
        # Scanner RAS and surface RAS must coincide, otherwise the landmarks
        # written into the BIDS T1w sidecar will not round-trip.
        np.testing.assert_allclose(
            img.affine, img.header.get_vox2ras_tkr(), atol=1e-6
        )

    def test_fiducials_round_trip(self, written):
        subjects_dir, head = written
        pts, frame = mne.io.read_fiducials(
            subjects_dir / "sub-999_ses-01" / "bem" / "sub-999_ses-01-fiducials.fif"
        )
        assert frame == mne.io.constants.FIFF.FIFFV_COORD_MRI
        by_ident = {p["ident"]: p["r"] for p in pts}
        np.testing.assert_allclose(
            by_ident[mne.io.constants.FIFF.FIFFV_POINT_NASION],
            head.fiducials["nasion"],
            atol=1e-6,
        )

    def test_surfaces_support_oct6_source_space(self, written):
        subjects_dir, _ = written
        src = mne.setup_source_space(
            "sub-999_ses-01",
            spacing="oct6",
            subjects_dir=str(subjects_dir),
            add_dist=False,
            verbose="error",
        )
        assert [s["nuse"] for s in src] == [4098, 4098]

    def test_bem_model_builds(self, written):
        subjects_dir, _ = written
        model = mne.make_bem_model(
            "sub-999_ses-01",
            subjects_dir=str(subjects_dir),
            conductivity=(0.3,),
            verbose="error",
        )
        assert len(model) == 1
        assert model[0]["np"] == 2562


# ---------------------------------------------------------------------------
# The committed dataset
# ---------------------------------------------------------------------------


@requires_dataset
class TestCommittedDataset:
    """Integrity of the subject checked into the repository."""

    @pytest.fixture(scope="class")
    def ground_truth(self):
        return json.loads((BIDS_ROOT / "ground_truth.json").read_text())

    @pytest.fixture(scope="class")
    def raw(self):
        path = (
            BIDS_ROOT
            / "sub-001/ses-01/meg/sub-001_ses-01_task-synth_run-01_meg.fif"
        )
        return mne.io.read_raw_fif(path, preload=False, verbose="error")

    @pytest.mark.parametrize(
        "relpath",
        [
            "dataset_description.json",
            "participants.tsv",
            "ground_truth.json",
            "sub-001/ses-01/meg/sub-001_ses-01_task-synth_run-01_meg.fif",
            "sub-001/ses-01/meg/sub-001_ses-01_task-synth_run-01_events.tsv",
            "sub-001/ses-01/meg/sub-001_ses-01_task-synth_run-01_channels.tsv",
            "sub-001/ses-01/meg/sub-001_ses-01_task-noise_meg.fif",
            "sub-001/ses-01/meg/sub-001_ses-01_coordsystem.json",
            "sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz",
            "sub-001/ses-01/anat/sub-001_ses-01_T1w.json",
        ],
    )
    def test_bids_file_exists(self, relpath):
        assert (BIDS_ROOT / relpath).exists()

    def test_freesurfer_subject_and_template_exist(self):
        for subject in (FS_SUBJECT, "fsaverage"):
            assert (SUBJECTS_DIR / subject / "mri" / "T1.mgz").exists()
            assert (SUBJECTS_DIR / subject / "bem" / "inner_skull.surf").exists()
            assert (SUBJECTS_DIR / subject / "surf" / "lh.sphere.reg").exists()

    def test_channel_composition(self, raw, ground_truth):
        n_meg = ground_truth["subjects"]["001"]["n_meg_channels"]
        assert len(mne.pick_types(raw.info, meg=True)) == n_meg
        # Enough channels for Maxwell filtering at mf_int_order=10 (120 bases)
        # plus mf_ext_order=2 (8 bases).
        assert n_meg >= 128
        assert len(mne.pick_types(raw.info, eog=True)) == 3
        # format_bids drops the trigger channels once they are annotations.
        assert not mne.pick_types(raw.info, stim=True).size

    def test_annotations_cover_the_generic_design(self, raw):
        descriptions = set(raw.annotations.description)
        assert {"trial/cond_a", "trial/cond_b", "ITI", "feedback"} <= descriptions
        assert {"response/left", "response/right"} & descriptions

    def test_metadata_matches_trial_events(self, raw):
        import pandas as pd

        from custom.preprocessing._io import count_condition_events_in_raw

        csv = sorted((DATASET_ROOT / "raw/synth_001/metadata").glob("sub-001_*.csv"))
        assert csv, "behavioural metadata CSV is missing"
        metadata = pd.concat([pd.read_csv(p) for p in csv], ignore_index=True)
        n_events, _ = count_condition_events_in_raw(raw, ("trial",))
        assert n_events == len(metadata)

    def test_events_tsv_agrees_with_the_fif(self, raw):
        from custom.preprocessing._io import (
            count_condition_events_in_raw,
            count_condition_events_in_tsv,
        )

        tsv = (
            BIDS_ROOT
            / "sub-001/ses-01/meg/sub-001_ses-01_task-synth_run-01_events.tsv"
        )
        assert count_condition_events_in_tsv(tsv, ("trial",))[0] == (
            count_condition_events_in_raw(raw, ("trial",))[0]
        )

    def test_planted_bad_channels_are_real_channels(self, raw, ground_truth):
        subject = ground_truth["subjects"]["001"]
        for name in subject["noisy_channels"] + [subject["flat_channel"]]:
            assert name in raw.ch_names

    def test_planted_defects_are_detectable(self, ground_truth):
        """The bad channels really are outliers, not just labelled as such."""
        path = (
            BIDS_ROOT
            / "sub-001/ses-01/meg/sub-001_ses-01_task-synth_run-01_meg.fif"
        )
        raw = mne.io.read_raw_fif(path, preload=True, verbose="error")
        picks = mne.pick_types(raw.info, meg=True)
        names = [raw.ch_names[p] for p in picks]
        std = raw.get_data(picks=picks).std(axis=1)
        median = np.median(std)

        subject = ground_truth["subjects"]["001"]
        for name in subject["noisy_channels"]:
            assert std[names.index(name)] > 3 * median
        assert std[names.index(subject["flat_channel"])] < 0.05 * median

    def test_ground_truth_sources_are_inside_the_head(self, ground_truth):
        for source in ground_truth["subjects"]["001"]["sources"]:
            position = np.asarray(source["position_mri_m"])
            assert np.linalg.norm(position) < 0.11
            assert source["hemi"] in ("lh", "rh")
            assert (position[0] < 0) == (source["hemi"] == "lh")

    def test_ground_truth_sources_are_magnetically_visible(self, ground_truth):
        """Sources must not sit on a radial normal.

        A dipole oriented radially in a near-spherical conductor produces
        almost no external field, so a "ground truth" placed there is one no
        beamformer could ever recover.
        """
        for source in ground_truth["subjects"]["001"]["sources"]:
            assert source["radiality"] < 0.8, (
                f"{source['name']} is too close to radial "
                f"(|n.radial| = {source['radiality']:.2f})"
            )

    def test_ground_truth_records_both_coordinate_frames(self, ground_truth):
        """Head and MRI positions must both be present and actually differ.

        Confusing the two is the classic way to misread a source estimate: a
        forward's source space is in head coordinates, one read from
        ``bem/*-src.fif`` is in MRI surface RAS.
        """
        for source in ground_truth["subjects"]["001"]["sources"]:
            head = np.asarray(source["position_head_m"])
            mri = np.asarray(source["position_mri_m"])
            assert head.shape == mri.shape == (3,)
            assert not np.allclose(head, mri)

    def test_head_mri_transform_round_trips_through_bids(self, ground_truth):
        """The landmarks in the T1w sidecar must reproduce the true transform.

        source/make_forward derives its head<->MRI transform this way, so if
        this drifts the beamformer silently localises to the wrong place.
        """
        import mne_bids

        recovered = mne_bids.get_head_mri_trans(
            mne_bids.BIDSPath(
                subject="001",
                session="01",
                task="synth",
                run="01",
                datatype="meg",
                root=BIDS_ROOT,
            ),
            fs_subject=FS_SUBJECT,
            fs_subjects_dir=SUBJECTS_DIR,
            verbose="error",
        )
        expected = mne.read_trans(
            SUBJECTS_DIR / FS_SUBJECT / "bem" / f"{FS_SUBJECT}-trans.fif"
        )
        np.testing.assert_allclose(
            recovered["trans"], expected["trans"], atol=1e-4
        )

    def test_empty_room_is_associated_with_the_task_run(self):
        sidecar = json.loads(
            (
                BIDS_ROOT
                / "sub-001/ses-01/meg/sub-001_ses-01_task-synth_run-01_meg.json"
            ).read_text()
        )
        associated = sidecar.get("AssociatedEmptyRoom", "")
        if isinstance(associated, list):  # mne-bids has used both shapes
            associated = associated[0] if associated else ""
        assert "task-noise" in associated

    def test_pipeline_config_imports(self):
        """The shipped config must load without the pipeline's env vars set."""
        from mne_bids_pipeline._config_import import _import_config

        config = _import_config(
            config_path=REPO_ROOT / "synthetic/config/synth/config-trial.py",
            overrides=None,
            check=False,
        )
        assert config.subjects == ["001"]
        assert config.ch_types == ["mag"]
        assert Path(config.subjects_dir).resolve() == SUBJECTS_DIR.resolve()


# ---------------------------------------------------------------------------
# Beamformer localisation
# ---------------------------------------------------------------------------


@requires_dataset
class TestLocalisation:
    """The whole point of simulating through a forward model."""

    def test_stc_positions_rejects_a_mismatched_source_space(self):
        """The vertex-mapping guard must actually fire.

        Indexing source-estimate rows into an unpruned source space is a silent
        error that looks like a beamformer failure, so the helper refuses it.
        """
        from custom.synthetic.validate import _stc_positions

        src = [
            dict(vertno=np.array([0, 2, 4]), rr=np.zeros((5, 3))),
            dict(vertno=np.array([1, 3]), rr=np.zeros((5, 3))),
        ]
        stc = type("Stc", (), {"vertices": [np.array([0, 1]), np.array([1])]})()
        with pytest.raises(ValueError, match="not a subset"):
            _stc_positions(stc, src)

    @slow
    def test_beamformer_recovers_the_planted_dipoles(self):
        """Requires the preprocessing pipeline to have been run first."""
        from custom.synthetic.validate import localization_errors

        derivatives = sorted((BIDS_ROOT / "derivatives").glob("trial__*"))
        if not derivatives:
            pytest.skip("no pipeline derivatives; run mne-opm.sh preproc first")

        errors = localization_errors(BIDS_ROOT, derivatives[0])
        assert set(errors) == {"occipital_visual", "left_temporal", "right_parietal"}
        for name, error in errors.items():
            assert error < 20.0, f"{name} localised {error:.1f} mm from truth"


# ---------------------------------------------------------------------------
# End-to-end generation
# ---------------------------------------------------------------------------


@slow
def test_generate_a_full_subject(tmp_path):
    """Build a short subject from scratch and check the outputs hang together."""
    from custom.synthetic import DatasetSpec, make_dataset

    spec = DatasetSpec(duration=25.0, noise_duration=8.0, n_slots=16, sfreq=150.0)
    summary = make_dataset(tmp_path, subjects=["001"], spec=spec, write_template=False)

    gt = summary["subjects"]["001"]
    assert gt["n_trials"] > 0
    assert gt["trans_roundtrip_error"] < 1e-4
    assert len(gt["sources"]) == len(default_sources())

    bids = tmp_path / "bids"
    raw = mne.io.read_raw_fif(
        bids / "sub-001/ses-01/meg/sub-001_ses-01_task-synth_run-01_meg.fif",
        verbose="error",
    )
    assert len(mne.pick_types(raw.info, meg=True)) == 48
    assert (bids / "sub-001/ses-01/anat/sub-001_ses-01_T1w.json").exists()
