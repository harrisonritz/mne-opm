"""Assemble a synthetic BIDS dataset from the phantom, array and simulation.

The conversion to BIDS deliberately goes through this repository's *own*
``format_bids.bids_conversion``, and the anatomical landmarks through its own
``preprocessing.coreg.CoregAnalysis``, rather than calling ``mne_bids``
directly.  That way the committed dataset is exactly what the real converter
produces -- if the converter changes, regenerating the dataset picks the change
up, and any drift between the two shows up as a diff.

Layout produced under ``--out``::

    <out>/
      raw/
        synth_<sub>/
          metadata/sub-<sub>_run-01.csv     behavioural metadata (committed)
          synth_<sub>_task/..._meg.fif      pre-BIDS recording (regenerable)
          synth_<sub>_noise/..._meg.fif     pre-BIDS empty room (regenerable)
      bids/
        sub-<sub>/ses-01/meg/               task + empty-room recordings
        sub-<sub>/ses-01/anat/              T1w with anatomical landmarks
        derivatives/freesurfer/subjects/
          sub-<sub>_ses-01/                 phantom "recon"
          fsaverage/                        group template for morphing
        ground_truth.json                   dipole positions, planted defects

Author: Harrison Ritz (2025)
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np


__all__ = ["DatasetSpec", "make_subject", "make_dataset"]


DEFAULT_TASK = "synth"
DEFAULT_SESSION = "01"
FS_TEMPLATE = "fsaverage"


@dataclass
class DatasetSpec:
    """Knobs for dataset generation.

    The defaults are the ones the committed subject was built with.  They are
    a compromise: 48 triaxial slots keeps Maxwell filtering
    (``mf_int_order = 10``) well posed, and 100 s at 200 Hz keeps the committed
    FIF near 12 MB while still yielding ~40 trials.
    """

    task: str = DEFAULT_TASK
    session: str = DEFAULT_SESSION
    sfreq: float = 200.0
    duration: float = 100.0
    # Long enough for a 30 s tSSS window (``mf_st_duration``): Maxwell
    # filtering is applied to the empty-room run too when
    # ``process_empty_room`` is set, and it refuses a window longer than the
    # recording.  A covariance estimate would be happy with far less.
    noise_duration: float = 40.0
    n_slots: int = 48
    line_freq: float = 60.0
    seed: int = 0
    #: Head-shape variability across a cohort.  Zero for the reference subject.
    head_jitter: float = 0.06
    #: Keep the pre-BIDS FIFs after conversion.  Off by default because they
    #: double the on-disk size and are regenerable.
    keep_raw: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _subject_label(subject: int | str) -> str:
    return f"{int(subject):03d}"


def _meas_date(subject: int) -> datetime:
    """A stable, distinct acquisition time per subject."""
    return datetime(2025, 1, 6 + (subject % 20), 10, 0, 0, tzinfo=timezone.utc)


def _write_raw_tree(
    raw_task, raw_noise, metadata, raw_dir: Path, subject: str, task: str
) -> Path:
    """Write the Cerca-style pre-BIDS folder that ``format_bids`` expects."""
    subj_dir = raw_dir / f"synth_{subject}"
    task_dir = subj_dir / f"synth_{subject}_task"
    noise_dir = subj_dir / f"synth_{subject}_noise"
    meta_dir = subj_dir / "metadata"
    for d in (task_dir, noise_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    raw_task.save(task_dir / f"synth_{subject}_{task}_meg.fif", overwrite=True)
    raw_noise.save(noise_dir / f"synth_{subject}_noise_meg.fif", overwrite=True)
    metadata.to_csv(meta_dir / f"sub-{subject}_run-01.csv", index=False)
    return subj_dir


def _bids_config(spec: DatasetSpec, subject: str, raw_dir: Path, bids_dir: Path):
    """The ``cfg`` namespace ``format_bids.bids_conversion`` consumes.

    Mirrors ``config/<exp>/bids/sub-XXX_config-bids.py`` in the analysis repo.
    """
    from .events import TRIGGER_DESC

    return SimpleNamespace(
        raw_dir=str(raw_dir),
        bids_dir=str(bids_dir),
        ids=int(subject),
        task=spec.task,
        session=spec.session,
        rename_annot=True,
        trigger_desc=dict(TRIGGER_DESC),
        response_desc={},
        line_freq=spec.line_freq,
        bads=[],
        crop=0,
        device_info=dict(
            type="Cerca_synthetic", site="mne-opm synthetic dataset", model="cMEG"
        ),
        verify_conditions=("trial",),
    )


def _write_landmarks(spec: DatasetSpec, subject: str, bids_dir: Path, subjects_dir: Path):
    """Run the repo's coregistration step against the ground-truth transform.

    ``_use_precomputed_trans`` makes ``CoregAnalysis`` skip ICP (there is no
    display to run the fiducial GUI on) and take the transform we already know
    exactly, but everything downstream -- ``get_anat_landmarks`` and the
    ``write_anat`` call that puts ``AnatomicalLandmarkCoordinates`` in the T1w
    sidecar -- is the production code path, and it is what
    ``mne_bids.get_head_mri_trans`` reads back during ``source/make_forward``.
    """
    from custom.preprocessing.coreg import CoregAnalysis

    fs_subject = f"sub-{subject}_ses-{spec.session}"
    cfg = SimpleNamespace(
        bids_root=str(bids_dir),
        subjects=[subject],
        sessions=[spec.session],
        subjects_dir=str(subjects_dir),
        task=spec.task,
        _use_precomputed_trans=True,
        _precomputed_trans_path=str(
            subjects_dir / fs_subject / "bem" / f"{fs_subject}-trans.fif"
        ),
    )
    CoregAnalysis(cfg).execute()


# ---------------------------------------------------------------------------
# One subject
# ---------------------------------------------------------------------------


def make_subject(
    out_root: Path | str,
    subject: int | str = 1,
    spec: DatasetSpec | None = None,
    *,
    reference: bool = False,
) -> dict:
    """Generate one synthetic subject, end to end.

    Parameters
    ----------
    out_root : path-like
        Dataset root.  ``raw/`` and ``bids/`` are created underneath.
    subject : int or str
        Numeric subject ID; zero-padded to three digits as elsewhere in the
        pipeline.
    spec : DatasetSpec, optional
        Generation parameters.
    reference : bool
        Build the canonical (un-jittered) head.  Used for the committed
        subject so its geometry matches the group template exactly.

    Returns
    -------
    ground_truth : dict
        Source positions, planted defects and file paths, also written to
        ``bids/ground_truth.json``.
    """
    import mne
    import mne_bids

    from custom import format_bids

    from .anatomy import build_head_model, write_freesurfer_subject
    from .events import build_schedule
    from .sensors import build_info
    from .simulate import simulate_empty_room, simulate_task

    spec = spec or DatasetSpec()
    label = _subject_label(subject)
    out_root = Path(out_root)
    raw_dir = out_root / "raw"
    bids_dir = out_root / "bids"
    subjects_dir = bids_dir / "derivatives" / "freesurfer" / "subjects"
    fs_subject = f"sub-{label}_ses-{spec.session}"
    seed = spec.seed + int(label)

    print(f"\n[synthetic] subject {label}: building anatomy")
    head = build_head_model(seed=seed, jitter=0.0 if reference else spec.head_jitter)
    write_freesurfer_subject(head, subjects_dir, fs_subject)

    # The ground-truth head <-> MRI transform, saved where coreg.py looks for it.
    trans = mne.transforms.invert_transform(head.mri_head_t)  # head -> mri
    mne.write_trans(
        subjects_dir / fs_subject / "bem" / f"{fs_subject}-trans.fif",
        trans,
        overwrite=True,
    )

    print(f"[synthetic] subject {label}: simulating {spec.duration:.0f} s of OPM data")
    info, _ = build_info(
        head, spec.sfreq, n_slots=spec.n_slots, line_freq=spec.line_freq, seed=seed
    )
    info.set_meas_date(_meas_date(int(label)))
    schedule = build_schedule(spec.duration, seed=seed)

    raw_task, ground_truth, _ = simulate_task(
        info, head, schedule, subjects_dir, fs_subject, seed=seed
    )

    meg_info = mne.pick_info(info, mne.pick_types(info, meg=True, exclude=()))
    raw_noise = simulate_empty_room(meg_info, head, spec.noise_duration, seed=seed)
    raw_noise.set_meas_date(_meas_date(int(label)))

    print(f"[synthetic] subject {label}: writing pre-BIDS tree")
    subj_raw_dir = _write_raw_tree(
        raw_task, raw_noise, schedule.metadata, raw_dir, label, spec.task
    )
    del raw_task, raw_noise

    print(f"[synthetic] subject {label}: converting to BIDS via format_bids")
    format_bids.bids_conversion(_bids_config(spec, label, raw_dir, bids_dir))

    print(f"[synthetic] subject {label}: writing anatomical landmarks")
    _write_landmarks(spec, label, bids_dir, subjects_dir)

    if not spec.keep_raw:
        for sub in (f"synth_{label}_task", f"synth_{label}_noise"):
            shutil.rmtree(subj_raw_dir / sub, ignore_errors=True)

    ground_truth.update(
        subject=label,
        session=spec.session,
        task=spec.task,
        fs_subject=fs_subject,
        head_mri_t=np.asarray(head.mri_head_t["trans"]).tolist(),
        fiducials_mri_m={k: list(map(float, v)) for k, v in head.fiducials.items()},
        n_meg_channels=int(spec.n_slots * 3),
        trans_path=str(
            Path("derivatives/freesurfer/subjects")
            / fs_subject
            / "bem"
            / f"{fs_subject}-trans.fif"
        ),
    )

    # Sanity check that the round-trip through the BIDS sidecar reproduces the
    # transform we started from.  A silent failure here would only surface much
    # later, as a beamformer pointing at the wrong hemisphere.
    recovered = mne_bids.get_head_mri_trans(
        mne_bids.BIDSPath(
            subject=label,
            session=spec.session,
            task=spec.task,
            run="01",
            datatype="meg",
            root=bids_dir,
        ),
        fs_subject=fs_subject,
        fs_subjects_dir=subjects_dir,
    )
    err = np.abs(np.asarray(recovered["trans"]) - np.asarray(trans["trans"])).max()
    if err > 1e-4:
        raise RuntimeError(
            f"head<->MRI transform did not survive the BIDS round-trip "
            f"(max deviation {err:.2e}); the landmarks and the FreeSurfer T1 "
            f"have gone out of sync."
        )
    ground_truth["trans_roundtrip_error"] = float(err)

    return ground_truth


# ---------------------------------------------------------------------------
# Whole dataset
# ---------------------------------------------------------------------------


def _write_dataset_files(bids_dir: Path, spec: DatasetSpec, subjects: list[str]) -> None:
    """Top-level BIDS metadata that ``mne_bids`` does not write for us."""
    desc_path = bids_dir / "dataset_description.json"
    desc = json.loads(desc_path.read_text()) if desc_path.exists() else {}
    desc.update(
        Name="mne-opm synthetic OPM-MEG dataset",
        BIDSVersion=desc.get("BIDSVersion", "1.7.0"),
        DatasetType="raw",
        Authors=["mne-opm synthetic data generator"],
        GeneratedBy=[
            dict(
                Name="mne-opm",
                Description=(
                    "Fully synthetic OPM-MEG data simulated through a forward "
                    "model on an analytic head phantom. No human subjects."
                ),
                CodeURL="https://github.com/harrisonritz/mne-opm",
            )
        ],
    )
    desc_path.write_text(json.dumps(desc, indent=4) + "\n")

    (bids_dir / "README").write_text(
        "Synthetic OPM-MEG dataset\n"
        "=========================\n\n"
        "Every file here was generated by `src/custom/make_synthetic.py`.\n"
        "There is no human data: the anatomy is an analytic phantom and the\n"
        "recordings are simulated through a BEM forward model.\n\n"
        f"Subjects: {', '.join(subjects)}\n"
        f"Task: {spec.task} | session: {spec.session} | "
        f"{spec.n_slots * 3} magnetometers at {spec.sfreq:g} Hz\n\n"
        "Ground-truth source positions and the planted artifacts are recorded\n"
        "in ground_truth.json.\n"
    )


def make_dataset(
    out_root: Path | str,
    subjects: list[int | str] | None = None,
    spec: DatasetSpec | None = None,
    *,
    n_subjects: int | None = None,
    write_template: bool = True,
) -> dict:
    """Generate a synthetic dataset with one or more subjects.

    Parameters
    ----------
    out_root : path-like
        Dataset root (the ``--data``/``EXPERIMENT`` directory).
    subjects : list, optional
        Subject IDs.  Defaults to ``[1]``, or ``1..n_subjects``.
    spec : DatasetSpec, optional
        Generation parameters.
    n_subjects : int, optional
        Convenience alternative to ``subjects`` for a cohort.
    write_template : bool
        Also write the synthetic ``fsaverage`` group template.

    Returns
    -------
    summary : dict
        Per-subject ground truth, also written to ``bids/ground_truth.json``.
    """
    from .template import write_group_template

    spec = spec or DatasetSpec()
    if subjects is None:
        subjects = list(range(1, (n_subjects or 1) + 1))

    out_root = Path(out_root)
    bids_dir = out_root / "bids"
    subjects_dir = bids_dir / "derivatives" / "freesurfer" / "subjects"

    results = {}
    for i, subject in enumerate(subjects):
        results[_subject_label(subject)] = make_subject(
            out_root, subject, spec, reference=(i == 0)
        )

    if write_template:
        print(f"[synthetic] writing group template '{FS_TEMPLATE}'")
        write_group_template(subjects_dir, spec.seed)

    _write_dataset_files(bids_dir, spec, [_subject_label(s) for s in subjects])

    summary = dict(spec=asdict(spec), template=FS_TEMPLATE, subjects=results)
    (bids_dir / "ground_truth.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
