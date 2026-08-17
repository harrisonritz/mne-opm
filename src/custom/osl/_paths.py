"""Path resolution for the osl-ephys OPM pipeline.

Maps the ``pipeline`` section of the config onto concrete BIDS inputs and
osl-ephys derivative outputs for one subject.

osl-ephys names every output after a single subject label, so the derivative
tree looks like::

    {outdir}/
        {subject_label}/
            {subject_label}_preproc-raw.fif
            {subject_label}_epo.fif
            {subject_label}_events.npy
            {subject_label}_event-id.yml
            rhino/ | fs_src/          # source-recon working files
            parc/                     # parcellated output
        logs/
        preproc_report/{subject_label}/
        src_report/{subject_label}/

Functions
---------
resolve_paths
    Resolve every input and output path for one subject.
find_smri
    Locate a subject's T1w image under a BIDS root.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from mne_bids import BIDSPath


# Extensions searched when auto-detecting a T1w image, in preference order.
_SMRI_EXTENSIONS: tuple[str, ...] = (".nii.gz", ".nii")


def _require(pipeline: SimpleNamespace, *keys: str) -> None:
    """Raise if any of ``keys`` is unset on the pipeline config."""
    missing = [k for k in keys if getattr(pipeline, k, None) in (None, "")]
    if missing:
        raise ValueError(
            f"pipeline.{{{', '.join(missing)}}} must be set in the osl pipeline "
            f"config (or supplied via the environment)."
        )


def find_smri(
    bids_root: str | Path,
    subject: str,
    session: Optional[str] = None,
) -> Optional[str]:
    """Locate a subject's T1w image under a BIDS root.

    Parameters
    ----------
    bids_root : str or Path
        BIDS dataset root.
    subject : str
        Subject label without the ``sub-`` prefix (e.g. ``'007'``).
    session : str, optional
        Session label without the ``ses-`` prefix (e.g. ``'01'``).

    Returns
    -------
    smri : str or None
        Path to the T1w image, or None if no candidate was found.

    Examples
    --------
    >>> find_smri("/data/bids", "007", "01")  # doctest: +SKIP
    '/data/bids/sub-007/ses-01/anat/sub-007_ses-01_T1w.nii.gz'
    """
    anat_dir = Path(bids_root) / f"sub-{subject}"
    stem = f"sub-{subject}"
    if session:
        anat_dir = anat_dir / f"ses-{session}"
        stem = f"{stem}_ses-{session}"
    anat_dir = anat_dir / "anat"

    for extension in _SMRI_EXTENSIONS:
        candidate = anat_dir / f"{stem}_T1w{extension}"
        if candidate.exists():
            return str(candidate)

    # Fall back to any T1w image in the anat directory (e.g. one carrying an
    # acquisition or reconstruction entity).
    for extension in _SMRI_EXTENSIONS:
        matches = sorted(anat_dir.glob(f"{stem}*_T1w{extension}"))
        if matches:
            return str(matches[0])

    return None


def resolve_paths(pipeline: SimpleNamespace) -> SimpleNamespace:
    """Resolve every input and output path for one subject.

    Parameters
    ----------
    pipeline : SimpleNamespace
        The ``pipeline`` section of a loaded config
        (:func:`custom.osl._config.load_config`).

    Returns
    -------
    paths : SimpleNamespace
        With attributes ``subject``, ``session``, ``subject_label``,
        ``input_fif``, ``smri``, ``outdir``, ``subject_dir``, ``preproc_fif``,
        ``epochs_fif``, ``source_input_fif``, ``logsdir``,
        ``preproc_reportdir``, ``src_reportdir``, ``freesurfer_subjects_dir``
        and ``trans``.

        Input paths are strings (or None when not applicable); output
        directories are :class:`~pathlib.Path` objects and are *not* created
        here.

    Raises
    ------
    ValueError
        If ``subject``, ``task``, ``bids_root`` or ``outdir`` is unset, or if
        the ``freesurfer`` backend is selected without
        ``freesurfer_subjects_dir``.

    Notes
    -----
    ``source_input_fif`` points at the epochs file when
    ``pipeline.source_input == 'epochs'`` and at the continuous preprocessed
    file otherwise.  It is the file the source stage reconstructs.
    """
    _require(pipeline, "subject", "task", "bids_root", "outdir")

    subject = str(pipeline.subject)
    session = str(pipeline.session) if pipeline.session else None
    subject_label = pipeline.subject_label

    bids_path = BIDSPath(
        root=pipeline.bids_root,
        subject=subject,
        session=session,
        task=pipeline.task,
        run=str(pipeline.run) if pipeline.run else None,
        datatype="meg",
        suffix="meg",
        extension=".fif",
    )

    smri = pipeline.smri or find_smri(pipeline.bids_root, subject, session)

    outdir = Path(pipeline.outdir)
    subject_dir = outdir / subject_label
    preproc_fif = subject_dir / f"{subject_label}_preproc-raw.fif"
    epochs_fif = subject_dir / f"{subject_label}_epo.fif"

    freesurfer_subjects_dir = pipeline.freesurfer_subjects_dir
    trans = pipeline.trans
    if pipeline.source_backend == "freesurfer":
        if not freesurfer_subjects_dir:
            raise ValueError(
                "pipeline.freesurfer_subjects_dir must be set when "
                "pipeline.source_backend == 'freesurfer' (typically $SUBJECTS_DIR)."
            )
        if not trans:
            trans = str(
                Path(freesurfer_subjects_dir)
                / subject_label
                / "bem"
                / f"{subject_label}-trans.fif"
            )

    return SimpleNamespace(
        subject=subject,
        session=session,
        subject_label=subject_label,
        # Inputs
        input_fif=str(bids_path.fpath),
        smri=smri,
        freesurfer_subjects_dir=freesurfer_subjects_dir,
        trans=trans,
        # Outputs
        outdir=outdir,
        subject_dir=subject_dir,
        preproc_fif=preproc_fif,
        epochs_fif=epochs_fif,
        source_input_fif=(
            epochs_fif if pipeline.source_input == "epochs" else preproc_fif
        ),
        logsdir=Path(pipeline.logsdir) if pipeline.logsdir else outdir / "logs",
        preproc_reportdir=(
            Path(pipeline.preproc_reportdir)
            if pipeline.preproc_reportdir
            else outdir / "preproc_report"
        ),
        src_reportdir=(
            Path(pipeline.src_reportdir)
            if pipeline.src_reportdir
            else outdir / "src_report"
        ),
    )
