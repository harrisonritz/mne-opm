"""Repair stage: restore condition names and the time axis on parcel files.

A one-off migration for parcel files written before
:mod:`custom.osl.parcel_epochs` replaced osl-ephys' parcel converter.  Those
files carry the right *data* but a header MNE synthesised: ``event_id`` is the
stringified codes (``{'201': 201}``) rather than the condition names, and
``tmin`` is 0 rather than the epoch start, so the group stage's condition
averaging finds nothing and fills every subject with NaN.

Both are recoverable from the subject's own sensor-level epochs, which still
carry them, so the whole sample can be fixed by rewriting headers -- seconds
per subject -- rather than by re-running source reconstruction.

Run it once over an existing output tree::

    mne-opm.sh osl --exp TSX --sub 007 --analysis trialResponse \\
        --stage repair-parc ...

then re-run the group stage.  It is idempotent: a file that already carries
its names is rebuilt to the same content, so re-running is harmless.

Both the parcellated file and the sign-flipped copy derived from it are
repaired, since the group stage reads the latter.

Functions
---------
run
    Run the repair stage over every subject with parcellated data.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import mne

from . import sign_flip
from ._paths import resolve_paths
from .parcel_epochs import is_placeholder_event_id, restore_epoch_metadata


def run(cfg: SimpleNamespace) -> bool:
    """Run the repair stage over every subject with parcellated data.

    Parameters
    ----------
    cfg : SimpleNamespace
        Config from :func:`custom.osl._config.load_config`.  Only
        ``pipeline`` is required; ``group.source_method`` is read when
        present, so that a config whose source stage used something other
        than LCMV finds its files.

    Returns
    -------
    success : bool
        True if every subject was repaired.  False if any subject failed;
        the rest are still repaired, and each failure is reported.

    Raises
    ------
    ValueError
        If ``pipeline.source_input`` is not ``epochs`` -- continuous parcel
        files have no condition names to lose -- or if no subject under
        ``outdir`` has parcellated data.
    """
    pipeline = cfg.pipeline
    paths = resolve_paths(pipeline)
    outdir = paths.outdir

    if pipeline.source_input != "epochs":
        raise ValueError(
            f"The repair stage restores epoch condition names and tmin, so it "
            f"only applies to pipeline.source_input: epochs, not "
            f"{pipeline.source_input!r}."
        )

    source_method = dict(cfg.group or {}).get("source_method", "lcmv")

    subjects = sign_flip.discover_subjects(
        outdir, epoched=True, source_method=source_method
    )
    if not subjects:
        raise ValueError(
            f"No subject under {outdir} has parcellated data. Has the source "
            f"stage run?"
        )
    print(f"[osl:repair-parc] {len(subjects)} subject(s) with parcellated data")

    n_failed = 0
    for subject in subjects:
        try:
            _repair_subject(outdir, subject, source_method)
        except Exception as exc:  # noqa: BLE001 -- one subject must not stop the rest
            n_failed += 1
            print(f"[osl:repair-parc] FAILED for {subject}: {exc}")

    print(
        f"[osl:repair-parc] repaired {len(subjects) - n_failed}/{len(subjects)} "
        f"subject(s)"
    )
    return n_failed == 0


def _repair_subject(outdir: Path, subject: str, source_method: str) -> None:
    """Rewrite one subject's parcel and sign-flipped files.

    Parameters
    ----------
    outdir : Path
        osl-ephys output directory.
    subject : str
        Subject label, as the output tree names it.
    source_method : str
        Inverse method prefix on the parcel files.

    Raises
    ------
    FileNotFoundError
        If the subject's sensor-level epochs, which supply the names and the
        time axis, are missing.
    """
    epochs_fif = outdir / subject / f"{subject}_epo.fif"
    if not epochs_fif.exists():
        raise FileNotFoundError(
            f"no sensor-level epochs at {epochs_fif}, so the condition names "
            f"cannot be recovered. Re-run the preproc stage for this subject."
        )

    # Header only: the sensor epochs supply event_id, tmin and metadata, and
    # at the acquisition rate they are far larger than the parcel data.
    source_epochs = mne.read_epochs(epochs_fif, preload=False, verbose="ERROR")

    targets = [
        Path(sign_flip.parc_file(outdir, subject, True, source_method)),
        Path(sign_flip.sflip_file(outdir, subject, True, source_method)),
    ]

    repaired = []
    for path in targets:
        if not path.exists():
            # The sign-flipped copy only exists once the group stage has run.
            continue
        was_placeholder = _repair_file(path, source_epochs)
        repaired.append(f"{path.name}{'' if was_placeholder else ' (already named)'}")

    print(f"[osl:repair-parc] {subject}: {', '.join(repaired)}")


def _repair_file(path: Path, source_epochs) -> bool:
    """Rewrite one parcel file in place, restoring its names and time axis.

    Written to a temporary file beside the target and moved into place only
    after it reads back correctly, so that a job killed mid-write cannot leave
    a truncated parcel file where a valid one used to be.

    Parameters
    ----------
    path : Path
        Parcel or sign-flipped parcel file to rewrite.
    source_epochs : mne.Epochs
        The subject's sensor-level epochs.

    Returns
    -------
    was_placeholder : bool
        Whether the file had lost its condition names, as opposed to already
        carrying them (in which case the rewrite is a no-op in content).

    Raises
    ------
    RuntimeError
        If the rewritten file does not read back with the expected names,
        time axis and shape, or if MNE had to split it -- the move into place
        handles a single file only, so the original is left untouched.
    """
    parc_epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
    was_placeholder = is_placeholder_event_id(parc_epochs.event_id)

    repaired = restore_epoch_metadata(parc_epochs, source_epochs)

    tmp = path.with_name(f"{path.stem}-repair-tmp{path.suffix}")
    _unlink_stale(tmp)
    repaired.save(tmp, overwrite=True, verbose="ERROR")

    splits = sorted(tmp.parent.glob(f"{tmp.stem}-[0-9]*{tmp.suffix}"))
    if splits:
        for split in [tmp, *splits]:
            _unlink_stale(split)
        raise RuntimeError(
            f"MNE split {path.name} across {len(splits) + 1} files while "
            f"rewriting it; the in-place repair handles a single file only. "
            f"Re-run the source stage for this subject instead."
        )

    _verify(tmp, repaired)
    os.replace(tmp, path)
    return was_placeholder


def _verify(path: Path, expected) -> None:
    """Check a rewritten parcel file reads back as what was written."""
    written = mne.read_epochs(path, preload=False, verbose="ERROR")

    problems = []
    if written.event_id != expected.event_id:
        problems.append(f"event_id {written.event_id} != {expected.event_id}")
    if abs(written.tmin - expected.tmin) > 1e-9:
        problems.append(f"tmin {written.tmin} != {expected.tmin}")
    if len(written) != len(expected):
        problems.append(f"{len(written)} epochs != {len(expected)}")
    if written.ch_names != expected.ch_names:
        problems.append("channel names differ")

    if problems:
        _unlink_stale(path)
        raise RuntimeError(
            f"the rewritten parcel file did not read back as written "
            f"({'; '.join(problems)}); the original was left in place."
        )


def _unlink_stale(path: Path) -> None:
    """Remove a temporary file if it is there, ignoring a missing one."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
