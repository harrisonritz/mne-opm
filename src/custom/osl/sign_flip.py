"""Dipole sign-flipping for the osl-ephys OPM pipeline.

A beamformer resolves each dipole's orientation only up to a sign, and it
resolves it independently per subject.  Averaging parcel time courses across
subjects without fixing that first cancels the signal.  osl-ephys fixes it by
picking a template subject and, for every other subject, searching for the set
of parcel sign flips that best matches that subject's parcel covariance to the
template's.

The search itself is osl-ephys' (:mod:`osl_ephys.source_recon.sign_flipping`);
this module supplies the surrounding file handling, because osl-ephys' own is
broken for epoched data.

.. warning::

   :func:`osl_ephys.source_recon.sign_flipping.apply_flips` reads
   ``{outdir}/{subject}/parc/parc-epo.fif`` in its ``epoched=True`` branch,
   while :func:`~osl_ephys.source_recon.wrappers.find_template_subject` and
   :func:`~osl_ephys.source_recon.wrappers.fix_sign_ambiguity` both write and
   look for ``{source_method}-parc-epo.fif`` -- ``lcmv-parc-epo.fif`` here.  The
   continuous branch gets the prefix right; the epoched one does not, so
   ``fix_sign_ambiguity(epoched=True)`` fails with ``FileNotFoundError`` on a
   file that never existed.  :func:`apply_flips` below is the corrected
   equivalent.

Functions
---------
parcel_channels
    Parcel channel names in a Raw or Epochs object.
parc_file
    Path to a subject's parcellated data.
sflip_file
    Path to a subject's sign-flipped parcellated data.
discover_subjects
    Subject labels with parcellated data under an output directory.
find_template
    Pick the template subject to align everyone else to.
find_flips_for_subject
    Search for one subject's parcel sign flips.
apply_flips
    Write a subject's sign-flipped parcellated data.
flip_subject
    Find and apply one subject's flips.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import logging
import os.path as op
from pathlib import Path
from typing import Sequence

import mne
import numpy as np

from osl_ephys.source_recon import sign_flipping


logger = logging.getLogger(__name__)


def parcel_channels(data) -> list[str]:
    """Return the parcel channel names in a Raw or Epochs object.

    :func:`osl_ephys.source_recon.sign_flipping._get_parc_chans` returns channel
    *names* when the ``parcel_X`` convention is used, but falls back to the
    literal string ``'misc'`` for older files.  MNE accepts either, but code
    that needs to count or index them does not, so resolve to names here.

    Parameters
    ----------
    data : mne.io.Raw or mne.Epochs
        Parcellated data.

    Returns
    -------
    names : list of str
        Parcel channel names, in file order.
    """
    chans = sign_flipping._get_parc_chans(data)
    if isinstance(chans, str):
        picks = mne.pick_types(data.info, meg=False, misc=True)
        return [data.ch_names[i] for i in picks]
    return list(chans)


def parc_file(
    outdir: str | Path,
    subject: str,
    epoched: bool = True,
    source_method: str = "lcmv",
) -> str:
    """Return the path to a subject's parcellated data.

    Matches the naming :func:`osl_ephys...wrappers.beamform_and_parcellate`
    writes, for both the continuous and epoched cases.
    """
    suffix = "epo" if epoched else "raw"
    return op.join(
        str(outdir), subject, "parc", f"{source_method}-parc-{suffix}.fif"
    )


def sflip_file(
    outdir: str | Path,
    subject: str,
    epoched: bool = True,
    source_method: str = "lcmv",
) -> str:
    """Return the path to a subject's sign-flipped parcellated data.

    Matches the naming osl-ephys uses for its continuous output, so downstream
    tooling written against osl-ephys finds these files where it expects them.
    """
    suffix = "epo" if epoched else "raw"
    return op.join(
        str(outdir), subject, f"{subject}_sflip_{source_method}-parc-{suffix}.fif"
    )


def discover_subjects(
    outdir: str | Path,
    epoched: bool = True,
    source_method: str = "lcmv",
) -> list[str]:
    """Return every subject label under ``outdir`` that has parcellated data.

    The stages that run once over the whole sample -- the group analysis, and
    the parcel repair -- take their subject list from the output tree rather
    than from a config, so that they pick up exactly the subjects whose source
    stage succeeded.

    Parameters
    ----------
    outdir : str or Path
        osl-ephys output directory.
    epoched : bool, optional
        Look for epoched rather than continuous parcellated data.
    source_method : str, optional
        Inverse method prefix on the parcel file.

    Returns
    -------
    subjects : list of str
        Subject labels, sorted.  Empty when ``outdir`` does not exist.
    """
    outdir = Path(outdir)
    if not outdir.is_dir():
        return []

    labels = sorted(
        child.name for child in outdir.iterdir() if (child / "parc").is_dir()
    )
    return available_subjects(outdir, labels, epoched, source_method)


def available_subjects(
    outdir: str | Path,
    subjects: Sequence[str],
    epoched: bool = True,
    source_method: str = "lcmv",
) -> list[str]:
    """Return the subjects that have parcellated data on disk.

    Parameters
    ----------
    outdir : str or Path
        osl-ephys output directory.
    subjects : sequence of str
        Subject labels to check.
    epoched : bool, optional
        Look for epoched rather than continuous parcellated data.
    source_method : str, optional
        Inverse method prefix on the parcel file.

    Returns
    -------
    found : list of str
        Subjects whose parcellated file exists, in the given order.
    """
    found = []
    for subject in subjects:
        path = parc_file(outdir, subject, epoched, source_method)
        if op.exists(path):
            found.append(subject)
        else:
            logger.warning("no parcellated data for %s (%s)", subject, path)
    return found


def find_template(
    outdir: str | Path,
    subjects: Sequence[str],
    n_embeddings: int = 15,
    standardize: bool = True,
    epoched: bool = True,
    source_method: str = "lcmv",
) -> str:
    """Pick the template subject to align everyone else to.

    The template is the subject whose parcel covariance is most similar to the
    others', so that the fewest flips are needed overall.

    Parameters
    ----------
    outdir : str or Path
        osl-ephys output directory.
    subjects : sequence of str
        Subjects to consider.  Those without parcellated data are skipped.
    n_embeddings : int, optional
        Time-delay embeddings used when forming the covariance.
    standardize : bool, optional
        Z-transform the parcel time courses before computing covariances.
    epoched : bool, optional
        Use epoched rather than continuous parcellated data.
    source_method : str, optional
        Inverse method prefix on the parcel file.

    Returns
    -------
    template : str
        The template subject's label.

    Raises
    ------
    ValueError
        If fewer than two subjects have parcellated data.

    Notes
    -----
    This is the same computation as
    :func:`osl_ephys...wrappers.find_template_subject`, but resolves the parcel
    files through :func:`parc_file` so that the epoched case works.
    """
    subjects = available_subjects(outdir, subjects, epoched, source_method)
    if len(subjects) < 2:
        raise ValueError(
            f"Sign flipping needs two or more subjects with parcellated data, "
            f"found {len(subjects)}."
        )

    files = [parc_file(outdir, s, epoched, source_method) for s in subjects]
    covs = sign_flipping.load_covariances(files, n_embeddings, standardize)
    index = sign_flipping.find_template_subject(covs, n_embeddings)

    template = subjects[index]
    logger.info("Template subject for sign flipping: %s", template)
    return template


def find_flips_for_subject(
    outdir: str | Path,
    subject: str,
    template: str,
    n_embeddings: int = 15,
    standardize: bool = True,
    n_init: int = 3,
    n_iter: int = 2500,
    max_flips: int = 20,
    epoched: bool = True,
    source_method: str = "lcmv",
) -> tuple[np.ndarray, list]:
    """Search for one subject's parcel sign flips.

    Returns
    -------
    flips : numpy.ndarray
        ``(n_parcels,)`` array of +1 / -1.
    metrics : list
        Covariance correlation with the template, per initialisation.
    """
    files = [
        parc_file(outdir, subject, epoched, source_method),
        parc_file(outdir, template, epoched, source_method),
    ]
    for path in files:
        if not op.exists(path):
            raise FileNotFoundError(f"Parcellated data not found: {path}")

    cov, template_cov = sign_flipping.load_covariances(
        files, n_embeddings, standardize, use_tqdm=False
    )

    return sign_flipping.find_flips(
        cov,
        template_cov,
        n_embeddings,
        n_init,
        n_iter,
        max_flips,
        use_tqdm=False,
    )


def apply_flips(
    outdir: str | Path,
    subject: str,
    flips: np.ndarray,
    epoched: bool = True,
    source_method: str = "lcmv",
) -> str:
    """Write a subject's sign-flipped parcellated data.

    The corrected equivalent of
    :func:`osl_ephys.source_recon.sign_flipping.apply_flips`; see the module
    docstring for what is wrong with that one.

    Parameters
    ----------
    outdir : str or Path
        osl-ephys output directory.
    subject : str
        Subject label.
    flips : numpy.ndarray
        ``(n_parcels,)`` array of +1 / -1.
    epoched : bool, optional
        Flip epoched rather than continuous parcellated data.
    source_method : str, optional
        Inverse method prefix on the parcel file.

    Returns
    -------
    outfile : str
        Path written.
    """
    infile = parc_file(outdir, subject, epoched, source_method)
    outfile = sflip_file(outdir, subject, epoched, source_method)

    if epoched:
        data = mne.read_epochs(infile, preload=True, verbose=False)
        # Epochs data is (epochs, channels, times); flips index the channels.
        def flip(values):
            return values * flips[np.newaxis, :, np.newaxis]
    else:
        data = mne.io.read_raw_fif(infile, preload=True, verbose=False)

        def flip(values):
            return values * flips[:, np.newaxis]

    picks = parcel_channels(data)
    if len(picks) != len(flips):
        raise ValueError(
            f"{subject}: got {len(flips)} flips for {len(picks)} parcel "
            f"channels in {infile}."
        )

    data.apply_function(flip, picks=picks, channel_wise=False)

    logger.info("saving %s", outfile)
    data.save(outfile, overwrite=True)
    return outfile


def flip_subject(
    outdir: str | Path,
    subject: str,
    template: str,
    n_embeddings: int = 15,
    standardize: bool = True,
    n_init: int = 3,
    n_iter: int = 2500,
    max_flips: int = 20,
    epoched: bool = True,
    source_method: str = "lcmv",
) -> dict:
    """Find and apply one subject's flips.

    The unit of work parallelised across the group; picklable, so it can be
    handed to a :class:`dask.distributed.Client`.

    Returns
    -------
    result : dict
        ``subject``, ``outfile``, ``n_flipped``, ``metrics``, and ``error``
        (None on success).  Failures are returned rather than raised so that
        one bad subject does not abort the group run.
    """
    try:
        if subject == template:
            # The template defines the reference signs, so it needs no flips --
            # but it still needs an sflip file, so the group stage can read
            # every subject from one place.
            path = parc_file(outdir, subject, epoched, source_method)
            reference = (
                mne.read_epochs(path, preload=False, verbose=False)
                if epoched
                else mne.io.read_raw_fif(path, preload=False, verbose=False)
            )
            flips = np.ones(len(parcel_channels(reference)))
            metrics = []
        else:
            flips, metrics = find_flips_for_subject(
                outdir,
                subject,
                template,
                n_embeddings=n_embeddings,
                standardize=standardize,
                n_init=n_init,
                n_iter=n_iter,
                max_flips=max_flips,
                epoched=epoched,
                source_method=source_method,
            )

        outfile = apply_flips(outdir, subject, flips, epoched, source_method)

        return {
            "subject": subject,
            "outfile": outfile,
            "n_flipped": int(np.sum(flips < 0)),
            "metrics": list(metrics),
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        logger.error("sign flipping failed for %s: %s", subject, exc)
        return {
            "subject": subject,
            "outfile": None,
            "n_flipped": None,
            "metrics": [],
            "error": str(exc),
        }
