"""FreeSurfer/MNE source-reconstruction wrappers for the osl-ephys pipeline.

osl-ephys' LCMV path is written against RHINO and therefore needs FSL: its
:func:`osl_ephys.source_recon.beamforming.make_lcmv` reads the forward model
from the RHINO file tree, ``transform_recon_timeseries`` needs RHINO's
``mni_mri_t``/``head_mri_t`` transforms and shells out to FSL ``flirt``, and
even :func:`osl_ephys.source_recon.parcellation.resample_parcellation` calls
``flirt``.  osl-ephys' ``surface_extraction_method='freesurfer'`` path avoids
FSL but only reaches minimum-norm estimates, not beamforming.

This module provides the missing combination: LCMV beamforming and volumetric
parcellation built on the FreeSurfer ``recon-all`` output and the MNE
``-trans.fif`` this repository already produces
(``custom.preprocessing.coreg``), with no FSL dependency.  The parcel
time-course maths is osl-ephys' own, so output is directly comparable with the
RHINO backend.

The wrappers follow osl-ephys' source-recon calling convention, so they slot
into a ``source_recon`` config like any built-in step::

    source_recon:
      - fs_coregister: {}
      - fs_forward_model: {gridstep: 5, mindist: 5}
      - fs_beamform_and_parcellate:
          chantypes: mag
          rank: {mag: 60}
          freq_range: [1, 32]
          decim: 6
          parcellation_file: Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
          method: spatial_basis
          orthogonalisation: symmetric

Functions
---------
fs_coregister
    Validate and report the existing FreeSurfer/MNE coregistration.
fs_forward_model
    Build a volumetric source space, BEM and forward solution with MNE.
fs_beamform_and_parcellate
    LCMV beamform onto the volumetric grid, morph to MNI and parcellate.

Constants
---------
SOURCE_EXTRA_FUNCS
    Wrappers passed to osl-ephys as ``extra_funcs`` for the source stage.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import logging
import os
import os.path as op
from pathlib import Path
from typing import Optional

import mne
import numpy as np

from osl_ephys.report import src_report
from osl_ephys.source_recon import parcellation
from osl_ephys.source_recon.rhino import utils as rhino_utils

from .parcel_epochs import convert2mne_epochs


logger = logging.getLogger(__name__)


# osl-ephys extends MNE with orientation options MNE itself does not have.
# Silently substituting a different estimator would change the science, so we
# reject them explicitly on this backend.
_OSL_ONLY_PICK_ORI: dict[str, str] = {
    "max-power-pre-weight-norm": "max-power",
}

_SUBDIR = "fs_src"
"""Sub-directory of ``{outdir}/{subject}`` holding this backend's files."""

_SOURCE_DTYPE = np.float32
"""Dtype of the source-space array.

Source reconstruction's peak memory is one ``(voxels, times, trials)`` array,
which at a full-rate epoched recording runs to tens of GiB, so the dtype is
worth a factor of two.  float32 carries ~7 significant digits, far more than
the beamformer output is meaningful to, and the parcel time courses that come
out of :func:`osl_ephys...parcellation._get_parcel_timeseries` are float64
regardless.
"""

_MEMORY_WARN_FRACTION = 0.6
"""Fraction of the job's memory budget at which to warn about the source array."""

_RANK_TOL = 1e-6
_RANK_TOL_KIND = "relative"
"""Tolerance for the data-driven rank estimate, matching ``run_beamformer``.

``tol='auto'`` is ``n_dim * max_s * eps_float64`` (~1e-13 relative), while the
directions SSS and ICA null return from a float32 FIF at ~1e-7 relative -- so
``'auto'`` counts every nulled direction and lands near the channel count.  An
explicit relative tolerance is what separates the two.
"""


# ---------------------------------------------------------------------------
# Report payloads
# ---------------------------------------------------------------------------


def _drop_missing_plots(payload: dict) -> dict:
    """Strip ``*_plot`` entries that name no file.

    :func:`osl_ephys.report.src_report.gen_html_data` copies every plot it
    finds in the report data by key, and -- ``parc_freqbands_plot`` aside --
    tests only that the key is *present*, not that it holds a path.  A key left
    at None therefore fails inside the report build rather than being skipped:
    ``coreg_plot`` raises ``TypeError: argument of type 'NoneType' is not
    iterable``, and the rest try to copy a file literally named "None".

    Every figure here is optional (rendering them depends on the 3D stack and
    on plotting libraries the pipeline treats as best-effort), so a figure that
    was not produced is dropped from the payload and simply does not appear in
    the report.
    """
    return {
        key: value
        for key, value in payload.items()
        if value is not None or not key.endswith("_plot")
    }


# ---------------------------------------------------------------------------
# File layout
# ---------------------------------------------------------------------------


def get_fs_filenames(outdir: str | Path, subject: str) -> dict[str, str]:
    """Return the paths this backend reads and writes for one subject.

    Parameters
    ----------
    outdir : str or Path
        osl-ephys output directory.
    subject : str
        Subject label (the ``{outdir}`` sub-directory name).

    Returns
    -------
    files : dict
        Keys ``basedir``, ``info_fif``, ``source_space``, ``bem``,
        ``fwd_model``, ``filters``, ``coreg_html``, ``parcdir``.
    """
    basedir = op.join(str(outdir), subject, _SUBDIR)
    return {
        "basedir": basedir,
        "info_fif": op.join(basedir, "info-raw.fif"),
        "source_space": op.join(basedir, "space-src.fif"),
        "bem": op.join(basedir, "bem-sol.fif"),
        "fwd_model": op.join(basedir, "model-fwd.fif"),
        "filters": op.join(basedir, "lcmv-filters.h5"),
        "coreg_html": op.join(basedir, "coreg.html"),
        "parcdir": op.join(str(outdir), subject, "parc"),
    }


def _resolve_subjects_dir(subjects_dir: Optional[str]) -> str:
    """Return the FreeSurfer subjects directory, falling back to the environment."""
    subjects_dir = subjects_dir or os.environ.get("SUBJECTS_DIR")
    if not subjects_dir:
        raise ValueError(
            "subjects_dir must be given for the freesurfer backend (either as a "
            "step option in the config, via pipeline.freesurfer_subjects_dir, or "
            "as the SUBJECTS_DIR environment variable)."
        )
    return str(subjects_dir)


def _resolve_trans(trans: Optional[str], subjects_dir: str, subject: str) -> str:
    """Return the path to the head<->MRI transform, using the FreeSurfer convention."""
    if not trans:
        trans = op.join(subjects_dir, subject, "bem", f"{subject}-trans.fif")
    if not op.exists(trans):
        raise FileNotFoundError(
            f"Coregistration transform not found: {trans}. Run the coreg stage "
            f"(mne-opm.sh coreg) first, or set 'trans' in the config."
        )
    return str(trans)


def _read_mri_head_t(trans_path: str) -> mne.Transform:
    """Read a ``-trans.fif`` and return it oriented as MRI -> head.

    ``mne.write_trans`` preserves whichever direction the transform was created
    in, so normalise here rather than assuming.
    """
    trans = mne.read_trans(trans_path)
    if trans["from"] == mne.io.constants.FIFF.FIFFV_COORD_MRI:
        return trans
    return mne.transforms.invert_transform(trans)


def _load_data(preproc_file: Optional[str], epoch_file: Optional[str]):
    """Load epochs when available, otherwise the continuous preprocessed data."""
    if epoch_file is not None:
        logger.info("using epoched data: %s", epoch_file)
        return mne.read_epochs(epoch_file, preload=True), True
    if preproc_file is None:
        raise ValueError("One of preproc_file or epoch_file must be given.")
    logger.info("using continuous data: %s", preproc_file)
    return mne.io.read_raw_fif(preproc_file, preload=True), False


def _bandpass(data, freq_range: Optional[list]):
    """Apply osl-ephys' IIR bandpass, so both backends filter identically."""
    if freq_range is None:
        return data
    logger.info("bandpass filtering: %s-%s Hz", freq_range[0], freq_range[1])
    return data.filter(
        l_freq=freq_range[0],
        h_freq=freq_range[1],
        method="iir",
        iir_params={"order": 5, "ftype": "butter"},
    )


def _decimate(data, decim: Optional[int], is_epochs: bool, freq_range):
    """Reduce the sampling rate before beamforming.

    Source reconstruction holds one ``(voxels, times, trials)`` array, so the
    sampling rate scales its size directly: at 1200 Hz a one-second epoch is
    1201 samples, and a few thousand trials on an 8 mm grid is tens of GiB.  A
    recording already low-passed at ``freq_range[1]`` carries no information
    above that, so decimating towards its Nyquist rate costs nothing.

    Epochs are decimated (a straight sample slice, the band-limit having
    already been imposed by :func:`_bandpass`); continuous data is resampled,
    which applies its own anti-alias filter.

    Parameters
    ----------
    data : mne.io.Raw or mne.Epochs
        Band-passed data.
    decim : int or None
        Factor to decimate by.  None or 1 leaves the data untouched.
    is_epochs : bool
        Whether ``data`` is epoched.
    freq_range : list or None
        ``[l_freq, h_freq]`` already applied, used to reject a factor that
        would alias the retained band.

    Returns
    -------
    data : mne.io.Raw or mne.Epochs
        The decimated data.

    Raises
    ------
    ValueError
        If ``decim`` is not a positive integer, or would take the sampling
        rate to or below twice the low-pass edge.
    """
    if decim is None or decim == 1:
        return data

    decim = int(decim)
    if decim < 1:
        raise ValueError(f"decim must be a positive integer, got {decim}.")

    sfreq = float(data.info["sfreq"])
    new_sfreq = sfreq / decim

    if freq_range is not None and new_sfreq <= 2 * freq_range[1]:
        raise ValueError(
            f"decim={decim} takes the sampling rate from {sfreq:.1f} Hz to "
            f"{new_sfreq:.1f} Hz, at or below twice the {freq_range[1]} Hz "
            f"low-pass, which would alias the band being reconstructed. The "
            f"largest safe factor here is "
            f"{max(1, int(sfreq // (2 * freq_range[1])))}."
        )

    logger.info(
        "decimating by %d: %.1f -> %.1f Hz", decim, sfreq, new_sfreq
    )
    if is_epochs:
        return data.decimate(decim)
    return data.resample(new_sfreq)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def _cgroup_memory_limit() -> Optional[int]:
    """Memory limit of this process' cgroup, in bytes, or None if unlimited.

    This is the limit SLURM enforces, and whose breach is reported as
    ``oom_kill event in StepId=...``.
    """
    candidates: list[Path] = []

    try:
        with open("/proc/self/cgroup") as f:
            for line in f:
                # cgroup v2: "0::/<path>"; v1: "<n>:memory:/<path>"
                parts = line.strip().split(":", 2)
                if len(parts) == 3 and parts[1] in ("", "memory"):
                    rel = parts[2].lstrip("/")
                    candidates.append(Path("/sys/fs/cgroup") / rel / "memory.max")
                    candidates.append(
                        Path("/sys/fs/cgroup/memory") / rel / "memory.limit_in_bytes"
                    )
    except OSError:
        pass

    # Fall back to the mount root, which is the job's own cgroup when SLURM
    # puts the step in a cgroup namespace.
    candidates.append(Path("/sys/fs/cgroup/memory.max"))
    candidates.append(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))

    for path in candidates:
        try:
            value = path.read_text().strip()
        except OSError:
            continue
        # "max" on v2, and a sentinel near 2**63 on v1, both mean unlimited.
        if value.isdigit() and 0 < int(value) < 2 ** 62:
            return int(value)

    return None


def _memory_budget() -> Optional[int]:
    """Memory this job may use, in bytes, or None if it cannot be determined.

    Takes the smaller of the cgroup limit and SLURM's ``--mem``.
    """
    limits = []

    cgroup = _cgroup_memory_limit()
    if cgroup is not None:
        limits.append(cgroup)

    per_node = os.environ.get("SLURM_MEM_PER_NODE", "").strip()
    if per_node.isdigit() and int(per_node) > 0:
        limits.append(int(per_node) * 1024 ** 2)  # SLURM reports MB

    return min(limits) if limits else None


def _check_source_memory(n_voxels: int, n_times: int, n_trials: int) -> None:
    """Warn or bail out before allocating a source array that cannot fit.

    The source-space array is ``(n_voxels, n_times, n_trials)``, and
    :func:`osl_ephys...parcellation._get_parcel_timeseries` makes one more of
    the same shape when it standardises the voxel time courses, so two of them
    is a floor on the peak.  ``numpy.zeros`` maps its pages lazily, so an
    array far larger than memory allocates without complaint and the process
    is only killed once the beamformer has spent an hour filling it -- with no
    Python traceback, just SIGKILL.  Checking the shape up front turns that
    into an error that says what to change.

    Parameters
    ----------
    n_voxels, n_times, n_trials : int
        Shape of the source-space array; ``n_trials`` is 1 for continuous data.

    Raises
    ------
    MemoryError
        If the floor on the peak already exceeds the job's memory budget.
    """
    itemsize = np.dtype(_SOURCE_DTYPE).itemsize
    array_bytes = n_voxels * n_times * n_trials * itemsize
    floor_bytes = 2 * array_bytes

    gib = 1024 ** 3
    logger.info(
        "source array: %d voxels x %d samples x %d trials = %.1f GiB (%s)",
        n_voxels, n_times, n_trials, array_bytes / gib,
        np.dtype(_SOURCE_DTYPE).name,
    )

    budget = _memory_budget()
    if budget is None:
        return

    advice = (
        f"Reduce it with `decim` (the source stage's own decimation factor), "
        f"which divides the sample count: the array is "
        f"{array_bytes / gib:.1f} GiB at {n_times} samples per trial. Failing "
        f"that, raise --mem, epoch a shorter window, or coarsen the source "
        f"grid with `gridstep`."
    )

    if floor_bytes > budget:
        raise MemoryError(
            f"Source reconstruction needs at least "
            f"{floor_bytes / gib:.1f} GiB (the {array_bytes / gib:.1f} GiB "
            f"source array, plus the copy the parcellation makes of it), and "
            f"this job's budget is {budget / gib:.1f} GiB. {advice}"
        )

    if floor_bytes > _MEMORY_WARN_FRACTION * budget:
        logger.warning(
            "source reconstruction will use at least %.1f GiB of this job's "
            "%.1f GiB. %s",
            floor_bytes / gib, budget / gib, advice,
        )


# ---------------------------------------------------------------------------
# Coregistration
# ---------------------------------------------------------------------------


def fs_coregister(
    outdir,
    subject,
    preproc_file=None,
    epoch_file=None,
    subjects_dir=None,
    trans=None,
    make_plot=True,
    reportdir=None,
):
    """Validate and report the existing FreeSurfer/MNE coregistration.

    This backend does not fit a coregistration -- that is done upstream by
    ``custom.preprocessing.coreg`` -- so this step only checks that the
    transform exists, records the digitisation-to-scalp distances for QC, and
    saves an alignment plot into the source-recon report.

    Parameters
    ----------
    outdir : str
        osl-ephys output directory.
    subject : str
        Subject label.
    preproc_file : str, optional
        Preprocessed fif file, used for its measurement info.
    epoch_file : str, optional
        Epoched fif file, used when ``preproc_file`` is None.
    subjects_dir : str, optional
        FreeSurfer subjects directory.  Defaults to ``$SUBJECTS_DIR``.
    trans : str, optional
        Path to the ``-trans.fif``.  Defaults to
        ``{subjects_dir}/{subject}/bem/{subject}-trans.fif``.
    make_plot : bool, optional
        Save an interactive alignment plot to the report.  Rendering failures
        are logged and ignored, since they must not fail a batch job.
    reportdir : str, optional
        Report directory.
    """
    logger.info("fs_coregister")

    subjects_dir = _resolve_subjects_dir(subjects_dir)
    trans_path = _resolve_trans(trans, subjects_dir, subject)

    files = get_fs_filenames(outdir, subject)
    os.makedirs(files["basedir"], exist_ok=True)

    info = mne.io.read_info(preproc_file or epoch_file)

    # osl-ephys' report and forward step both read the info back from disk.
    mne.io.RawArray(np.zeros((len(info["ch_names"]), 1)), info).save(
        files["info_fif"], overwrite=True
    )

    dists = mne.dig_mri_distances(
        info, trans_path, subject, subjects_dir=subjects_dir
    )
    logger.info(
        "Distance between HSP and MRI (mean/min/max): %.2f mm / %.2f mm / %.2f mm",
        np.mean(dists * 1e3),
        np.min(dists * 1e3),
        np.max(dists * 1e3),
    )

    coreg_plot = None
    if make_plot:
        try:
            fig = mne.viz.plot_alignment(
                info,
                trans=trans_path,
                subject=subject,
                subjects_dir=subjects_dir,
                surfaces="head",
                dig=True,
                show_axes=True,
                meg=("sensors",),
            )
            fig.plotter.export_html(files["coreg_html"])
            coreg_plot = op.relpath(files["coreg_html"], str(outdir))
            logger.info("saved %s", files["coreg_html"])
        except Exception as exc:  # pragma: no cover - depends on 3D rendering
            logger.warning("Could not render the coregistration plot: %s", exc)

    if reportdir is not None:
        src_report.add_to_data(
            f"{reportdir}/{subject}/data.pkl",
            _drop_missing_plots({
                "coregister": True,
                "surface_extraction_method": "freesurfer",
                "already_coregistered": True,
                "use_headshape": True,
                "use_nose": False,
                "allow_smri_scaling": False,
                "n_init_coreg": None,
                # osl-ephys' report expects [nasion, lpa, rpa, rms] in cm; only
                # the RMS is meaningful for a transform we did not fit here.
                "fid_err": np.array(
                    [np.nan, np.nan, np.nan, np.sqrt(np.mean(dists**2)) * 1e2]
                ),
                "coreg_plot": coreg_plot,
                "trans_file": trans_path,
            }),
        )


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------


def fs_forward_model(
    outdir,
    subject,
    preproc_file=None,
    epoch_file=None,
    subjects_dir=None,
    trans=None,
    gridstep=5,
    model="Single Layer",
    conductivity=None,
    ico=4,
    mindist=0.0,
    bound_by_inner_skull=True,
    reportdir=None,
):
    """Build a volumetric source space, BEM and forward solution with MNE.

    Parameters
    ----------
    outdir : str
        osl-ephys output directory.
    subject : str
        Subject label; must also be the FreeSurfer subject directory name.
    preproc_file : str, optional
        Preprocessed fif file, used for its measurement info.
    epoch_file : str, optional
        Epoched fif file, used when ``preproc_file`` is None.
    subjects_dir : str, optional
        FreeSurfer subjects directory.  Defaults to ``$SUBJECTS_DIR``.
    trans : str, optional
        Path to the ``-trans.fif``.
    gridstep : float, optional
        Source grid spacing in mm.
    model : str, optional
        ``'Single Layer'`` (brain only) or ``'Triple Layer'`` (scalp, skull,
        brain).  MEG is insensitive to skull conductivity, so single layer is
        the usual choice for OPM data.
    conductivity : list of float, optional
        BEM conductivities.  Defaults to ``(0.3,)`` for a single layer and
        ``(0.3, 0.006, 0.3)`` for three layers.
    ico : int, optional
        BEM surface subdivision.
    mindist : float, optional
        Discard sources closer than this many mm to the inner skull.
    bound_by_inner_skull : bool, optional
        Restrict the grid to the inner skull surface.  When False the grid
        covers the whole MRI bounding box, which is much larger and slower.
    reportdir : str, optional
        Report directory.

    Notes
    -----
    Writes ``space-src.fif``, ``bem-sol.fif`` and ``model-fwd.fif`` into
    ``{outdir}/{subject}/fs_src/``.
    """
    logger.info("fs_forward_model")

    subjects_dir = _resolve_subjects_dir(subjects_dir)
    trans_path = _resolve_trans(trans, subjects_dir, subject)

    files = get_fs_filenames(outdir, subject)
    os.makedirs(files["basedir"], exist_ok=True)

    info = mne.io.read_info(preproc_file or epoch_file)

    if conductivity is None:
        conductivity = (0.3,) if model == "Single Layer" else (0.3, 0.006, 0.3)
    conductivity = tuple(conductivity)

    logger.info("setting up a %s mm volume source space", gridstep)
    surface = None
    if bound_by_inner_skull:
        surface = op.join(subjects_dir, subject, "bem", "inner_skull.surf")
        if not op.exists(surface):
            raise FileNotFoundError(
                f"Inner skull surface not found: {surface}. Run the FreeSurfer "
                f"stage (which builds the watershed BEM surfaces) first, or set "
                f"bound_by_inner_skull: false."
            )

    src = mne.setup_volume_source_space(
        subject=subject,
        subjects_dir=subjects_dir,
        pos=gridstep,
        surface=surface,
        add_interpolator=True,
    )
    mne.write_source_spaces(files["source_space"], src, overwrite=True)

    logger.info("making the BEM (conductivity=%s, ico=%s)", conductivity, ico)
    bem_model = mne.make_bem_model(
        subject=subject,
        subjects_dir=subjects_dir,
        conductivity=conductivity,
        ico=ico,
    )
    bem = mne.make_bem_solution(bem_model)
    mne.write_bem_solution(files["bem"], bem, overwrite=True)

    logger.info("making the forward solution")
    fwd = mne.make_forward_solution(
        info,
        trans=trans_path,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=mindist,
    )
    mne.write_forward_solution(files["fwd_model"], fwd, overwrite=True)
    logger.info(
        "forward solution has %d sources", fwd["src"][0]["nuse"]
    )

    if reportdir is not None:
        src_report.add_to_data(
            f"{reportdir}/{subject}/data.pkl",
            {
                "forward_model": True,
                "surface_extraction_method": "freesurfer",
                "model": model,
                "gridstep": gridstep,
                "eeg": False,
                "conductivity": conductivity,
                "ico": ico,
                "mindist": mindist,
                "n_sources": int(fwd["src"][0]["nuse"]),
            },
        )


# ---------------------------------------------------------------------------
# MNI grid and parcellation, without FSL
# ---------------------------------------------------------------------------


def _resample_to_isotropic(img, gridstep: float, interpolation: str = "continuous"):
    """Resample a NIfTI image to isotropic ``gridstep`` mm voxels.

    Stands in for osl-ephys' ``flirt -applyisoxfm``.  nilearn recomputes the
    bounding box, but voxel-to-world coordinates stay in the image's own (MNI)
    space, which is all the nearest-neighbour matching downstream needs.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        Image to resample.
    gridstep : float
        Target voxel size in mm.
    interpolation : str, optional
        ``'continuous'`` matches ``flirt``'s default trilinear resampling, and
        is what osl-ephys' RHINO path does.  ``'nearest'`` is right when the
        image is only being used as a coverage mask.

    Notes
    -----
    nilearn stamps the output with sform code 2 (ALIGNED_ANAT) regardless of
    the input, which loses the fact that the image is in MNI space.  The codes
    are restored here so the saved file stays valid for tools that check them
    (osl-ephys' own ``get_sform`` rejects code 2).
    """
    import warnings

    from nilearn.image import resample_img

    with warnings.catch_warnings():
        # A binary parcellation resampled with trilinear interpolation is
        # exactly what the RHINO path does, and the resulting partial-volume
        # weights are used deliberately by the 'spatial_basis' parcel method.
        warnings.filterwarnings("ignore", message=".*Resampling binary images.*")
        resampled = resample_img(
            img,
            target_affine=np.diag([gridstep] * 3),
            interpolation=interpolation,
            force_resample=True,
            copy_header=True,
        )

    header = resampled.header
    header.set_sform(resampled.affine, code=int(img.header["sform_code"]) or 4)
    header.set_qform(resampled.affine, code=int(img.header["qform_code"]) or 4)
    return resampled


def _nii_pointcloud(img, volindex: Optional[int] = None):
    """Return the non-zero voxels of an image as an ``(3, n)`` mm point cloud.

    The in-memory equivalent of osl-ephys'
    :func:`~osl_ephys.source_recon.rhino.utils.niimask2mmpointcloud`, which
    only accepts a filename and reads the transform back off the sform.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        Image to read.  A 4D image needs ``volindex``.
    volindex : int, optional
        Volume to take from a 4D image.

    Returns
    -------
    coords : numpy.ndarray
        ``(3, n)`` coordinates in mm.
    values : numpy.ndarray
        ``(n,)`` voxel values.
    """
    import nibabel as nib

    data = img.get_fdata()
    if data.ndim == 4:
        if volindex is None:
            raise ValueError("volindex is required for a 4D image.")
        data = data[:, :, :, volindex]

    indices = np.asarray(np.where(data != 0))
    values = np.asarray(data[data != 0])
    coords = nib.affines.apply_affine(img.affine, indices.T).T
    return coords, values


def _drop_interpolation_dust(img, weight_tol: float = 1e-6):
    """Zero out negligible parcel weights left behind by interpolation.

    Resampling a parcellation trilinearly spreads a trace of every parcel into
    the voxels around it -- for the shipped 8 mm atlases, values down to 1e-42.
    They are numerically meaningless, but they are non-zero, and the
    nearest-neighbour assignment in :func:`_resample_parcellation` treats any
    non-zero voxel as part of the parcel.  Left in, they put nearly every voxel
    within reach of nearly every parcel.

    Each parcel is thresholded relative to its own maximum, so this is safe for
    probabilistic parcellations (whose real weights are far above the
    threshold) as well as binary ones.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        Resampled parcellation, 3D or 4D (parcels along the last axis).
    weight_tol : float, optional
        Relative threshold, as a fraction of each parcel's maximum weight.

    Returns
    -------
    img : nibabel.Nifti1Image
        The parcellation with negligible weights set to zero.
    """
    import nibabel as nib

    data = np.asarray(img.get_fdata())
    if data.ndim == 3:
        data = data[..., np.newaxis]
        squeeze = True
    else:
        squeeze = False

    peaks = np.abs(data).max(axis=(0, 1, 2), keepdims=True)
    data = np.where(np.abs(data) < weight_tol * peaks, 0.0, data)

    if squeeze:
        data = data[..., 0]

    return nib.Nifti1Image(data, img.affine, img.header)


def _resample_parcellation(
    parcellation_file: str,
    voxel_coords: np.ndarray,
    working_dir: str,
    weight_tol: float = 1e-6,
) -> np.ndarray:
    """FSL-free replacement for :func:`osl_ephys...parcellation.resample_parcellation`.

    Parameters
    ----------
    parcellation_file : str
        Parcellation NIfTI, in the same space as ``voxel_coords``.
    voxel_coords : numpy.ndarray
        ``(3, nvoxels)`` coordinates in mm.
    working_dir : str
        Directory for the resampled parcellation.
    weight_tol : float, optional
        Relative threshold for discarding interpolation dust; see
        :func:`_drop_interpolation_dust`.

    Returns
    -------
    parcellation_asmatrix : numpy.ndarray
        ``(nvoxels, nparcels)`` parcellation resampled onto ``voxel_coords``.

    Notes
    -----
    Mirrors osl-ephys' logic -- resample to the grid resolution, then assign
    each target voxel the value of the nearest parcellation voxel within one
    grid step -- substituting nilearn for ``flirt``, and thresholding the
    interpolation dust that nilearn's reorientation leaves behind.
    """
    import nibabel as nib
    from scipy.spatial import KDTree

    gridstep = int(rhino_utils.get_gridstep(voxel_coords.T) / 1000)
    logger.info("gridstep = %d mm", gridstep)

    parcellation_file = parcellation.find_file(parcellation_file)
    path, parcellation_name = op.split(
        op.splitext(op.splitext(parcellation_file)[0])[0]
    )

    os.makedirs(working_dir, exist_ok=True)
    parcellation_resampled = op.join(
        working_dir, f"{parcellation_name}_{gridstep}mm.nii.gz"
    )

    resampled = _resample_to_isotropic(nib.load(parcellation_file), gridstep)
    resampled = _drop_interpolation_dust(resampled, weight_tol)
    # Kept for provenance and for inspecting the grid the parcels were read on.
    nib.save(resampled, parcellation_resampled)

    nparcels = resampled.shape[3]
    parcellation_asmatrix = np.zeros((voxel_coords.shape[1], nparcels))

    for parcel_index in range(nparcels):
        parcellation_coords, parcellation_vals = _nii_pointcloud(
            resampled, parcel_index
        )
        if parcellation_coords.shape[1] == 0:
            continue

        kdtree = KDTree(parcellation_coords.T)
        distances, indices = kdtree.query(voxel_coords.T)

        within = distances < gridstep
        parcellation_asmatrix[within, parcel_index] = parcellation_vals[
            indices[within]
        ]

    return parcellation_asmatrix


def _mni_grid_from_reference(
    reference_brain: str, parcellation_file: str, spatial_resolution: int
):
    """Build a regular MNI grid at ``spatial_resolution`` mm.

    Parameters
    ----------
    reference_brain : str
        ``'parcellation'`` to derive the grid from the parcellation's own
        coverage (the default -- it guarantees every grid point is inside a
        parcel), or a path to a NIfTI in MNI space.
    parcellation_file : str
        Parcellation NIfTI, used when ``reference_brain == 'parcellation'``.
    spatial_resolution : int
        Grid spacing in mm.

    Returns
    -------
    coords : numpy.ndarray
        ``(3, nvoxels)`` grid coordinates in mm.
    """
    import nibabel as nib

    if reference_brain == "parcellation":
        reference_file = parcellation.find_file(parcellation_file)
    else:
        reference_file = parcellation.find_file(reference_brain)

    img = nib.load(reference_file)

    data = img.get_fdata()
    if data.ndim == 4:
        # Collapse a 4D parcellation to its overall coverage mask before
        # resampling, so overlapping parcels do not cancel.
        img = nib.Nifti1Image(np.abs(data).sum(axis=3), img.affine, img.header)

    # A coverage mask only needs to know which voxels are inside the brain, so
    # nearest-neighbour keeps the boundary crisp.
    resampled = _resample_to_isotropic(img, spatial_resolution, interpolation="nearest")

    coords, _ = _nii_pointcloud(resampled)
    return coords


def _transform_to_mni(
    fwd: mne.Forward,
    recon_timeseries: np.ndarray,
    subject: str,
    subjects_dir: str,
    trans_path: str,
    parcellation_file: str,
    reference_brain: str,
    spatial_resolution: Optional[int],
):
    """Morph reconstructed dipole time courses onto a regular MNI grid.

    The FSL-free counterpart of
    :func:`osl_ephys...beamforming.transform_recon_timeseries`.

    Parameters
    ----------
    fwd : mne.Forward
        Forward solution the beamformer was built from.  Its source positions
        are in head coordinates, in metres.
    recon_timeseries : numpy.ndarray
        ``(ndipoles, ntpts)`` or ``(ndipoles, ntpts, ntrials)``.
    subject, subjects_dir, trans_path : str
        FreeSurfer subject, subjects directory, and head<->MRI transform.
    parcellation_file : str
        Parcellation, used to define the target grid.
    reference_brain : str
        See :func:`_mni_grid_from_reference`.
    spatial_resolution : int or None
        Target grid spacing in mm.  Defaults to the forward model's own
        grid step.

    Returns
    -------
    timeseries_mni : numpy.ndarray
        Time courses on the MNI grid, same trailing dimensions as the input.
    coords_mni : numpy.ndarray
        ``(3, nvoxels)`` grid coordinates in mm.
    """
    coords_mni, grid_index, dipole_index = _mni_mapping(
        fwd,
        subject=subject,
        subjects_dir=subjects_dir,
        trans_path=trans_path,
        parcellation_file=parcellation_file,
        reference_brain=reference_brain,
        spatial_resolution=spatial_resolution,
    )

    timeseries_mni = np.zeros(
        (coords_mni.shape[1],) + recon_timeseries.shape[1:],
        dtype=recon_timeseries.dtype,
    )
    timeseries_mni[grid_index] = recon_timeseries[dipole_index]

    return timeseries_mni, coords_mni


def _mni_mapping(
    fwd: mne.Forward,
    subject: str,
    subjects_dir: str,
    trans_path: str,
    parcellation_file: str,
    reference_brain: str,
    spatial_resolution: Optional[int],
):
    """Match each MNI grid point to its nearest reconstructed dipole.

    Split out from :func:`_transform_to_mni` so the mapping can be built
    before the beamformer runs, which lets :func:`_apply_lcmv_to_mni` write
    straight onto the MNI grid and lets the size of the source array be
    checked before an hour is spent filling it.

    Parameters
    ----------
    fwd : mne.Forward
        Forward solution the beamformer was built from.  Its source positions
        are in head coordinates, in metres.
    subject, subjects_dir, trans_path : str
        FreeSurfer subject, subjects directory, and head<->MRI transform.
    parcellation_file : str
        Parcellation, used to define the target grid.
    reference_brain : str
        See :func:`_mni_grid_from_reference`.
    spatial_resolution : int or None
        Target grid spacing in mm.  Defaults to the forward model's own
        grid step.

    Returns
    -------
    coords_mni : numpy.ndarray
        ``(3, nvoxels)`` grid coordinates in mm.
    grid_index : numpy.ndarray
        Indices of the grid points that have a dipole within one grid step;
        every other grid point stays at zero.
    dipole_index : numpy.ndarray
        The dipole feeding each of those grid points, same length.
    """
    from scipy.spatial import KDTree

    vs = fwd["src"][0]
    recon_coords_head = vs["rr"][vs["vertno"]]  # metres, head coordinates

    if spatial_resolution is None:
        spatial_resolution = rhino_utils.get_gridstep(vs["rr"])
    spatial_resolution = int(spatial_resolution)
    logger.info("spatial_resolution = %d mm", spatial_resolution)

    # head -> MRI -> MNI, in mm
    mri_head_t = _read_mri_head_t(trans_path)
    recon_coords_mni = mne.head_to_mni(
        recon_coords_head, subject, mri_head_t, subjects_dir=subjects_dir
    )

    coords_mni = _mni_grid_from_reference(
        reference_brain, parcellation_file, spatial_resolution
    )

    # For each grid point take the nearest reconstructed dipole, leaving grid
    # points with no dipole within one grid step at zero.
    kdtree = KDTree(recon_coords_mni)
    distances, indices = kdtree.query(coords_mni.T)
    within = distances < spatial_resolution
    grid_index = np.flatnonzero(within)

    logger.info(
        "mapped %d/%d MNI grid points onto %d dipoles",
        grid_index.size,
        coords_mni.shape[1],
        recon_coords_mni.shape[0],
    )

    return coords_mni, grid_index, indices[grid_index]


# ---------------------------------------------------------------------------
# Beamforming
# ---------------------------------------------------------------------------


def _resolve_pick_ori(pick_ori: Optional[str]) -> Optional[str]:
    """Validate the dipole orientation option for this backend.

    Rejects orientations that only osl-ephys' own beamformer implements, and
    ``'vector'``, which produces a three-component estimate that the volumetric
    parcellation cannot reduce to a parcel time course.
    """
    if pick_ori in _OSL_ONLY_PICK_ORI:
        raise ValueError(
            f"pick_ori={pick_ori!r} is implemented by osl-ephys' own beamformer "
            f"and is not available on the freesurfer backend, which uses "
            f"mne.beamformer.make_lcmv. Use "
            f"{_OSL_ONLY_PICK_ORI[pick_ori]!r} here, or switch to "
            f"pipeline.source_backend: rhino."
        )

    if pick_ori == "vector":
        raise ValueError(
            "pick_ori='vector' gives each dipole three components, which the "
            "volumetric parcellation cannot collapse into a parcel time "
            "course. Use 'max-power' (or 'normal') for the osl-ephys pipeline; "
            "for vector output use the mne-bids-pipeline route's "
            "_beamformer_pick_ori instead."
        )

    return pick_ori


def _resolve_rank(rank, data):
    """Resolve the rank used for the covariance and for ``make_lcmv``.

    Mirrors :func:`custom.run_beamformer.resolve_rank`, so both routes through
    this repository regularise the beamformer the same way.

    Parameters
    ----------
    rank : str or dict or None
        ``'data'`` estimates the rank from this subject's own data; anything
        else (an explicit ``{ch_type: n}``, ``'info'``, None) is handed to MNE
        untouched.
    data : mne.io.Raw or mne.Epochs
        The data the beamformer is built from, already picked to ``chantypes``.

    Returns
    -------
    rank : dict or str or None
        A ``{ch_type: n}`` dict when estimated, otherwise ``rank`` unchanged.

    Notes
    -----
    ``'data'`` takes the element-wise **minimum** of two estimates, because
    neither is trustworthy alone:

    * ``rank='info'`` reads the SSS bookkeeping out of ``info['proc_history']``,
      so it knows the Maxwell basis dimension but nothing about what ICA
      removed afterwards -- it *overstates* the rank of cleaned data.
    * the data-driven estimate sees ICA, but only at a sensible tolerance
      (:data:`_RANK_TOL`); it can still be fooled, and is then capped by the
      info rank.
    """
    if rank != "data":
        logger.info("using the configured rank: %s", rank)
        return rank

    candidates = {}
    try:
        candidates["info"] = mne.compute_rank(data, rank="info", verbose=False)
    except Exception as exc:  # noqa: BLE001 -- no proc_history / no projections
        logger.info("no info-based rank available (%s)", exc)
    candidates["data"] = mne.compute_rank(
        data, tol=_RANK_TOL, tol_kind=_RANK_TOL_KIND, verbose=False
    )

    for name, value in candidates.items():
        extra = f" (tol={_RANK_TOL}, tol_kind={_RANK_TOL_KIND!r})" if name == "data" else ""
        logger.info("%s rank%s: %s", name, extra, value)

    keys = set().union(*(c.keys() for c in candidates.values()))
    resolved = {
        key: min(c[key] for c in candidates.values() if key in c) for key in keys
    }
    logger.info("using the element-wise minimum rank: %s", resolved)

    # A data estimate at or above the info rank means the tolerance did not
    # separate the nulled directions from the retained ones, so the result is
    # just the info rank and whatever ICA removed goes unaccounted for.
    info_rank = candidates.get("info")
    if info_rank is not None and any(
        candidates["data"].get(key, 0) >= value for key, value in info_rank.items()
    ):
        logger.warning(
            "the data-driven rank is not below the info rank, so it is not "
            "detecting the rank lost to SSS/ICA -- check it against "
            "src/custom/rank_check.py before trusting the result."
        )

    return resolved


def _n_parcels(parcellation_file: str) -> int:
    """Number of parcels in a parcellation NIfTI (its 4th dimension)."""
    import nibabel as nib

    return int(nib.load(parcellation.find_file(parcellation_file)).shape[3])


def _check_orthogonalisation_rank(
    rank, orthogonalisation: Optional[str], parcellation_file: str
) -> None:
    """Fail before beamforming when symmetric orthogonalisation cannot work.

    :func:`osl_ephys...parcellation.symmetric_orthogonalise` requires the
    parcel time courses to be linearly independent, and they inherit the rank
    the beamformer was regularised to.  A rank below the parcel count therefore
    always ends in "Not full rank, rank required is N, but rank is only M" --
    but only after the forward model, the beamformer and the parcellation have
    all been computed, half an hour into the job.
    """
    if orthogonalisation != "symmetric" or not isinstance(rank, dict):
        return

    n_parcels = _n_parcels(parcellation_file)
    too_low = {key: value for key, value in rank.items() if value < n_parcels}
    if too_low:
        raise ValueError(
            f"orthogonalisation='symmetric' needs the {n_parcels} parcel time "
            f"courses of {op.basename(parcellation_file)} to be linearly "
            f"independent, but the beamformer rank is {too_low}, which caps "
            f"them at that many dimensions. Raise the rank (rank: data "
            f"estimates it from this subject's data), use a parcellation with "
            f"at most {min(too_low.values())} parcels, or set "
            f"orthogonalisation to 'local' or null."
        )


def _compute_data_cov(data, is_epochs: bool, cov_method: str, rank):
    """Compute the data covariance the beamformer is built from.

    osl-ephys' ``beamforming.make_lcmv`` estimates this internally; MNE's
    expects it as an argument, so it is computed here over the whole of
    whatever was passed in (every epoch, or the full continuous recording
    excluding bad segments).

    ``rank`` is the resolved rank, passed on so the covariance is regularised
    in the same subspace the beamformer inverts it in.
    """
    logger.info("computing the data covariance (method=%s, rank=%s)", cov_method, rank)
    if is_epochs:
        return mne.compute_covariance(data, method=cov_method, rank=rank, verbose=False)
    return mne.compute_raw_covariance(
        data, method=cov_method, rank=rank, verbose=False
    )


def _apply_lcmv(data, filters, is_epochs: bool) -> np.ndarray:
    """Apply the beamformer, returning ``(ndipoles, ntpts[, ntrials])``.

    The pipeline itself goes through :func:`_apply_lcmv_to_mni`, which fuses
    this with :func:`_transform_to_mni` to avoid holding both arrays; this
    stays as the unfused reference the two are checked against.
    """
    if not is_epochs:
        return mne.beamformer.apply_lcmv_raw(data, filters).data

    n_trials = len(data)
    stcs = mne.beamformer.apply_lcmv_epochs(data, filters, return_generator=True)

    out = None
    for trial, stc in enumerate(stcs):
        if out is None:
            # Allocate once from the first estimate rather than materialising
            # every trial's estimate before stacking: at a 8 mm grid with a few
            # hundred trials the full list is several GB.
            out = np.zeros(stc.data.shape + (n_trials,), dtype=stc.data.dtype)
        out[..., trial] = stc.data

    if out is None:
        raise ValueError("No epochs to beamform.")
    return out


def _apply_lcmv_to_mni(data, filters, is_epochs: bool, mapping) -> np.ndarray:
    """Beamform straight onto the MNI grid, returning ``(nvoxels, ntpts[, ntrials])``.

    The two-step route -- reconstruct every trial into a
    ``(ndipoles, ntpts, ntrials)`` array with :func:`_apply_lcmv`, then morph
    that whole array with :func:`_transform_to_mni` -- holds both arrays at
    once, and on an epoched recording each runs to tens of GiB.  Writing each
    trial onto the grid as it leaves the beamformer means only one of them
    ever exists, halving the peak for identical output.

    Parameters
    ----------
    data : mne.io.Raw or mne.Epochs
        Data to beamform, already picked to the beamformed channel types.
    filters : instance of Beamformer
        LCMV filters from :func:`mne.beamformer.make_lcmv`.
    is_epochs : bool
        Whether ``data`` is epoched.
    mapping : tuple
        ``(coords_mni, grid_index, dipole_index)`` from :func:`_mni_mapping`.

    Returns
    -------
    timeseries_mni : numpy.ndarray
        ``(nvoxels, ntpts)`` or ``(nvoxels, ntpts, ntrials)``, in
        :data:`_SOURCE_DTYPE`.

    Raises
    ------
    ValueError
        If ``data`` is epoched but holds no epochs.
    """
    coords_mni, grid_index, dipole_index = mapping
    n_voxels = coords_mni.shape[1]

    if not is_epochs:
        stc = mne.beamformer.apply_lcmv_raw(data, filters)
        out = np.zeros((n_voxels, stc.data.shape[1]), dtype=_SOURCE_DTYPE)
        out[grid_index] = stc.data[dipole_index]
        return out

    n_trials = len(data)
    if n_trials == 0:
        raise ValueError("No epochs to beamform.")

    stcs = mne.beamformer.apply_lcmv_epochs(data, filters, return_generator=True)

    out = None
    for trial, stc in enumerate(stcs):
        if out is None:
            # Allocate once from the first estimate, so the time axis comes
            # from the beamformer rather than being assumed.
            out = np.zeros(
                (n_voxels, stc.data.shape[1], n_trials), dtype=_SOURCE_DTYPE
            )
        out[grid_index, :, trial] = stc.data[dipole_index]

    return out


def fs_beamform_and_parcellate(
    outdir,
    subject,
    preproc_file,
    epoch_file,
    chantypes,
    rank,
    parcellation_file,
    method,
    orthogonalisation,
    subjects_dir=None,
    trans=None,
    freq_range=None,
    decim=None,
    weight_norm="unit-noise-gain-invariant",
    pick_ori="max-power",
    reg=0.05,
    cov_method="empirical",
    reduce_rank=False,
    spatial_resolution=None,
    reference_brain="parcellation",
    extra_chans="stim",
    neighbour_distance=None,
    reportdir=None,
):
    """LCMV beamform onto the volumetric grid, morph to MNI and parcellate.

    The FSL-free counterpart of
    :func:`osl_ephys...wrappers.beamform_and_parcellate`.  Beamforming uses
    :func:`mne.beamformer.make_lcmv` against the forward model written by
    :func:`fs_forward_model`; parcellation reuses osl-ephys' own parcel
    time-course maths so output matches the RHINO backend.

    Parameters
    ----------
    outdir : str
        osl-ephys output directory.
    subject : str
        Subject label.
    preproc_file : str
        Preprocessed fif file.
    epoch_file : str
        Epoched fif file.  When given, epochs are reconstructed rather than
        continuous data.
    chantypes : str or list of str
        Channel types to beamform (``'mag'`` for OPM data).
    rank : str or dict
        Rank used to regularise the covariance and the beamformer.  ``'data'``
        estimates it from this subject's own data (see :func:`_resolve_rank`),
        which is what an array job wants since the rank differs per subject.
        An explicit ``{ch_type: n}``, ``'info'`` or None is passed to MNE
        unchanged.
    parcellation_file : str
        Parcellation NIfTI, by name (resolved inside osl-ephys) or path.
    method : str
        Parcel time-course method: ``'spatial_basis'`` or ``'pca'``.
    orthogonalisation : str or None
        ``'symmetric'``, ``'local'`` or None.
    subjects_dir : str, optional
        FreeSurfer subjects directory.  Defaults to ``$SUBJECTS_DIR``.
    trans : str, optional
        Path to the ``-trans.fif``.
    freq_range : list, optional
        ``[l_freq, h_freq]`` bandpass applied before beamforming.
    decim : int, optional
        Decimate in time by this factor after band-passing, before the
        covariance and the beamformer.  Peak memory is one
        ``(voxels, times, trials)`` array, so this divides it: an epoched
        recording kept at an acquisition rate far above ``freq_range[1]``
        needs tens of GiB it cannot use, since the band-pass has already
        removed everything the extra samples could carry.  Defaults to None,
        which leaves the sampling rate alone.
    weight_norm : str, optional
        Beamformer weight normalisation, as accepted by
        :func:`mne.beamformer.make_lcmv`.
    pick_ori : str, optional
        Dipole orientation, as accepted by :func:`mne.beamformer.make_lcmv`.
        osl-ephys' ``'max-power-pre-weight-norm'`` is not available here.
    reg : float, optional
        Covariance regularisation.
    cov_method : str, optional
        Covariance estimator passed to :func:`mne.compute_covariance`.
    reduce_rank : bool, optional
        Drop the smallest singular value of each source's leadfield.  MEG is
        blind to radially oriented sources, so a free-orientation forward from
        a single-layer BEM can be rank-deficient, and
        :func:`mne.beamformer.make_lcmv` then fails with "Singular matrix
        detected when estimating spatial filters".  Set this True if that
        happens.  Defaults to False to match ``_reduce_rank`` in the
        mne-bids-pipeline configs.
    spatial_resolution : int, optional
        MNI grid spacing in mm.  Defaults to the forward model's grid step.
    reference_brain : str, optional
        ``'parcellation'`` (default) or a path to a NIfTI in MNI space.
    extra_chans : str or list of str, optional
        Extra channels carried into the parcellated Raw file.
    neighbour_distance : float, optional
        Required when ``orthogonalisation='local'``.
    reportdir : str, optional
        Report directory.

    Raises
    ------
    ValueError
        If ``pick_ori`` is an osl-ephys-only option, ``orthogonalisation`` is
        not recognised, ``decim`` would alias the retained band, or the
        resolved rank is below the parcel count while
        ``orthogonalisation='symmetric'``.
    FileNotFoundError
        If the forward model has not been built yet.
    MemoryError
        If the source-space array cannot fit in the job's memory budget; see
        :func:`_check_source_memory`.
    """
    logger.info("fs_beamform_and_parcellate")

    subjects_dir = _resolve_subjects_dir(subjects_dir)
    trans_path = _resolve_trans(trans, subjects_dir, subject)
    pick_ori = _resolve_pick_ori(pick_ori)

    if isinstance(chantypes, str):
        chantypes = [chantypes]

    files = get_fs_filenames(outdir, subject)
    if not op.exists(files["fwd_model"]):
        raise FileNotFoundError(
            f"Forward model not found: {files['fwd_model']}. Add fs_forward_model "
            f"to the source_recon config before fs_beamform_and_parcellate."
        )

    data, is_epochs = _load_data(preproc_file, epoch_file)
    data = _bandpass(data, freq_range)
    data = _decimate(data, decim, is_epochs, freq_range)
    chantype_data = data.copy().pick(chantypes)

    fwd = mne.read_forward_solution(files["fwd_model"])

    # Built before the beamformer rather than after it: the grid size is what
    # sets peak memory, and this is the last cheap moment to refuse a job that
    # cannot fit.
    mapping = _mni_mapping(
        fwd,
        subject=subject,
        subjects_dir=subjects_dir,
        trans_path=trans_path,
        parcellation_file=parcellation_file,
        reference_brain=reference_brain,
        spatial_resolution=spatial_resolution,
    )
    coords_mni = mapping[0]
    _check_source_memory(
        coords_mni.shape[1],
        n_times=len(data.times),
        n_trials=len(data) if is_epochs else 1,
    )

    # --- Beamformer filters ---
    rank = _resolve_rank(rank, chantype_data)
    _check_orthogonalisation_rank(rank, orthogonalisation, parcellation_file)

    logger.info("mne.beamformer.make_lcmv (chantypes=%s, rank=%s)", chantypes, rank)
    data_cov = _compute_data_cov(chantype_data, is_epochs, cov_method, rank)
    filters = mne.beamformer.make_lcmv(
        chantype_data.info,
        fwd,
        data_cov,
        reg=reg,
        noise_cov=None,
        pick_ori=pick_ori,
        weight_norm=weight_norm,
        rank=rank,
        reduce_rank=reduce_rank,
    )
    filters.save(files["filters"], overwrite=True)

    filters_cov_plot = _plot_cov(data_cov, files["basedir"], outdir)

    # --- Apply, morph to MNI, parcellate ---
    logger.info("applying the beamformer")
    bf_data_mni = _apply_lcmv_to_mni(chantype_data, filters, is_epochs, mapping)

    logger.info("parcellation using %s", parcellation_file)
    parcellation_asmatrix = _resample_parcellation(
        parcellation_file, coords_mni, files["parcdir"]
    )
    # osl-ephys' own parcel time-course maths, so both backends agree.
    parcel_data, _, _ = parcellation._get_parcel_timeseries(
        bf_data_mni, parcellation_asmatrix, method=method
    )

    parcel_data = _orthogonalise(
        parcel_data, orthogonalisation, parcellation_file, neighbour_distance
    )

    parc_fif_file, _ = _save_parcellated(
        parcel_data, data, is_epochs, files["parcdir"], extra_chans
    )

    parc_psd_plot, parc_corr_plot = _plot_parcellation(
        parcel_data, data, parcellation_file, freq_range, files["parcdir"], outdir
    )

    if reportdir is not None:
        src_report.add_to_data(
            f"{reportdir}/{subject}/data.pkl",
            _drop_missing_plots({
                "beamform_and_parcellate": True,
                "beamform": True,
                "parcellate": True,
                "surface_extraction_method": "freesurfer",
                "chantypes": chantypes,
                "rank": rank,
                "reg": reg,
                "freq_range": freq_range,
                "pick_ori": pick_ori,
                "weight_norm": weight_norm,
                "reduce_rank": reduce_rank,
                "filters_cov_plot": filters_cov_plot,
                "parcellation_file": parcellation_file,
                "method": method,
                "reference_brain": reference_brain,
                "orthogonalisation": orthogonalisation,
                "parc_fif_file": str(parc_fif_file),
                "n_samples": parcel_data.shape[1],
                "n_parcels": parcel_data.shape[0],
                "n_epochs": parcel_data.shape[2] if parcel_data.ndim == 3 else None,
                "parc_psd_plot": parc_psd_plot,
                "parc_corr_plot": parc_corr_plot,
                "parc_freqbands_plot": None,
            }),
        )


def _orthogonalise(
    parcel_data: np.ndarray,
    orthogonalisation: Optional[str],
    parcellation_file: str,
    neighbour_distance: Optional[float],
) -> np.ndarray:
    """Apply osl-ephys' leakage correction, if requested."""
    if orthogonalisation in (None, "none", "None"):
        return parcel_data

    if orthogonalisation == "symmetric":
        logger.info("symmetric orthogonalisation")
        return parcellation.symmetric_orthogonalise(
            parcel_data, maintain_magnitudes=True
        )

    if orthogonalisation == "local":
        logger.info("local orthogonalisation")
        if neighbour_distance is None:
            raise ValueError(
                "neighbour_distance must be set when orthogonalisation='local'."
            )
        return parcellation.local_orthogonalise(
            parcel_data, parcellation_file, neighbour_distance
        )

    raise ValueError(
        f"Unknown orthogonalisation {orthogonalisation!r}. Valid options: "
        f"'symmetric', 'local', None."
    )


def _save_parcellated(parcel_data, data, is_epochs, parcdir, extra_chans):
    """Write the parcellated data as an MNE Raw or Epochs file.

    Epochs go through :func:`custom.osl.parcel_epochs.convert2mne_epochs`
    rather than osl-ephys' own, which drops the condition names and the epoch
    time axis; see that module for what goes wrong downstream when they are
    lost.  ``data`` is the decimated object the beamformer was applied to, so
    its sampling rate matches ``parcel_data``.
    """
    os.makedirs(parcdir, exist_ok=True)

    if is_epochs:
        parc_fif_file = op.join(parcdir, "lcmv-parc-epo.fif")
        parc_obj = convert2mne_epochs(parcel_data, data)
    else:
        parc_fif_file = op.join(parcdir, "lcmv-parc-raw.fif")
        parc_obj = parcellation.convert2mne_raw(
            parcel_data, data, extra_chans=extra_chans
        )

    logger.info("saving %s", parc_fif_file)
    parc_obj.save(parc_fif_file, overwrite=True)
    return parc_fif_file, parc_obj


def _plot_cov(data_cov, basedir: str, outdir) -> Optional[str]:
    """Save a data-covariance figure for the report."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(data_cov.data, cmap="RdBu_r")
        ax.set_title("Data covariance")
        fig.colorbar(image, ax=ax)
        path = op.join(basedir, "filters_cov.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return op.relpath(path, str(outdir))
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        logger.warning("Could not render the covariance plot: %s", exc)
        return None


def _plot_parcellation(
    parcel_data, data, parcellation_file, freq_range, parcdir, outdir
):
    """Save the parcel PSD and correlation figures for the report."""
    os.makedirs(parcdir, exist_ok=True)
    psd_path = op.join(parcdir, "psd.png")
    corr_path = op.join(parcdir, "corr.png")

    try:
        parcellation.plot_psd(
            parcel_data,
            fs=data.info["sfreq"],
            freq_range=freq_range,
            parcellation_file=parcellation_file,
            filename=psd_path,
            freesurfer=False,
        )
        parcellation.plot_correlation(parcel_data, filename=corr_path)
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        logger.warning("Could not render the parcellation plots: %s", exc)
        return None, None

    return op.relpath(psd_path, str(outdir)), op.relpath(corr_path, str(outdir))


SOURCE_EXTRA_FUNCS = [fs_coregister, fs_forward_model, fs_beamform_and_parcellate]
"""Custom wrappers passed to osl-ephys as ``extra_funcs`` for the source stage."""
