"""Source-reconstruction stage of the osl-ephys OPM pipeline.

Runs an osl-ephys ``source_recon`` chain over one subject: surfaces,
coregistration, forward model, LCMV beamforming and parcellation.

Two backends are supported, selected by ``pipeline.source_backend``:

``rhino``
    osl-ephys' native path (``surface_extraction_method='fsl'``).  RHINO
    extracts surfaces from the T1 and fits its own coregistration.  Requires
    FSL, and does not use the FreeSurfer/MNE coregistration this repository
    produces elsewhere.

``freesurfer``
    The wrappers in :mod:`custom.osl.fs_bridge`, which reuse the existing
    ``recon-all`` output and ``-trans.fif`` and need no FSL.

As in :mod:`custom.osl.preproc`, group-level report pages are left to the
``collate`` stage so that concurrent array tasks never write the same file.

Functions
---------
run
    Run the source-reconstruction stage for one subject.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import os
from copy import deepcopy
from types import SimpleNamespace
from typing import Optional

from ._config import source_config
from ._headless import setup_headless_3d
from ._paths import resolve_paths
from .fs_bridge import SOURCE_EXTRA_FUNCS
from .parcel_epochs import preserving_epoch_metadata


# Steps that take the resolved FreeSurfer paths, and the pipeline keys they
# come from.  Injected only when the step does not set them itself.
_FS_STEP_DEFAULTS: dict[str, tuple[str, ...]] = {
    "fs_coregister": ("subjects_dir", "trans"),
    "fs_forward_model": ("subjects_dir", "trans"),
    "fs_beamform_and_parcellate": ("subjects_dir", "trans"),
}


def run(cfg: SimpleNamespace) -> bool:
    """Run the source-reconstruction stage for one subject.

    Parameters
    ----------
    cfg : SimpleNamespace
        Config from :func:`custom.osl._config.load_config`.

    Returns
    -------
    success : bool
        True if source reconstruction completed.  osl-ephys catches step
        failures and writes them to ``{logsdir}/{subject}_src.error.log``.

    Raises
    ------
    FileNotFoundError
        If the preprocessed input, or the structural MRI needed for
        coregistration, is missing.
    ValueError
        If the config has no ``source_recon`` section.
    """
    from osl_ephys.source_recon import run_src_chain

    # Must follow the import above: it pulls in cv2, whose bundled Qt plugins
    # would otherwise abort the process the moment a 3D figure is drawn.
    setup_headless_3d()

    pipeline = cfg.pipeline
    paths = resolve_paths(pipeline)
    config = source_config(cfg)

    source_input = str(paths.source_input_fif)
    if not os.path.exists(source_input):
        raise FileNotFoundError(
            f"Preprocessed input not found: {source_input}. Run the preproc "
            f"stage first, or set pipeline.source_input "
            f"({pipeline.source_input!r} selects "
            f"{'epochs' if pipeline.source_input == 'epochs' else 'continuous data'})."
        )

    backend = pipeline.source_backend
    if backend == "rhino":
        surface_extraction_method = "fsl"
        extra_funcs: Optional[list] = None
    else:
        surface_extraction_method = "freesurfer"
        extra_funcs = SOURCE_EXTRA_FUNCS
        _ensure_freesurfer_env(paths)
        config = _inject_fs_defaults(config, paths)

    _check_smri(config, paths, backend)

    # run_src_chain infers which of the two it was handed; pass exactly one.
    use_epochs = pipeline.source_input == "epochs"

    print(f"[osl:source] subject:  {paths.subject_label}")
    print(f"[osl:source] backend:  {backend} ({surface_extraction_method})")
    print(f"[osl:source] input:    {source_input}")
    print(f"[osl:source] smri:     {paths.smri}")
    print(f"[osl:source] steps:    {[next(iter(s)) for s in config['source_recon']]}")

    # osl-ephys' own parcel converter drops the condition names and the epoch
    # time axis, which the group stage needs to average by condition. The
    # rhino backend runs osl-ephys' beamform_and_parcellate, so the fix has to
    # be installed around the chain rather than in a wrapper of ours.
    with preserving_epoch_metadata():
        flag = run_src_chain(
            config,
            outdir=str(paths.outdir),
            subject=paths.subject_label,
            preproc_file=None if use_epochs else source_input,
            smri_file=paths.smri,
            epoch_file=source_input if use_epochs else None,
            surface_extraction_method=surface_extraction_method,
            logsdir=str(paths.logsdir),
            reportdir=str(paths.src_reportdir),
            gen_report=False,
            extra_funcs=extra_funcs,
            random_seed=(
                pipeline.random_seed if pipeline.random_seed is not None else "auto"
            ),
        )

    if not flag:
        print(
            f"[osl:source] FAILED for {paths.subject_label}; see "
            f"{paths.logsdir}/{paths.subject_label}_src.error.log"
        )
        return False

    if pipeline.gen_report:
        _gen_report_data(config, paths, extra_funcs)

    print(f"[osl:source] wrote source-recon output under {paths.subject_dir}")
    return True


def _check_smri(config: dict, paths: SimpleNamespace, backend: str) -> None:
    """Fail early when a coregistration step needs a structural MRI we don't have."""
    steps = [next(iter(step)) for step in config["source_recon"]]
    needs_smri = any(
        step in ("compute_surfaces", "coregister", "make_watershed_bem")
        for step in steps
    )
    if needs_smri and not paths.smri:
        raise FileNotFoundError(
            f"No T1w image found for subject {paths.subject}; the "
            f"{backend} backend needs one for {steps}. Set pipeline.smri, or "
            f"check that the anat/ directory under {paths.subject_label} is "
            f"populated."
        )


def _ensure_freesurfer_env(paths: SimpleNamespace) -> None:
    """Satisfy osl-ephys' FreeSurfer check and point MNE at the subjects directory.

    osl-ephys gates its ``freesurfer`` path on ``$FREESURFERDIR``, which its own
    :func:`osl_ephys.source_recon.setup_freesurfer` sets.  The wrappers in
    :mod:`custom.osl.fs_bridge` only read ``recon-all`` *output* -- no
    FreeSurfer binaries -- so mirroring ``$FREESURFER_HOME`` is enough.
    """
    if not os.environ.get("FREESURFERDIR"):
        freesurfer_home = os.environ.get("FREESURFER_HOME")
        if not freesurfer_home:
            raise ValueError(
                "The freesurfer backend needs $FREESURFER_HOME (or $FREESURFERDIR) "
                "set, so that osl-ephys' FreeSurfer check passes. Pass --fs to "
                "mne-opm.sh, or export it before running."
            )
        os.environ["FREESURFERDIR"] = freesurfer_home

    if paths.freesurfer_subjects_dir:
        os.environ["SUBJECTS_DIR"] = str(paths.freesurfer_subjects_dir)


def _inject_fs_defaults(config: dict, paths: SimpleNamespace) -> dict:
    """Fill the resolved FreeSurfer paths into the fs_* steps that want them.

    osl-ephys only forwards a fixed set of arguments to source-recon wrappers,
    so anything else -- here the FreeSurfer subjects directory and the
    coregistration transform -- has to arrive through the step's own options.
    Values written explicitly in the YAML always win.
    """
    config = deepcopy(config)
    available = {
        "subjects_dir": paths.freesurfer_subjects_dir,
        "trans": paths.trans,
    }

    for step in config["source_recon"]:
        name, userargs = next(iter(step.items()))
        if name not in _FS_STEP_DEFAULTS:
            continue
        if userargs is None:
            userargs = {}
            step[name] = userargs
        for key in _FS_STEP_DEFAULTS[name]:
            if key not in userargs and available.get(key):
                userargs[key] = available[key]

    return config


def _gen_report_data(
    config: dict, paths: SimpleNamespace, extra_funcs: Optional[list]
) -> None:
    """Generate this subject's source-recon report data.

    Mirrors what :func:`osl_ephys.source_recon.run_src_chain` does internally
    when ``gen_report=True``, minus the group-level page build.
    """
    from osl_ephys.report import src_report

    os.makedirs(paths.src_reportdir / paths.subject_label, exist_ok=True)
    src_report.gen_html_data(
        config,
        str(paths.outdir),
        paths.subject_label,
        str(paths.src_reportdir),
        extra_funcs=extra_funcs,
        logsdir=str(paths.logsdir),
    )
    print(f"[osl:source] report data in {paths.src_reportdir / paths.subject_label}")
