"""Preprocessing stage of the osl-ephys OPM pipeline.

Runs an osl-ephys ``preproc`` chain over one subject's BIDS raw file, writing
``{subject_label}_preproc-raw.fif`` (and ``_epo.fif``, ``_events.npy``,
``_event-id.yml``, ``_ica.fif`` when the chain produces them) into
``{outdir}/{subject_label}/``.

The stage deliberately calls :func:`osl_ephys.preprocessing.run_proc_chain`
with ``gen_report=False`` and generates the subject's report *data* itself.
osl-ephys would otherwise also rebuild the group-level ``subject_report.html``
on every call, from inside its own try/except -- which in a SLURM array means
every task rewriting one shared file, and a rendering hiccup marking an
otherwise-successful subject as failed.  Building the group pages is the
``collate`` stage's job instead.

Functions
---------
run
    Run the preprocessing stage for one subject.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import os
from types import SimpleNamespace

from ._config import preproc_config
from ._paths import resolve_paths
from .extra_funcs import PREPROC_EXTRA_FUNCS


def run(cfg: SimpleNamespace) -> bool:
    """Run the preprocessing stage for one subject.

    Parameters
    ----------
    cfg : SimpleNamespace
        Config from :func:`custom.osl._config.load_config`.

    Returns
    -------
    success : bool
        True if preprocessing completed.  osl-ephys catches step failures and
        writes them to ``{logsdir}/{subject}_preproc.error.log``, so a False
        here means "check that log", not "an exception was raised".

    Raises
    ------
    FileNotFoundError
        If the subject's BIDS raw file does not exist.
    ValueError
        If the config has no ``preproc`` section.
    """
    from osl_ephys.preprocessing import run_proc_chain

    pipeline = cfg.pipeline
    paths = resolve_paths(pipeline)
    config = preproc_config(cfg)

    if not os.path.exists(paths.input_fif):
        raise FileNotFoundError(
            f"BIDS raw file not found: {paths.input_fif}. Run the bids stage "
            f"(mne-opm.sh bids) for subject {paths.subject} first."
        )

    print(f"[osl:preproc] subject:  {paths.subject_label}")
    print(f"[osl:preproc] input:    {paths.input_fif}")
    print(f"[osl:preproc] outdir:   {paths.outdir}")
    print(f"[osl:preproc] steps:    {[next(iter(s)) for s in cfg.preproc]}")

    dataset = run_proc_chain(
        config,
        paths.input_fif,
        subject=paths.subject_label,
        outdir=str(paths.outdir),
        logsdir=str(paths.logsdir),
        reportdir=str(paths.preproc_reportdir),
        ret_dataset=True,
        gen_report=False,
        overwrite=pipeline.overwrite,
        extra_funcs=PREPROC_EXTRA_FUNCS,
        random_seed=(
            pipeline.random_seed if pipeline.random_seed is not None else "auto"
        ),
    )

    # run_proc_chain returns an empty dict when a step raised.
    if not dataset:
        print(
            f"[osl:preproc] FAILED for {paths.subject_label}; see "
            f"{paths.logsdir}/{paths.subject_label}_preproc.error.log"
        )
        return False

    if pipeline.gen_report:
        _gen_report_data(dataset, paths)

    print(f"[osl:preproc] wrote {paths.preproc_fif}")
    return True


def _gen_report_data(dataset: dict, paths: SimpleNamespace) -> None:
    """Generate this subject's preprocessing report data.

    Mirrors what :func:`osl_ephys.preprocessing.run_proc_chain` does internally
    when ``gen_report=True``, minus the group-level page build.
    """
    import matplotlib

    from osl_ephys.report import gen_html_data

    report_data_dir = paths.preproc_reportdir / paths.subject_label
    os.makedirs(report_data_dir, exist_ok=True)

    figures = dataset.get("fig") or None

    previous_backend = matplotlib.pyplot.get_backend()
    matplotlib.use("Agg")
    try:
        gen_html_data(
            dataset["raw"],
            str(report_data_dir),
            ica=dataset.get("ica"),
            events=dataset.get("events"),
            event_id=dataset.get("event_id"),
            preproc_fif_filename=str(paths.preproc_fif),
            logsdir=str(paths.logsdir),
            run_id=paths.subject_label,
            custom_figures=figures,
        )
    finally:
        matplotlib.use(previous_backend)

    print(f"[osl:preproc] report data in {report_data_dir}")
