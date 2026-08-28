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

import functools
import os
from contextlib import contextmanager
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


# Plot helpers that :func:`osl_ephys.report.gen_html_data` calls as module
# globals of ``osl_ephys.report.preproc_report``, mapped to the value to fall
# back on when one raises.  gen_html_data unpacks ``plot_spectra`` into two
# names, so its fallback has to keep that arity; every other helper already
# returns None when it has nothing to plot, and the report templates handle
# that.
_REPORT_PLOTS: dict[str, object] = {
    "plot_flowchart": None,
    "plot_rawdata": None,
    "plot_channel_time_series": None,
    "plot_sensors": None,
    "plot_channel_dists": None,
    "plot_spectra": (None, None),
    "plot_freqbands": None,
    "plot_digitisation_2d": None,
    "plot_eog_summary": None,
    "plot_ecg_summary": None,
    "plot_events": None,
    "plot_bad_ica": None,
    "plot_custom_figures": None,
}


def _skip_on_failure(name: str, func, fallback):
    """Wrap one report plot helper so a failure drops just that figure."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- a figure is not worth a run
            import matplotlib.pyplot as plt

            # The failed call may have left a figure open part-way through.
            plt.close("all")
            detail = str(exc).splitlines()[0][:200] or type(exc).__name__
            print(f"[osl:preproc] report plot {name} failed, skipping: {detail}")
            return fallback

    return wrapper


@contextmanager
def _skip_failing_report_plots():
    """Make the osl-ephys report plots best-effort for the enclosed block.

    :func:`osl_ephys.report.gen_html_data` renders a dozen figures in one
    pass with no error handling of its own, so a single unsupported figure
    takes the whole subject's report -- and, in this pipeline, an otherwise
    successful preprocessing run -- with it.  Several of them are unsupported
    for triaxial OPM data: MNE refuses to build a topography when channels
    share a sensor position, which is exactly how an OPM triplet is laid out
    (``plot_freqbands`` raises "electrodes have overlapping positions").

    Each helper is replaced for the duration of the block by a version that
    reports the failure and returns the value gen_html_data expects for "no
    figure", then restored.
    """
    from osl_ephys.report import preproc_report

    originals = {}
    try:
        for name, fallback in _REPORT_PLOTS.items():
            original = getattr(preproc_report, name, None)
            if original is None:
                # osl-ephys renamed or dropped this helper; nothing to guard.
                continue
            originals[name] = original
            setattr(preproc_report, name, _skip_on_failure(name, original, fallback))
        yield
    finally:
        for name, original in originals.items():
            setattr(preproc_report, name, original)


def _gen_report_data(dataset: dict, paths: SimpleNamespace) -> None:
    """Generate this subject's preprocessing report data.

    Mirrors what :func:`osl_ephys.preprocessing.run_proc_chain` does internally
    when ``gen_report=True``, minus the group-level page build.

    Notes
    -----
    :func:`osl_ephys.report.gen_html_data` is picky about the types of its two
    directory arguments:

    * ``outdir`` is indexed with ``/`` (``outdir / '{0}.png'``), so it has to
      be a :class:`~pathlib.Path` -- a string raises ``TypeError``.
    * ``logsdir`` is treated as the log *base*: the function appends
      ``'.log'`` / ``'.error.log'`` to it.  Only when it is a ``Path`` does the
      function build that base itself, and it builds it from the report
      directory name (``{logsdir}/{subject_label}``), which misses the
      ``_preproc`` suffix osl-ephys actually writes its logs with.  Passing the
      base as a string bypasses that and picks the log files up.
    """
    import matplotlib

    from osl_ephys.report import gen_html_data

    report_data_dir = paths.preproc_reportdir / paths.subject_label
    os.makedirs(report_data_dir, exist_ok=True)

    # osl-ephys writes {logsdir}/{subject_label}_preproc{.log,.error.log}
    # (batch.run_proc_chain formats its log name with ftype minus "-raw").
    logs_base = str(paths.logsdir / f"{paths.subject_label}_preproc")

    figures = dataset.get("fig") or None

    previous_backend = matplotlib.pyplot.get_backend()
    matplotlib.use("Agg")
    try:
        with _skip_failing_report_plots():
            gen_html_data(
                dataset["raw"],
                report_data_dir,
                ica=dataset.get("ica"),
                events=dataset.get("events"),
                event_id=dataset.get("event_id"),
                preproc_fif_filename=str(paths.preproc_fif),
                logsdir=logs_base,
                run_id=paths.subject_label,
                custom_figures=figures,
            )
    finally:
        matplotlib.use(previous_backend)

    print(f"[osl:preproc] report data in {report_data_dir}")
