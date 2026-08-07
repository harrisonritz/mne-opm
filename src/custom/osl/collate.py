"""Report collation stage of the osl-ephys OPM pipeline.

The ``preproc`` and ``source`` stages write only their own subject's report
*data* (a ``data.pkl`` per subject).  Rendering the shared HTML pages --
``subject_report.html`` and ``summary_report.html`` -- is deferred here so that
concurrent SLURM array tasks never write the same file.

Run this once after the array finishes, e.g. with
``sbatch --dependency=afterany:<array-job-id>``.

Functions
---------
run
    Rebuild the group-level preprocessing and source-recon reports.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ._paths import resolve_paths


def run(cfg: SimpleNamespace) -> bool:
    """Rebuild the group-level preprocessing and source-recon reports.

    Parameters
    ----------
    cfg : SimpleNamespace
        Config from :func:`custom.osl._config.load_config`.

    Returns
    -------
    success : bool
        True if at least one report was rebuilt.  A report directory that does
        not exist yet is skipped rather than treated as an error, so this can
        be run after a preproc-only array.

    Notes
    -----
    Report generation is best-effort per report: a failure in one is logged and
    does not prevent the others from being built, because a collate job that
    dies half-way leaves the reports in a more confusing state than one that
    reports partial success.
    """
    from osl_ephys.report import gen_html_page, gen_html_summary, src_report

    paths = resolve_paths(cfg.pipeline)

    reports = [
        ("preproc", paths.preproc_reportdir, gen_html_page, gen_html_summary),
        (
            "source",
            paths.src_reportdir,
            src_report.gen_html_page,
            src_report.gen_html_summary,
        ),
    ]

    any_built = False
    for name, reportdir, page_func, summary_func in reports:
        if not _has_subject_data(reportdir):
            print(f"[osl:collate] no {name} report data under {reportdir}; skipping")
            continue

        print(f"[osl:collate] building {name} report in {reportdir}")
        built = _build(name, reportdir, page_func, summary_func)
        any_built = any_built or built

    if not any_built:
        print("[osl:collate] nothing to collate")

    return any_built


def _has_subject_data(reportdir: Path) -> bool:
    """Is there at least one subject's ``data.pkl`` under this report directory?"""
    reportdir = Path(reportdir)
    if not reportdir.is_dir():
        return False
    return any(child.joinpath("data.pkl").exists() for child in reportdir.iterdir())


def _build(name: str, reportdir: Path, page_func, summary_func) -> bool:
    """Build one report's subject page and summary page."""
    built = False

    try:
        page_func(str(reportdir))
        print(f"[osl:collate] {name}: {reportdir}/subject_report.html")
        built = True
    except Exception as exc:
        print(f"[osl:collate] {name}: could not build the subject report: {exc}")

    try:
        summary_func(str(reportdir))
        print(f"[osl:collate] {name}: {reportdir}/summary_report.html")
        built = True
    except Exception as exc:
        print(f"[osl:collate] {name}: could not build the summary report: {exc}")

    return built
