#!/usr/bin/env python
"""rank_check.py — track the data rank of task vs empty-room derivatives.

Reports the numerical rank of the task recording and the empty-room ("noise")
recording at one or more preprocessing stages, so you can watch how — and where —
their ranks diverge as spatial filtering / cleaning progresses.

The stage is selected by its ``proc-<tag>`` entity (the same tag mne-bids-pipeline
and the custom steps write into the derivative file name).  For each requested tag
the CLI finds the task raw and the noise raw, preloads them, and computes:

  - ``rank(data)`` : ``mne.compute_rank(raw, tol="auto")`` — the SVD data rank.
                     This is the rank the beamformer data / noise covariance uses
                     (see run_beamformer.py), i.e. where task and empty-room ranks
                     actually diverge.
  - ``rank(info)`` : ``mne.compute_rank(raw, rank="info")`` — the declared rank
                     from ``raw.info`` (SSS / HFC / projection bookkeeping).  Fast,
                     header-derived; will not reveal numerical rank collapse caused
                     by bad channels.

Both are reported side-by-side, along with the channel and bad-channel counts, and
appended to a per-subject TSV so you get a rank-vs-step table across the whole run.

Configuration (``deriv_root`` / ``task`` / ``subject`` / ``session``) is loaded from
the pipeline config via ``--config``, exactly as the custom preprocessing steps do,
so paths are always resolved the same way the pipeline resolves them.

Usage
-----
    python rank_check.py --config=/path/to/config.py proc-filt
    python rank_check.py --config=/path/to/config.py filt            # 'proc-' optional
    python rank_check.py --config=/path/to/config.py filt sss        # several stages
    python rank_check.py --config=/path/to/config.py clean --all-subjects
    python rank_check.py --config=/path/to/config.py init --no-tsv

Intended to be dropped between steps in run_preproc.sh, e.g.::

    python "$ROOT_DIR/src/custom/rank_check.py" --config="$CONFIG_PATH" filt sss

It is a diagnostic: it never raises on missing files (a stage that has not run yet is
reported as "—" and skipped), so it is safe to call unconditionally between steps.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure ``src/`` is importable so ``custom.preprocessing._config`` resolves the
# same way it does for custom_preproc.py (whether run as a script or a module).
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import mne  # noqa: E402

from custom.preprocessing._config import load_config  # noqa: E402

mne.set_log_level("ERROR")  # keep routine MNE chatter out of the per-step logs

# Label used for the empty-room recording's BIDS ``task`` entity.
NOISE_TASK = "noise"

# Columns written to the per-subject rank-check TSV (append-only).
TSV_FIELDS = [
    "timestamp",
    "model",
    "subject",
    "session",
    "proc",
    "kind",
    "run",
    "n_ch",
    "n_bad",
    "rank_data",
    "rank_info",
    "file",
]


# ============================================================================
# PURE HELPERS (no MNE / filesystem state — unit tested)
# ============================================================================


def normalize_proc_tag(tag: str) -> str:
    """Strip a leading ``proc-`` (and surrounding whitespace) from a stage tag.

    Accepts both the bare tag (``filt``) and the full entity (``proc-filt``) so
    the CLI is forgiving about how the stage is passed on the command line.

    Examples
    --------
    >>> normalize_proc_tag("proc-filt")
    'filt'
    >>> normalize_proc_tag("sss")
    'sss'
    """
    tag = tag.strip()
    if tag.startswith("proc-"):
        tag = tag[len("proc-") :]
    return tag


def is_primary_split(name: str) -> bool:
    """Return True unless ``name`` is a secondary FIF split.

    MNE writes large recordings as a chain of split files; reading the first
    split transparently loads the rest.  Both split conventions are handled:

    - BIDS style : ``..._split-01_raw.fif`` (primary), ``..._split-02_raw.fif`` …
    - MNE style  : ``..._raw.fif`` (primary), ``..._raw-1.fif`` …

    Only the primary is a valid entry point; secondaries are skipped so each
    recording is opened (and ranked) exactly once.

    Examples
    --------
    >>> is_primary_split("sub-007_ses-01_task-TSX_run-01_proc-filt_split-01_raw.fif")
    True
    >>> is_primary_split("sub-007_ses-01_task-TSX_run-01_proc-filt_split-02_raw.fif")
    False
    >>> is_primary_split("sub-007_ses-01_task-noise_proc-filt_raw.fif")
    True
    """
    m = re.search(r"_split-(\d+)_", name)
    if m is not None and int(m.group(1)) != 1:
        return False
    # MNE-style continuation files (``_raw-1.fif``) are excluded by the glob
    # pattern (``*_raw.fif``); nothing more to do here.
    return True


def extract_run(name: str) -> Optional[str]:
    """Return the run entity label (e.g. ``run-01``) from a file name, or None.

    Examples
    --------
    >>> extract_run("sub-007_ses-01_task-TSX_run-01_proc-filt_raw.fif")
    'run-01'
    >>> extract_run("sub-007_ses-01_task-noise_proc-filt_raw.fif") is None
    True
    """
    m = re.search(r"_run-([A-Za-z0-9]+)_", name)
    return f"run-{m.group(1)}" if m is not None else None


def meg_dir_for(deriv_root: str, subject: str, session: str) -> Path:
    """Build the BIDS ``meg`` directory path for a subject/session.

    ``subject``/``session`` may be passed with or without their BIDS prefixes.
    """
    subject = subject.replace("sub-", "")
    session = session.replace("ses-", "")
    return Path(deriv_root) / f"sub-{subject}" / f"ses-{session}" / "meg"


def raw_glob_pattern(subject: str, session: str, task: str, proc: str) -> str:
    """Glob pattern matching the primary-and-secondary raw FIFs for a stage.

    The trailing ``_*raw.fif`` (``*`` may match nothing) captures both the plain
    ``proc-<tag>_raw.fif`` and the split ``proc-<tag>_split-NN_raw.fif`` names,
    while the underscore right after ``proc-<tag>`` keeps the tag exact so that,
    e.g., ``proc-ica`` never matches ``proc-icafit``.  MNE continuation files
    (``_raw-1.fif``) do not end in ``raw.fif`` and are therefore excluded.
    """
    subject = subject.replace("sub-", "")
    session = session.replace("ses-", "")
    return (
        f"sub-{subject}_ses-{session}_task-{task}_*proc-{proc}_*raw.fif"
    )


def find_primary_raws(
    meg_dir: Path, subject: str, session: str, task: str, proc: str
) -> list[Path]:
    """Return the primary raw FIF(s) for one task+stage, one per run.

    Secondary splits are filtered out via :func:`is_primary_split`.  The result
    is sorted for deterministic ordering (and stable multi-run reporting).
    """
    if not meg_dir.is_dir():
        return []
    matches = meg_dir.glob(raw_glob_pattern(subject, session, task, proc))
    primary = [p for p in matches if is_primary_split(p.name)]
    return sorted(primary)


# ============================================================================
# RANK COMPUTATION (needs MNE + data)
# ============================================================================


def rank_of_raw(path: Path) -> dict:
    """Preload a raw FIF and return its channel counts and data/info ranks.

    Returns a dict with ``n_ch`` (good data channels), ``n_bad`` (bad channels),
    ``rank_data`` (SVD rank via ``tol="auto"``) and ``rank_info`` (declared rank
    via ``rank="info"``).  Rank fields are ``nan`` if that computation fails; the
    function itself does not raise, so one unreadable file never aborts a sweep.
    """
    result = {
        "n_ch": float("nan"),
        "n_bad": float("nan"),
        "rank_data": float("nan"),
        "rank_info": float("nan"),
    }
    try:
        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    except Exception as e:  # noqa: BLE001
        print(f"    WARNING: could not read {path.name}: {e}")
        return result

    try:
        good = mne.pick_types(raw.info, meg=True, ref_meg=False, exclude="bads")
        result["n_ch"] = len(good)
        result["n_bad"] = len(raw.info["bads"])

        # SVD data rank — matches the beamformer's covariance rank (tol="auto").
        try:
            rd = mne.compute_rank(raw, tol="auto", verbose="ERROR")
            result["rank_data"] = int(sum(rd.values()))
        except Exception as e:  # noqa: BLE001
            print(f"    WARNING: data-rank failed for {path.name}: {e}")

        # Declared rank from info (SSS / HFC / projection bookkeeping).
        try:
            ri = mne.compute_rank(raw, rank="info", verbose="ERROR")
            result["rank_info"] = int(sum(ri.values()))
        except Exception as e:  # noqa: BLE001
            print(f"    WARNING: info-rank failed for {path.name}: {e}")
    finally:
        del raw

    return result


# ============================================================================
# COLLECTION
# ============================================================================


def collect_rows(cfg, subject: str, session: str, proc: str) -> list[dict]:
    """Build one result row per (task-run, noise) raw found for a stage.

    An empty list means neither the task nor the noise raw exists for this
    stage yet (the step has not run) — the caller reports it as skipped.
    """
    meg_dir = meg_dir_for(cfg.deriv_root, subject, session)
    subj = subject.replace("sub-", "")
    rows = []

    # Task recording(s) — one primary raw per run.
    for path in find_primary_raws(meg_dir, subj, session, cfg.task, proc):
        run = extract_run(path.name)
        kind = f"task {run}" if run else "task"
        rows.append({"kind": kind, "run": run or "", "path": path, **rank_of_raw(path)})

    # Empty-room recording (never epoched, no run entity).
    for path in find_primary_raws(meg_dir, subj, session, NOISE_TASK, proc):
        rows.append({"kind": "noise", "run": "", "path": path, **rank_of_raw(path)})

    return rows


# ============================================================================
# OUTPUT — CONSOLE + TSV
# ============================================================================


def _fmt(v) -> str:
    """Format an int/nan cell for the console table."""
    if isinstance(v, float) and math.isnan(v):
        return "—"
    return str(int(v)) if isinstance(v, (int, float)) else str(v)


def print_block(proc: str, model: str, subject: str, rows: list[dict]) -> None:
    """Print the per-subject, per-stage rank table (and task−noise deltas)."""
    print(f"\n[rank_check]  proc-{proc}  |  sub-{subject.replace('sub-', '')}  |  {model}")
    if not rows:
        print("    (no task/noise derivatives for this stage yet — skipped)")
        return

    print(f"    {'kind':<14}{'n_ch':>6}{'n_bad':>7}{'rank(data)':>12}{'rank(info)':>12}")
    for r in rows:
        print(
            f"    {r['kind']:<14}{_fmt(r['n_ch']):>6}{_fmt(r['n_bad']):>7}"
            f"{_fmt(r['rank_data']):>12}{_fmt(r['rank_info']):>12}"
        )

    # task − noise data-rank delta (first task run vs the noise recording), which
    # is the headline number when task and empty-room ranks disagree.
    task_rows = [r for r in rows if r["kind"].startswith("task")]
    noise_rows = [r for r in rows if r["kind"] == "noise"]
    if task_rows and noise_rows:
        t, n = task_rows[0]["rank_data"], noise_rows[0]["rank_data"]
        if not (isinstance(t, float) and math.isnan(t)) and not (
            isinstance(n, float) and math.isnan(n)
        ):
            print(f"    Δ task−noise rank(data): {int(t) - int(n):+d}")


def append_tsv(cfg, subject: str, session: str, proc: str, rows: list[dict]) -> Path:
    """Append the collected rows to the subject's rank-check TSV (create+header).

    Writing per-subject keeps concurrent SLURM-array jobs from contending for one
    file while still giving a full rank-over-steps history for each subject.
    """
    meg_dir = meg_dir_for(cfg.deriv_root, subject, session)
    meg_dir.mkdir(parents=True, exist_ok=True)
    subj = subject.replace("sub-", "")
    sess = session.replace("ses-", "")
    tsv_path = meg_dir / f"sub-{subj}_ses-{sess}_rank-check.tsv"
    model = Path(cfg.deriv_root).name
    ts = datetime.now().isoformat(timespec="seconds")

    write_header = not tsv_path.exists()
    with tsv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TSV_FIELDS, delimiter="\t")
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "timestamp": ts,
                    "model": model,
                    "subject": subj,
                    "session": sess,
                    "proc": proc,
                    "kind": r["kind"],
                    "run": r["run"],
                    "n_ch": _fmt(r["n_ch"]),
                    "n_bad": _fmt(r["n_bad"]),
                    "rank_data": _fmt(r["rank_data"]),
                    "rank_info": _fmt(r["rank_info"]),
                    "file": r["path"].name,
                }
            )
    return tsv_path


# ============================================================================
# CLI
# ============================================================================


def _discover_subjects(deriv_root: str) -> list[str]:
    """Return all ``sub-*`` subject IDs under a model's derivatives root."""
    root = Path(deriv_root)
    if not root.is_dir():
        return []
    return sorted(
        p.name.replace("sub-", "") for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-")
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to the pipeline config .py")
    p.add_argument(
        "proc",
        nargs="+",
        help="One or more stage tags, e.g. 'proc-filt' or 'filt sss' "
        "('proc-' optional). Stages with no derivatives yet are skipped.",
    )
    p.add_argument(
        "--all-subjects",
        action="store_true",
        help="Scan every sub-* directory under deriv_root instead of only the "
        "subject(s) named in the config.",
    )
    p.add_argument(
        "--no-tsv",
        action="store_true",
        help="Print to stdout only; do not append to the per-subject rank-check TSV.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)

    session = cfg.sessions[0].replace("ses-", "")
    if args.all_subjects:
        subjects = _discover_subjects(cfg.deriv_root)
        if not subjects:
            print(f"[rank_check] no sub-* directories under {cfg.deriv_root}")
            return 0
    else:
        subjects = [s.replace("sub-", "") for s in cfg.subjects]

    model = Path(cfg.deriv_root).name
    tags = [normalize_proc_tag(t) for t in args.proc]

    for proc in tags:
        for subject in subjects:
            rows = collect_rows(cfg, subject, session, proc)
            print_block(proc, model, subject, rows)
            if rows and not args.no_tsv:
                tsv_path = append_tsv(cfg, subject, session, proc, rows)
                print(f"    appended → {tsv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
