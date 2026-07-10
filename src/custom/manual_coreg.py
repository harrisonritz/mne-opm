"""Manual MRI-MEG coregistration loop for OPM-MEG data.

Loops over subjects and launches ``mne.gui.coregistration`` so the user
can place fiducials, fit ICP, and save a trans file by hand.

Trans files should be saved to::

    {FS_DIR}/{fs_subject}/bem/{fs_subject}-trans.fif

which is the default path checked by
:class:`custom.preprocessing.coreg.CoregAnalysis` when
``_use_precomputed_trans=True``. The save path is printed before each
GUI launch so it can be copy-pasted in the GUI's save dialog.

Usage
-----
Run every subject under the BIDS directory::

    python src/custom/manual_coreg.py

Run a specific list (any of ``019``, ``19``, ``sub-019`` are accepted)::

    python src/custom/manual_coreg.py --subjects 019 020 021

Override the session (defaults to ``01``)::

    python src/custom/manual_coreg.py --subjects 019 --session 02
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import mne
import mne_bids


# ---------------------------------------------------------------------------
# Paths / constants — edit these to match your machine
# ---------------------------------------------------------------------------

ROOT_DIR = "/Volumes/fileset-NDAW/harrison_ritz/TSX/data/TSX"
FS_DIR = f"{ROOT_DIR}/freesurfer"
BIDS_DIR = f"{ROOT_DIR}/bids"
TASK = "TSX"
SESSION = "01"

assert os.path.isdir(BIDS_DIR), f"BIDS_DIR does not exist: {BIDS_DIR}"
assert os.path.isdir(FS_DIR), f"FS_DIR does not exist: {FS_DIR}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _list_all_subjects(bids_dir: str) -> List[str]:
    """Return zero-padded subject IDs (e.g. ``'019'``) sorted numerically."""
    if not os.path.isdir(bids_dir):
        raise FileNotFoundError(f"BIDS directory does not exist: {bids_dir}")
    subs = [
        d.replace("sub-", "")
        for d in os.listdir(bids_dir)
        if d.startswith("sub-") and os.path.isdir(os.path.join(bids_dir, d))
    ]
    return sorted(subs, key=lambda s: int(s))


def _normalize_subject(sub: str) -> str:
    """Accept ``19``, ``019``, or ``sub-019`` → ``'019'``."""
    sub = sub.replace("sub-", "").lstrip("0") or "0"
    return sub.zfill(3)


def _fs_subject(subject: str, session: str) -> str:
    return f"sub-{subject}_ses-{session}"


def _trans_path(subject: str, session: str) -> Path:
    fs_sub = _fs_subject(subject, session)
    return Path(FS_DIR) / fs_sub / "bem" / f"{fs_sub}-trans.fif"


def _find_raw(subject: str, session: str) -> Optional[Path]:
    paths = mne_bids.find_matching_paths(
        root=BIDS_DIR,
        subjects=subject,
        sessions=session,
        tasks=TASK,
        datatypes="meg",
        extensions=".fif",
        ignore_nosub=True,
    )
    if not paths:
        return None
    return Path(paths[0].fpath)


def _mtime(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Per-subject driver
# ---------------------------------------------------------------------------


def _print_subject_status(subject: str, session: str) -> Dict[str, object]:
    """Print and return pre-coreg status info for one subject."""
    fs_sub = _fs_subject(subject, session)
    fs_root = Path(FS_DIR) / fs_sub
    bem_dir = fs_root / "bem"

    raw_path = _find_raw(subject, session)
    trans_path = _trans_path(subject, session)
    t1_path = fs_root / "mri" / "T1.mgz"
    head_dense = bem_dir / f"{fs_sub}-head-dense.fif"
    head = bem_dir / f"{fs_sub}-head.fif"
    outer_skin = bem_dir / "outer_skin.surf"
    fiducials = sorted(bem_dir.glob("*fiducials.fif")) if bem_dir.is_dir() else []

    print(f"  fs_subject     : {fs_sub}")
    print(f"  FS dir         : {fs_root}  [{'OK' if fs_root.is_dir() else 'MISSING'}]")
    print(f"  raw MEG        : {raw_path if raw_path else 'NOT FOUND'}")
    print(f"  T1.mgz         : {t1_path}  [{'OK' if t1_path.exists() else 'MISSING'}]")

    if head_dense.exists():
        head_surf_msg = f"{head_dense.name} (dense)"
    elif head.exists():
        head_surf_msg = f"{head.name} (low-res)"
    elif outer_skin.exists():
        head_surf_msg = "outer_skin.surf (FreeSurfer)"
    else:
        head_surf_msg = "NONE FOUND — GUI may fail to render scalp"
    print(f"  head surface   : {head_surf_msg}")

    print(
        f"  fiducials      : "
        f"{fiducials[0].name if fiducials else 'none (will be placed in GUI)'}"
    )

    if trans_path.exists():
        print(f"  existing trans : {trans_path} (modified {_mtime(trans_path)})")
    else:
        print(f"  existing trans : none (target: {trans_path})")

    # Quick read-only summary of the raw file if it exists
    if raw_path is not None:
        try:
            info = mne.io.read_info(raw_path, verbose="ERROR")
            n_meg = len(mne.pick_types(info, meg=True, eeg=False, ref_meg=False))
            n_dig = len(info.get("dig") or [])
            n_hsp = sum(
                1
                for d in (info.get("dig") or [])
                if d["kind"] == mne.io.constants.FIFF.FIFFV_POINT_EXTRA
            )
            print(f"  meg channels   : {n_meg}")
            print(f"  dig points     : {n_dig} total ({n_hsp} head-shape)")
        except Exception as e:
            print(f"  raw info       : could not read ({e})")

    return {
        "fs_subject": fs_sub,
        "raw_path": raw_path,
        "trans_path": trans_path,
        "fs_dir_ok": fs_root.is_dir(),
        "t1_ok": t1_path.exists(),
        "trans_before": trans_path.exists(),
        "trans_mtime_before": _mtime(trans_path) if trans_path.exists() else None,
    }


def run_subject(subject: str, session: str = SESSION) -> str:
    """Launch the GUI for one subject. Returns a short status string."""
    print("\n" + "=" * 72)
    print(f"SUBJECT sub-{subject}  (session ses-{session})")
    print("=" * 72)
    info = _print_subject_status(subject, session)

    if not info["fs_dir_ok"]:
        print("→ Skipping: FreeSurfer subject directory does not exist.")
        return "no_fs"
    if info["raw_path"] is None:
        print("→ Skipping: no raw MEG .fif found in BIDS.")
        return "no_raw"
    if not info["t1_ok"]:
        print("→ Skipping: T1.mgz missing — GUI requires it.")
        return "no_t1"

    print(
        f"\n------------------------------------------------\n"
        f"→ Save the trans file to:\n  {info['trans_path']}\n"
        f"  (this is the default checked by CoregAnalysis when "
        f"_use_precomputed_trans=True)"
        f"\n------------------------------------------------\n"
    )
    print("Launching mne.gui.coregistration ... close the window to continue.\n")

    try:
        trans_path = _trans_path(subject, session)
        if trans_path.exists():
            mne.gui.coregistration(
                inst=str(info["raw_path"]),
                subject=info["fs_subject"],
                subjects_dir=FS_DIR,
                block=True,
                trans=trans_path,
            )
        else:
            mne.gui.coregistration(
                inst=str(info["raw_path"]),
                subject=info["fs_subject"],
                subjects_dir=FS_DIR,
                block=True,
            )

    except Exception as e:
        print(f"→ GUI failed: {e}")
        return "gui_error"

    if not info["trans_path"].exists():
        print("→ No trans file at the expected path after GUI exit.")
        return "not_saved"

    new_mtime = _mtime(info["trans_path"])
    if info["trans_before"] and new_mtime == info["trans_mtime_before"]:
        print(f"→ Existing trans left unchanged ({new_mtime}).")
        return "unchanged"
    print(f"→ Trans saved: {info['trans_path']} ({new_mtime})")
    return "saved"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--subjects",
        nargs="+",
        default=["all"],
        help="Subject IDs to run, or 'all' (default). "
        "Accepts '019', '19', or 'sub-019'.",
    )
    p.add_argument(
        "--session",
        default=SESSION,
        help=f"Session id without the 'ses-' prefix (default: {SESSION}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.subjects) == 1 and args.subjects[0].lower() == "all":
        subjects = _list_all_subjects(BIDS_DIR)
        print(f"\nFound {len(subjects)} subjects in {BIDS_DIR}")
    else:
        subjects = [_normalize_subject(s) for s in args.subjects]
        print(f"\nRunning {len(subjects)} subject(s): {subjects}")

    print(f"BIDS dir : {BIDS_DIR}")
    print(f"FS dir   : {FS_DIR}")
    print(f"Task     : {TASK}")
    print(f"Session  : ses-{args.session}")

    summary: Dict[str, List[str]] = {}
    try:
        for sub in subjects:
            status = run_subject(sub, session=args.session)
            summary.setdefault(status, []).append(sub)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user — showing partial summary.")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if not summary:
        print("  (no subjects processed)")
    else:
        for status in sorted(summary):
            subs = summary[status]
            print(f"  {status:11s}: {len(subs):3d}  {subs}")
    print()


if __name__ == "__main__":
    main()
