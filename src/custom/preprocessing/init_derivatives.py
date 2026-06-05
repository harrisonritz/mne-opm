"""Clear leftover custom-proc derivatives before a fresh preprocessing run.

When ``cfg.custom_proc`` is set (e.g. ``"init"``), the custom preprocessing
steps read their inputs from ``deriv_root`` using the ``proc-<custom_proc>``
label, falling back to the raw BIDS recordings only when no such derivative
exists yet (see :func:`custom.preprocessing._io.find_custom_input_paths`).

That fallback is what chains the custom steps together *within* a single
pipeline run: the first step reads the raw BIDS data, every later step reads
the ``proc-<custom_proc>`` file the previous step wrote.  But it also means
that **stale** ``proc-<custom_proc>`` files left over from a previous run are
silently picked up as the input to the first custom step of the next run.
Because each step writes back to the same label, re-running the pipeline then
processes already-processed data (e.g. regressing twice), producing
different — and wrong — results.

This module removes the ``proc-<custom_proc>`` files for the configured
subject(s)/session(s) from ``deriv_root`` so that the next run starts from the
canonical raw BIDS recordings.  It is intentionally surgical: only files
carrying the ``proc-<custom_proc>`` entity are deleted.  Expensive or manual
derivatives that live alongside them in the same ``meg`` folder — most notably
the coregistration ``*_trans.fif`` — are left untouched.

Run via the CLI dispatcher::

    python src/custom/custom_preproc.py --analysis=init --config=config.py

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ._io import get_custom_proc


def _as_list(value) -> list:
    """Normalize a scalar / list / None config value to a list.

    Parameters
    ----------
    value : Any
        ``cfg.subjects`` or ``cfg.sessions`` — may be a scalar, a list/tuple,
        or ``None``.

    Returns
    -------
    list
        The values as a list (empty if ``value`` is ``None``).
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _meg_dirs_for_subject(
    deriv_root: Path, subject: str, sessions: list
) -> list[Path]:
    """Collect the ``meg`` derivative folders for one subject.

    Parameters
    ----------
    deriv_root : Path
        Root of the analysis derivatives tree.
    subject : str
        Subject label without the ``sub-`` prefix (e.g. ``"011"``).
    sessions : list
        Session labels without the ``ses-`` prefix.  When empty, every
        session folder (and a session-less ``meg`` folder, if present) is
        matched.

    Returns
    -------
    list of Path
        Unique, existing ``meg`` directories for the subject.
    """
    sub_dir = deriv_root / f"sub-{subject}"
    if not sub_dir.exists():
        return []

    meg_dirs: set[Path] = set()
    if sessions:
        for ses in sessions:
            meg_dirs.update(sub_dir.glob(f"ses-{ses}/meg"))
    else:
        meg_dirs.update(sub_dir.glob("ses-*/meg"))
        meg_dirs.update(sub_dir.glob("meg"))

    return sorted(d for d in meg_dirs if d.is_dir())


def run(cfg: SimpleNamespace) -> None:
    """Remove stale ``proc-<custom_proc>`` derivatives before preprocessing.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration object.  Uses ``cfg.custom_proc``,
        ``cfg.deriv_root``, ``cfg.subjects`` and ``cfg.sessions``.

    Notes
    -----
    Safe to run when nothing needs clearing: missing ``custom_proc``, a
    not-yet-created ``deriv_root``, or absent subject folders all result in a
    no-op with an informative message.
    """
    proc = get_custom_proc(cfg)
    if proc is None:
        print(
            "[init_derivatives] custom_proc is not set; no custom derivatives "
            "to clear."
        )
        return

    deriv_root = getattr(cfg, "deriv_root", None)
    if deriv_root is None or not Path(deriv_root).exists():
        print(
            f"[init_derivatives] deriv_root does not exist yet ({deriv_root}); "
            "nothing to clear."
        )
        return
    deriv_root = Path(deriv_root)

    subjects = _as_list(getattr(cfg, "subjects", None))
    sessions = _as_list(getattr(cfg, "sessions", None))
    if not subjects:
        print("[init_derivatives] cfg.subjects is empty; nothing to clear.")
        return

    n_removed = 0
    for subject in subjects:
        for meg_dir in _meg_dirs_for_subject(deriv_root, subject, sessions):
            # Restrict to this subject's files so a shared folder (should not
            # happen, but be safe) cannot delete another subject's data.
            sub_pattern = f"sub-{subject}_*_proc-{proc}_*"
            for path in sorted(meg_dir.glob(sub_pattern)):
                if path.is_file():
                    print(f"[init_derivatives] removing {path}")
                    path.unlink()
                    n_removed += 1

    print(
        f"[init_derivatives] removed {n_removed} stale proc-{proc} file(s) for "
        f"subject(s) {subjects}, session(s) {sessions or 'any'}."
    )
