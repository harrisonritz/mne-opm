#!/usr/bin/env python
"""trunk.py — run the first preprocessing steps once, then branch variants.

``run_preproc.sh`` splits into a shared *trunk* and per-variant *branches*
with ``PREPROC_STAGE`` (see ``src/run/README.md``):

trunk
    INIT, annotate_breaks, select_trial_response, bad_segments_1 and
    bad_channels write the ``proc-<custom_proc>`` derivatives under the trunk
    config's ``deriv_root``; ``save-state`` then records which of the
    environment flags exported by those steps are set.
branch
    ``seed`` clears the stale ``proc-<custom_proc>`` files in the variant
    config's own ``deriv_root`` (as INIT does), copies the trunk's
    ``proc-<custom_proc>`` files and participants metadata into it, and prints
    the recorded flags for ``run_preproc.sh`` to export before it runs
    manual_channel onwards.

The state file sits next to the files it describes::

    <deriv_root>/sub-<S>/ses-<X>/meg/sub-<S>_ses-<X>_task-<T>_proc-<P>_desc-trunk_state.json

    {"custom_proc": "init",
     "flags": {"MNE_OPM_BREAKS_ALREADY_ANNOTATED": true,
               "MNE_OPM_REQUIRE_RESPONSE_METADATA": true,
               "MNE_OPM_STAGE1_BREAKS_SAVED": false}}

Its name carries ``proc-<custom_proc>``, so INIT clears it with the trunk
files when the trunk is re-run.  Seeding does not copy it, so a variant is
never mistaken for a trunk.

Usage
-----
    python src/custom/trunk.py save-state --config=/path/to/trunk-config.py
    MNE_OPM_TRUNK_DERIV=/abs/path/to/trunk/deriv_root \\
        python src/custom/trunk.py seed --config=/path/to/variant-config.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import uuid
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

# Ensure ``src/`` is importable, as custom_preproc.py does.
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from mne_bids import BIDSPath  # noqa: E402

from custom.preprocessing import init_derivatives  # noqa: E402
from custom.preprocessing._config import load_config  # noqa: E402
from custom.preprocessing._io import get_custom_proc  # noqa: E402

# Environment variable naming the trunk's deriv_root for a branch run.
TRUNK_DERIV_ENV = "MNE_OPM_TRUNK_DERIV"

# Flags run_preproc.sh exports during the trunk steps (set means "1").
TRUNK_STATE_FLAGS = (
    "MNE_OPM_BREAKS_ALREADY_ANNOTATED",
    "MNE_OPM_REQUIRE_RESPONSE_METADATA",
    "MNE_OPM_STAGE1_BREAKS_SAVED",
)

# Dataset-level files written next to the first proc-<custom_proc> derivative
# (see _io._seed_sidecars); later steps only seed them from their own input.
DATASET_FILES = ("participants.tsv", "participants.json")


def trunk_state_path(
    deriv_root, subject: str, session: str | None, task: str, proc: str
) -> Path:
    """Return the trunk state file for one subject/session.

    Parameters
    ----------
    deriv_root : str | Path
        The trunk's derivatives root.
    subject, session, task : str
        BIDS entities (``session`` may be None).
    proc : str
        The ``custom_proc`` label.

    Returns
    -------
    Path
        ``.../meg/sub-<S>_ses-<X>_task-<T>_proc-<P>_desc-trunk_state.json``.
    """
    return BIDSPath(
        root=deriv_root,
        subject=subject,
        session=session,
        task=task,
        processing=proc,
        description="trunk",
        suffix="state",
        extension=".json",
        datatype="meg",
        check=False,
    ).fpath


def _replace_atomically(dst: Path, write: Callable[[Path], object]) -> None:
    """Write ``dst`` through a temporary file in its folder, then rename it."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}.tmp")
    try:
        write(tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def write_trunk_state(path, custom_proc: str, environ=None) -> dict:
    """Record which trunk flags are set (``== "1"``) in ``environ``.

    Parameters
    ----------
    path : str | Path
        Destination, written atomically.
    custom_proc : str
        The trunk's ``custom_proc`` label.
    environ : Mapping | None
        Environment to read (default ``os.environ``).

    Returns
    -------
    dict
        The state written: ``{"custom_proc": str, "flags": {name: bool}}``.
    """
    environ = os.environ if environ is None else environ
    state = {
        "custom_proc": custom_proc,
        "flags": {name: environ.get(name) == "1" for name in TRUNK_STATE_FLAGS},
    }
    text = json.dumps(state, indent=2) + "\n"
    _replace_atomically(Path(path), lambda tmp: tmp.write_text(text))
    return state


def read_trunk_state(path) -> dict:
    """Read and validate a trunk state file written by :func:`write_trunk_state`.

    Raises
    ------
    FileNotFoundError
        If the file does not exist (the trunk has not finished).
    ValueError
        If the file is not a valid trunk state.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trunk state file {path} is missing; run run_preproc.sh with "
            "PREPROC_STAGE=trunk for this subject first."
        )
    state = json.loads(path.read_text())
    flags = state.get("flags") if isinstance(state, dict) else None
    if (
        not isinstance(flags, dict)
        or not isinstance(state.get("custom_proc"), str)
        or set(flags) != set(TRUNK_STATE_FLAGS)
        or not all(isinstance(value, bool) for value in flags.values())
    ):
        raise ValueError(f"Malformed trunk state file {path}: {state!r}")
    return state


def save_trunk_state(cfg: SimpleNamespace) -> list[Path]:
    """Write the state file for every configured subject/session.

    Parameters
    ----------
    cfg : SimpleNamespace
        The trunk config (``deriv_root``, ``subjects``, ``sessions``,
        ``task``, ``custom_proc``).

    Returns
    -------
    list of Path
        The state files written.
    """
    proc = get_custom_proc(cfg)
    if proc is None:
        raise ValueError("cfg.custom_proc must be set to save a trunk state.")
    sessions = init_derivatives._as_list(getattr(cfg, "sessions", None)) or [None]
    paths = []
    for subject in init_derivatives._as_list(cfg.subjects):
        for session in sessions:
            path = trunk_state_path(cfg.deriv_root, subject, session, cfg.task, proc)
            state = write_trunk_state(path, proc)
            print(f"[trunk] wrote {path}: {state['flags']}")
            paths.append(path)
    return paths


def seed_from_trunk(cfg: SimpleNamespace, trunk_root) -> dict[str, bool]:
    """Seed ``cfg.deriv_root`` with a trunk's ``proc-<custom_proc>`` files.

    Everything is checked before anything is changed.  The variant's stale
    ``proc-<custom_proc>`` files are then removed (as INIT does) and the
    trunk's files are copied to the same relative paths, each atomically, so
    the trunk is only ever read.

    Parameters
    ----------
    cfg : SimpleNamespace
        The variant config (``deriv_root``, ``subjects``, ``sessions``,
        ``task``, ``custom_proc``; the trunk must use the same
        ``custom_proc``).
    trunk_root : str | Path
        Absolute path of the trunk's ``deriv_root``.

    Returns
    -------
    dict
        The trunk's flags, ``{name: bool}``.
    """
    proc = get_custom_proc(cfg)
    if proc is None:
        raise ValueError("cfg.custom_proc must be set to seed from a trunk.")
    if not trunk_root:
        raise ValueError(f"{TRUNK_DERIV_ENV} must be set to the trunk's deriv_root.")
    trunk_root = Path(trunk_root)
    deriv_root = Path(cfg.deriv_root)
    if not trunk_root.is_absolute():
        raise ValueError(f"{TRUNK_DERIV_ENV} must be absolute, got {trunk_root}.")
    if not trunk_root.is_dir():
        raise FileNotFoundError(f"Trunk deriv_root {trunk_root} does not exist.")
    if trunk_root.resolve() == deriv_root.resolve():
        raise ValueError(
            f"The variant's deriv_root is the trunk's ({trunk_root}); give the "
            "variant config its own deriv_root."
        )
    subjects = init_derivatives._as_list(cfg.subjects)
    sessions = init_derivatives._as_list(getattr(cfg, "sessions", None))
    if not subjects:
        raise ValueError("cfg.subjects is empty; nothing to seed.")

    flags = None
    copies = []
    for subject in subjects:
        state_names = set()
        for session in sessions or [None]:
            state_path = trunk_state_path(trunk_root, subject, session, cfg.task, proc)
            state = read_trunk_state(state_path)
            if state["custom_proc"] != proc:
                raise ValueError(
                    f"The trunk used custom_proc={state['custom_proc']!r} but this "
                    f"config uses {proc!r} ({state_path})."
                )
            if flags is not None and state["flags"] != flags:
                raise ValueError(f"Trunk state files disagree ({state_path}).")
            flags = state["flags"]
            state_names.add(state_path.name)
        files = [
            path
            for path in init_derivatives.custom_proc_files(
                trunk_root, subject, sessions, proc
            )
            if path.name not in state_names
        ]
        if not any(
            f"_task-{cfg.task}_" in path.name and path.name.endswith("_raw.fif")
            for path in files
        ):
            raise FileNotFoundError(
                f"The trunk {trunk_root} has no proc-{proc} raw FIF for "
                f"sub-{subject} task-{cfg.task}."
            )
        copies += [(path, deriv_root / path.relative_to(trunk_root)) for path in files]
    copies += [
        (trunk_root / name, deriv_root / name)
        for name in DATASET_FILES
        if (trunk_root / name).is_file()
    ]

    init_derivatives.run(cfg)
    for src, dst in copies:
        _replace_atomically(dst, partial(shutil.copy2, src))
        print(f"[trunk] copied {src} -> {dst}")
    print(f"[trunk] seeded {len(copies)} file(s) from {trunk_root}; flags: {flags}")
    return flags


def _load_cfg(config_path: str) -> SimpleNamespace:
    """Load a config without the behavioural metadata, which is not used here.

    The variant's ``proc-<custom_proc>`` FIF the metadata is aligned to does
    not exist until it has been seeded.
    """
    previous = os.environ.get("MNE_OPM_SKIP_METADATA")
    os.environ["MNE_OPM_SKIP_METADATA"] = "1"
    try:
        return load_config(config_path)
    finally:
        if previous is None:
            os.environ.pop("MNE_OPM_SKIP_METADATA", None)
        else:
            os.environ["MNE_OPM_SKIP_METADATA"] = previous


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the exit code.

    ``seed`` logs to stderr and prints only the names of the set flags on
    stdout, one per line, for ``run_preproc.sh`` to export.
    """
    parser = argparse.ArgumentParser(
        description="Save a preprocessing trunk's state or seed a branch from it",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("command", choices=["save-state", "seed"])
    parser.add_argument("--config", required=True, metavar="PATH")
    args = parser.parse_args(argv)

    try:
        if args.command == "save-state":
            save_trunk_state(_load_cfg(args.config))
            return 0
        with contextlib.redirect_stdout(sys.stderr):
            flags = seed_from_trunk(
                _load_cfg(args.config), os.environ.get(TRUNK_DERIV_ENV)
            )
    except (FileNotFoundError, ValueError) as e:
        print(f"[trunk] ERROR: {e}", file=sys.stderr)
        return 1
    for name, is_set in flags.items():
        if is_set:
            print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
