"""Configuration loading for the osl-ephys OPM pipeline.

The pipeline is driven by a single YAML file per analysis, holding three
top-level sections:

.. code-block:: yaml

    pipeline:            # mne-opm settings: paths, subject, source backend
      subject: "${SUBJECT}"
      bids_root: "${BIDS_DIR}"
      outdir: "${BIDS_DIR}/derivatives/osl-trial"

    meta:                # osl-ephys metadata (event codes)
      event_codes:
        trial/read_read: 9

    preproc:             # osl-ephys preprocessing chain
      - filter: {l_freq: 0.5, h_freq: 32}

    source_recon:        # osl-ephys source reconstruction chain
      - compute_surfaces: {include_nose: false}

``meta``/``preproc`` and ``source_recon`` are handed to osl-ephys verbatim (see
:func:`preproc_config` and :func:`source_config`); osl-ephys never sees the
``pipeline`` section.

Because the two source backends need different step lists, ``source_recon``
may instead be a mapping keyed by backend, so that one file describes both:

.. code-block:: yaml

    source_recon:
      rhino:
        - compute_surfaces: {include_nose: false}
        - coregister: {use_nose: false}
      freesurfer:
        - fs_coregister: {}
        - fs_forward_model: {gridstep: 5}

:func:`source_config` then selects the list matching
``pipeline.source_backend``.

String values anywhere in the file may reference environment variables as
``${VAR}`` or ``${VAR:-default}``.  Strings in the ``pipeline`` section may
additionally use ``{subject}``, ``{session}``, ``{task}`` and ``{analysis}``
placeholders, which are substituted after environment expansion.

Functions
---------
expand_env
    Recursively expand ``${VAR}`` references in a nested structure.
load_config
    Load, expand and validate a pipeline YAML file.
preproc_config
    Extract the osl-ephys preprocessing config from a loaded pipeline config.
source_config
    Extract the osl-ephys source-recon config from a loaded pipeline config.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ${VAR} or ${VAR:-default}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

TOP_LEVEL_SECTIONS: tuple[str, ...] = ("pipeline", "meta", "preproc", "source_recon")
"""Sections recognised at the top level of a pipeline YAML file."""

SOURCE_BACKENDS: tuple[str, ...] = ("rhino", "freesurfer")
"""Valid values for ``pipeline.source_backend``."""

# Defaults for every recognised key of the ``pipeline`` section.  A value of
# ``None`` means "no default"; whether that is an error depends on the stage
# being run, so it is enforced in _paths / the stage runners rather than here.
PIPELINE_DEFAULTS: dict[str, Any] = {
    # --- identity ---
    "analysis": None,  # analysis name, used only for {analysis} substitution
    "subject": None,  # BIDS subject label without the "sub-" prefix, e.g. "007"
    "session": "01",
    "task": None,  # BIDS task entity
    "run": "01",  # BIDS run entity; None for no run entity
    # --- input ---
    "bids_root": None,
    "smri": None,  # T1w path; auto-detected under bids_root/.../anat when None
    # --- output ---
    "outdir": None,  # osl-ephys output directory (one sub-directory per subject)
    "logsdir": None,  # defaults to {outdir}/logs
    "preproc_reportdir": None,  # defaults to {outdir}/preproc_report
    "src_reportdir": None,  # defaults to {outdir}/src_report
    # osl-ephys names every output file after this label, and for the
    # freesurfer backend it must also be the FreeSurfer subject directory name.
    "subject_label": "sub-{subject}_ses-{session}",
    # --- source reconstruction ---
    "source_backend": "rhino",
    "freesurfer_subjects_dir": None,  # required by the freesurfer backend
    "trans": None,  # -trans.fif; defaults to the FreeSurfer bem/ convention
    # Whether the source stage reconstructs epochs (``{subject}_epo.fif``) or
    # continuous data (``{subject}_preproc-raw.fif``).
    "source_input": "epochs",
    # --- behaviour ---
    "overwrite": True,
    "gen_report": True,
    "random_seed": None,
}

_SOURCE_INPUTS: tuple[str, ...] = ("epochs", "raw")


# ---------------------------------------------------------------------------
# Environment expansion
# ---------------------------------------------------------------------------


def expand_env(value: Any, env: Optional[Mapping[str, str]] = None) -> Any:
    """Recursively expand ``${VAR}`` references in a nested structure.

    Parameters
    ----------
    value : Any
        A string, mapping, sequence or scalar.  Mappings and sequences are
        walked recursively; strings are expanded; everything else is returned
        unchanged.
    env : Mapping, optional
        Environment to read from.  Defaults to ``os.environ``.

    Returns
    -------
    expanded : Any
        The input with all ``${VAR}`` references replaced.

    Raises
    ------
    KeyError
        If a referenced variable is unset and no ``:-default`` was given.

    Examples
    --------
    >>> expand_env("${HOME}/data", {"HOME": "/root"})
    '/root/data'
    >>> expand_env("${MISSING:-fallback}", {})
    'fallback'
    """
    if env is None:
        env = os.environ

    if isinstance(value, str):
        return _expand_env_str(value, env)
    if isinstance(value, Mapping):
        return {k: expand_env(v, env) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        expanded = [expand_env(v, env) for v in value]
        return type(value)(expanded) if isinstance(value, tuple) else expanded
    return value


def _expand_env_str(value: str, env: Mapping[str, str]) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` references in a single string."""
    missing: list[str] = []

    def _replace(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        if name in env and env[name] != "":
            return env[name]
        if default is not None:
            return default
        missing.append(name)
        return ""

    expanded = _ENV_PATTERN.sub(_replace, value)

    if missing:
        raise KeyError(
            f"Environment variable(s) {missing} referenced in {value!r} are unset. "
            f"Either export them or give a default as ${{VAR:-default}}."
        )

    return expanded


def _format_placeholders(value: Any, fields: Mapping[str, Any]) -> Any:
    """Substitute ``{subject}``-style placeholders in strings, recursively.

    Unknown placeholders are left untouched rather than raising, so that
    values destined for osl-ephys (which has its own ``{run_id}`` templating)
    survive unharmed.
    """
    if isinstance(value, str):
        try:
            return value.format(**fields)
        except (KeyError, IndexError):
            return value
    if isinstance(value, Mapping):
        return {k: _format_placeholders(v, fields) for k, v in value.items()}
    if isinstance(value, list):
        return [_format_placeholders(v, fields) for v in value]
    return value


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_config(
    config_path: str | Path, env: Optional[Mapping[str, str]] = None
) -> SimpleNamespace:
    """Load, expand and validate a pipeline YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to the pipeline YAML file.
    env : Mapping, optional
        Environment used for ``${VAR}`` expansion.  Defaults to ``os.environ``.

    Returns
    -------
    cfg : SimpleNamespace
        With attributes:

        ``pipeline``
            :class:`~types.SimpleNamespace` of the ``pipeline`` section, with
            every key in :data:`PIPELINE_DEFAULTS` present.
        ``meta``
            The ``meta`` mapping (``{}`` when absent).
        ``preproc``
            The ``preproc`` step list (``None`` when absent).
        ``source_recon``
            The ``source_recon`` step list (``None`` when absent).
        ``path``
            The path the config was loaded from, as a string.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    ValueError
        If the file is empty, is not a mapping, contains an unrecognised
        top-level section or ``pipeline`` key, or holds an invalid
        ``source_backend`` / ``source_input``.

    Examples
    --------
    >>> cfg = load_config("config/TSX/osl/trial.yaml")  # doctest: +SKIP
    >>> cfg.pipeline.source_backend  # doctest: +SKIP
    'rhino'
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"osl pipeline config not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"osl pipeline config is empty: {config_path}")
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"osl pipeline config must be a mapping, got {type(raw).__name__}: "
            f"{config_path}"
        )

    unknown_sections = sorted(set(raw) - set(TOP_LEVEL_SECTIONS))
    if unknown_sections:
        raise ValueError(
            f"Unrecognised top-level section(s) {unknown_sections} in {config_path}. "
            f"Valid sections: {list(TOP_LEVEL_SECTIONS)}."
        )

    raw = expand_env(dict(raw), env)

    pipeline = _build_pipeline_section(raw.get("pipeline") or {}, config_path)
    source_recon = raw.get("source_recon")
    _validate_source_recon(source_recon, config_path)

    return SimpleNamespace(
        pipeline=pipeline,
        meta=raw.get("meta") or {},
        preproc=raw.get("preproc"),
        source_recon=source_recon,
        path=str(config_path),
    )


def _validate_source_recon(source_recon: Any, config_path: Path) -> None:
    """Check the shape of the ``source_recon`` section, if present."""
    if source_recon is None:
        return

    if isinstance(source_recon, Mapping):
        unknown = sorted(set(source_recon) - set(SOURCE_BACKENDS))
        if unknown:
            raise ValueError(
                f"'source_recon' in {config_path} is keyed by backend, but has "
                f"unrecognised key(s) {unknown}. Valid backends: "
                f"{list(SOURCE_BACKENDS)}."
            )
        return

    if not isinstance(source_recon, list):
        raise ValueError(
            f"'source_recon' in {config_path} must be a list of steps, or a "
            f"mapping from backend to a list of steps, got "
            f"{type(source_recon).__name__}."
        )


def _build_pipeline_section(
    section: Mapping[str, Any], config_path: Path
) -> SimpleNamespace:
    """Validate the ``pipeline`` section and fill in defaults."""
    if not isinstance(section, Mapping):
        raise ValueError(
            f"'pipeline' section must be a mapping, got {type(section).__name__}: "
            f"{config_path}"
        )

    unknown_keys = sorted(set(section) - set(PIPELINE_DEFAULTS))
    if unknown_keys:
        raise ValueError(
            f"Unrecognised pipeline key(s) {unknown_keys} in {config_path}. "
            f"Valid keys: {sorted(PIPELINE_DEFAULTS)}."
        )

    merged = {**PIPELINE_DEFAULTS, **dict(section)}

    # Placeholder substitution: resolve identity fields first, then use them to
    # fill the remaining strings (paths, subject_label, ...).
    fields = {
        "subject": merged["subject"],
        "session": merged["session"],
        "task": merged["task"],
        "analysis": merged["analysis"],
    }
    merged = {k: _format_placeholders(v, fields) for k, v in merged.items()}
    # subject_label may itself appear in path templates, so make it available.
    fields["subject_label"] = merged["subject_label"]
    merged = {k: _format_placeholders(v, fields) for k, v in merged.items()}

    backend = merged["source_backend"]
    if backend not in SOURCE_BACKENDS:
        raise ValueError(
            f"Invalid pipeline.source_backend {backend!r} in {config_path}. "
            f"Must be one of {list(SOURCE_BACKENDS)}."
        )

    source_input = merged["source_input"]
    if source_input not in _SOURCE_INPUTS:
        raise ValueError(
            f"Invalid pipeline.source_input {source_input!r} in {config_path}. "
            f"Must be one of {list(_SOURCE_INPUTS)}."
        )

    return SimpleNamespace(**merged)


# ---------------------------------------------------------------------------
# osl-ephys config extraction
# ---------------------------------------------------------------------------


def preproc_config(cfg: SimpleNamespace) -> dict:
    """Extract the osl-ephys preprocessing config from a pipeline config.

    Parameters
    ----------
    cfg : SimpleNamespace
        Config returned by :func:`load_config`.

    Returns
    -------
    config : dict
        A ``{'meta': ..., 'preproc': [...]}`` dict, ready to pass to
        :func:`osl_ephys.preprocessing.run_proc_chain`.

    Raises
    ------
    ValueError
        If the pipeline config has no ``preproc`` section.
    """
    if not cfg.preproc:
        raise ValueError(
            f"No 'preproc' section in {cfg.path}; cannot run the preproc stage."
        )

    return {"meta": dict(cfg.meta), "preproc": cfg.preproc}


def source_config(cfg: SimpleNamespace) -> dict:
    """Extract the osl-ephys source-recon config from a pipeline config.

    Parameters
    ----------
    cfg : SimpleNamespace
        Config returned by :func:`load_config`.

    Returns
    -------
    config : dict
        A ``{'source_recon': [...]}`` dict, ready to pass to
        :func:`osl_ephys.source_recon.run_src_chain`.

    Raises
    ------
    ValueError
        If the pipeline config has no ``source_recon`` section, or if the
        section is keyed by backend but has no entry for
        ``pipeline.source_backend``.

    Notes
    -----
    ``source_recon`` may be a plain list of steps, or a mapping from backend
    name (``rhino`` / ``freesurfer``) to a list of steps.  The mapping form lets
    one config describe both backends, since they need different steps.
    """
    steps = cfg.source_recon
    if not steps:
        raise ValueError(
            f"No 'source_recon' section in {cfg.path}; cannot run the source stage."
        )

    if isinstance(steps, Mapping):
        backend = cfg.pipeline.source_backend
        if backend not in steps:
            raise ValueError(
                f"'source_recon' in {cfg.path} is keyed by backend but has no "
                f"'{backend}' entry (pipeline.source_backend is {backend!r}). "
                f"Found: {sorted(steps)}."
            )
        steps = steps[backend]
        if not steps:
            raise ValueError(
                f"'source_recon.{backend}' in {cfg.path} is empty; "
                f"cannot run the source stage."
            )

    return {"source_recon": steps}
