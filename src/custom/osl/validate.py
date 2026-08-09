"""Config validation for the osl-ephys OPM pipeline.

Checks a pipeline config without running anything, so that a typo costs a
second rather than a place in the cluster queue and however long the chain runs
before it reaches the bad step.

It reports these classes of problem:

* **Unresolvable steps** -- a step name that neither osl-ephys nor the custom
  wrappers provide.  osl-ephys logs ``Function not found!`` and then fails with
  a ``TypeError`` some way into the chain, which is a slow way to find a typo.

* **Unsatisfied arguments** -- a source-recon step whose required arguments are
  neither supplied by osl-ephys nor set in the config.  This replicates the
  check inside :func:`osl_ephys.source_recon.run_src_chain`, which is stricter
  than it looks: it rejects any wrapper declared as ``(*args, **kwargs)``, so
  the ``extract_fiducials_from_fif`` alias used by the osl-ephys tutorials
  cannot actually be used in a config (use ``extract_polhemus_from_info``).

* **Unknown options** -- a step option the wrapper does not accept.

* **Malformed group contrasts** -- mismatched condition/weight lengths, or a
  contrast referencing a condition the group stage will not compute.

Missing input files are reported as warnings, not errors, since a config is
routinely validated before the data it points at exists.

Functions
---------
run
    Validate a config and print a report.
validate_config
    Validate a config and return the list of problems found.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

from ._config import preproc_config, source_config
from ._paths import resolve_paths
from .extra_funcs import PREPROC_EXTRA_FUNCS
from .fs_bridge import SOURCE_EXTRA_FUNCS


# Arguments osl-ephys supplies to every source-recon wrapper itself.
_SRC_PROVIDED_ARGS: frozenset[str] = frozenset(
    {
        "outdir",
        "subject",
        "surface_extraction_method",
        "preproc_file",
        "smri_file",
        "epoch_file",
        "reportdir",
        "logsdir",
    }
)


def run(cfg: SimpleNamespace) -> bool:
    """Validate a config and print a report.

    Parameters
    ----------
    cfg : SimpleNamespace
        Config from :func:`custom.osl._config.load_config`.

    Returns
    -------
    valid : bool
        True if no errors were found.  Warnings do not make it False.
    """
    errors, warnings = validate_config(cfg)

    paths = resolve_paths(cfg.pipeline)
    print(f"[osl:validate] config:   {cfg.path}")
    print(f"[osl:validate] subject:  {paths.subject_label}")
    print(f"[osl:validate] backend:  {cfg.pipeline.source_backend}")
    print(f"[osl:validate] outdir:   {paths.outdir}")
    print()

    for warning in warnings:
        print(f"[osl:validate] warning: {warning}")
    for error in errors:
        print(f"[osl:validate] ERROR:   {error}")

    print()
    if errors:
        print(f"[osl:validate] {len(errors)} error(s) found")
    else:
        print("[osl:validate] config is valid")

    return not errors


def validate_config(cfg: SimpleNamespace) -> tuple[list[str], list[str]]:
    """Validate a config and return the problems found.

    Parameters
    ----------
    cfg : SimpleNamespace
        Config from :func:`custom.osl._config.load_config`.

    Returns
    -------
    errors : list of str
        Problems that would make a run fail.
    warnings : list of str
        Things worth knowing that would not necessarily fail, chiefly inputs
        that do not exist yet.
    """
    errors: list[str] = []
    warnings: list[str] = []

    try:
        paths = resolve_paths(cfg.pipeline)
    except ValueError as exc:
        return [str(exc)], warnings

    _check_inputs(cfg, paths, warnings)

    if cfg.preproc:
        errors.extend(_check_preproc_steps(cfg))
    else:
        warnings.append("no 'preproc' section; the preproc stage cannot run")

    if cfg.source_recon:
        errors.extend(_check_source_steps(cfg))
    else:
        warnings.append("no 'source_recon' section; the source stage cannot run")

    if cfg.group:
        errors.extend(_check_group(cfg))
    else:
        warnings.append("no 'group' section; the group stage cannot run")

    return errors, warnings


def _check_group(cfg: SimpleNamespace) -> list[str]:
    """Check the group section's contrasts are well formed."""
    errors: list[str] = []
    group = cfg.group

    if cfg.pipeline.source_input != "epochs":
        errors.append(
            f"the group stage averages epochs by condition, so it needs "
            f"pipeline.source_input: epochs, not "
            f"{cfg.pipeline.source_input!r}"
        )

    declared = group.get("conditions")

    for index, contrast in enumerate(group.get("contrasts") or []):
        label = contrast.get("name", f"#{index}")

        missing_keys = sorted({"name", "conditions", "weights"} - set(contrast))
        if missing_keys:
            errors.append(f"group contrast '{label}' is missing {missing_keys}")
            continue

        conditions, weights = contrast["conditions"], contrast["weights"]
        if len(conditions) != len(weights):
            errors.append(
                f"group contrast '{label}' has {len(conditions)} condition(s) "
                f"but {len(weights)} weight(s)"
            )

        if declared:
            unknown = [c for c in conditions if c not in declared]
            if unknown:
                errors.append(
                    f"group contrast '{label}' references {unknown}, which "
                    f"is not in group.conditions {list(declared)}"
                )

    return errors


def _check_inputs(
    cfg: SimpleNamespace, paths: SimpleNamespace, warnings: list[str]
) -> None:
    """Warn about inputs that are not present yet."""
    if not os.path.exists(paths.input_fif):
        warnings.append(f"BIDS raw file does not exist yet: {paths.input_fif}")

    if not paths.smri:
        warnings.append(
            f"no T1w image found for sub-{paths.subject}; coregistration steps "
            f"will fail"
        )
    elif not os.path.exists(paths.smri):
        warnings.append(f"T1w image does not exist yet: {paths.smri}")

    if cfg.pipeline.source_backend == "freesurfer" and paths.trans:
        if not os.path.exists(paths.trans):
            warnings.append(
                f"coregistration transform does not exist yet: {paths.trans} "
                f"(produced by `mne-opm.sh coreg`)"
            )


def _check_preproc_steps(cfg: SimpleNamespace) -> list[str]:
    """Check that every preprocessing step resolves to a function."""
    from osl_ephys.preprocessing.batch import find_func

    errors: list[str] = []
    for step in preproc_config(cfg)["preproc"]:
        name, userargs = next(iter(step.items()))
        userargs = userargs or {}
        target = userargs.get("target", "raw")

        if find_func(name, target=target, extra_funcs=PREPROC_EXTRA_FUNCS) is None:
            errors.append(
                f"preproc step '{name}' is not an osl-ephys wrapper, an MNE "
                f"{target} method, or a custom function"
            )

    return errors


def _check_source_steps(cfg: SimpleNamespace) -> list[str]:
    """Check that every source-recon step resolves and can be called."""
    from osl_ephys.source_recon.batch import find_func

    extra_funcs = (
        SOURCE_EXTRA_FUNCS if cfg.pipeline.source_backend == "freesurfer" else None
    )

    errors: list[str] = []
    for step in source_config(cfg)["source_recon"]:
        name, userargs = next(iter(step.items()))
        userargs = userargs or {}

        func = find_func(name, extra_funcs=extra_funcs)
        if func is None:
            errors.append(
                f"source_recon step '{name}' is not an osl-ephys source-recon "
                f"wrapper or a custom function"
            )
            continue

        errors.extend(_check_signature(name, func, userargs))

    return errors


def _check_signature(name: str, func, userargs: dict) -> list[str]:
    """Replicate osl-ephys' argument check for one source-recon step."""
    errors: list[str] = []
    parameters = inspect.signature(func).parameters

    accepts_kwargs = any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )

    # osl-ephys builds its required-argument list from every parameter except
    # one literally named "kwargs", so a *args parameter counts as required and
    # can never be satisfied.
    required = [
        param_name
        for param_name, param in parameters.items()
        if param_name != "kwargs"
        and param.default is inspect.Parameter.empty
    ]

    for param_name in required:
        if parameters[param_name].kind is inspect.Parameter.VAR_POSITIONAL:
            errors.append(
                f"source_recon step '{name}' is declared with *{param_name}, "
                f"which run_src_chain cannot satisfy: it will raise "
                f"\"{param_name} needs to be passed to {name}\". Use the "
                f"underlying wrapper instead of this alias."
            )
        elif param_name not in _SRC_PROVIDED_ARGS and param_name not in userargs:
            errors.append(
                f"source_recon step '{name}' needs '{param_name}', which is "
                f"neither supplied by osl-ephys nor set in the config"
            )

    if not accepts_kwargs:
        unknown = sorted(set(userargs) - set(parameters))
        if unknown:
            errors.append(
                f"source_recon step '{name}' does not accept option(s) {unknown}"
            )

    return errors
