#!/usr/bin/env python
"""osl-ephys pipeline CLI for OPM-MEG data.

Runs one stage of the osl-ephys pipeline for a single subject, which is what
makes it usable as the body of a SLURM array job.

Stages
------
preproc
    Run the osl-ephys ``preproc`` chain over the subject's BIDS raw file.
source
    Run the osl-ephys ``source_recon`` chain: surfaces, coregistration,
    forward model, LCMV beamforming and parcellation.
all
    ``preproc`` then ``source``.  Stops if preprocessing fails.
group
    Sign-flip every subject against a template, then compute group condition
    averages and contrasts.  Run once, after the array finishes.
collate
    Rebuild the group-level HTML reports across all subjects.  Run once, after
    the array finishes -- not per subject.
validate
    Check the config without running anything: that every step resolves and
    every required argument is satisfied.  Worth doing before submitting an
    array job.

Usage
-----
    python src/custom/run_osl.py --stage=<name> --config=/path/to/config.yaml

Examples
--------
    # Check the config before burning a place in the queue
    SUBJECT=007 python src/custom/run_osl.py --stage=validate --config=osl/trial.yaml

    # One subject, end to end
    SUBJECT=007 python src/custom/run_osl.py --stage=all --config=osl/trial.yaml

    # Just re-run source reconstruction with new beamformer settings
    python src/custom/run_osl.py --stage=source --config=osl/trial.yaml

    # After the array job, build the group reports
    python src/custom/run_osl.py --stage=collate --config=osl/trial.yaml

Configuration
-------------
The config is a YAML file (see ``custom.osl._config``) holding a ``pipeline``
section for paths and backends, and ``meta``/``preproc``/``source_recon``
sections passed through to osl-ephys.  Values may reference environment
variables as ``${VAR}``, so one file serves every subject in an array.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the parent of this file's directory (i.e. ``src/``) is on sys.path so
# that ``custom.osl.*`` resolves, matching custom_preproc.py.
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from custom.osl import collate as collate_stage
from custom.osl import group as group_stage
from custom.osl import preproc as preproc_stage
from custom.osl import source as source_stage
from custom.osl import validate as validate_stage
from custom.osl._config import load_config


STAGE_CHOICES: list[str] = [
    "preproc",
    "source",
    "all",
    "group",
    "collate",
    "validate",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Argument list.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments with ``stage``, ``config`` and ``subject`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Run an osl-ephys pipeline stage for one subject",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=STAGE_CHOICES,
        metavar="STAGE",
        help=f"Pipeline stage to run. Choices: {', '.join(STAGE_CHOICES)}",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        metavar="PATH",
        help="Path to the pipeline YAML config",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "Override pipeline.subject (e.g. '007'). Normally supplied through "
            "the SUBJECT environment variable referenced by the config."
        ),
    )

    return parser.parse_args(argv)


def run_stage(stage: str, cfg) -> bool:
    """Run one stage against a loaded config.

    Parameters
    ----------
    stage : str
        One of :data:`STAGE_CHOICES`.
    cfg : SimpleNamespace
        Config from :func:`custom.osl._config.load_config`.

    Returns
    -------
    success : bool
        Whether the stage (and, for ``all``, every sub-stage) succeeded.

    Raises
    ------
    ValueError
        If ``stage`` is not recognised.
    """
    if stage == "preproc":
        return preproc_stage.run(cfg)
    if stage == "source":
        return source_stage.run(cfg)
    if stage == "group":
        return group_stage.run(cfg)
    if stage == "collate":
        return collate_stage.run(cfg)
    if stage == "validate":
        return validate_stage.run(cfg)
    if stage == "all":
        if not preproc_stage.run(cfg):
            print("[osl] preprocessing failed; not running source reconstruction")
            return False
        return source_stage.run(cfg)

    raise ValueError(f"Unknown stage: {stage}. Valid options: {STAGE_CHOICES}")


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    Returns
    -------
    exit_code : int
        0 on success, 1 on failure.
    """
    args = parse_args(argv)

    print()
    print("=" * 70)
    print("MNE-OPM osl-ephys Pipeline")
    print("=" * 70)
    print(f"  Stage:  {args.stage}")
    print(f"  Config: {args.config}")
    print("=" * 70)
    print()

    try:
        cfg = load_config(args.config)

        if args.subject:
            cfg.pipeline.subject = args.subject

        success = run_stage(args.stage, cfg)

    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        return 1
    except (KeyError, ValueError) as e:
        print(f"\n[ERROR] Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Stage '{args.stage}' failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print()
    print("=" * 70)
    if success:
        print(f"Stage '{args.stage}' completed successfully")
    else:
        print(f"Stage '{args.stage}' did not complete; see the logs above")
    print("=" * 70)
    print()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
