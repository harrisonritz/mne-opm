#!/usr/bin/env python
"""Modular auxiliary preprocessing CLI for OPM-MEG data.

This is the main entry point for custom preprocessing analyses. It parses
command-line arguments and dispatches to the appropriate analysis module
in the preprocessing subpackage.

Provided analyses (CLI --analysis):
    regress_ref    -> Regress out reference channel signals
    bad_segments   -> Detect & annotate bad raw data segments
    bad_channels   -> Statistical detection of bad channels
    manual_channel -> Interactive visual marking of bad channels
    apply_hfc      -> Apply homogeneous field correction (HFC) projections
    bad_epochs     -> Drop bad epochs post-epoching
    auto_ica       -> Automatic ICA component labeling
    manual_ica     -> Interactive ICA component review

Usage
-----
    python src/custom/custom_preproc.py --analysis=<name> --config=/path/to/config.py

Examples
--------
    # Run reference regression
    python src/custom/custom_preproc.py --analysis=regress_ref --config=config.py

    # Run bad segment detection
    python src/custom/custom_preproc.py --analysis=bad_segments --config=config.py

    # Run interactive ICA review
    python src/custom/custom_preproc.py --analysis=manual_ica --config=config.py

Internal Notes
--------------
Internal normalized keys remove underscores (e.g., bad_segments -> badsegments).
This mapping is handled by the normalize_analysis_key function.

Outputs are written back into the BIDS structure using mne-bids utilities,
re-using existing derivative locations produced by mne-bids-pipeline.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Callable

from preprocessing._config import load_config, normalize_analysis_key


# -----------------------------------------------------------------------------
# Analysis Registry
# -----------------------------------------------------------------------------

# Mapping of normalized analysis keys to their module names
# Each module must have a run(cfg) function
ANALYSIS_REGISTRY: dict[str, str] = {
    "regressref": "regress_ref",
    "badsegments": "bad_segments",
    "badchannels": "bad_channels",
    "manualchannel": "manual_channel",
    "applyhfc": "apply_hfc",
    "badepochs": "bad_epochs",
    "autoica": "auto_ica",
    "manualica": "manual_ica",
}

# Human-readable names for CLI choices (with underscores)
ANALYSIS_CHOICES: list[str] = [
    "regress_ref",
    "bad_segments",
    "bad_channels",
    "manual_channel",
    "apply_hfc",
    "bad_epochs",
    "auto_ica",
    "manual_ica",
]


# -----------------------------------------------------------------------------
# CLI Parsing
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments with 'analysis' and 'config' attributes.
    """
    parser = argparse.ArgumentParser(
        description="Modular OPM auxiliary preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--analysis",
        required=True,
        choices=ANALYSIS_CHOICES,
        metavar="ANALYSIS",
        help=(
            f"Analysis type to run. Choices: {', '.join(ANALYSIS_CHOICES)}"
        ),
    )

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        metavar="PATH",
        help="Path to configuration Python file",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Module Import
# -----------------------------------------------------------------------------


def import_analysis_module(analysis_key: str) -> Callable:
    """Dynamically import an analysis module and return its run function.

    Parameters
    ----------
    analysis_key : str
        Normalized analysis key (e.g., 'regressref').

    Returns
    -------
    run_func : callable
        The run(cfg) function from the analysis module.

    Raises
    ------
    ValueError
        If the analysis key is unknown.
    ImportError
        If the module cannot be imported.
    """
    if analysis_key not in ANALYSIS_REGISTRY:
        raise ValueError(
            f"Unknown analysis: {analysis_key}. "
            f"Valid options: {list(ANALYSIS_REGISTRY.keys())}"
        )

    module_name = ANALYSIS_REGISTRY[analysis_key]
    full_module_path = f"preprocessing.{module_name}"

    # Import the module
    module = importlib.import_module(full_module_path, package=".")

    # Get the run function
    if not hasattr(module, "run"):
        raise ImportError(
            f"Module {full_module_path} does not have a run() function"
        )

    return module.run


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------


def main() -> int:
    """Main entry point for the CLI.

    Returns
    -------
    exit_code : int
        0 for success, 1 for error.
    """
    args = parse_args()

    # Normalize analysis key (remove underscores)
    analysis_key = normalize_analysis_key(args.analysis)

    # Print header
    print()
    print("=" * 70)
    print("MNE-OPM Custom Preprocessing")
    print("=" * 70)
    print(f"  Analysis: {args.analysis}")
    print(f"  Config:   {args.config}")
    print("=" * 70)
    print()

    try:
        # Load configuration
        cfg = load_config(args.config)

        # Import and run analysis module
        run_func = import_analysis_module(analysis_key)
        run_func(cfg)

        # Print footer
        print()
        print("=" * 70)
        print(f"Analysis '{args.analysis}' completed successfully")
        print("=" * 70)
        print()

        return 0

    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        return 1
    except ValueError as e:
        print(f"\n[ERROR] Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
