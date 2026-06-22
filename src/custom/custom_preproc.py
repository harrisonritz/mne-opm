#!/usr/bin/env python
"""Modular auxiliary preprocessing CLI for OPM-MEG data.

This is the main entry point for custom preprocessing analyses. It parses
command-line arguments and dispatches to the appropriate analysis module
in the preprocessing subpackage.

Provided analyses (CLI --analysis):
    init           -> Clear leftover proc-<custom_proc> derivatives from a
                      previous run so preprocessing starts from the raw data
    regress        -> Regress out a configurable list of sensor signals
    bad_segments   -> Detect & annotate bad raw data segments (legacy)
    bad_segments_1 -> Stage 1: coarse bad segment detection (pre-spatial filter)
    bad_segments_2 -> Stage 2: fine bad segment detection (post-spatial filter)
    bad_channels   -> Statistical detection of bad channels
    manual_channel -> Interactive visual marking of bad channels
    apply_hfc      -> Apply homogeneous field correction (HFC) projections
    select_trial_response -> Keep only the first response following each trial
                      (response-locked analyses; gated by _select_trial_response)
    bad_epochs     -> Drop bad epochs post-epoching
    bad_ICs       -> Automatic ICA component labeling
    manual_ica     -> Interactive ICA component review

Usage
-----
    python src/custom/custom_preproc.py --analysis=<name> --config=/path/to/config.py

Examples
--------
    # Run sensor regression
    python src/custom/custom_preproc.py --analysis=regress --config=config.py

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
import json
import sys
from pathlib import Path
from typing import Callable

# Ensure the parent of this file's directory (i.e. ``src/``) is on sys.path
# so that both ``custom.preprocessing.*`` and ``preprocessing.*`` resolve.
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from custom.preprocessing._config import load_config, normalize_analysis_key


# -----------------------------------------------------------------------------
# Analysis Registry
# -----------------------------------------------------------------------------

# Mapping of normalized analysis keys to their module names
# Each module must have a run(cfg) function
ANALYSIS_REGISTRY: dict[str, str] = {
    "init": "init_derivatives",
    "initderivatives": "init_derivatives",
    "regress": "regress",
    "badsegments": "bad_segments",
    "badsegments1": "bad_segments",
    "badsegments2": "bad_segments",
    "badchannels": "bad_channels",
    "manualchannel": "manual_channel",
    "applyhfc": "apply_hfc",
    "zcafilter": "zca_filter",
    "applyzca": "zca_filter",  # Alias for zca_filter
    "selecttrialresponse": "select_trial_response",
    "badepochs": "bad_epochs",
    "badICs": "bad_ICs",
    "manualica": "manual_ica",
    "coreg": "coreg",
}

# Human-readable names for CLI choices (with underscores)
ANALYSIS_CHOICES: list[str] = [
    "init",
    "init_derivatives",
    "regress",
    "bad_segments",
    "bad_segments_1",
    "bad_segments_2",
    "bad_channels",
    "bad_epochs",
    "bad_ICs",
    "manual_channel",
    "apply_hfc",
    "zca_filter",
    "apply_zca",  # Alias for zca_filter
    "select_trial_response",
    "manual_ica",
    "coreg",
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
    full_module_path = f"custom.preprocessing.{module_name}"

    # Import the module
    module = importlib.import_module(full_module_path)

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

        # Extract stage suffix for staged analyses (e.g., badsegments1 -> "1")
        if analysis_key.startswith("badsegments") and len(analysis_key) > len("badsegments"):
            cfg._bad_segments_stage = analysis_key[len("badsegments"):]

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
    except json.JSONDecodeError as e:
        print(f"\n[ERROR] JSON parse error (possible NFS race condition): {e}")
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
