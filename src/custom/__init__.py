"""Custom preprocessing and analysis modules for OPM-MEG data.

This package provides modular preprocessing analyses for OPM-MEG data,
including bad segment/channel detection, reference regression, ICA, and more.

Main entry point:
    python -m src.custom.custom_preproc --analysis=<name> --config=<path>

Subpackages
-----------
preprocessing
    Individual analysis modules and shared utilities.

Author: Harrison Ritz, 2025
"""

from . import preprocessing

__all__ = ["preprocessing"]
