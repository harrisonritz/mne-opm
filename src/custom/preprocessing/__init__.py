"""Preprocessing analysis modules for OPM-MEG data.

This subpackage contains individual analysis modules and shared utilities
for preprocessing OPM-MEG data. Each analysis can be run independently
via the CLI dispatcher in custom_preproc.py.

Available Analyses
------------------
regress_ref
    Regress out reference channel signals from MEG data.
bad_segments
    Detect and annotate bad raw data segments.
bad_channels
    Statistical detection of bad channels using GESD.
manual_channel
    Interactive visual marking of bad channels.
apply_hfc
    Apply homogeneous field correction (HFC) projections.
bad_epochs
    Drop bad epochs post-epoching using GESD.
auto_ica
    Automatic ICA component labeling based on reference sensors.
manual_ica
    Interactive ICA component review and selection.

Shared Utilities
----------------
_base
    BaseAnalysis class and shared constants.
_config
    Configuration loading and validation utilities.
_io
    Data loading and saving utilities for BIDS.
_bids_utils
    BIDS path construction helper functions.

Author: Harrison Ritz, 2025
"""

# Import analysis modules for convenient access
from . import (
    regress_ref,
    bad_segments,
    bad_channels,
    manual_channel,
    apply_hfc,
    bad_epochs,
    auto_ica,
    manual_ica,
)

# Import shared utilities
from ._base import BaseAnalysis, SEGMENT_LEN_SEC
from ._config import load_config, normalize_analysis_key
from ._io import save_ica_bids, get_bids_path_for_task
from ._bids_utils import get_bids_path

__all__ = [
    # Analysis modules
    "regress_ref",
    "bad_segments",
    "bad_channels",
    "manual_channel",
    "apply_hfc",
    "bad_epochs",
    "auto_ica",
    "manual_ica",
    # Shared utilities
    "BaseAnalysis",
    "SEGMENT_LEN_SEC",
    "load_config",
    "normalize_analysis_key",
    "save_ica_bids",
    "get_bids_path_for_task",
    "get_bids_path",
]
