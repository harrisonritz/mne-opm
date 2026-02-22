"""Configuration loading and validation utilities for OPM-MEG preprocessing.

This module provides utilities for loading configuration from mne-bids-pipeline
config files and validating that required attributes are present.

Functions
---------
load_config
    Load configuration from a Python config file.
normalize_analysis_key
    Normalize analysis name by removing underscores.
check_analysis_enabled
    Check if a specific analysis is enabled in the configuration.
get_analysis_config_flag
    Get the config flag name for a given analysis.

Constants
---------
ANALYSIS_CONFIG_FLAGS
    Mapping of analysis keys to their configuration enable flags.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Optional

from mne_bids_pipeline._config_import import (
    _update_config_from_path,
    _import_config,
)


# Mapping of normalized analysis keys to their config enable flags.
# Analyses not in this mapping are always enabled (no config flag required).
ANALYSIS_CONFIG_FLAGS: dict[str, str] = {
    "manualchannel": "_manual_channels",
    "autoica": "_auto_ica",
    "manualica": "_manual_ica",
    "regressref": "_regress_ref",
    "applyhfc": "_do_HFC",
    "zcafilter": "_do_ZCA",
}

# Analyses that require spatial_filter="ica" to be set
ICA_ANALYSES: set[str] = {"autoica", "manualica"}


def load_config(config_path: str) -> SimpleNamespace:
    """Load configuration from a Python config file.

    Uses mne-bids-pipeline's config import utilities to load and merge
    configuration from the specified Python file.

    Parameters
    ----------
    config_path : str
        Path to the Python configuration file.

    Returns
    -------
    cfg : SimpleNamespace
        Configuration object with all settings as attributes.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ImportError
        If the config file cannot be imported.

    Examples
    --------
    >>> cfg = load_config("/path/to/config.py")
    >>> print(cfg.subjects)
    ['sub-01', 'sub-02']
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    cfg = _import_config(config_path=config_path)
    _update_config_from_path(config=cfg, config_path=config_path)

    return cfg


def normalize_analysis_key(analysis: str) -> str:
    """Normalize an analysis name by removing underscores.

    The CLI accepts analysis names with underscores (e.g., 'bad_segments'),
    but internally we use normalized keys without underscores ('badsegments').

    Parameters
    ----------
    analysis : str
        Analysis name from CLI (e.g., 'bad_segments', 'regress_ref').

    Returns
    -------
    key : str
        Normalized key (e.g., 'badsegments', 'regressref').

    Examples
    --------
    >>> normalize_analysis_key('bad_segments')
    'badsegments'
    >>> normalize_analysis_key('regress_ref')
    'regressref'
    """
    return analysis.replace("_", "")


def get_analysis_config_flag(analysis_key: str) -> Optional[str]:
    """Get the configuration flag name for a given analysis.

    Parameters
    ----------
    analysis_key : str
        Normalized analysis key (e.g., 'regressref').

    Returns
    -------
    flag : str or None
        Configuration attribute name that enables this analysis,
        or None if the analysis has no enable flag.

    Examples
    --------
    >>> get_analysis_config_flag('regressref')
    '_regress_ref'
    >>> get_analysis_config_flag('badsegments')
    None
    """
    return ANALYSIS_CONFIG_FLAGS.get(analysis_key)


def check_analysis_enabled(cfg: SimpleNamespace, analysis_key: str) -> bool:
    """Check if a specific analysis is enabled in the configuration.

    Some analyses require specific config flags to be set to True.
    ICA analyses additionally require spatial_filter='ica'.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object.
    analysis_key : str
        Normalized analysis key (e.g., 'regressref', 'manualica').

    Returns
    -------
    enabled : bool
        True if the analysis is enabled, False otherwise.

    Notes
    -----
    Analyses without a config flag in ANALYSIS_CONFIG_FLAGS are always
    considered enabled (e.g., 'badsegments', 'badchannels').

    Examples
    --------
    >>> cfg = SimpleNamespace(_regress_ref=True)
    >>> check_analysis_enabled(cfg, 'regressref')
    True
    >>> check_analysis_enabled(cfg, 'badsegments')
    True
    """
    # Check for analysis-specific enable flag
    flag = get_analysis_config_flag(analysis_key)
    if flag is not None:
        if not getattr(cfg, flag, False):
            return False

    # ICA analyses require spatial_filter='ica'
    if analysis_key in ICA_ANALYSES:
        if getattr(cfg, "spatial_filter", None) != "ica":
            return False

    return True


def validate_required_config(
    cfg: SimpleNamespace, required_attrs: list[str], analysis_name: str
) -> None:
    """Validate that required configuration attributes are present.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object to validate.
    required_attrs : list of str
        List of attribute names that must be present.
    analysis_name : str
        Name of the analysis (for error messages).

    Raises
    ------
    ValueError
        If any required attribute is missing from the configuration.
    """
    missing = [attr for attr in required_attrs if not hasattr(cfg, attr)]
    if missing:
        raise ValueError(
            f"[{analysis_name}] Missing required config attributes: {missing}"
        )
