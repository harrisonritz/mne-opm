"""Base classes and shared constants for OPM-MEG preprocessing.

This module provides the BaseAnalysis abstract base class that all
analysis modules inherit from, ensuring a consistent interface across
the preprocessing pipeline.

Classes
-------
BaseAnalysis
    Abstract base class for preprocessing analyses.

Constants
---------
SEGMENT_LEN_SEC
    Default segment length for segment-based detection (1.0 second).

Module Initialization
---------------------
This module also handles global setup for MNE visualization:
- Sets Qt as the browser backend (if available)
- Checks for Qt browser availability

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any, Dict

import mne


# -----------------------------------------------------------------------------
# Global Constants
# -----------------------------------------------------------------------------

SEGMENT_LEN_SEC: float = 1.0
"""Default segment length in seconds for segment-based detection."""


# -----------------------------------------------------------------------------
# Qt Browser Setup
# -----------------------------------------------------------------------------

# Check for Qt browser availability
try:
    import mne_qt_browser

    _HAVE_QT_BROWSER: bool = True
except Exception:
    _HAVE_QT_BROWSER: bool = False

# Try to set Qt as the browser backend
try:
    mne.viz.set_browser_backend("qt")
except Exception:
    # Fallback silently if Qt is not available
    pass


def have_qt_browser() -> bool:
    """Check if the Qt browser is available.

    Returns
    -------
    available : bool
        True if mne_qt_browser is installed and importable.

    Examples
    --------
    >>> if have_qt_browser():
    ...     raw.plot(block=True)
    ... else:
    ...     print("Qt browser not available")
    """
    return _HAVE_QT_BROWSER


# -----------------------------------------------------------------------------
# Base Analysis Class
# -----------------------------------------------------------------------------


class BaseAnalysis(ABC):
    """Abstract base class for OPM-MEG preprocessing analyses.

    This class defines the interface that all analysis modules must implement.
    It provides a consistent structure for loading data, running analyses,
    and saving results.

    Subclasses must implement:
        - is_enabled(): Check if analysis is enabled in config
        - load_data(): Load required data
        - run(): Execute the analysis
        - save_results(): Save results to BIDS

    The execute() method provides the full pipeline: load -> run -> save.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object containing all settings for the analysis.

    Attributes
    ----------
    ANALYSIS_KEY : str
        Normalized key for this analysis (e.g., 'badsegments').
        Must be defined by subclass.
    ANALYSIS_NAME : str
        Human-readable name for logging (e.g., 'bad_segments').
        Must be defined by subclass.
    cfg : SimpleNamespace
        Configuration object passed to constructor.

    Examples
    --------
    Implementing a custom analysis:

    >>> class MyAnalysis(BaseAnalysis):
    ...     ANALYSIS_KEY = 'myanalysis'
    ...     ANALYSIS_NAME = 'my_analysis'
    ...
    ...     def is_enabled(self) -> bool:
    ...         return getattr(self.cfg, '_my_analysis', False)
    ...
    ...     def load_data(self) -> Dict[str, Any]:
    ...         return {'task': load_raw_bids(self.cfg, self.cfg.task)}
    ...
    ...     def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
    ...         # Process data...
    ...         return {'task': processed_data}
    ...
    ...     def save_results(self, results: Dict[str, Any]) -> None:
    ...         save_raw_bids(results['task'], self.cfg, self.cfg.task)

    Running an analysis:

    >>> analysis = MyAnalysis(cfg)
    >>> if analysis.is_enabled():
    ...     analysis.execute()
    """

    ANALYSIS_KEY: str = ""
    """Normalized key for this analysis (must be overridden)."""

    ANALYSIS_NAME: str = ""
    """Human-readable name for logging (must be overridden)."""

    def __init__(self, cfg: SimpleNamespace) -> None:
        """Initialize the analysis with configuration.

        Parameters
        ----------
        cfg : SimpleNamespace
            Configuration object containing all settings.
        """
        self.cfg = cfg

    def log(self, message: str) -> None:
        """Print a log message with the analysis name prefix.

        Parameters
        ----------
        message : str
            Message to log.
        """
        print(f"[{self.ANALYSIS_NAME}] {message}")

    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if this analysis is enabled in the configuration.

        Returns
        -------
        enabled : bool
            True if the analysis should run, False to skip.

        Notes
        -----
        This method should check for analysis-specific config flags.
        For example, regress_ref checks cfg._regress_ref.
        """
        ...

    @abstractmethod
    def load_data(self) -> Dict[str, Any]:
        """Load data required for this analysis.

        Returns
        -------
        data : dict
            Dictionary containing loaded data. Keys depend on the analysis
            type (e.g., task names for raw data, 'ica' for ICA objects).

        Notes
        -----
        This method should use the utilities in _io.py to load data
        from the BIDS structure.
        """
        ...

    @abstractmethod
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the analysis on loaded data.

        Parameters
        ----------
        data : dict
            Dictionary from load_data() containing input data.

        Returns
        -------
        results : dict
            Dictionary containing analysis results. Must include all data
            that needs to be saved, plus any metadata (e.g., 'bads' for
            bad channel lists).

        Notes
        -----
        This method contains the core analysis logic. It should not
        perform any I/O operations.
        """
        ...

    @abstractmethod
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results to BIDS structure.

        Parameters
        ----------
        results : dict
            Dictionary from run() containing results to save.

        Notes
        -----
        This method should use the utilities in _io.py to save data
        back to the BIDS structure.
        """
        ...

    def execute(self) -> None:
        """Execute the full analysis pipeline: load -> run -> save.

        This is the main entry point for running an analysis. It calls
        load_data(), run(), and save_results() in sequence.

        Examples
        --------
        >>> analysis = BadSegmentsAnalysis(cfg)
        >>> if analysis.is_enabled():
        ...     analysis.execute()
        """
        self.log("Starting analysis...")

        self.log("Loading data...")
        data = self.load_data()

        self.log("Running analysis...")
        results = self.run(data)

        self.log("Saving results...")
        self.save_results(results)

        self.log("Analysis complete!")
