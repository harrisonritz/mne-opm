"""Adaptive multipole models (AMM) for OPM-MEG data.

Projects the data onto the internal space of a prolate spheroidal harmonic
model fitted to the sensor array, after removing its external space, and
optionally removes the remaining interference with a temporal extension
(Tierney et al., 2024).  The projection is stored as SSP projectors, so the
forward model sees it too.  The method is MNE's
(``mne.preprocessing.apply_amm`` and ``compute_proj_amm``, an independent
implementation of SPM's ``spm_opm_amm`` by Tim Tierney); see their docstrings
for the details and attribution.

Usage
-----
CLI:
    python src/custom/custom_preproc.py --analysis=apply_amm --config=/path/to/config.py

Programmatic:
    >>> from preprocessing.apply_amm import run
    >>> run(cfg)

Configuration Attributes
------------------------
Required:
    ch_types, bids_root, subjects, sessions, task : as for apply_hfc

Optional:
    _do_AMM : bool
        Enable/disable AMM. Default: False.
    _amm_li : int
        Order of the internal harmonics. Default: 9 (99 harmonics).
    _amm_le : int
        Order of the external harmonics. Default: 2.
    _amm_corr_lim : float
        Correlation limit of the temporal extension; 1 turns it off.
        Default: 1.0 (SPM's default; the paper used 0.98).
    _amm_window : float
        Window of the temporal extension in seconds. Default: 10.
    _amm_coord_frame : str
        Frame the spheroid is fitted in, "head" or "meg". Default: "head".
    _amm_center_basis : bool
        Centre the basis across channels, as SPM does. Default: False (the
        paper's equations).
    process_empty_room : bool
        Apply the task's projection to the noise recording. Default: False.

Reference
---------
Tierney, T. M., Seedat, Z., St Pier, K., Mellor, S., & Barnes, G. R. (2024).
Adaptive multipole models of optically pumped magnetometer data. Human Brain
Mapping, 45(4), e26596. https://doi.org/10.1002/hbm.26596

Author: Harrison Ritz, 2026
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

from mne.preprocessing import apply_amm, compute_proj_amm

from .apply_hfc import ApplyHFCAnalysis


class ApplyAMMAnalysis(ApplyHFCAnalysis):
    """Apply adaptive multipole models (AMM).

    Reads and writes the task (and empty-room) recordings as apply_hfc does;
    the model is fitted to the task recording's good channels and its
    projection applied to both.

    Attributes
    ----------
    ANALYSIS_KEY : str
        'applyamm'
    ANALYSIS_NAME : str
        'apply_amm'
    """

    ANALYSIS_KEY = "applyamm"
    ANALYSIS_NAME = "apply_amm"

    def is_enabled(self) -> bool:
        """True if ``cfg._do_AMM`` is True."""
        return getattr(self.cfg, "_do_AMM", False)

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fit AMM to the task recording and apply it to every recording.

        Parameters
        ----------
        data : dict
            Dictionary from load_data() with raw data per task.

        Returns
        -------
        results : dict
            Dictionary with the AMM-processed raw data per task.
        """
        cfg = self.cfg
        corr_lim = getattr(cfg, "_amm_corr_lim", 1.0)
        window = getattr(cfg, "_amm_window", 10.0)
        kwargs = dict(
            int_order=getattr(cfg, "_amm_li", 9),
            ext_order=getattr(cfg, "_amm_le", 2),
            coord_frame=getattr(cfg, "_amm_coord_frame", "head"),
            center_basis=getattr(cfg, "_amm_center_basis", False),
        )
        task = data[cfg.task]
        projs = compute_proj_amm(task.info, **kwargs)
        raw = apply_amm(
            task,
            st_duration=window if corr_lim < 1 else None,
            st_correlation=corr_lim,
            copy=False,
            **kwargs,
        )
        n_chan = len(projs[0]["data"]["col_names"]) if projs else 0
        self.log(
            f"AMM int_order={kwargs['int_order']} ext_order={kwargs['ext_order']}: "
            f"keeps {n_chan - len(projs)} of {n_chan} dimensions; temporal extension "
            + (f"corr_lim={corr_lim}, window={window} s" if corr_lim < 1 else "off")
        )
        results: Dict[str, Any] = {"bads": [], cfg.task: raw}
        if data.get("noise") is not None:
            results["noise"] = data["noise"].add_proj(projs).apply_proj()
            self.log("Applied the task's AMM projectors to the noise recording")
        return results


def run(cfg: SimpleNamespace) -> None:
    """Module entry point for CLI dispatcher.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration object.
    """
    analysis = ApplyAMMAnalysis(cfg)

    if not analysis.is_enabled():
        print(f"\n[{analysis.ANALYSIS_NAME}] Disabled in configuration; exiting")
        return

    analysis.execute()
