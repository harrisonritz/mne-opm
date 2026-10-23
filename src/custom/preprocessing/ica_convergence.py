"""Check whether the pipeline ICA fit converged.

mne-bids-pipeline fits ICA with a fixed iteration limit
(``ica_max_iterations``) and only logs the number of iterations used. This
step reads the saved ICA solution, reports ``ICA.converged_`` and
``ICA.n_iter_``, and saves a figure of iterations used against the limit
next to the other ICA figures. A non-converged fit raises a warning, or an
error when ``_ica_fail_on_nonconvergence`` is True.

Usage
-----
CLI:
    python src/custom/custom_preproc.py --analysis=ica_convergence --config=/path/to/config.py

Configuration Attributes
------------------------
Required:
    deriv_root, subjects, sessions, task, spatial_filter

Optional:
    _ica_fail_on_nonconvergence : bool
        Raise instead of warning when ICA did not converge. Default: False.

Notes
-----
``converged_`` is stored with the ICA solution by MNE-Python >= 1.14 (dev).
It is ``None`` for solutions saved by older versions, which this step
reports as unknown. For Picard, MNE infers convergence from whether the fit
stopped before the iteration limit.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any, Dict

import mne

from ._base import BaseAnalysis
from ._io import get_ica_bids_path


class ICAConvergenceAnalysis(BaseAnalysis):
    """Report and plot the convergence of the pipeline ICA fit.

    Attributes
    ----------
    ANALYSIS_KEY : str
        'icaconvergence'
    ANALYSIS_NAME : str
        'ica_convergence'
    """

    ANALYSIS_KEY = "icaconvergence"
    ANALYSIS_NAME = "ica_convergence"

    def is_enabled(self) -> bool:
        """Run whenever the pipeline uses ICA."""
        return getattr(self.cfg, "spatial_filter", None) == "ica"

    def load_data(self) -> Dict[str, Any]:
        """Load the saved ICA solution."""
        bp = get_ica_bids_path(self.cfg)
        return {"ica": mne.preprocessing.read_ica(bp.fpath), "bids_path": bp}

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect the convergence status of the fit."""
        ica = data["ica"]
        # read_ica restores the limit used for the fit in fit_params only
        max_iter = ica.fit_params.get("max_iter", ica.max_iter)
        converged = getattr(ica, "converged_", None)
        status = {True: "converged", False: "NOT converged", None: "unknown"}
        self.log(
            f"{ica.method}: {status[converged]} after {ica.n_iter_} of "
            f"{max_iter} iterations"
        )
        return dict(data, max_iter=max_iter, converged=converged)

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save the figure, then warn or raise if the fit did not converge."""
        import matplotlib.pyplot as plt

        ica, bp = results["ica"], results["bids_path"]
        max_iter, converged = results["max_iter"], results["converged"]
        out_dir = bp.directory / "ICA"
        out_dir.mkdir(exist_ok=True)
        fname = out_dir / f"{bp.fpath.stem}_convergence.png"

        color = {True: "tab:green", False: "tab:red", None: "tab:gray"}[converged]
        status = {True: "converged", False: "not converged", None: "unknown"}
        fig, ax = plt.subplots(figsize=(6, 1.8), layout="constrained")
        ax.barh([0], [ica.n_iter_], color=color)
        ax.axvline(max_iter, color="k", ls="--", label=f"max_iter = {max_iter}")
        ax.set(
            xlim=(0, max_iter * 1.05),
            yticks=[],
            xlabel="Iterations",
            title=f"ICA ({ica.method}): {status[converged]}, {ica.n_iter_} iterations",
        )
        ax.legend(loc="lower right")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        self.log(f"Saved {fname}")

        if converged is False:
            msg = (
                f"ICA ({ica.method}) did not converge after {ica.n_iter_} of "
                f"{max_iter} iterations; consider increasing ica_max_iterations."
            )
            if getattr(self.cfg, "_ica_fail_on_nonconvergence", False):
                raise RuntimeError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)


def run(cfg: SimpleNamespace) -> None:
    """Module entry point for CLI dispatcher.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration object.
    """
    analysis = ICAConvergenceAnalysis(cfg)

    if not analysis.is_enabled():
        print(f"\n[{analysis.ANALYSIS_NAME}] Disabled in configuration; exiting")
        return

    analysis.execute()
