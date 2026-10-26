"""Spectral-shape metrics shared by bad-channel and ICA component detection.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import kurtosis


def log10_power(psd: np.ndarray) -> np.ndarray:
    """``log10`` of a power spectrum, with zeros floored at the smallest float.

    No fixed offset is added: sensor spectra in T²/Hz sit near 1e-27, so an
    offset such as 1e-20 would replace the spectrum with a constant.
    """
    return np.log10(np.maximum(np.asarray(psd, dtype=float), np.finfo(float).tiny))


def fit_power_law(freqs: np.ndarray, psd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a ``1/f^n`` baseline to each spectrum in log-log space.

    Parameters
    ----------
    freqs : np.ndarray
        Frequencies (Hz), shape ``(n_freqs,)``; all must be positive.  Select
        the fit band (and drop any bins to ignore) before calling.
    psd : np.ndarray
        Power spectra, shape ``(n_items, n_freqs)``.

    Returns
    -------
    slope : np.ndarray
        Log-log slope per item, shape ``(n_items,)``.  Near 0 for a flat
        (white) spectrum, more negative for a steeper 1/f rolloff.
    resid_kurtosis : np.ndarray
        Fisher kurtosis of the residuals from the fitted line, shape
        ``(n_items,)``.  Narrow-band peaks above the baseline make the
        residuals heavy-tailed, so the kurtosis is high.
    """
    freqs = np.asarray(freqs, dtype=float)
    if np.any(freqs <= 0):
        raise ValueError("fit_power_law needs positive frequencies")
    log_freqs = np.log10(freqs)
    log_psd = log10_power(psd)
    X = np.column_stack([log_freqs, np.ones_like(log_freqs)])
    coeffs = np.linalg.lstsq(X, log_psd.T, rcond=None)[0]  # (2, n_items)
    residuals = log_psd - (X @ coeffs).T  # (n_items, n_freqs)
    return coeffs[0], kurtosis(residuals, axis=1, fisher=True)
