"""Spectral-shape helpers and the scale-free log transform."""

import numpy as np
import pytest

from custom.preprocessing._spectral import fit_power_law, log10_power
from custom.preprocessing.pca_gesd import log_transform


def test_log_transform_ignores_units():
    """Rescaling the input only shifts the log: no fixed offset swamps it."""
    x = np.linspace(1.0, 3.0, 20)
    for scale in (1.0, 1e-12, 1e6):
        np.testing.assert_allclose(
            log_transform(scale * x) - np.log(scale), np.log(x), atol=1e-5
        )


def test_log_transform_keeps_zero_finite_and_lowest():
    out = log_transform(np.array([0.0, 1e-12, 2e-12]))
    assert np.isfinite(out).all()
    assert out[0] < out[1] - 10


def test_log10_power_has_no_offset_at_tesla_scale():
    psd = np.array([[1e-27, 1e-26], [0.0, 1e-27]])
    out = log10_power(psd)
    np.testing.assert_allclose(out[0], [-27.0, -26.0])
    assert np.isfinite(out[1, 0])


def test_fit_power_law_recovers_slope_and_flags_peak():
    freqs = np.arange(2.0, 80.0, 0.5)
    pink = 1e-26 / freqs  # slope -1
    white = np.full_like(freqs, 1e-27)  # slope 0
    peaked = pink.copy()
    peaked[np.abs(freqs - 23.0) < 1.0] *= 50.0
    slope, resid_kurtosis = fit_power_law(freqs, np.vstack([pink, white, peaked]))
    np.testing.assert_allclose(slope[:2], [-1.0, 0.0], atol=1e-10)
    assert resid_kurtosis[2] > 5.0


def test_fit_power_law_rejects_zero_frequency():
    with pytest.raises(ValueError):
        fit_power_law(np.arange(0.0, 10.0), np.ones((1, 10)))
