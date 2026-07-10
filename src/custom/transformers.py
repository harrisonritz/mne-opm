"""Reusable scikit-learn transformers for MEG decoding pipelines.

Classes defined here are importable from any script or worker process,
which avoids ``AttributeError`` when ``joblib`` / ``loky`` tries to
unpickle objects in spawned workers.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# MultivariateNoiseNormalizer
# ---------------------------------------------------------------------------


class MultivariateNoiseNormalizer(BaseEstimator, TransformerMixin):
    """Pre-whiten data by a LedoitWolf estimate of the noise covariance.

    Noise is estimated from within-class residuals: each class mean is
    subtracted from training samples before fitting LedoitWolf, so the
    covariance captures noise variance rather than signal variance.

    Handles both 2-D and 3-D inputs:

    * **2-D** ``(n_samples, n_features)`` — used directly.
      Called per time-step inside ``SlidingEstimator``.

    * **3-D** ``(n_samples, n_channels, n_times)`` — time points are pooled
      to estimate a channel-space covariance (n_channels x n_channels),
      then whitening is applied channel-wise at every time step.
      This keeps computation tractable before ``Vectorizer``.

    Implements ``inverse_transform`` so ``get_coef(..., inverse_transform=True)``
    can propagate patterns back to sensor space through the full pipeline.
    """

    def fit(self, X, y=None):
        if X.ndim == 3:
            n_epochs, n_channels, n_times = X.shape
            # Pool (epoch, time) pairs → (n_epochs*n_times, n_channels).
            X_2d = X.transpose(0, 2, 1).reshape(-1, n_channels)
            # Tile labels so every (epoch, time) sample has a class label.
            y_2d = np.tile(y, n_times) if y is not None else None
        else:
            X_2d = X
            y_2d = y

        # Subtract per-class means to isolate noise.
        X_res = X_2d.copy()
        if y_2d is not None:
            for cls in np.unique(y_2d):
                mask = y_2d == cls
                X_res[mask] -= X_res[mask].mean(axis=0)

        lw = LedoitWolf().fit(X_res)
        cov = lw.covariance_

        # Symmetric matrix square root via eigendecomposition.
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-10)  # guard against numerical negatives

        # whitening  : multiply by Σ^{-1/2}
        # "coloring" : multiply by Σ^{+1/2}  (inverse of whitening)
        self.whitening_ = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        self.coloring_ = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        return self

    def transform(self, X, y=None):
        if X.ndim == 3:
            # (n_epochs, n_times, n_channels) @ W → back to (n_epochs, n_channels, n_times)
            return (X.transpose(0, 2, 1) @ self.whitening_).transpose(0, 2, 1)
        return X @ self.whitening_

    def inverse_transform(self, X):
        if X.ndim == 3:
            return (X.transpose(0, 2, 1) @ self.coloring_).transpose(0, 2, 1)
        return X @ self.coloring_


# ---------------------------------------------------------------------------
# FlexPCA
# ---------------------------------------------------------------------------


class FlexPCA(PCA):
    """PCA that caps n_components at min(n_samples, n_features) during fit.

    When n_components is an integer exceeding the data dimensions (e.g. the
    data rank is larger than the number of CV-fold samples), it is silently
    reduced to min(n_samples, n_features) so the fit never raises.
    Float values (variance-ratio mode) are left untouched.
    """

    def _clamp(self, X):
        max_comp = min(X.shape)
        if (
            isinstance(self.n_components, (int, np.integer))
            and self.n_components > max_comp
        ):
            self.n_components = max_comp

    def fit(self, X, y=None):
        self._clamp(X)
        return super().fit(X, y)

    def fit_transform(self, X, y=None):
        self._clamp(X)
        return super().fit_transform(X, y)
