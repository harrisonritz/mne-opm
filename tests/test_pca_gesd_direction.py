"""PC-tail direction must respect the arbitrary sign of PCA loadings."""

import numpy as np

from custom.preprocessing.pca_gesd import MetricSpec, run_pca_gesd


def test_high_side_outlier_with_negative_loading_is_found():
    values = np.random.default_rng(0).normal(size=40)
    values[0] = 12.0

    result = run_pca_gesd([MetricSpec("high_bad", values, 1)], p_out=0.5)

    assert result.loadings[0, 0] < 0  # Regression case: SVD flips the PC axis.
    assert result.pc_sides.tolist() == [-1]
    assert result.flagged[0]


def test_mixed_metric_directions_use_both_pc_tails():
    values = np.random.default_rng(1).normal(size=40)
    result = run_pca_gesd(
        [
            MetricSpec("upper_bad", values, 1),
            MetricSpec("lower_bad", values, -1),
        ],
        n_pcs=1,
    )

    assert result.pc_sides.tolist() == [0]
