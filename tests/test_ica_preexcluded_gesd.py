"""Pre-excluded ICA components can be included in or omitted from PCA-GESD."""

from types import SimpleNamespace

import numpy as np

from custom.preprocessing.bad_ICs import BadICAnalysis
from custom.preprocessing.pca_gesd import MetricSpec


def _analysis(tmp_path, include_preexcluded=None):
    cfg = SimpleNamespace(
        deriv_root=str(tmp_path),
        subjects=["001"],
        sessions=["01"],
        task="example",
    )
    if include_preexcluded is not None:
        cfg._bad_ICs_include_preexcluded = include_preexcluded
    return BadICAnalysis(cfg)


def _scores():
    values = np.random.default_rng(0).normal(size=40)
    values[0] = 15.0  # Already excluded, but still has a spatial score.
    values[5] = 8.0  # New outlier; mapping must retain its original IC index.
    return [MetricSpec("spatial_kurtosis", values, 1)]


def test_default_keeps_preexcluded_in_gesd(tmp_path):
    analysis = _analysis(tmp_path)
    ica = SimpleNamespace(n_components_=40, exclude=[0])

    ica, result = analysis._run_unified_gesd(ica, None, _scores())

    assert result.item_indices.tolist() == list(range(40))
    assert result.flagged[0]
    assert {0, 5}.issubset(ica.exclude)


def test_omit_preexcluded_preserves_original_ids_and_raw_scores(tmp_path):
    analysis = _analysis(tmp_path, include_preexcluded=False)
    ica = SimpleNamespace(n_components_=40, exclude=[0])

    ica, result = analysis._run_unified_gesd(ica, None, _scores())
    analysis._gesd_result = result
    components = analysis._build_components_tsv(ica)

    assert result.item_indices.tolist() == list(range(1, 40))
    assert result.n_items == 39
    assert result.M.shape == (1, 39)
    assert 5 in ica.exclude
    assert analysis._component_labels[5] == ["GESD_PC1"]
    assert components.loc[0, "score_spatial_kurtosis"] == 15.0
    assert np.isnan(components.loc[0, "gesd_score_PC1"])
    assert components.loc[5, "method_gesd"] == 1


def test_full_ica_labeling_maps_subset_flags_to_original_ids(tmp_path, monkeypatch):
    analysis = _analysis(tmp_path, include_preexcluded=False)
    analysis.cfg._bad_ICs_overlay = False
    ica = SimpleNamespace(n_components_=40, exclude=[0])
    monkeypatch.setattr(analysis, "_compute_ic_scores", lambda ica, raw: _scores())

    labeled = analysis._bad_ICs(ica, None)

    assert labeled.exclude == [0, 5]
    assert analysis._gesd_result.item_indices[4] == 5
