"""ICA file routing for shared pipeline fits and legacy task-specific fits."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from custom.preprocessing._io import get_ica_bids_path
from custom.preprocessing.bad_ICs import BadICAnalysis
from custom.preprocessing.manual_ica import ManualICAAnalysis


@pytest.fixture()
def cfg(tmp_path):
    return SimpleNamespace(
        deriv_root=tmp_path, subjects=["007"], sessions=["01"], task="TSX"
    )


@pytest.mark.parametrize("layout", ["shared", "legacy"])
@pytest.mark.parametrize("analysis_cls", [BadICAnalysis, ManualICAAnalysis])
def test_loads_existing_ica(cfg, layout, analysis_cls):
    path = get_ica_bids_path(cfg)
    if layout == "legacy":
        path.update(task=cfg.task)
    path.directory.mkdir(parents=True)
    path.fpath.touch()
    processing = "filt" if analysis_cls is BadICAnalysis else "clean"
    raw_path = path.copy().update(
        task=cfg.task, run="01", processing=processing, suffix="raw"
    )
    fake_raw, fake_ica = MagicMock(), MagicMock()
    analysis = analysis_cls(cfg)
    module = analysis_cls.__module__
    with (
        patch(f"{module}.find_matching_paths", return_value=[raw_path]),
        patch(f"{module}.mne.io.read_raw_fif", return_value=fake_raw),
        patch(f"{module}.mne.preprocessing.read_ica", return_value=fake_ica) as read,
    ):
        data = analysis.load_data()
    read.assert_called_once_with(path.fpath)
    assert data == {cfg.task: fake_raw, "ica": fake_ica}


def test_bad_ICs_scores_filtered_raw_first_split(cfg):
    """Labeling reads pre-ICA (filt) raw: in ICA-cleaned raw, components already
    excluded have zero time courses. Later splits load via the first split."""
    path = get_ica_bids_path(cfg)
    path.directory.mkdir(parents=True)
    path.fpath.touch()
    raw = path.copy().update(task=cfg.task, run="01", processing="filt", suffix="raw")
    splits = [raw.copy().update(split=s) for s in ("01", "02")]
    analysis = BadICAnalysis(cfg)
    module = BadICAnalysis.__module__
    with (
        patch(f"{module}.find_matching_paths", return_value=splits) as find,
        patch(f"{module}.mne.io.read_raw_fif", return_value=MagicMock()) as read_raw,
        patch(f"{module}.mne.preprocessing.read_ica", return_value=MagicMock()),
    ):
        analysis.load_data()
    assert find.call_args.kwargs["processings"] == "filt"
    read_raw.assert_called_once_with(splits[0].fpath, preload=True)


@pytest.mark.parametrize(
    "found, error",
    [
        ([], FileNotFoundError),
        ([("01", "01"), ("01", "02"), ("02", "01")], ValueError),
    ],
)
def test_bad_ICs_requires_one_filtered_run(cfg, found, error):
    raw = get_ica_bids_path(cfg).update(task=cfg.task, processing="filt", suffix="raw")
    paths = [raw.copy().update(run=run, split=split) for run, split in found]
    with patch(f"{BadICAnalysis.__module__}.find_matching_paths", return_value=paths):
        with pytest.raises(error):
            BadICAnalysis(cfg).load_data()


def test_prefers_shared_solution_and_ignores_icafit(cfg):
    path = get_ica_bids_path(cfg)
    path.directory.mkdir(parents=True)
    path.copy().update(task=cfg.task).fpath.touch()
    path.copy().update(processing="icafit").fpath.touch()
    assert get_ica_bids_path(cfg).task == cfg.task
    path.fpath.touch()
    assert get_ica_bids_path(cfg).task is None


def test_retains_shared_pipeline_labels(cfg):
    path = get_ica_bids_path(cfg)
    path.directory.mkdir(parents=True)
    tsv_path = path.copy().update(suffix="components", extension=".tsv")
    pd.DataFrame(
        {
            "component": [0],
            "status_description": ["Auto-detected eye blink (MNE-ICALabel)"],
        }
    ).to_csv(tsv_path.fpath, sep="\t", index=False)
    analysis = BadICAnalysis(cfg)
    analysis._component_labels = {0: []}
    df = analysis._build_components_tsv(SimpleNamespace(n_components_=1, exclude=[0]))
    assert df.loc[0, "method_pipeline_icalabel"] == "eye blink"
    assert df.loc[0, "status"] == "bad"
