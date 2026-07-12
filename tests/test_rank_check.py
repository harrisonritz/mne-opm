"""Tests for the rank_check CLI's pure helpers.

Covers the file-discovery and parsing logic that decides which derivative
FIF is ranked for a given proc stage:

1. ``normalize_proc_tag`` accepts both ``proc-<tag>`` and bare ``<tag>``.
2. ``is_primary_split`` keeps the primary FIF split and drops secondaries.
3. ``extract_run`` recovers the run label (task) and returns None (noise).
4. ``find_primary_raws`` globs exactly the primary raw(s) for a task+stage,
   respecting the ``proc-<tag>`` boundary (``proc-ica`` != ``proc-icafit``)
   and excluding secondary splits and epoch files.
5. ``append_tsv`` creates a per-subject TSV with a header and appends rows.

These exercise the logic that has no MNE/data dependency; the rank
computation itself is a thin wrapper over ``mne.compute_rank``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from custom import rank_check


# ---------------------------------------------------------------------------
# normalize_proc_tag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("proc-filt", "filt"),
        ("filt", "filt"),
        ("  proc-sss  ", "sss"),
        ("proc-clean", "clean"),
    ],
)
def test_normalize_proc_tag(raw, expected):
    assert rank_check.normalize_proc_tag(raw) == expected


# ---------------------------------------------------------------------------
# is_primary_split
# ---------------------------------------------------------------------------


def test_is_primary_split():
    assert rank_check.is_primary_split(
        "sub-007_ses-01_task-TSX_run-01_proc-filt_split-01_raw.fif"
    )
    assert not rank_check.is_primary_split(
        "sub-007_ses-01_task-TSX_run-01_proc-filt_split-02_raw.fif"
    )
    # No split entity at all is still primary.
    assert rank_check.is_primary_split("sub-007_ses-01_task-noise_proc-filt_raw.fif")


# ---------------------------------------------------------------------------
# extract_run
# ---------------------------------------------------------------------------


def test_extract_run():
    assert (
        rank_check.extract_run("sub-007_ses-01_task-TSX_run-01_proc-filt_raw.fif")
        == "run-01"
    )
    assert (
        rank_check.extract_run("sub-007_ses-01_task-noise_proc-filt_raw.fif") is None
    )


# ---------------------------------------------------------------------------
# find_primary_raws
# ---------------------------------------------------------------------------


def _touch(d: Path, name: str) -> Path:
    p = d / name
    p.touch()
    return p


def test_find_primary_raws_task_and_splits(tmp_path):
    meg = tmp_path / "sub-007" / "ses-01" / "meg"
    meg.mkdir(parents=True)

    # Primary + secondary split of the same task run, plus decoys.
    _touch(meg, "sub-007_ses-01_task-TSX_run-01_proc-filt_split-01_raw.fif")
    _touch(meg, "sub-007_ses-01_task-TSX_run-01_proc-filt_split-02_raw.fif")
    _touch(meg, "sub-007_ses-01_task-TSX_run-01_proc-filt_epo.fif")  # not raw
    _touch(meg, "sub-007_ses-01_task-noise_proc-filt_raw.fif")  # different task

    found = rank_check.find_primary_raws(meg, "007", "01", "TSX", "filt")
    names = [p.name for p in found]
    assert names == ["sub-007_ses-01_task-TSX_run-01_proc-filt_split-01_raw.fif"]


def test_find_primary_raws_tag_boundary(tmp_path):
    """proc-ica must not match proc-icafit."""
    meg = tmp_path / "sub-009" / "ses-01" / "meg"
    meg.mkdir(parents=True)
    ica = _touch(meg, "sub-009_ses-01_task-TSX_proc-ica_raw.fif")
    _touch(meg, "sub-009_ses-01_task-TSX_proc-icafit_raw.fif")

    found = rank_check.find_primary_raws(meg, "009", "01", "TSX", "ica")
    assert found == [ica]


def test_find_primary_raws_noise_no_run(tmp_path):
    meg = tmp_path / "sub-007" / "ses-01" / "meg"
    meg.mkdir(parents=True)
    noise = _touch(meg, "sub-007_ses-01_task-noise_proc-clean_raw.fif")

    found = rank_check.find_primary_raws(meg, "007", "01", "noise", "clean")
    assert found == [noise]


def test_find_primary_raws_missing_dir(tmp_path):
    # Absent meg dir -> empty list, never raises.
    assert rank_check.find_primary_raws(tmp_path / "nope", "007", "01", "TSX", "filt") == []


# ---------------------------------------------------------------------------
# append_tsv
# ---------------------------------------------------------------------------


def test_append_tsv_creates_header_and_appends(tmp_path):
    deriv_root = tmp_path / "derivatives" / "mymodel"
    cfg = SimpleNamespace(deriv_root=str(deriv_root))
    rows = [
        {
            "kind": "task run-01",
            "run": "run-01",
            "path": Path("sub-007_ses-01_task-TSX_run-01_proc-filt_raw.fif"),
            "n_ch": 144,
            "n_bad": 2,
            "rank_data": 86,
            "rank_info": 120,
        },
        {
            "kind": "noise",
            "run": "",
            "path": Path("sub-007_ses-01_task-noise_proc-filt_raw.fif"),
            "n_ch": 148,
            "n_bad": 1,
            "rank_data": 88,
            "rank_info": 120,
        },
    ]

    tsv = rank_check.append_tsv(cfg, "007", "01", "filt", rows)
    assert tsv.exists()
    lines = tsv.read_text().strip().splitlines()
    # header + 2 rows
    assert lines[0].split("\t") == rank_check.TSV_FIELDS
    assert len(lines) == 3

    # A second call appends without re-writing the header.
    rank_check.append_tsv(cfg, "007", "01", "sss", rows)
    lines = tsv.read_text().strip().splitlines()
    assert len(lines) == 5
    assert lines[0].split("\t") == rank_check.TSV_FIELDS
    # model name is derived from the deriv_root folder name
    assert "mymodel" in lines[1]
