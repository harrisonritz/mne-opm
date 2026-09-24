"""Behavioral row trimming is opt-in and restricted to small surpluses."""

import pandas as pd
import pytest

from custom.preprocessing._io import reconcile_trial_metadata


def test_metadata_mismatch_fails_by_default():
    rows = [pd.DataFrame({"trial": [1, 2, 3]})]

    with pytest.raises(RuntimeError, match="refusing an unverified positional join"):
        reconcile_trial_metadata(rows, 2)


def test_known_restart_trim_removes_end_of_first_file():
    rows = [
        pd.DataFrame({"trial": [1, 2, 3]}),
        pd.DataFrame({"trial": [4, 5]}),
    ]

    metadata, location = reconcile_trial_metadata(rows, 4, policy="trim_restart")

    assert metadata["trial"].tolist() == [1, 2, 4, 5]
    assert location == "end of first metadata file (restart boundary)"


def test_restart_trim_rejects_large_or_negative_mismatches():
    rows = [pd.DataFrame({"trial": list(range(25))})]

    with pytest.raises(RuntimeError, match="25 more rows"):
        reconcile_trial_metadata(rows, 0, policy="trim_restart")
    with pytest.raises(RuntimeError, match="1 fewer rows"):
        reconcile_trial_metadata(rows, 26, policy="trim_restart")
