"""Behavioral rows are aligned to trial triggers by their inter-trial intervals.

:func:`reconcile_trial_metadata` places surplus behavioral rows (trials with no
trigger) where the trigger and behavioral ISIs say they belong, and verifies
the ISI agreement in every case.
"""

import numpy as np
import pandas as pd
import pytest

from custom.preprocessing._io import reconcile_trial_metadata

CLOCK = 5000.0  # behavioral onsets run on a different clock from the triggers


def _log(n, seed=0):
    """``n`` behavioral rows with jittered ITIs; returns (rows, onsets)."""
    rng = np.random.default_rng(seed)
    onsets = np.cumsum(rng.uniform(1.0, 5.0, n))
    rows = pd.DataFrame({"trial": np.arange(n), "stim_start_time": onsets + CLOCK})
    return rows, onsets


def _reconcile(meta_dfs, onsets, **kwargs):
    kwargs.setdefault("policy", "align")
    return reconcile_trial_metadata(meta_dfs, onsets, **kwargs)


def test_matching_counts_are_verified_and_kept():
    rows, onsets = _log(30)
    metadata, alignment = _reconcile([rows], onsets, policy="error")
    assert metadata["trial"].tolist() == list(range(30))
    assert alignment.dropped_rows == ()
    assert (alignment.n_isi_agree, alignment.n_isi) == (29, 29)
    assert alignment.isi_agreement == 1.0


@pytest.mark.parametrize("dropped", [(0, 1), (13, 14, 15), (27, 28, 29)])
def test_surplus_rows_are_located_by_timing(dropped):
    # Start of the log (recording started late), the middle (a restart), or
    # the end: the rows without a trigger are the ones dropped.
    rows, onsets = _log(30)
    kept = [i for i in range(30) if i not in dropped]
    metadata, alignment = _reconcile([rows], onsets[kept])
    assert metadata["trial"].tolist() == kept
    assert alignment.dropped_rows == dropped
    assert alignment.isi_agreement == 1.0


def test_surplus_at_a_restart_boundary_spans_files():
    rows, onsets = _log(30)
    first, second = rows.iloc[:12], rows.iloc[12:].reset_index(drop=True)
    kept = [i for i in range(30) if i not in (10, 11)]
    metadata, alignment = _reconcile([first, second], onsets[kept])
    assert metadata["trial"].tolist() == kept
    assert alignment.dropped_rows == (10, 11)


def test_clipped_first_trigger_prefers_the_start_of_the_log():
    # The recording started mid-trial: the first trigger's onset is clipped, so
    # dropping rows 0-1 and dropping rows 1-2 align the other ISIs equally.
    rows, onsets = _log(30)
    triggers = onsets[2:].copy()
    triggers[0] += 0.4
    metadata, alignment = _reconcile([rows], triggers)
    assert alignment.dropped_rows == (0, 1)
    assert metadata["trial"].tolist() == list(range(2, 30))
    assert (alignment.n_isi_agree, alignment.n_isi) == (26, 27)


def test_unrelated_timing_is_refused():
    rows, _ = _log(30, seed=0)
    _, other = _log(30, seed=1)
    with pytest.raises(RuntimeError, match="inter-trial intervals agree"):
        _reconcile([rows], other)


def test_ambiguous_surplus_location_is_refused():
    # Constant ITIs: every block position aligns the intervals equally well.
    onsets = np.arange(20) * 2.0
    rows = pd.DataFrame({"trial": np.arange(22), "stim_start_time": np.arange(22) * 2.0})
    with pytest.raises(RuntimeError, match="equally well"):
        _reconcile([rows], onsets)


def test_error_policy_refuses_any_surplus():
    rows, onsets = _log(10)
    with pytest.raises(RuntimeError, match="1 more rows"):
        _reconcile([rows], onsets[1:], policy="error")


def test_large_or_negative_mismatches_are_refused():
    rows, onsets = _log(40)
    with pytest.raises(RuntimeError, match="25 more rows"):
        _reconcile([rows], onsets[:15])
    with pytest.raises(RuntimeError, match="1 fewer rows"):
        _reconcile([rows.iloc[:39]], onsets)


def test_missing_onset_column_is_refused():
    rows = [pd.DataFrame({"trial": [1, 2, 3]})]
    with pytest.raises(ValueError, match="stim_start_time"):
        _reconcile(rows, [0.0, 1.0, 2.0])


def test_unknown_policy_is_refused():
    rows, onsets = _log(5)
    with pytest.raises(ValueError, match="Unknown metadata mismatch policy"):
        _reconcile([rows], onsets, policy="trim_restart")
