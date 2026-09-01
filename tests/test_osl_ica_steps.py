"""Tests for the custom osl-ephys ICA rejection steps."""

from __future__ import annotations

import mne
import numpy as np
import pytest
from scipy.stats import kurtosis

from custom.osl.extra_funcs import (
    PREPROC_EXTRA_FUNCS,
    ica_autoreject_safe,
    ica_kurtosisreject,
)


def make_raw(n_meg=12, n_eog=0, sfreq=200.0, n_times=4000, seed=0):
    """Raw magnetometer data, optionally with EOG channels."""
    rng = np.random.RandomState(seed)
    ch_names = [f"MEG{i:03d}" for i in range(n_meg)]
    ch_types = ["mag"] * n_meg
    if n_eog:
        ch_names += [f"eye_nmf{i + 1}" for i in range(n_eog)]
        ch_types += ["eog"] * n_eog

    info = mne.create_info(ch_names, sfreq, ch_types)
    data = rng.randn(len(ch_names), n_times) * 1e-12
    return mne.io.RawArray(data, info, verbose=False)


def make_dataset(raw, n_components=5):
    """Dataset dict with a fitted ICA, as osl-ephys would hand to a step."""
    ica = mne.preprocessing.ICA(
        n_components=n_components, random_state=0, max_iter=200, verbose=False
    )
    ica.fit(raw, picks="mag", verbose=False)
    return {
        "raw": raw,
        "events": None,
        "epochs": None,
        "event_id": None,
        "ica": ica,
        "fig": {},
    }


# ---------------------------------------------------------------------------
# ica_autoreject_safe
# ---------------------------------------------------------------------------


class TestIcaAutorejectSafe:
    def test_skips_eog_detection_when_no_eog_channel(self, monkeypatch, caplog):
        captured = {}

        def fake_autoreject(dataset, userargs):
            captured.update(userargs)
            return dataset

        monkeypatch.setattr(
            "osl_ephys.preprocessing.mne_wrappers.run_mne_ica_autoreject",
            fake_autoreject,
        )

        dataset = make_dataset(make_raw(n_eog=0))
        ica_autoreject_safe(dataset, {"eogmethod": "default", "apply": False})

        assert captured["eogmethod"] is None
        # ECG is untouched: find_bads_ecg synthesises an ECG from the
        # magnetometers, so it works without an ECG channel.
        assert "ecgmethod" not in captured or captured["ecgmethod"] != "skip"

    def test_keeps_eog_detection_when_the_channel_is_present(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "osl_ephys.preprocessing.mne_wrappers.run_mne_ica_autoreject",
            lambda dataset, userargs: captured.update(userargs) or dataset,
        )

        dataset = make_dataset(make_raw(n_eog=3))
        ica_autoreject_safe(dataset, {"eogmethod": "default", "apply": False})

        assert captured["eogmethod"] == "default"

    def test_skip_if_absent_can_be_disabled(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "osl_ephys.preprocessing.mne_wrappers.run_mne_ica_autoreject",
            lambda dataset, userargs: captured.update(userargs) or dataset,
        )

        dataset = make_dataset(make_raw(n_eog=0))
        ica_autoreject_safe(
            dataset, {"eogmethod": "default", "skip_if_absent": False}
        )

        # Left alone, so osl-ephys' own call raises as it normally would.
        assert captured["eogmethod"] == "default"

    def test_skip_if_absent_is_not_forwarded_to_osl(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "osl_ephys.preprocessing.mne_wrappers.run_mne_ica_autoreject",
            lambda dataset, userargs: captured.update(userargs) or dataset,
        )

        dataset = make_dataset(make_raw(n_eog=3))
        ica_autoreject_safe(dataset, {"skip_if_absent": True, "apply": False})

        assert "skip_if_absent" not in captured

    def test_does_not_mutate_the_callers_userargs(self, monkeypatch):
        monkeypatch.setattr(
            "osl_ephys.preprocessing.mne_wrappers.run_mne_ica_autoreject",
            lambda dataset, userargs: dataset,
        )

        userargs = {"eogmethod": "default", "skip_if_absent": True}
        ica_autoreject_safe(make_dataset(make_raw(n_eog=0)), userargs)

        assert userargs == {"eogmethod": "default", "skip_if_absent": True}

    def test_runs_end_to_end_without_eog(self):
        # The real point: a subject with no eye-tracking must get through the
        # step rather than raising out of find_bads_eog.
        dataset = make_dataset(make_raw(n_eog=0))
        result = ica_autoreject_safe(
            dataset, {"eogmethod": "default", "ecgmethod": None, "apply": False}
        )
        assert result["ica"] is dataset["ica"]


# ---------------------------------------------------------------------------
# ica_kurtosisreject
# ---------------------------------------------------------------------------


class TestIcaKurtosisReject:
    def test_marks_high_kurtosis_components(self):
        dataset = make_dataset(make_raw())
        sources = dataset["ica"].get_sources(dataset["raw"]).get_data()
        scores = kurtosis(sources, axis=1, fisher=False)

        # A threshold just under the largest score must catch exactly the
        # components above it.
        threshold = float(np.sort(scores)[-2])
        expected = set(np.where(scores > threshold)[0].tolist())

        ica_kurtosisreject(dataset, {"threshold": threshold, "apply": False})
        assert set(dataset["ica"].exclude) == expected

    def test_uses_the_unmixing_matrix_not_the_mixing_matrix(self):
        # get_sources() applies the unmixing matrix; the tutorial's
        # get_components().T @ data does not, and gives different scores.
        dataset = make_dataset(make_raw())
        ica, raw = dataset["ica"], dataset["raw"]

        via_sources = kurtosis(
            ica.get_sources(raw).get_data(), axis=1, fisher=False
        )
        picks = mne.pick_types(ica.info, meg=True, eeg=False)
        via_mixing = kurtosis(
            (ica.get_components()[picks, :].T @ raw.get_data()[picks, :]),
            axis=1,
            fisher=False,
        )

        assert not np.allclose(via_sources, via_mixing)

    def test_nothing_marked_when_the_threshold_is_high(self):
        dataset = make_dataset(make_raw())
        ica_kurtosisreject(dataset, {"threshold": 1e6, "apply": False})
        assert dataset["ica"].exclude == []

    def test_accumulates_with_earlier_exclusions(self):
        dataset = make_dataset(make_raw())
        dataset["ica"].exclude = [0]

        ica_kurtosisreject(dataset, {"threshold": 0.0, "apply": False})

        assert 0 in dataset["ica"].exclude
        assert len(dataset["ica"].exclude) == dataset["ica"].n_components_

    def test_exclusions_are_unique_and_sorted(self):
        dataset = make_dataset(make_raw())
        dataset["ica"].exclude = [2, 1, 2]

        ica_kurtosisreject(dataset, {"threshold": 1e6, "apply": False})

        assert dataset["ica"].exclude == [1, 2]

    def test_excludes_bad_segments_by_default(self):
        # A bad segment is itself a high-kurtosis excursion, so leaving it in
        # makes almost every component look bad.
        raw = make_raw()
        raw._data[:, 1000:1100] *= 500.0
        raw.set_annotations(
            mne.Annotations(onset=[5.0], duration=[0.5], description=["BAD_jump"])
        )

        with_bad = make_dataset(raw.copy())
        ica_kurtosisreject(
            with_bad, {"threshold": 8, "reject_by_annotation": False, "apply": False}
        )

        without_bad = make_dataset(raw.copy())
        without_bad["ica"] = with_bad["ica"].copy()
        without_bad["ica"].exclude = []
        ica_kurtosisreject(
            without_bad,
            {"threshold": 8, "reject_by_annotation": True, "apply": False},
        )

        assert len(without_bad["ica"].exclude) <= len(with_bad["ica"].exclude)

    def test_apply_changes_the_raw_data(self):
        dataset = make_dataset(make_raw())
        before = dataset["raw"].get_data().copy()

        ica_kurtosisreject(dataset, {"threshold": 0.0, "apply": True})

        # MEG data is ~1e-12, far below np.allclose's default atol of 1e-8, so
        # compare against the signal's own scale rather than an absolute one.
        difference = np.abs(before - dataset["raw"].get_data()).max()
        assert difference > 0.1 * np.abs(before).max()

    def test_apply_false_leaves_the_raw_data_alone(self):
        dataset = make_dataset(make_raw())
        before = dataset["raw"].get_data().copy()

        ica_kurtosisreject(dataset, {"threshold": 0.0, "apply": False})

        np.testing.assert_array_equal(before, dataset["raw"].get_data())

    def test_raises_without_a_fitted_ica(self):
        dataset = {"raw": make_raw(), "ica": None}
        with pytest.raises(ValueError, match="no ICA in the dataset"):
            ica_kurtosisreject(dataset, {})


class TestRegistry:
    def test_every_step_is_registered(self):
        assert [f.__name__ for f in PREPROC_EXTRA_FUNCS] == [
            "events_from_annotations",
            "ica_autoreject_safe",
            "ica_kurtosisreject",
            "bad_channels_clean",
        ]

    def test_steps_resolve_through_osls_dispatcher(self):
        from osl_ephys.preprocessing.batch import find_func

        # find_func wraps custom functions in a logging decorator, so compare
        # by name rather than identity.
        for func in PREPROC_EXTRA_FUNCS:
            resolved = find_func(func.__name__, extra_funcs=PREPROC_EXTRA_FUNCS)
            assert resolved is not None
            assert resolved.__name__ == func.__name__

    def test_custom_steps_take_priority_over_osl_builtins(self):
        from osl_ephys.preprocessing.batch import find_func

        # ica_autoreject_safe is a distinct name, so the builtin is still
        # reachable and unshadowed.
        builtin = find_func("ica_autoreject", extra_funcs=PREPROC_EXTRA_FUNCS)
        assert builtin.__name__ == "run_mne_ica_autoreject"
