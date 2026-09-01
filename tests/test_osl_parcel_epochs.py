"""Tests for parcel Epochs that keep their condition names and time axis.

The defect these guard against is silent: osl-ephys' own converter builds an
Epochs object of the right shape whose ``event_id`` is the stringified codes
MNE synthesises and whose time axis starts at zero, so nothing downstream
raises -- the group stage simply averages nothing.
"""

from __future__ import annotations

import logging

import mne
import numpy as np
import pandas as pd
import pytest

from custom.osl import parcel_epochs


EVENT_ID_FIXTURE = {"response/left": 201, "response/right": 202}

N_PARCELS = 4
N_TIMES = 12
N_EPOCHS = 6
SFREQ = 100.0
TMIN = -0.05


def make_source_epochs(n_epochs=N_EPOCHS, sfreq=SFREQ, tmin=TMIN, n_times=N_TIMES):
    """Sensor-level Epochs standing in for what the beamformer was applied to."""
    rng = np.random.RandomState(0)
    info = mne.create_info(["MEG0", "MEG1", "MEG2"], sfreq, "mag")
    events = np.column_stack(
        [
            np.arange(n_epochs) * n_times + 100,
            np.zeros(n_epochs, int),
            np.tile([201, 202], n_epochs // 2 + 1)[:n_epochs],
        ]
    )
    return mne.EpochsArray(
        rng.randn(n_epochs, 3, n_times) * 1e-12,
        info,
        events=events,
        event_id={"response/left": 201, "response/right": 202},
        tmin=tmin,
        metadata=pd.DataFrame({"rt": np.arange(n_epochs, dtype=float)}),
        verbose=False,
    )


def make_parc_data(n_parcels=N_PARCELS, n_times=N_TIMES, n_epochs=N_EPOCHS, seed=1):
    """(parcels, times, epochs) parcel data, the layout osl-ephys produces."""
    return np.random.RandomState(seed).randn(n_parcels, n_times, n_epochs)


class TestIsPlaceholderEventId:
    def test_stringified_codes_are_a_placeholder(self):
        assert parcel_epochs.is_placeholder_event_id({"201": 201, "202": 202})

    def test_real_names_are_not(self):
        assert not parcel_epochs.is_placeholder_event_id({"response/left": 201})

    def test_a_partly_named_mapping_is_not(self):
        assert not parcel_epochs.is_placeholder_event_id(
            {"response/left": 201, "202": 202}
        )

    def test_an_empty_mapping_has_lost_nothing(self):
        assert not parcel_epochs.is_placeholder_event_id({})


class TestBuildParcelEpochs:
    def test_data_lands_in_mne_epoch_channel_time_order(self):
        parc_data = make_parc_data()
        built = parcel_epochs.build_parcel_epochs(
            parc_data,
            sfreq=SFREQ,
            events=make_source_epochs().events,
            event_id={"response/left": 201, "response/right": 202},
        )
        assert built.get_data(copy=False).shape == (N_EPOCHS, N_PARCELS, N_TIMES)
        # parcel p, time t, epoch e must land at [e, p, t].
        assert built.get_data(copy=False)[3, 2, 5] == pytest.approx(
            parc_data[2, 5, 3]
        )

    def test_channels_default_to_the_parcel_convention(self):
        built = parcel_epochs.build_parcel_epochs(
            make_parc_data(), sfreq=SFREQ, events=make_source_epochs().events
        )
        assert built.ch_names == [f"parcel_{i}" for i in range(N_PARCELS)]
        assert set(built.get_channel_types()) == {"misc"}

    def test_explicit_parcel_names_are_used(self):
        names = [f"Glasser_{i}" for i in range(N_PARCELS)]
        built = parcel_epochs.build_parcel_epochs(
            make_parc_data(),
            sfreq=SFREQ,
            events=make_source_epochs().events,
            parcel_names=names,
        )
        assert built.ch_names == names

    def test_rejects_data_that_is_not_three_dimensional(self):
        with pytest.raises(ValueError, match="parcels, times, epochs"):
            parcel_epochs.build_parcel_epochs(
                np.zeros((N_PARCELS, N_TIMES)),
                sfreq=SFREQ,
                events=make_source_epochs().events,
            )

    def test_rejects_an_events_length_mismatch(self):
        with pytest.raises(ValueError, match="events for"):
            parcel_epochs.build_parcel_epochs(
                make_parc_data(n_epochs=N_EPOCHS - 1),
                sfreq=SFREQ,
                events=make_source_epochs().events,
            )

    def test_rejects_a_parcel_name_count_mismatch(self):
        with pytest.raises(ValueError, match="parcel name"):
            parcel_epochs.build_parcel_epochs(
                make_parc_data(),
                sfreq=SFREQ,
                events=make_source_epochs().events,
                parcel_names=["only_one"],
            )

    def test_drops_conditions_with_no_surviving_epochs(self, caplog):
        # A FIF keeps the event_id it was written with, so a condition whose
        # epochs were all rejected is still named in it. EpochsArray refuses
        # such an entry, which would otherwise fail the whole subject.
        events = make_source_epochs().events
        with caplog.at_level(logging.WARNING, logger=parcel_epochs.__name__):
            built = parcel_epochs.build_parcel_epochs(
                make_parc_data(),
                sfreq=SFREQ,
                events=events,
                event_id={**EVENT_ID_FIXTURE, "feedback": 1, "ITI": 2},
            )

        assert built.event_id == EVENT_ID_FIXTURE
        assert "feedback" in caplog.text and "ITI" in caplog.text

    def test_keeps_every_condition_that_has_epochs(self, caplog):
        with caplog.at_level(logging.WARNING, logger=parcel_epochs.__name__):
            built = parcel_epochs.build_parcel_epochs(
                make_parc_data(),
                sfreq=SFREQ,
                events=make_source_epochs().events,
                event_id=dict(EVENT_ID_FIXTURE),
            )
        assert built.event_id == EVENT_ID_FIXTURE
        assert caplog.text == ""

    def test_does_not_baseline_correct(self):
        parc_data = make_parc_data() + 5.0
        built = parcel_epochs.build_parcel_epochs(
            parc_data, sfreq=SFREQ, events=make_source_epochs().events, tmin=TMIN
        )
        assert built.baseline is None
        assert built.get_data(copy=False).mean() == pytest.approx(parc_data.mean())


class TestConvert2mneEpochs:
    def test_condition_names_survive(self):
        source = make_source_epochs()
        parc = parcel_epochs.convert2mne_epochs(make_parc_data(), source)

        assert parc.event_id == source.event_id
        assert len(parc["response/left"]) == len(source["response/left"])

    def test_the_time_axis_survives(self):
        source = make_source_epochs()
        parc = parcel_epochs.convert2mne_epochs(make_parc_data(), source)

        assert parc.tmin == pytest.approx(source.tmin)
        assert parc.times == pytest.approx(source.times)

    def test_metadata_and_description_survive(self):
        source = make_source_epochs()
        source.info["description"] = "sub-007_ses-01"
        parc = parcel_epochs.convert2mne_epochs(make_parc_data(), source)

        assert parc.info["description"] == "sub-007_ses-01"
        pd.testing.assert_frame_equal(parc.metadata, source.metadata)

    def test_the_sampling_rate_comes_from_the_beamformed_epochs(self):
        # The source stage decimates before beamforming, so the rate must be
        # read from the object handed in, not from the acquisition rate.
        source = make_source_epochs(sfreq=200.0)
        parc = parcel_epochs.convert2mne_epochs(make_parc_data(), source)
        assert parc.info["sfreq"] == pytest.approx(200.0)

    def test_the_data_matches_osl_ephys_own_converter(self):
        # Only the header should differ from what osl-ephys writes.
        from osl_ephys.source_recon import parcellation

        source = make_source_epochs()
        parc_data = make_parc_data()

        ours = parcel_epochs.convert2mne_epochs(parc_data, source)
        theirs = parcellation.convert2mne_epochs(parc_data, source)

        assert ours.get_data(copy=False) == pytest.approx(
            theirs.get_data(copy=False)
        )
        assert ours.ch_names == theirs.ch_names

    def test_osl_ephys_own_converter_loses_what_this_module_keeps(self):
        # The regression under test: if osl-ephys ever fixes this upstream,
        # this test fails and the patching below can be retired.
        from osl_ephys.source_recon import parcellation

        source = make_source_epochs()
        theirs = parcellation.convert2mne_epochs(make_parc_data(), source)

        assert parcel_epochs.is_placeholder_event_id(theirs.event_id)
        assert theirs.tmin == pytest.approx(0.0)

    def test_survives_a_fif_round_trip(self, tmp_path):
        source = make_source_epochs()
        parc = parcel_epochs.convert2mne_epochs(make_parc_data(), source)

        path = tmp_path / "lcmv-parc-epo.fif"
        parc.save(path, overwrite=True, verbose="ERROR")
        read = mne.read_epochs(path, preload=True, verbose="ERROR")

        assert read.event_id == source.event_id
        assert read.tmin == pytest.approx(source.tmin)


class TestRestoreEpochMetadata:
    def _placeholder(self, source, sfreq=SFREQ):
        """A parcel Epochs as osl-ephys' converter would have written it."""
        from osl_ephys.source_recon import parcellation

        return parcellation.convert2mne_epochs(make_parc_data(), source)

    def test_names_and_time_axis_come_back(self):
        source = make_source_epochs()
        broken = self._placeholder(source)

        repaired = parcel_epochs.restore_epoch_metadata(broken, source)

        assert repaired.event_id == source.event_id
        assert repaired.tmin == pytest.approx(source.tmin)
        assert len(repaired["response/left"]) == len(source["response/left"])

    def test_the_parcel_data_is_untouched(self):
        source = make_source_epochs()
        broken = self._placeholder(source)

        repaired = parcel_epochs.restore_epoch_metadata(broken, source)

        assert repaired.get_data(copy=False) == pytest.approx(
            broken.get_data(copy=False)
        )
        assert repaired.ch_names == broken.ch_names

    def test_the_parcel_sampling_rate_wins_over_the_sensor_rate(self, caplog):
        # The parcel file is decimated by the source stage; the sensor file is
        # not, so the rate has to come from the parcel file. Both describe the
        # same 0.01 s window: 13 samples at 1200 Hz, 3 at 200 Hz.
        source = make_source_epochs(sfreq=1200.0, n_times=13)
        broken = parcel_epochs.build_parcel_epochs(
            make_parc_data(n_times=3),
            sfreq=200.0,
            events=source.events,
            tmin=0.0,
        )

        with caplog.at_level(logging.WARNING, logger=parcel_epochs.__name__):
            repaired = parcel_epochs.restore_epoch_metadata(broken, source)

        assert repaired.info["sfreq"] == pytest.approx(200.0)
        assert len(repaired.times) == 3
        assert repaired.tmin == pytest.approx(source.tmin)
        assert caplog.text == ""

    def test_metadata_comes_from_the_sensor_epochs(self):
        source = make_source_epochs()
        repaired = parcel_epochs.restore_epoch_metadata(
            self._placeholder(source), source
        )
        pd.testing.assert_frame_equal(repaired.metadata, source.metadata)

    def test_is_idempotent(self):
        source = make_source_epochs()
        once = parcel_epochs.restore_epoch_metadata(self._placeholder(source), source)
        twice = parcel_epochs.restore_epoch_metadata(once, source)

        assert twice.event_id == once.event_id
        assert twice.tmin == pytest.approx(once.tmin)
        assert twice.get_data(copy=False) == pytest.approx(once.get_data(copy=False))

    def test_survives_a_sensor_file_naming_conditions_with_no_epochs(self):
        # sub-011 in the TSX sample: heavy epoch rejection left 5 of its 13
        # conditions with no surviving epochs, but the FIF still names them.
        source = make_source_epochs()
        broken = self._placeholder(source)
        source.event_id = {**source.event_id, "feedback": 1, "ITI": 2}

        repaired = parcel_epochs.restore_epoch_metadata(broken, source)

        assert repaired.event_id == EVENT_ID_FIXTURE
        assert len(repaired["response/left"]) == 3

    def test_rejects_a_different_number_of_epochs(self):
        source = make_source_epochs()
        broken = self._placeholder(source)

        with pytest.raises(ValueError, match="not a matching pair"):
            parcel_epochs.restore_epoch_metadata(broken, source[:-1])

    def test_rejects_a_different_sequence_of_codes(self):
        source = make_source_epochs()
        broken = self._placeholder(source)
        broken.events[0, 2] = 999

        with pytest.raises(ValueError, match="disagree on the event codes"):
            parcel_epochs.restore_epoch_metadata(broken, source)

    def test_warns_when_the_epoch_windows_do_not_line_up(self, caplog):
        source = make_source_epochs()
        broken = self._placeholder(source)
        # Half the samples, at the same rate: a window half as long.
        broken = broken.crop(tmax=broken.times[N_TIMES // 2])

        with caplog.at_level(logging.WARNING, logger=parcel_epochs.__name__):
            parcel_epochs.restore_epoch_metadata(broken, source)

        assert "sensor epochs span" in caplog.text

    def test_silent_when_the_windows_agree(self, caplog):
        source = make_source_epochs()
        with caplog.at_level(logging.WARNING, logger=parcel_epochs.__name__):
            parcel_epochs.restore_epoch_metadata(self._placeholder(source), source)
        assert caplog.text == ""


class TestPreservingEpochMetadata:
    def test_the_patch_makes_osl_ephys_keep_the_names(self):
        from osl_ephys.source_recon import parcellation

        source = make_source_epochs()
        with parcel_epochs.preserving_epoch_metadata():
            parc = parcellation.convert2mne_epochs(make_parc_data(), source)

        assert parc.event_id == source.event_id
        assert parc.tmin == pytest.approx(source.tmin)

    def test_the_original_is_restored_afterwards(self):
        from osl_ephys.source_recon import parcellation

        original = parcellation.convert2mne_epochs
        with parcel_epochs.preserving_epoch_metadata():
            assert parcellation.convert2mne_epochs is not original
        assert parcellation.convert2mne_epochs is original

    def test_the_original_is_restored_when_the_block_raises(self):
        from osl_ephys.source_recon import parcellation

        original = parcellation.convert2mne_epochs
        with pytest.raises(RuntimeError):
            with parcel_epochs.preserving_epoch_metadata():
                raise RuntimeError("boom")
        assert parcellation.convert2mne_epochs is original

    def test_the_wrapper_reaches_osl_ephys_beamforming_step(self):
        # beamform_and_parcellate looks the converter up on the module, which
        # is what makes patching the attribute enough.
        from osl_ephys.source_recon import wrappers

        source = inspect_source(wrappers.beamform_and_parcellate)
        assert "parcellation.convert2mne_epochs(" in source


def inspect_source(func) -> str:
    import inspect

    return inspect.getsource(func)
