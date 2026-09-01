"""Tests for the repair-parc stage.

Rewrites parcel files that osl-ephys' own converter wrote without their
condition names or epoch time axis, so that an existing output tree can be
used by the group stage without re-running source reconstruction.
"""

from __future__ import annotations

import textwrap

import mne
import numpy as np
import pytest

from custom.osl import repair_parc, sign_flip
from custom.osl._config import load_config
from custom.osl.parcel_epochs import build_parcel_epochs


N_PARCELS = 4
N_TIMES = 10
N_EPOCHS = 6
SENSOR_SFREQ = 60.0
PARC_SFREQ = 60.0
TMIN = -0.05

EVENT_ID = {"response/left": 201, "response/right": 202}


CONFIG = """
pipeline:
  subject: "000"
  task: TSX
  bids_root: {bids_root}
  outdir: {outdir}
  subject_label: "sub-{{subject}}"
  source_input: {source_input}
"""


def _events(n_epochs=N_EPOCHS):
    return np.column_stack(
        [
            np.arange(n_epochs) * N_TIMES + 100,
            np.zeros(n_epochs, int),
            np.tile([201, 202], n_epochs // 2 + 1)[:n_epochs],
        ]
    )


def write_subject(outdir, subject, *, with_sflip=True, labelled=False, seed=0):
    """Write a subject's sensor epochs and its parcel files.

    ``labelled=False`` writes the parcel files the way osl-ephys' converter
    did: right data, no condition names, time axis starting at zero.
    """
    subject_dir = outdir / subject
    (subject_dir / "parc").mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    events = _events()

    sensor = mne.EpochsArray(
        rng.randn(N_EPOCHS, 3, N_TIMES) * 1e-12,
        mne.create_info(["MEG0", "MEG1", "MEG2"], SENSOR_SFREQ, "mag"),
        events=events,
        event_id=EVENT_ID,
        tmin=TMIN,
        verbose=False,
    )
    sensor.save(subject_dir / f"{subject}_epo.fif", overwrite=True, verbose="ERROR")

    parc_data = rng.randn(N_PARCELS, N_TIMES, N_EPOCHS)
    parc = build_parcel_epochs(
        parc_data,
        sfreq=PARC_SFREQ,
        events=events,
        event_id=EVENT_ID if labelled else None,
        tmin=TMIN if labelled else 0.0,
    )

    parc.save(
        sign_flip.parc_file(outdir, subject, True, "lcmv"),
        overwrite=True,
        verbose="ERROR",
    )
    if with_sflip:
        parc.save(
            sign_flip.sflip_file(outdir, subject, True, "lcmv"),
            overwrite=True,
            verbose="ERROR",
        )
    return parc_data


def make_cfg(tmp_path, source_input="epochs"):
    outdir = tmp_path / "osl"
    outdir.mkdir(exist_ok=True)
    path = tmp_path / "cfg.yaml"
    path.write_text(
        textwrap.dedent(
            CONFIG.format(
                bids_root=tmp_path / "bids",
                outdir=outdir,
                source_input=source_input,
            )
        )
    )
    return load_config(path, env={}), outdir


def read(path):
    return mne.read_epochs(path, preload=True, verbose="ERROR")


@pytest.fixture
def repair_cfg(tmp_path):
    cfg, outdir = make_cfg(tmp_path)
    data = {
        f"sub-{i:03d}": write_subject(outdir, f"sub-{i:03d}", seed=i) for i in range(3)
    }
    return cfg, outdir, data


class TestRepairStage:
    def test_restores_the_condition_names(self, repair_cfg):
        cfg, outdir, _ = repair_cfg
        assert repair_parc.run(cfg) is True

        parc = read(sign_flip.parc_file(outdir, "sub-000", True, "lcmv"))
        assert parc.event_id == EVENT_ID
        assert len(parc["response/left"]) == 3

    def test_restores_the_time_axis(self, repair_cfg):
        cfg, outdir, _ = repair_cfg
        assert read(sign_flip.parc_file(outdir, "sub-000", True, "lcmv")).tmin == 0.0

        repair_parc.run(cfg)

        parc = read(sign_flip.parc_file(outdir, "sub-000", True, "lcmv"))
        assert parc.tmin == pytest.approx(TMIN)

    def test_repairs_the_sign_flipped_copy_too(self, repair_cfg):
        # The group stage reads the sign-flipped file, not the parcel file.
        cfg, outdir, _ = repair_cfg
        repair_parc.run(cfg)

        sflip = read(sign_flip.sflip_file(outdir, "sub-000", True, "lcmv"))
        assert sflip.event_id == EVENT_ID
        assert sflip.tmin == pytest.approx(TMIN)

    def test_the_parcel_data_is_unchanged(self, repair_cfg):
        cfg, outdir, data = repair_cfg
        repair_parc.run(cfg)

        for subject, parc_data in data.items():
            parc = read(sign_flip.parc_file(outdir, subject, True, "lcmv"))
            expected = np.swapaxes(parc_data.T, 1, 2)
            assert parc.get_data(copy=False) == pytest.approx(expected)

    def test_repairs_every_subject(self, repair_cfg):
        cfg, outdir, data = repair_cfg
        repair_parc.run(cfg)

        for subject in data:
            assert read(
                sign_flip.parc_file(outdir, subject, True, "lcmv")
            ).event_id == EVENT_ID

    def test_is_idempotent(self, repair_cfg):
        cfg, outdir, _ = repair_cfg
        assert repair_parc.run(cfg) is True
        first = read(sign_flip.parc_file(outdir, "sub-000", True, "lcmv"))

        assert repair_parc.run(cfg) is True
        second = read(sign_flip.parc_file(outdir, "sub-000", True, "lcmv"))

        assert second.event_id == first.event_id
        assert second.tmin == pytest.approx(first.tmin)
        assert second.get_data(copy=False) == pytest.approx(first.get_data(copy=False))

    def test_an_already_labelled_subject_is_left_correct(self, tmp_path):
        cfg, outdir = make_cfg(tmp_path)
        write_subject(outdir, "sub-000", labelled=True)
        write_subject(outdir, "sub-001", labelled=True)

        assert repair_parc.run(cfg) is True
        parc = read(sign_flip.parc_file(outdir, "sub-000", True, "lcmv"))
        assert parc.event_id == EVENT_ID
        assert parc.tmin == pytest.approx(TMIN)

    def test_a_subject_without_a_sign_flipped_copy_is_still_repaired(self, tmp_path):
        # The group stage has not run yet, so only the parcel file exists.
        cfg, outdir = make_cfg(tmp_path)
        write_subject(outdir, "sub-000", with_sflip=False)
        write_subject(outdir, "sub-001", with_sflip=False)

        assert repair_parc.run(cfg) is True
        assert read(
            sign_flip.parc_file(outdir, "sub-000", True, "lcmv")
        ).event_id == EVENT_ID


class TestFailures:
    def test_a_subject_missing_its_sensor_epochs_does_not_stop_the_rest(
        self, repair_cfg, capsys
    ):
        cfg, outdir, _ = repair_cfg
        (outdir / "sub-000" / "sub-000_epo.fif").unlink()

        assert repair_parc.run(cfg) is False

        out = capsys.readouterr().out
        assert "FAILED for sub-000" in out
        assert "repaired 2/3" in out
        # The other subjects were still repaired.
        assert read(
            sign_flip.parc_file(outdir, "sub-001", True, "lcmv")
        ).event_id == EVENT_ID

    def test_a_failed_subject_keeps_its_original_file(self, repair_cfg):
        cfg, outdir, data = repair_cfg
        (outdir / "sub-000" / "sub-000_epo.fif").unlink()

        repair_parc.run(cfg)

        parc = read(sign_flip.parc_file(outdir, "sub-000", True, "lcmv"))
        expected = np.swapaxes(data["sub-000"].T, 1, 2)
        assert parc.get_data(copy=False) == pytest.approx(expected)

    def test_no_temporary_files_are_left_behind(self, repair_cfg):
        cfg, outdir, _ = repair_cfg
        repair_parc.run(cfg)
        assert list(outdir.rglob("*repair-tmp*")) == []

    def test_rejects_a_mismatched_pair(self, repair_cfg):
        cfg, outdir, _ = repair_cfg
        # Rewrite one subject's sensor epochs with different codes.
        path = outdir / "sub-000" / "sub-000_epo.fif"
        sensor = read(path)
        sensor.events[0, 2] = 999
        sensor.save(path, overwrite=True, verbose="ERROR")

        assert repair_parc.run(cfg) is False

    def test_requires_epoched_source_input(self, tmp_path):
        cfg, outdir = make_cfg(tmp_path, source_input="raw")
        write_subject(outdir, "sub-000")

        with pytest.raises(ValueError, match="source_input: epochs"):
            repair_parc.run(cfg)

    def test_raises_when_no_subject_has_parcellated_data(self, tmp_path):
        cfg, _ = make_cfg(tmp_path)
        with pytest.raises(ValueError, match="has parcellated data"):
            repair_parc.run(cfg)
