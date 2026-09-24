"""Recording breaks are annotated once before artifact scoring."""

from types import SimpleNamespace

import mne
import mne_bids
import numpy as np
from mne_bids import BIDSPath

from custom.preprocessing.annotate_breaks import AnnotateBreaksAnalysis


def test_annotate_breaks_is_idempotent():
    raw = mne.io.RawArray(
        np.zeros((3, 3000)),
        mne.create_info(["M1", "M2", "M3"], 100, "mag"),
        verbose="ERROR",
    )
    raw.set_annotations(mne.Annotations([1, 2, 20, 21], [0] * 4, ["trial"] * 4))
    cfg = SimpleNamespace(
        task="example",
        find_breaks=True,
        min_break_duration=6.0,
        t_break_annot_start_after_previous_event=1.5,
        t_break_annot_stop_before_next_event=1.5,
    )
    analysis = AnnotateBreaksAnalysis(cfg)

    first = analysis.run({"example": raw})["example"]
    second = analysis.run({"example": first})["example"]

    assert analysis.is_enabled()
    assert list(second.annotations.description).count("BAD_break") == 2


def test_breaks_are_saved_before_later_custom_steps(tmp_path):
    raw = mne.io.RawArray(
        np.zeros((3, 3000)),
        mne.create_info(["M1", "M2", "M3"], 100, "mag"),
        verbose="ERROR",
    )
    raw.set_meas_date(None)
    raw.set_annotations(mne.Annotations([1, 2, 20, 21], [0] * 4, ["trial"] * 4))
    bids_path = BIDSPath(
        root=tmp_path / "bids",
        subject="001",
        session="01",
        task="example",
        datatype="meg",
        suffix="meg",
        extension=".fif",
    )
    mne_bids.write_raw_bids(
        raw, bids_path, allow_preload=True, overwrite=True, format="FIF"
    )
    cfg = SimpleNamespace(
        bids_root=str(tmp_path / "bids"),
        deriv_root=str(tmp_path / "derivatives"),
        custom_proc="init",
        subjects=["001"],
        sessions=["01"],
        task="example",
        conditions=["trial"],
        find_breaks=True,
        min_break_duration=6.0,
        t_break_annot_start_after_previous_event=1.5,
        t_break_annot_stop_before_next_event=1.5,
    )

    AnnotateBreaksAnalysis(cfg).execute()

    saved = next((tmp_path / "derivatives").rglob("*_proc-init_raw.fif"))
    derivative = mne.io.read_raw_fif(saved, verbose="ERROR")
    source = mne.io.read_raw_fif(bids_path.fpath, verbose="ERROR")
    assert list(derivative.annotations.description).count("BAD_break") == 2
    assert "BAD_break" not in source.annotations.description
