"""Annotate task breaks before segment and channel artifact detection.

The annotations are written to the custom raw derivative, so subsequent
custom steps and mne-bids-pipeline see the same ``BAD_break`` intervals.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import mne

from ._base import BaseAnalysis
from ._io import (
    find_custom_input_paths,
    read_raw_bids_with_retry,
    trim_raw_to_events_tsv,
    write_raw_bids_custom_step,
)


class AnnotateBreaksAnalysis(BaseAnalysis):
    """Find and save breaks in the task recording before artifact scoring."""

    ANALYSIS_KEY = "annotatebreaks"
    ANALYSIS_NAME = "annotate_breaks"

    def is_enabled(self) -> bool:
        return bool(getattr(self.cfg, "find_breaks", False))

    def load_data(self) -> Dict[str, Any]:
        paths = find_custom_input_paths(self.cfg, task=self.cfg.task)
        if not paths:
            raise FileNotFoundError(f"No raw data found for task={self.cfg.task}")
        self._source_bp = paths[0]
        raw = read_raw_bids_with_retry(paths[0], extra_params={"preload": True})
        events_path = paths[0].copy().update(
            suffix="events", extension=".tsv", split=None, check=False
        ).fpath
        conditions = tuple(
            sorted(
                set(getattr(self.cfg, "conditions", ("trial",)) or ("trial",))
                | set(
                    getattr(self.cfg, "_trial_conditions", ("trial",))
                    or ("trial",)
                )
            )
        )
        trim_raw_to_events_tsv(
            raw, events_path, conditions=conditions, context="annotate_breaks"
        )
        return {self.cfg.task: raw}

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raw = data[self.cfg.task]
        breaks = mne.preprocessing.annotate_break(
            raw,
            min_break_duration=self.cfg.min_break_duration,
            t_start_after_previous=self.cfg.t_break_annot_start_after_previous_event,
            t_stop_before_next=self.cfg.t_break_annot_stop_before_next_event,
        )
        existing = {
            (ann["onset"], ann["duration"], ann["description"])
            for ann in raw.annotations
        }
        new_breaks = [
            ann for ann in breaks
            if (ann["onset"], ann["duration"], ann["description"]) not in existing
        ]
        if new_breaks:
            raw.annotations.append(
                [ann["onset"] for ann in new_breaks],
                [ann["duration"] for ann in new_breaks],
                [ann["description"] for ann in new_breaks],
            )
        self.log(f"Added {len(new_breaks)} BAD_break annotation(s)")
        return {self.cfg.task: raw}

    def save_results(self, results: Dict[str, Any]) -> None:
        output = write_raw_bids_custom_step(
            results[self.cfg.task], self.cfg, self._source_bp
        )
        self.log(f"Saved task={self.cfg.task} → {output.fpath}")


def run(cfg: SimpleNamespace) -> None:
    """Module entry point for the custom preprocessing CLI."""
    analysis = AnnotateBreaksAnalysis(cfg)
    if not analysis.is_enabled():
        print("[annotate_breaks] Disabled in configuration; exiting")
        return
    analysis.execute()
