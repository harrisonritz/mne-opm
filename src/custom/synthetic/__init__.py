"""Generate a synthetic, BIDS-formatted OPM subject for development.

The pipeline in this repository can only be exercised end to end against a
real dataset, which is exactly what an agent or a new contributor working from
a fresh clone does not have.  This subpackage builds one from nothing: an
analytic head phantom, a Cerca-style triaxial OPM array, and data simulated
through a genuine forward model, written out as a BIDS dataset with a matching
FreeSurfer subject.

The result is drop-in: point ``mne-opm.sh`` at it and the preprocessing,
source and beamformer stages run unmodified.

Typical use::

    python src/custom/make_synthetic.py --out synthetic/datasets/synth
    python src/custom/make_synthetic.py --out /tmp/cohort --n-subjects 12

See ``synthetic/README.md`` for the committed dataset and how to run against it.

Author: Harrison Ritz (2025)
"""

from __future__ import annotations

from .anatomy import HeadModel, build_head_model, write_freesurfer_subject
from .dataset import DatasetSpec, make_dataset, make_subject
from .events import TRIGGER_DESC, build_schedule
from .sensors import build_info
from .simulate import SourceSpec, default_sources, simulate_empty_room, simulate_task


__all__ = [
    "DatasetSpec",
    "HeadModel",
    "SourceSpec",
    "TRIGGER_DESC",
    "build_head_model",
    "build_info",
    "build_schedule",
    "default_sources",
    "make_dataset",
    "make_subject",
    "simulate_empty_room",
    "simulate_task",
    "write_freesurfer_subject",
]
