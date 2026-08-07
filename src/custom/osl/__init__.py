"""osl-ephys pipeline for OPM-MEG data.

This subpackage runs an `osl-ephys <https://osl-ephys.readthedocs.io>`_
pipeline over BIDS OPM data, from preprocessing through to LCMV beamforming
and parcellation.  It is an alternative to the mne-bids-pipeline route in
``custom.preprocessing``; the two share the same BIDS input (written by
``custom.format_bids``) but keep entirely separate derivative trees.

The pipeline runs one subject per invocation, which is what makes it suitable
for a SLURM array job.  Group-level HTML reports are deferred to a separate
``collate`` stage so that concurrent array tasks never write the same file.

Stages
------
preproc
    Run an osl-ephys ``preproc`` chain over the subject's BIDS raw file.
source
    Run an osl-ephys ``source_recon`` chain: surfaces, coregistration,
    forward model, LCMV beamforming and parcellation.
collate
    Regenerate the group-level preprocessing and source-recon HTML reports
    across every subject present in the report directories.
validate
    Check a config without running anything.

Source backends
---------------
``rhino`` (default)
    osl-ephys' native path.  RHINO extracts surfaces from the T1 and runs its
    own coregistration, then ``beamform_and_parcellate`` reconstructs onto a
    volumetric grid in MNI space.  Requires FSL.
``freesurfer``
    Reuses the FreeSurfer ``recon-all`` output and the MNE ``-trans.fif``
    produced by ``custom.preprocessing.coreg``, and reproduces the beamform +
    parcellate step with MNE and nilearn.  Requires no FSL.

Modules
-------
_config
    Load and validate the pipeline YAML.
_paths
    Resolve BIDS inputs and derivative outputs for a subject.
extra_funcs
    Custom osl-ephys preprocessing wrappers (notably ``events_from_annotations``).
preproc
    The ``preproc`` stage.
source
    The ``source`` stage.
fs_bridge
    FreeSurfer/MNE source-recon wrappers used by the ``freesurfer`` backend.
collate
    The ``collate`` stage.
validate
    Config validation.

Author: Harrison Ritz, 2025
"""

from . import _config, _paths, collate, extra_funcs, fs_bridge, preproc, source, validate
from ._config import load_config, preproc_config, source_config
from ._paths import resolve_paths
from .extra_funcs import PREPROC_EXTRA_FUNCS, events_from_annotations
from .fs_bridge import SOURCE_EXTRA_FUNCS
from .validate import validate_config

__all__ = [
    # Modules
    "_config",
    "_paths",
    "collate",
    "extra_funcs",
    "fs_bridge",
    "preproc",
    "source",
    "validate",
    # Config
    "load_config",
    "preproc_config",
    "source_config",
    # Paths
    "resolve_paths",
    # Custom osl-ephys functions
    "PREPROC_EXTRA_FUNCS",
    "SOURCE_EXTRA_FUNCS",
    "events_from_annotations",
    "validate_config",
]
