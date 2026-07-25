"""LCMV Beamformer source reconstruction for OPM-MEG data.

This script performs LCMV (Linearly Constrained Minimum Variance) beamformer
analysis on preprocessed MEG data. It supports two types of analyses:

1. Time-locked analysis: Apply beamformer to evoked responses (PRIMARY)
2. Power analysis: Apply beamformer to covariance matrices (SECONDARY)

The beamformer provides spatial filtering to reconstruct source activity,
particularly well-suited for OPM-MEG data.

Either or both source spaces can be reconstructed in one pass
(``_beamformer_source_space``).  They differ in how the otherwise-arbitrary
orientation sign is resolved, which matters a great deal once estimates are
averaged across subjects:

* **surface** — the forward is rotated into surface orientation
  (:func:`surface_orient_forward`) so ``pick_ori='max-power'`` signs are anchored
  to the outward cortical normal.
* **volume** — a grid has no cortical normal (MNE falls back to the +Z / superior
  direction), so it is fit with ``pick_ori='vector'``, keeping all three
  components.  Downstream, ``beamformer_volume.project_vol_stc_to_surface`` reads
  them out along the fsaverage cortical normals to recover a signed estimate.

Usage:
    python run_beamformer.py --config=/path/to/config.py

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import mne
import numpy as np
from mne.beamformer import apply_lcmv, apply_lcmv_cov, make_lcmv
from mne_bids import BIDSPath, get_head_mri_trans

# Add mne-bids-pipeline to path for importing utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mne-bids-pipeline"))

from mne_bids_pipeline._config_import import _update_config_from_path, _import_config
from mne_bids_pipeline._config_utils import (
    _get_bem_conductivity,
    get_fs_subject,
    get_fs_subjects_dir,
    get_noise_cov_bids_path,
    sanitize_cond_name,
)
from mne_bids_pipeline._io import _read_json
from mne_bids_pipeline._report import _all_conditions, _open_report, _sanitize_cond_tag


# --------------------------------------------------------------------------------------
# Configuration and Environment
# --------------------------------------------------------------------------------------


def resolve_source_spaces(cfg: SimpleNamespace) -> list[str]:
    """Return the beamformer source spaces to run, as an ordered unique list.

    ``_beamformer_source_space`` may be a single string (``"surface"`` |
    ``"volume"``) or a list/tuple of them (e.g. ``["volume", "surface"]``) to run
    both reconstructions in one invocation.  Duplicates are collapsed and the
    given order is preserved.
    """
    val = getattr(cfg, "_beamformer_source_space", "surface")
    spaces = [val] if isinstance(val, str) else list(val)
    resolved: list[str] = []
    for s in spaces:
        if s not in ("surface", "volume"):
            raise ValueError(
                f"Invalid _beamformer_source_space entry: {s!r}. "
                f"Must be 'surface' or 'volume'."
            )
        if s not in resolved:
            resolved.append(s)
    if not resolved:
        raise ValueError("_beamformer_source_space must not be empty.")
    return resolved


def resolve_per_space(value: Any, space: str, default: Any = None) -> Any:
    """Resolve a setting that may be global or per source space.

    Orientation-related settings differ between source spaces: a surface space has
    a cortical normal to anchor ``pick_ori='max-power'`` signs to, a volume grid
    does not (MNE falls back to the +Z / superior direction).  So
    ``_beamformer_pick_ori`` / ``_beamformer_weight_norm`` accept either

    * a plain value  -> used for every source space (back-compatible), or
    * a ``dict`` keyed by ``'surface'`` / ``'volume'``.

    ``default`` is returned when a dict is given but does not mention ``space``.
    """
    if isinstance(value, dict):
        if space not in value and default is None:
            raise ValueError(
                f"No entry for source space {space!r} in {value!r}, and no default."
            )
        return value.get(space, default)
    return value


def surface_orient_forward(forward: mne.Forward, cfg: SimpleNamespace) -> mne.Forward:
    """Rotate a surface forward into surface orientation, for the sign convention.

    ``mne.read_forward_solution`` returns a free-orientation forward with
    ``surf_ori=False`` and ``source_nn = kron(ones, eye(3))``, and MNE only
    converts to surface orientation for *constrained* (``loose < 1``) inverses —
    which a beamformer never is.  ``_prepare_beamformer_input`` therefore ends up
    with ``nn = [0, 0, 1]`` in **head** coordinates, and
    ``_compute_beamformer`` resolves the otherwise-arbitrary ``max-power`` sign as
    ``sign(max_power_ori @ nn)`` — i.e. against head-superior rather than against
    the cortical normal.  Sources oriented tangentially to head +Z (the central
    sulcus walls, lateral sensorimotor, most of temporal cortex) then get a
    noise-determined sign, which flips vertex-to-vertex within a subject and
    subject-to-subject in the group average.

    Converting to surface orientation makes ``nn`` the local surface normal, so
    the sign is anchored to the outward cortical normal instead.  LCMV with free
    orientation is invariant to an orthogonal rotation of the source coordinate
    frame, so this changes the sign convention and *nothing else*.

    Only meaningful for surface source spaces; volume grids have no normal.
    """
    if not getattr(cfg, "_beamformer_surf_ori", True):
        print("[surface_orient_forward] _beamformer_surf_ori=False; leaving forward as-is")
        return forward
    if forward["surf_ori"]:
        return forward
    print("[surface_orient_forward] Converting forward to surface orientation (use_cps)")
    return mne.convert_forward_solution(
        forward, surf_ori=True, force_fixed=False, use_cps=True
    )


# --------------------------------------------------------------------------------------
# Volume Forward Solution
# --------------------------------------------------------------------------------------


def _find_bem_solution(fs_subjects_dir: str, fs_subject: str, tag: str) -> Path | None:
    """Locate an existing BEM solution in the FreeSurfer subject ``bem/`` directory.

    Prefers the conductivity-tagged file the pipeline writes
    (``{fs_subject}-{tag}-bem-sol.fif``), then falls back to any ``*bem-sol.fif``.
    Mirrors ``coreg_diagnostics._find_bem_solution`` but honours the pipeline's
    ``_get_bem_conductivity`` tag so the OPM single-layer solution is found first.
    """
    bem_dir = Path(fs_subjects_dir) / fs_subject / "bem"
    if not bem_dir.exists():
        return None
    tagged = bem_dir / f"{fs_subject}-{tag}-bem-sol.fif"
    if tagged.exists():
        return tagged
    matches = sorted(bem_dir.glob("*bem-sol.fif"))
    if not matches:
        return None
    for m in matches:
        if m.name.startswith(fs_subject):
            return m
    return matches[0]


def build_volume_forward(cfg: SimpleNamespace, info: mne.Info) -> mne.Forward:
    """Build (or load a cached) volume-source-space forward solution.

    mne-bids-pipeline only ever builds *surface* forward solutions, so a volume
    beamformer needs its own forward.  This adapts the on-the-fly forward pattern
    from ``coreg_diagnostics._compute_forward``, swapping ``setup_source_space``
    for :func:`mne.setup_volume_source_space` (a regular 3D grid bounded by the
    BEM inner-skull surface).

    The result is cached to ``*_acq-vol_fwd.fif`` (the forward file embeds its
    volume ``src``) and reused on subsequent runs when
    ``_beamformer_volume_cache`` is set.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object.  Reads ``_beamformer_volume_pos`` (grid spacing,
        mm), ``_beamformer_volume_mindist`` (mm from inner skull),
        ``_beamformer_volume_bem_conductivity``, ``_beamformer_volume_bem_ico``
        and ``_beamformer_volume_cache``.
    info : mne.Info
        Measurement info (sensor geometry) the forward is computed for.  Bad
        channels are dropped so the forward's channel set matches the one the
        filters are built on.

    Returns
    -------
    fwd : mne.Forward
        Volume-source-space forward solution.
    """
    print("\n[build_volume_forward] Building volume-source-space forward...")

    if info["bads"]:
        print(f"[build_volume_forward] Excluding {len(info['bads'])} bad channel(s)")
        info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False, exclude="bads"))

    subject = cfg.subjects[0]
    session = cfg.sessions[0]

    fs_subject = get_fs_subject(config=cfg, subject=subject, session=session)
    fs_subjects_dir = get_fs_subjects_dir(config=cfg)
    print(
        f"[build_volume_forward] FreeSurfer subject={fs_subject}, "
        f"subjects_dir={fs_subjects_dir}"
    )

    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        root=cfg.deriv_root,
        datatype=cfg.datatype,
        check=False,
    )

    pos = float(getattr(cfg, "_beamformer_volume_pos", 5.0))
    mindist = float(getattr(cfg, "_beamformer_volume_mindist", getattr(cfg, "mindist", 5.0)))
    cache = getattr(cfg, "_beamformer_volume_cache", True)

    # Cache -----------------------------------------------------------------
    vol_fwd_path = bids_path.copy().update(
        suffix="fwd", acquisition="vol", extension=".fif"
    )
    if cache and vol_fwd_path.fpath.exists():
        print(f"[build_volume_forward] Loading cached volume forward: {vol_fwd_path.fpath}")
        return mne.read_forward_solution(vol_fwd_path.fpath)

    # BEM solution ----------------------------------------------------------
    # _get_bem_conductivity reads per-step fields (fs_subject / use_template_mri
    # / ch_types) that the raw imported config does not carry; supply them via a
    # lightweight shim so we still reuse the pipeline's conductivity-tag
    # convention (rather than re-implementing it) without mutating cfg.
    bem_cfg = SimpleNamespace(
        fs_subject=fs_subject,
        use_template_mri=getattr(cfg, "use_template_mri", None),
        ch_types=cfg.ch_types,
    )
    conductivity_default, tag = _get_bem_conductivity(bem_cfg)
    bem_path = _find_bem_solution(fs_subjects_dir, fs_subject, tag)
    if bem_path is not None:
        print(f"[build_volume_forward] Loading BEM solution: {bem_path}")
        bem = mne.read_bem_solution(bem_path)
    else:
        conductivity = tuple(
            getattr(cfg, "_beamformer_volume_bem_conductivity", (0.3,))
        )
        ico = int(getattr(cfg, "_beamformer_volume_bem_ico", 4))
        print(
            f"[build_volume_forward] No BEM on disk; building model "
            f"(conductivity={conductivity}, ico={ico})"
        )
        model = mne.make_bem_model(
            subject=fs_subject,
            ico=ico,
            conductivity=conductivity,
            subjects_dir=fs_subjects_dir,
        )
        bem = mne.make_bem_solution(model)

    # Head <-> MRI transform ------------------------------------------------
    # Prefer the trans the pipeline already wrote next to the surface forward;
    # fall back to recomputing it from the BIDS anatomical landmarks.
    trans = None
    trans_path = bids_path.copy().update(suffix="trans", extension=".fif")
    if trans_path.fpath.exists():
        print(f"[build_volume_forward] Loading head-MRI trans: {trans_path.fpath}")
        trans = mne.read_trans(trans_path.fpath)
    else:
        print("[build_volume_forward] No trans on disk; deriving from BIDS landmarks")
        t1_bids_path = BIDSPath(
            subject=subject,
            session=session,
            root=cfg.bids_root,
            datatype="anat",
            suffix="T1w",
            extension=".nii.gz",
            check=False,
        )
        trans = get_head_mri_trans(
            bids_path,
            fs_subject=fs_subject,
            fs_subjects_dir=fs_subjects_dir,
            t1_bids_path=t1_bids_path,
        )

    # Volume source space (grid bounded by the BEM inner skull) -------------
    print(f"[build_volume_forward] Setting up volume source space (pos={pos} mm)")
    src = mne.setup_volume_source_space(
        subject=fs_subject,
        pos=pos,
        bem=bem,
        subjects_dir=fs_subjects_dir,
        add_interpolator=True,
        n_jobs=getattr(cfg, "n_jobs", -1),
    )
    print(f"[build_volume_forward] Volume source space: {sum(s['nuse'] for s in src)} sources")

    # Forward solution ------------------------------------------------------
    print("[build_volume_forward] Computing forward solution...")
    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        ignore_ref=True,
        mindist=mindist,
        n_jobs=getattr(cfg, "n_jobs", -1),
    )

    # Persist for reuse -----------------------------------------------------
    if cache:
        try:
            vol_fwd_path.fpath.parent.mkdir(parents=True, exist_ok=True)
            mne.write_forward_solution(vol_fwd_path.fpath, fwd, overwrite=True)
            print(f"[build_volume_forward] Cached volume forward to {vol_fwd_path.fpath}")
        except Exception as e:
            print(f"[build_volume_forward] WARNING: could not cache forward: {e}")

    return fwd


# --------------------------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------------------------


def load_beamformer_data(cfg: SimpleNamespace) -> Dict[str, Any]:
    """Load all required input files for beamformer analysis.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object.

    Returns
    -------
    data : dict
        Dictionary containing:
        - 'forwards': dict of str -> mne.Forward, keyed by source space
        - 'forward': mne.Forward (the first requested space; back-compat)
        - 'epochs': mne.Epochs
        - 'noise_cov': mne.Covariance or None (None means "use MNE's ad-hoc")
        - 'noise_rank': dict or None, the rank the pipeline stored with the cov

    Notes
    -----
    The measurement info is *not* returned separately: everything downstream uses
    ``data['epochs'].info`` so that the projectors carried by the data covariance
    and by the info handed to :func:`mne.beamformer.make_lcmv` can never diverge.
    """
    print("\n[load_beamformer_data] Loading data files...")

    subject = cfg.subjects[0]
    session = cfg.sessions[0]
    data = {}

    # Construct base BIDS path
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        root=cfg.deriv_root,
        datatype=cfg.datatype,
        check=False,
    )

    # Source space(s) to reconstruct: 'surface' (mne-bids-pipeline forward),
    # 'volume' (built on the fly from the measurement info), or a list of both.
    source_spaces = resolve_source_spaces(cfg)
    print(f"[load_beamformer_data] Source space(s): {source_spaces}")
    data["forwards"] = {}

    # Load surface forward up front so a missing one errors clearly (volume
    # forwards are built after epochs/info, which they require).
    if "surface" in source_spaces:
        fwd_path = bids_path.copy().update(suffix="fwd", extension=".fif")
        if not fwd_path.fpath.exists():
            raise FileNotFoundError(
                f"Forward solution not found at {fwd_path.fpath}\n"
                f"Run forward modeling first with:\n"
                f"  mne_bids_pipeline --steps=source/make_forward --config=<config>"
            )
        print(f"[load_beamformer_data] Loading forward solution: {fwd_path.fpath}")
        data["forwards"]["surface"] = mne.read_forward_solution(fwd_path)

    # Load clean epochs
    epochs_path = bids_path.copy().update(
        suffix="epo", processing="clean", extension=".fif"
    )
    if not epochs_path.fpath.exists():
        raise FileNotFoundError(
            f"Clean epochs not found at {epochs_path.fpath}\nRun preprocessing first."
        )
    print(f"[load_beamformer_data] Loading epochs: {epochs_path.fpath}")
    data["epochs"] = mne.read_epochs(epochs_path, preload=True)

    # Build (or load cached) volume forward from the measurement info.
    if "volume" in source_spaces:
        data["forwards"]["volume"] = build_volume_forward(cfg, data["epochs"].info)

    # Back-compat: expose the first requested space's forward as data["forward"].
    data["forward"] = data["forwards"][source_spaces[0]]

    # Noise covariance ------------------------------------------------------
    # Reuse the covariance mne-bids-pipeline already computed and wrote (plus the
    # rank it stored alongside it), exactly as source/_05_make_inverse does.  That
    # keeps the beamformer's whitening consistent with the pipeline's minimum-norm
    # solutions and avoids recomputing an empty-room covariance with a different
    # rank convention.
    #
    # `noise_cov = 'ad-hoc'` deliberately leaves this as None: MNE then builds its
    # own ad-hoc covariance internally (std=1.0, allow_mismatch=True).  Building
    # one *here* and passing it in would make it a real covariance as far as MNE
    # is concerned, and because an ad-hoc covariance is isotropic its eigenvectors
    # are the canonical channel basis — so rank truncation in `_get_ch_whitener`
    # would zero out whole OPM channels rather than a meaningful subspace.
    data["noise_cov"] = None
    data["noise_rank"] = None
    if cfg.noise_cov == "ad-hoc":
        print("[load_beamformer_data] Using MNE's internal ad-hoc noise covariance")
    else:
        cov_path = get_noise_cov_bids_path(cfg=cfg, subject=subject, session=session)
        rank_path = cov_path.copy().update(suffix="rank", extension=".json")
        if not Path(cov_path.fpath).exists():
            raise FileNotFoundError(
                f"Noise covariance not found at {cov_path.fpath}\n"
                f"With noise_cov={cfg.noise_cov!r} the pipeline must compute it first:\n"
                f"  mne_bids_pipeline --steps=sensor/make_cov --config=<config>"
            )
        print(f"[load_beamformer_data] Loading noise covariance: {cov_path.fpath}")
        data["noise_cov"] = mne.read_cov(cov_path.fpath)
        if Path(rank_path.fpath).exists():
            data["noise_rank"] = _read_json(rank_path)
            print(f"[load_beamformer_data] Noise covariance rank: {data['noise_rank']}")
        else:
            print(
                f"[load_beamformer_data] WARNING: no rank sidecar at {rank_path.fpath}; "
                f"rank will be estimated from the data alone"
            )

    print(f"[load_beamformer_data] Data loading complete")
    print(f"  - Forward: {len(data['forward']['src'])} source spaces")
    print(
        f"  - Epochs: {len(data['epochs'])} epochs, {len(data['epochs'].ch_names)} channels"
    )
    print(
        f"  - Noise cov: {'ad-hoc' if data['noise_cov'] is None else 'loaded from file'}"
    )

    return data


# --------------------------------------------------------------------------------------
# Rank
# --------------------------------------------------------------------------------------


def resolve_rank(
    cfg: SimpleNamespace,
    epochs: mne.Epochs,
    noise_cov: mne.Covariance | None,
    noise_rank: dict | None,
) -> Any:
    """Resolve the rank handed to covariance estimation and to ``make_lcmv``.

    ``_beamformer_rank`` accepts:

    ``"data"``
        Take the element-wise **minimum** of the info rank, a data-driven rank
        estimate, and (when present) the rank the pipeline stored with the noise
        covariance.  Neither of the first two is trustworthy on its own:

        * ``rank="info"`` reads the SSS bookkeeping out of
          ``info['proc_history']``, so it knows the Maxwell basis dimension but
          nothing about components ICA removed afterwards — it *overstates* the
          rank of cleaned data.
        * a data-driven estimate sees ICA, but only if its tolerance is set
          sensibly.  ``tol="auto"`` is **not**: it works out to
          ``n_dim * max_s * eps_float64`` ~ 1e-13 relative
          (:func:`mne.rank._estimate_rank_from_s`), while directions nulled by
          SSS or ICA come back from a float32 FIF at ~1e-7 relative.  Every
          nulled direction therefore clears the threshold and the estimate
          returns near-full rank — which is why ``tol="auto"`` reports a rank
          well *above* the info rank on Maxwell-filtered data.  (MNE says as
          much in a comment in ``rank.py``: the default "should be float32
          probably due to how we save and load data".)

        So the default tolerance here is an explicit relative one
        (``_beamformer_rank_tol`` / ``_beamformer_rank_tol_kind``), and the
        minimum guards both ways: if the data estimate is still fooled, the info
        rank caps it; if ICA cut deeper than the SSS basis, the data estimate
        catches that.
    ``"empty_room"``
        Use the rank stored with the empty-room covariance verbatim.
    ``"info"`` / ``None`` / ``dict``
        Passed straight through to MNE.

    Returning one explicit ``{ch_type: n}`` dict (rather than the string ``"info"``)
    also guarantees the data and noise ranks agree, which ``make_lcmv`` requires
    whenever a real noise covariance is supplied.
    """
    setting = getattr(cfg, "_beamformer_rank", "data")

    if setting == "empty_room":
        if noise_rank is None:
            raise ValueError(
                "_beamformer_rank='empty_room' requires a noise covariance with a "
                "rank sidecar; set noise_cov to 'emptyroom' (or 'rest') and run "
                "mne_bids_pipeline --steps=sensor/make_cov first."
            )
        print(f"[resolve_rank] Using empty-room noise covariance rank: {noise_rank}")
        return dict(noise_rank)

    if setting != "data":
        print(f"[resolve_rank] Using configured beamformer rank: {setting}")
        return setting

    tol = getattr(cfg, "_beamformer_rank_tol", 1e-6)
    tol_kind = getattr(cfg, "_beamformer_rank_tol_kind", "relative")

    candidates = {}
    try:
        candidates["info"] = mne.compute_rank(epochs, rank="info")
    except Exception as e:  # noqa: BLE001 — no proc_history / no projections
        print(f"[resolve_rank] No info-based rank available ({e})")
    candidates["data"] = mne.compute_rank(epochs, tol=tol, tol_kind=tol_kind)
    if noise_cov is not None and noise_rank is not None:
        candidates["noise"] = dict(noise_rank)

    for name, value in candidates.items():
        extra = f" (tol={tol}, tol_kind={tol_kind!r})" if name == "data" else ""
        print(f"[resolve_rank] {name} rank{extra}: {value}")

    keys = set().union(*(c.keys() for c in candidates.values()))
    rank = {
        key: min(c[key] for c in candidates.values() if key in c) for key in keys
    }
    print(f"[resolve_rank] Using element-wise minimum: {rank}")

    # A data estimate at (or above) the info rank means the tolerance did not
    # separate the nulled directions from the retained ones — say so, because the
    # result is then just the info rank and ICA is going unaccounted for.
    info_rank = candidates.get("info")
    if info_rank is not None:
        if any(
            candidates["data"].get(key, 0) >= value for key, value in info_rank.items()
        ):
            print(
                "[resolve_rank] NOTE: the data-driven estimate is not below the "
                "info rank, so it is not detecting the rank lost to SSS/ICA. "
                "Raise _beamformer_rank_tol (e.g. 1e-5) and compare against "
                "rank_check.py before trusting it."
            )
    return rank


# --------------------------------------------------------------------------------------
# Beamformer Computation
# --------------------------------------------------------------------------------------


def compute_lcmv_filters(
    forward: mne.Forward,
    data_cov: mne.Covariance,
    noise_cov: mne.Covariance | None,
    info: mne.Info,
    cfg: SimpleNamespace,
    rank: Any = None,
    source_space: str = "surface",
) -> dict:
    """Compute LCMV spatial filters.

    Parameters
    ----------
    forward : mne.Forward
        Forward solution.
    data_cov : mne.Covariance
        Data covariance matrix.
    noise_cov : mne.Covariance or None
        Noise covariance matrix.  ``None`` is passed straight through to
        :func:`~mne.beamformer.make_lcmv`, which then builds its own ad-hoc
        covariance (``std=1.0``, ``allow_mismatch=True``).  Do *not* build one
        here: an ad-hoc covariance is isotropic, so its eigenvectors are the
        canonical channel basis and rank truncation would zero whole channels.
    info : mne.Info
        Measurement info.
    cfg : SimpleNamespace
        Configuration with beamformer parameters.
    rank : dict | str | int | None
        Rank of the covariance matrices, from :func:`resolve_rank`.
    source_space : str
        ``'surface'`` or ``'volume'``.  Selects the per-space entry of
        ``_beamformer_pick_ori`` / ``_beamformer_weight_norm`` when those are
        given as dicts.

    Returns
    -------
    filters : dict
        LCMV filters object.
    """
    pick_ori = resolve_per_space(cfg._beamformer_pick_ori, source_space, "max-power")
    weight_norm = resolve_per_space(cfg._beamformer_weight_norm, source_space, "nai")

    print("\n[compute_lcmv_filters] Computing LCMV spatial filters...")
    print(f"  - Source space: {source_space}")
    print(f"  - Regularization: {cfg._beamformer_reg}")
    print(f"  - Pick orientation: {pick_ori}")
    print(f"  - Weight normalization: {weight_norm}")
    print(f"  - Depth weighting: {cfg._beamformer_depth}")
    print(f"  - Rank: {rank}")

    # Validate parameters
    valid_ori = ["max-power", "vector", None]
    if pick_ori not in valid_ori:
        raise ValueError(
            f"Invalid _beamformer_pick_ori: {pick_ori}. Must be one of {valid_ori}"
        )

    valid_norm = ["unit-noise-gain", "nai", "unit-noise-gain-invariant", None]
    if weight_norm not in valid_norm:
        raise ValueError(
            f"Invalid _beamformer_weight_norm: {weight_norm}. "
            f"Must be one of {valid_norm}"
        )

    # Warn about suboptimal combinations
    if pick_ori == "vector" and weight_norm == "unit-noise-gain":
        print("  [WARNING] Using 'unit-noise-gain' with vector beamformer.")
        print("  Consider using 'unit-noise-gain-invariant' instead.")

    # Compute filters
    filters = make_lcmv(
        info,
        forward,
        data_cov,
        reg=cfg._beamformer_reg,
        noise_cov=noise_cov,
        pick_ori=pick_ori,
        weight_norm=weight_norm,
        depth=cfg._beamformer_depth,
        rank=rank,
        reduce_rank=cfg._reduce_rank,
        verbose=True,
    )

    print("[compute_lcmv_filters] Filters computed successfully")

    return filters


def run_beamformer_timecourse(
    epochs: mne.Epochs,
    filters: dict,
    cfg: SimpleNamespace,
) -> Dict[str, mne.SourceEstimate]:
    """Apply beamformer to evoked responses (time-locked analysis).

    This is the PRIMARY beamformer analysis, following the same pattern
    as the MNE inverse solution in _05_make_inverse.py.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data.
    filters : dict
        LCMV filters from make_lcmv.
    cfg : SimpleNamespace
        Configuration object.

    Returns
    -------
    stcs : dict
        Dictionary mapping condition names to source estimates.
    """
    print("\n[run_beamformer_timecourse] Running time-locked beamformer analysis...")

    stcs = {}
    conditions = _all_conditions(cfg=cfg)

    print(f"[run_beamformer_timecourse] Processing {len(conditions)} conditions")

    for condition in conditions:
        print(f"  - Processing condition: {condition}")

        # Check if this is a contrast
        is_contrast = condition not in cfg.conditions

        if is_contrast:
            # Find the contrast definition
            contrast_def = None
            for contrast in cfg.contrasts:
                if contrast["name"] == condition:
                    contrast_def = contrast
                    break

            if contrast_def is None:
                print(
                    f"    [WARNING] Could not find contrast definition for '{condition}'. Skipping."
                )
                continue

            # Average epochs for each condition in the contrast
            evokeds = []
            for cond_name in contrast_def["conditions"]:
                try:
                    epochs_subset = epochs[cond_name].copy()
                    if len(epochs_subset) == 0:
                        print(
                            f"    [WARNING] No epochs for condition '{cond_name}'. Skipping contrast."
                        )
                        continue
                    evokeds.append(epochs_subset.average())
                except KeyError:
                    print(
                        f"    [WARNING] Condition '{cond_name}' not found in epochs. Skipping contrast."
                    )
                    continue

            if len(evokeds) != len(contrast_def["conditions"]):
                print(
                    f"    [WARNING] Could not load all conditions for contrast. Skipping."
                )
                continue

            # Combine evoked responses with weights
            evoked = mne.combine_evoked(evokeds, weights=contrast_def["weights"])
            print(f"    - Created contrast from {len(evokeds)} conditions")

        else:
            # Simple condition - just average
            try:
                epochs_subset = epochs[condition].copy()
                if len(epochs_subset) == 0:
                    print(
                        f"    [WARNING] No epochs for condition '{condition}'. Skipping."
                    )
                    continue
                evoked = epochs_subset.average()
                print(f"    - Averaged {len(epochs_subset)} epochs")
            except KeyError:
                print(
                    f"    [WARNING] Condition '{condition}' not found in epochs. Skipping."
                )
                continue

        # Set EEG reference if needed
        if "eeg" in cfg.ch_types:
            evoked.set_eeg_reference("average", projection=True)

        # Apply beamformer
        stc = apply_lcmv(evoked, filters)
        stcs[condition] = stc

        print(f"    - STC shape: {stc.data.shape}")

    print(
        f"[run_beamformer_timecourse] Completed. Generated {len(stcs)} source estimates."
    )

    return stcs


def run_beamformer_power(
    epochs: mne.Epochs,
    filters: dict,
    cfg: SimpleNamespace,
) -> Dict[str, mne.SourceEstimate]:
    """Apply beamformer to covariance matrices (power analysis).

    This is the SECONDARY beamformer analysis, following the pattern
    from fit_beamformer.py.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data.
    filters : dict
        LCMV filters from make_lcmv.
    cfg : SimpleNamespace
        Configuration object.

    Returns
    -------
    stcs : dict
        Dictionary mapping condition names to source estimates.
    """
    print("\n[run_beamformer_power] Running power beamformer analysis...")
    print(
        f"  - Time window: {cfg._beamformer_power_tmin} to {cfg._beamformer_power_tmax} s"
    )

    cov_method = getattr(cfg, "_beamformer_cov_method", "empirical")

    def _condition_power(condition):
        """Beamformed power for one condition, or None if it has no epochs."""
        try:
            epochs_subset = epochs[condition].copy()
        except KeyError:
            print(
                f"    [WARNING] Condition '{condition}' not found in epochs. Skipping."
            )
            return None
        if len(epochs_subset) == 0:
            print(f"    [WARNING] No epochs for condition '{condition}'. Skipping.")
            return None
        cov = mne.compute_covariance(
            epochs_subset,
            method=cov_method,
            tmin=cfg._beamformer_power_tmin,
            tmax=cfg._beamformer_power_tmax,
            n_jobs=cfg.n_jobs,
        )
        print(f"    - Computed from {len(epochs_subset)} epochs")
        # apply_lcmv_cov traces over orientations (_compute_power), so this is a
        # scalar, sign-free power estimate even for vector filters.
        return apply_lcmv_cov(cov, filters)

    stcs = {}
    conditions = cfg.conditions  # Only run on base conditions, not contrasts for power

    print(f"\n\n[run_beamformer_power] Processing {len(conditions)} conditions")

    for condition in conditions:
        print(f"\n  - Computing power for condition: {condition}")
        stc = _condition_power(condition)
        if stc is None:
            continue
        stcs[condition] = stc
        print(f"    - STC shape: {stc.data.shape}")

    # CONTRASTS -------------------------------------
    print(f"\n\n[run_beamformer_power] Processing {len(cfg.contrasts)} contrasts")
    for contrast in cfg.contrasts:
        print("-" * 10)
        contrast_name = contrast["name"]
        contrast_conditions = contrast["conditions"]

        print(f"  - Computing contrast: {contrast_name}")
        print(f"    [{contrast}]")

        # Keep weights paired with their STC: a condition that drops out must not
        # shift the remaining conditions onto the wrong weights.
        weighted = []
        for condition, weight in zip(contrast_conditions, contrast["weights"]):
            print(f"    contrast_condition: {condition}")
            stc = stcs.get(condition)
            if stc is None:
                stc = _condition_power(condition)
            if stc is None:
                continue
            weighted.append((stc, weight))

        if len(weighted) != len(contrast_conditions):
            print(
                f"    [WARNING] Could not compute all conditions for contrast "
                f"'{contrast_name}'. Skipping."
            )
            continue

        # Normalised difference: sum(w_i * P_i) / sum(|w_i| * P_i).  Both
        # accumulators start at zero — seeding them with the first STC (as this
        # previously did) double-counts it, turning (A - B) / (A + B) into
        # (2A - B) / (2A + B).
        stc_contrast = weighted[0][0].copy()
        stc_norm = weighted[0][0].copy()
        stc_contrast.data = np.zeros_like(stc_contrast.data)
        stc_norm.data = np.zeros_like(stc_norm.data)

        for stc, weight in weighted:
            stc_contrast.data += weight * stc.data
            stc_norm.data += abs(weight) * stc.data

        # Power is non-negative, so the denominator only vanishes where every
        # contributing condition is exactly zero; leave those vertices at zero
        # rather than emitting inf/nan.
        nonzero = stc_norm.data != 0
        out = np.zeros_like(stc_contrast.data)
        np.divide(stc_contrast.data, stc_norm.data, out=out, where=nonzero)
        stc_contrast.data = out

        stcs[contrast_name] = stc_contrast
        print(f"    - Normalized difference contrast created")

    print(f"[run_beamformer_power] Completed. Generated {len(stcs)} source estimates.")

    return stcs


# --------------------------------------------------------------------------------------
# Result Saving
# --------------------------------------------------------------------------------------


def save_beamformer_results(
    cfg: SimpleNamespace,
    filters: dict,
    stcs: Dict[str, mne.SourceEstimate],
    analysis_type: str,
    source_space: str | None = None,
) -> Dict[str, Path]:
    """Save beamformer results to BIDS derivatives.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object.
    filters : dict
        LCMV filters.
    stcs : dict
        Source estimates for each condition.
    analysis_type : str
        'time' or 'power' to distinguish analysis types.
    source_space : str or None
        'surface' or 'volume'.  Controls the STC/filter naming so both
        reconstructions can coexist.  When ``None``, the first entry of
        ``_beamformer_source_space`` is used (single-space back-compat).

    Returns
    -------
    out_files : dict
        Dictionary mapping condition names to output file paths.
    """
    if source_space is None:
        source_space = resolve_source_spaces(cfg)[0]
    print(
        f"\n[save_beamformer_results] Saving {analysis_type} beamformer results "
        f"({source_space})..."
    )

    subject = cfg.subjects[0]
    session = cfg.sessions[0]
    out_files = {}

    # Construct base BIDS path
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        root=cfg.deriv_root,
        datatype=cfg.datatype,
        check=False,
    )

    # Save filters (once per analysis; volume filters get an acq-vol tag so a
    # combined surface+volume run does not overwrite one with the other).
    if cfg._beamformer_save_filters and analysis_type == "time":
        filter_path = bids_path.copy().update(suffix="lcmv", extension=".h5")
        if source_space == "volume":
            filter_path = filter_path.update(acquisition="vol")
        filters.save(filter_path.fpath, overwrite=True)
        out_files["filters"] = filter_path.fpath

    # Volume STCs are stored under a distinct "+vol" token (and save as "-vl.h5")
    # so downstream tooling can tell them apart from surface "+hemi" ("-stc.h5").
    space_tag = "vol" if source_space == "volume" else "hemi"

    # Save source estimates
    for condition, stc in stcs.items():
        # Create suffix based on analysis type
        cond_sanitized = sanitize_cond_name(condition)
        if analysis_type == "time":
            suffix = f"{cond_sanitized}+lcmv+{space_tag}"
        else:  # power
            suffix = f"{cond_sanitized}+lcmv-power+{space_tag}"

        stc_path = bids_path.copy().update(suffix=suffix)
        print(f"  - Saving {condition} to: {stc_path.fpath}")

        stc.save(stc_path.fpath, ftype="h5", overwrite=True)
        out_files[condition] = stc_path.fpath

    print(f"[save_beamformer_results] Saved {len(stcs)} source estimates")

    return out_files


# --------------------------------------------------------------------------------------
# Report Generation
# --------------------------------------------------------------------------------------


def add_to_report(
    cfg: SimpleNamespace,
    stcs: Dict[str, Path],
    analysis_type: str,
    src: "mne.SourceSpaces | None" = None,
    source_space: str | None = None,
) -> None:
    """Add beamformer results to MNE-BIDS-Pipeline HTML report.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object.
    stcs : dict
        Dictionary mapping condition names to STC file paths.
    analysis_type : str
        'time' or 'power' to distinguish analysis types.
    src : mne.SourceSpaces or None
        Volume source space.  Required when ``source_space == 'volume'``:
        ``report.add_stc`` cannot render volume estimates (it has no ``src``
        parameter), so each volume STC is plotted with
        ``stc.plot(src=..., mode='stat_map')`` and added as a figure instead.
    source_space : str or None
        'surface' or 'volume'.  Selects the rendering path and namespaces the
        report titles/tags so a combined run keeps both.  When ``None``, the
        first entry of ``_beamformer_source_space`` is used.
    """
    if not cfg._beamformer_add_to_report:
        print(f"\n[add_to_report] Report generation disabled. Skipping.")
        return

    if source_space is None:
        source_space = resolve_source_spaces(cfg)[0]
    is_volume = source_space == "volume"

    print(
        f"\n[add_to_report] Adding {analysis_type} beamformer results "
        f"({source_space}) to report..."
    )

    subject = cfg.subjects[0]
    session = cfg.sessions[0]

    # Strip BIDS prefixes if present (report system adds them back)
    subject_clean = (
        subject.replace("sub-", "") if subject.startswith("sub-") else subject
    )
    session_clean = (
        session.replace("ses-", "") if session.startswith("ses-") else session
    )
    print(f"[add_to_report] Clean subject: {subject_clean}, session: {session_clean}")

    # fs subject and subjects_dir
    fs_subject = get_fs_subject(config=cfg, subject=subject, session=session)
    fs_subjects_dir = get_fs_subjects_dir(config=cfg)
    print(
        f"[add_to_report] FreeSurfer subject: {fs_subject}, subjects_dir: {fs_subjects_dir}"
    )

    try:
        with _open_report(
            cfg=cfg,
            exec_params=cfg.exec_params,
            subject=subject_clean,
            session=session_clean,
        ) as report:
            print(f"[add_to_report] Report opened successfully")

            n_added = 0
            for condition, stc_path in stcs.items():
                if condition == "filters":
                    continue  # Skip the filters entry

                print(f"  - Adding {condition} to report")

                # Determine tags (namespaced by source space so a combined
                # surface+volume run keeps both entries in the report).
                tag_prefix = f"beamformer-{analysis_type}-{source_space}"
                tags = (tag_prefix, _sanitize_cond_tag(condition))
                report_title = (
                    f"Beamformer {source_space} ({analysis_type}): {condition}"
                )

                # Add 'contrast' tag if this is a contrast
                if condition not in cfg.conditions:
                    tags = tags + ("contrast",)

                if is_volume:
                    # report.add_stc has no `src` argument and cannot render a
                    # VolSourceEstimate, so plot it with nilearn and add the
                    # resulting figure instead.
                    if src is None:
                        print(
                            f"    [WARNING] Volume STC but no source space provided; "
                            f"skipping report figure for {condition}"
                        )
                        continue
                    import matplotlib.pyplot as plt

                    stc = mne.read_source_estimate(str(stc_path))
                    # nilearn renders a scalar map; a vector volume beamformer
                    # (pick_ori='vector') has three components per grid point.
                    if stc.data.ndim == 3:
                        stc = stc.magnitude()
                    # Representative slice at the peak of the mean-over-sources signal.
                    peak_time = stc.times[
                        int(np.argmax(np.abs(stc.data).mean(axis=0)))
                    ]
                    fig = stc.plot(
                        src=src,
                        subject=fs_subject,
                        subjects_dir=fs_subjects_dir,
                        mode="stat_map",
                        initial_time=peak_time,
                        show=False,
                    )
                    report.add_figure(
                        fig=fig,
                        title=report_title,
                        tags=tags,
                        replace=True,
                    )
                    plt.close(fig)
                else:
                    report.add_stc(
                        stc=stc_path,
                        title=report_title,
                        subject=fs_subject,
                        subjects_dir=fs_subjects_dir,
                        n_time_points=getattr(
                            cfg,
                            "_beamformer_report_n_time_points",
                            cfg.report_stc_n_time_points,
                        ),
                        tags=tags,
                        replace=True,
                    )
                n_added += 1
                print(f"    - tags: {tags}")

            print(
                f"[add_to_report] Successfully added {n_added} source estimates to report"
            )

    except Exception as e:
        # The STCs are already on disk at this point, so a report failure must not
        # take the run down with it.
        print(f"[add_to_report] Warning: Could not add to report: {e}")
        print(f"[add_to_report] Continuing without report update...")


# --------------------------------------------------------------------------------------
# Main Function
# --------------------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="LCMV Beamformer source reconstruction for OPM-MEG data"
    )

    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    return p.parse_args()


def main():
    """Main entry point for beamformer analysis."""

    # Parse command-line arguments
    args = parse_args()

    # load configuration
    #
    # _import_config validates the config but then reduces it to the names the
    # *default* pipeline config declares (`keep_names` in _config_import), which
    # drops every custom `_beamformer_*` / `_run_beamformer` setting this script
    # reads.  Re-merging the config file afterwards puts the private names back
    # (_update_config_from_path setattrs everything not starting with `__`).
    cfg = _import_config(config_path=args.config)
    _update_config_from_path(config=cfg, config_path=args.config)
    cfg.data_type = "meg"
    cfg.datatype = "meg"

    # Check if beamformer is enabled
    if not cfg._run_beamformer:
        print("\n[main] Beamformer disabled in configuration (_run_beamformer=False)")
        print("[main] Exiting without running analysis.")
        return

    # Load data (epochs, forward(s), and the pipeline's noise covariance)
    data = load_beamformer_data(cfg)
    epochs = data["epochs"]
    noise_cov = data["noise_cov"]

    # One info from here on: the projectors that end up in the data covariance
    # and the ones handed to make_lcmv have to be the same objects, or the
    # whitener double-applies them.
    epochs.info.normalize_proj()

    # One rank for both covariances.  make_lcmv requires the data and noise ranks
    # to agree whenever a real noise covariance is supplied, so resolving it once
    # and passing the same dict everywhere removes that whole failure mode.
    rank = resolve_rank(cfg, epochs, noise_cov, data["noise_rank"])

    # Compute data covariance (shared by every source space and both analyses)
    print("\n[main] Computing data covariance matrix...")
    data_cov = mne.compute_covariance(
        epochs,
        method=cfg._beamformer_cov_method,
        rank=rank,
        n_jobs=cfg.n_jobs,
    )
    print(f"[main] Data covariance computed from {len(epochs)} epochs")

    # Reconstruct each requested source space (surface, volume, or both).  The
    # data / noise covariance and rank above are shared; the forward, filters,
    # STCs, saved filenames and report entries are per source space.
    source_spaces = resolve_source_spaces(cfg)
    output_type = getattr(cfg, "_beamformer_output_type", "both")

    for space in source_spaces:
        print("\n" + "#" * 80)
        print(f"SOURCE SPACE: {space.upper()}")
        print("#" * 80)

        forward = data["forwards"][space]

        # Anchor the max-power sign to the cortical normal rather than head +Z.
        # Only surface spaces have a normal to anchor to (MNE falls back to the
        # +Z / superior direction for volume grids), which is why the volume
        # reconstruction is fit with pick_ori='vector' instead — see
        # surface_orient_forward and the per-space _beamformer_pick_ori.
        if space == "surface":
            forward = surface_orient_forward(forward, cfg)

        # Compute LCMV filters (shared by this space's time and power analyses)
        filters = compute_lcmv_filters(
            forward=forward,
            data_cov=data_cov,
            noise_cov=noise_cov,
            info=epochs.info,
            cfg=cfg,
            rank=rank,
            source_space=space,
        )

        # Run Time-locked beamformer --------------------------------
        if output_type in ["time", "both"]:
            print("\n" + "=" * 80)
            print(f"TIME-LOCKED BEAMFORMER ({space})")
            print("=" * 80)
            stcs_time = run_beamformer_timecourse(
                epochs=epochs,
                filters=filters,
                cfg=cfg,
            )

            out_files_time = save_beamformer_results(
                cfg=cfg,
                filters=filters,
                stcs=stcs_time,
                analysis_type="time",
                source_space=space,
            )

            add_to_report(
                cfg=cfg,
                stcs=out_files_time,
                analysis_type="time",
                src=forward["src"],
                source_space=space,
            )

        # Run Power beamformer --------------------------------
        if output_type in ["power", "both"]:
            print("\n" + "=" * 80)
            print(f"POWER BEAMFORMER ({space})")
            print("=" * 80)

            stcs_power = run_beamformer_power(
                epochs=epochs,
                filters=filters,
                cfg=cfg,
            )

            out_files_power = save_beamformer_results(
                cfg=cfg,
                filters=filters,
                stcs=stcs_power,
                analysis_type="power",
                source_space=space,
            )

            add_to_report(
                cfg=cfg,
                stcs=out_files_power,
                analysis_type="power",
                src=forward["src"],
                source_space=space,
            )

    print("\n" + "=" * 80)
    print("BEAMFORMER ANALYSIS COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
