"""Coregistration & forward-model diagnostics for OPM-MEG data.

Sister script to ``run_beamformer.py``.  Loads the same per-subject artifacts
and produces a bundle of diagnostic figures for assessing the quality of the
structural assignment underpinning OPM-MEG source reconstruction:

1. BEM reconstruction slice plots (``mne.viz.plot_bem``) plus a FreeSurfer
   inventory and per-surface geometry metrics.
2. Multi-view ``mne.viz.plot_alignment`` screenshots showing sensors, helmet,
   dense scalp, optical head-shape points and brain surfaces.
3. Digitization-to-scalp distance histogram.
4. Sensitivity maps (``mne.sensitivity_map``) on the inflated cortex.

Designed to run headless on a cluster.  All figures are written to disk under
``{deriv_root}/sub-XX/ses-YY/meg/coreg_diagnostics/`` along with a JSON summary.

Usage:
    python coreg_diagnostics.py --config=/path/to/config.py

Author: Harrison Ritz, 2026
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne_bids import BIDSPath, get_head_mri_trans

# Add mne-bids-pipeline to path for importing utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mne-bids-pipeline"))

from mne_bids_pipeline._config_import import (  # noqa: E402
    _import_config,
    _update_config_from_path,
)
from mne_bids_pipeline._config_utils import (  # noqa: E402
    get_fs_subject,
    get_fs_subjects_dir,
)


# Canonical 3D viewpoints used for the alignment screenshots.  Tuples are
# (azimuth, elevation, roll, distance) as accepted by ``mne.viz.set_3d_view``.
# ``distance`` is left None to let MNE autoscale per scene.
_VIEWS: Dict[str, Dict[str, float]] = {
    "frontal": dict(azimuth=90, elevation=90, roll=0),
    "posterior": dict(azimuth=-90, elevation=90, roll=0),
    "lateral_left": dict(azimuth=180, elevation=90, roll=0),
    "lateral_right": dict(azimuth=0, elevation=90, roll=0),
    "superior": dict(azimuth=90, elevation=0, roll=0),
    "oblique": dict(azimuth=45, elevation=60, roll=0),
}

_DEFAULT_VIEWS: List[str] = [
    "frontal",
    "posterior",
    "lateral_left",
    "lateral_right",
    "superior",
    "oblique",
]


# --------------------------------------------------------------------------------------
# Headless backend setup
# --------------------------------------------------------------------------------------


def _setup_3d_backend() -> None:
    """Configure MNE's 3D backend for off-screen rendering.

    Prefers the pure ``pyvista`` backend for headless/cluster use because
    ``pyvistaqt`` requires a live Qt event loop and attempts to open X windows
    even when ``QT_QPA_PLATFORM=offscreen`` is set, causing BadWindow X errors.
    Falls back to ``pyvistaqt`` only when ``pyvista`` is unavailable.

    Order matters: ``set_3d_backend`` must be called first, then
    ``set_3d_options``.  ``mne.viz.plot_alignment`` reads ``MNE_3D_OPTION_OFFSCREEN``
    (written by ``set_3d_options``) to decide whether to create an on-screen X
    window.  If ``set_3d_options`` is called before the backend is set,
    ``set_3d_backend`` resets those options and ``plot_alignment`` falls back to
    creating a real X window, which triggers a BadWindow error on headless nodes.
    ``stc.plot`` (used in plot_beamformer) is unaffected because Brain always
    passes ``off_screen=True`` directly from ``pyvista.OFF_SCREEN``.
    """
    for backend in ("pyvista", "pyvistaqt"):
        try:
            mne.viz.set_3d_backend(backend)
            print(f"[_setup_3d_backend] Using 3D backend: {backend}")
            break
        except Exception as e:
            print(f"[_setup_3d_backend] Backend {backend!r} unavailable: {e}")
    else:
        print("[_setup_3d_backend] WARNING: no usable 3D backend; 3D figures will fail.")
        return

    try:
        # antialias=False: multisampling requires GPU features absent in software
        # rendering and can cause VTK to fall back to X11 paths.
        mne.viz.set_3d_options(offscreen=True, depth_peeling=False, antialias=False)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[_setup_3d_backend] Could not set 3D options: {e}")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


def load_config(config_path: str) -> SimpleNamespace:
    """Load configuration from a Python config file.

    Mirrors :func:`run_beamformer.load_config` so the two scripts can share
    the same config files.

    Parameters
    ----------
    config_path : str
        Path to the configuration Python file.

    Returns
    -------
    cfg : SimpleNamespace
        Configuration object with all settings.
    """
    print(f"\n[load_config] Loading configuration from: {config_path}")

    cfg = SimpleNamespace()
    _update_config_from_path(config=cfg, config_path=config_path)

    print(f"[load_config] Subject: {cfg.subjects[0]}, Session: {cfg.sessions[0]}")
    print(f"[load_config] Task: {cfg.task}")
    print(
        f"[load_config] Coreg diagnostics enabled: "
        f"{getattr(cfg, '_run_coreg_diagnostics', True)}"
    )

    return cfg


# --------------------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------------------


def _bids_path(cfg: SimpleNamespace) -> BIDSPath:
    """Build the standard derivative ``BIDSPath`` for the configured subject."""
    return BIDSPath(
        subject=cfg.subjects[0],
        session=cfg.sessions[0],
        task=cfg.task,
        root=cfg.deriv_root,
        datatype=getattr(cfg, "datatype", "meg"),
        check=False,
    )


def _diag_paths(cfg: SimpleNamespace) -> Dict[str, Any]:
    """Resolve the output folder and BIDS-style basename for diagnostics."""
    bp = _bids_path(cfg)
    out_dir = Path(bp.directory) / "coreg_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_clean = cfg.subjects[0].replace("sub-", "")
    session_clean = cfg.sessions[0].replace("ses-", "")
    basename = f"sub-{subject_clean}_ses-{session_clean}_task-{cfg.task}"

    print(f"[_diag_paths] Output directory: {out_dir}")
    return {
        "out_dir": out_dir,
        "bids_path": bp,
        "subject_clean": subject_clean,
        "session_clean": session_clean,
        "basename": basename,
    }


def _save_fig(
    fig: Any,
    basename: str,
    cfg: SimpleNamespace,
    out_dir: Path,
    kind: str = "mpl",
) -> List[Path]:
    """Save a figure to disk using the configured output formats.

    Parameters
    ----------
    fig
        Either a matplotlib ``Figure`` (kind="mpl"), an ``mne.viz.Brain``
        (kind="brain"), or an object with a ``.plotter`` attribute returned by
        ``mne.viz.plot_alignment`` (kind="pyvista").
    basename
        Filename stem (no extension).
    cfg
        Loaded configuration; used for ``_coreg_diag_dpi`` and
        ``_coreg_diag_output_formats``.
    out_dir
        Destination folder.
    kind
        One of ``"mpl"``, ``"pyvista"``, ``"brain"``.

    Returns
    -------
    paths : list of Path
        Files actually written.
    """
    formats = list(getattr(cfg, "_coreg_diag_output_formats", ["png"]))
    dpi = int(getattr(cfg, "_coreg_diag_dpi", 200))
    written: List[Path] = []

    if kind == "mpl":
        for ext in formats:
            path = out_dir / f"{basename}.{ext}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            written.append(path)
        plt.close(fig)
    elif kind in ("pyvista", "brain"):
        # 3D screenshots are raster only.
        non_png = [ext for ext in formats if ext.lower() != "png"]
        if non_png:
            warnings.warn(
                f"[_save_fig] 3D figure '{basename}' supports PNG only; "
                f"ignoring formats {non_png}.",
                stacklevel=2,
            )
        path = out_dir / f"{basename}.png"
        try:
            if kind == "pyvista":
                fig.plotter.screenshot(str(path))
                fig.plotter.close()
            else:  # brain
                fig.save_image(str(path), mode="rgb")
                fig.close()
            written.append(path)
        except Exception as e:
            err_path = out_dir / f"{basename}.error.txt"
            err_path.write_text(f"Failed to save 3D figure: {e}\n")
            print(f"[_save_fig] WARNING: 3D save failed for {basename}: {e}")
    else:
        raise ValueError(f"Unknown figure kind: {kind!r}")

    return written


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------


def _find_bem_solution(subjects_dir: str, fs_subject: str) -> Optional[Path]:
    """Glob for an existing BEM solution file in the FreeSurfer subject dir."""
    bem_dir = Path(subjects_dir) / fs_subject / "bem"
    if not bem_dir.exists():
        return None
    matches = sorted(bem_dir.glob("*bem-sol.fif"))
    if not matches:
        return None
    for m in matches:
        if m.name.startswith(fs_subject):
            return m
    return matches[0]


def _find_source_space(subjects_dir: str, fs_subject: str) -> Optional[Path]:
    """Glob for an existing source space file in the FreeSurfer subject dir."""
    bem_dir = Path(subjects_dir) / fs_subject / "bem"
    if not bem_dir.exists():
        return None
    matches = sorted(bem_dir.glob("*-src.fif"))
    if not matches:
        return None
    for m in matches:
        if m.name.startswith(fs_subject):
            return m
    return matches[0]


def _compute_forward(
    cfg: SimpleNamespace,
    info: mne.Info,
    trans: Optional[mne.transforms.Transform],
    fs_subject: str,
    fs_subjects_dir: str,
    fwd_path: Path,
) -> mne.Forward:
    """Build a forward solution on the fly when no ``-fwd.fif`` is on disk.

    Loads existing BEM solution / source space when present in the FreeSurfer
    subject directory; otherwise builds them from FreeSurfer surfaces.
    """
    print("[_compute_forward] No forward solution on disk; computing one.")
    if trans is None:
        raise RuntimeError(
            "Cannot compute forward solution without a head-MRI transform."
        )

    # BEM solution -----------------------------------------------------------
    bem_path = _find_bem_solution(fs_subjects_dir, fs_subject)
    if bem_path is not None:
        print(f"[_compute_forward] Loading BEM solution: {bem_path}")
        bem = mne.read_bem_solution(bem_path)
    else:
        conductivity = tuple(
            getattr(cfg, "_coreg_diag_bem_conductivity", (0.3,))
        )
        ico = int(getattr(cfg, "_coreg_diag_bem_ico", 4))
        print(
            f"[_compute_forward] Building BEM model "
            f"(conductivity={conductivity}, ico={ico})"
        )
        model = mne.make_bem_model(
            subject=fs_subject,
            ico=ico,
            conductivity=conductivity,
            subjects_dir=fs_subjects_dir,
        )
        bem = mne.make_bem_solution(model)

    # Source space -----------------------------------------------------------
    src_path = _find_source_space(fs_subjects_dir, fs_subject)
    if src_path is not None:
        print(f"[_compute_forward] Loading source space: {src_path}")
        src = mne.read_source_spaces(src_path)
    else:
        spacing = getattr(cfg, "_coreg_diag_src_spacing", "oct6")
        print(f"[_compute_forward] Setting up source space (spacing={spacing})")
        src = mne.setup_source_space(
            subject=fs_subject,
            spacing=spacing,
            subjects_dir=fs_subjects_dir,
            n_jobs=getattr(cfg, "n_jobs", 1),
            add_dist=False,
        )

    # Forward solution -------------------------------------------------------
    print("[_compute_forward] Computing forward solution...")
    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        n_jobs=getattr(cfg, "n_jobs", 1),
    )

    # Persist for downstream pipeline steps.
    try:
        fwd_path.parent.mkdir(parents=True, exist_ok=True)
        mne.write_forward_solution(fwd_path, fwd, overwrite=True)
        print(f"[_compute_forward] Saved forward solution to {fwd_path}")
    except Exception as e:
        print(f"[_compute_forward] WARNING: could not save forward: {e}")

    return fwd


def load_diagnostic_data(cfg: SimpleNamespace) -> Dict[str, Any]:
    """Load the per-subject artifacts required for the diagnostics.

    Returns
    -------
    data : dict
        Keys: ``info``, ``forward`` (None on failure), ``trans`` (None on
        failure), ``fs_subject``, ``fs_subjects_dir``, ``bids_path``.
    """
    print("\n[load_diagnostic_data] Loading data...")
    bp = _bids_path(cfg)

    # Info — required.
    epochs_path = bp.copy().update(
        suffix="epo", processing="clean", extension=".fif"
    )
    if not epochs_path.fpath.exists():
        raise FileNotFoundError(
            f"Clean epochs not found at {epochs_path.fpath}\n"
            f"Run preprocessing first."
        )
    info = mne.io.read_info(epochs_path.fpath)
    print(f"[load_diagnostic_data] Loaded info from {epochs_path.fpath}")

    # FreeSurfer paths.
    fs_subject = get_fs_subject(
        config=cfg, subject=cfg.subjects[0], session=cfg.sessions[0]
    )
    fs_subjects_dir = get_fs_subjects_dir(config=cfg)
    print(
        f"[load_diagnostic_data] FreeSurfer subject={fs_subject}, "
        f"subjects_dir={fs_subjects_dir}"
    )

    # Trans — optional.
    trans: Optional[mne.transforms.Transform]
    try:
        t1_bp = BIDSPath(
            subject=cfg.subjects[0],
            session=cfg.sessions[0],
            root=cfg.bids_root,
            datatype="anat",
            suffix="T1w",
            extension=".nii.gz",
            check=False,
        )
        trans = get_head_mri_trans(
            bp,
            fs_subject=fs_subject,
            fs_subjects_dir=fs_subjects_dir,
            t1_bids_path=t1_bp,
        )
        print("[load_diagnostic_data] Loaded head-MRI trans from BIDS landmarks")
    except Exception as e:
        print(f"[load_diagnostic_data] WARNING: could not load trans: {e}")
        trans = None

    # Forward — optional, computed on the fly when missing.
    forward: Optional[mne.Forward]
    fwd_path = bp.copy().update(suffix="fwd", extension=".fif")
    if fwd_path.fpath.exists():
        print(f"[load_diagnostic_data] Loading forward solution: {fwd_path.fpath}")
        forward = mne.read_forward_solution(fwd_path.fpath)
    else:
        try:
            forward = _compute_forward(
                cfg, info, trans, fs_subject, fs_subjects_dir,
                Path(fwd_path.fpath),
            )
        except Exception as e:
            print(f"[load_diagnostic_data] WARNING: forward not available: {e}")
            forward = None

    return {
        "info": info,
        "forward": forward,
        "trans": trans,
        "fs_subject": fs_subject,
        "fs_subjects_dir": fs_subjects_dir,
        "bids_path": bp,
    }


# --------------------------------------------------------------------------------------
# BEM diagnostics
# --------------------------------------------------------------------------------------


_BEM_SURFACES = ("inner_skull", "outer_skull", "outer_skin")
_FS_INVENTORY_FILES = (
    "bem/inner_skull.surf",
    "bem/outer_skull.surf",
    "bem/outer_skin.surf",
    "mri/T1.mgz",
    "mri/aparc+aseg.mgz",
    "surf/lh.pial",
    "surf/rh.pial",
    "surf/lh.white",
    "surf/rh.white",
    "surf/lh.inflated",
    "surf/rh.inflated",
)


def _inventory_freesurfer(
    subjects_dir: str, fs_subject: str
) -> Dict[str, Dict[str, Any]]:
    """Report which expected FreeSurfer artifacts exist for ``fs_subject``."""
    root = Path(subjects_dir) / fs_subject
    inventory: Dict[str, Dict[str, Any]] = {}
    for rel in _FS_INVENTORY_FILES:
        p = root / rel
        inventory[rel] = {
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else None,
        }
    # Also list dynamic globs.
    for pattern, key in (
        ("bem/*bem-sol.fif", "bem_solution_glob"),
        ("bem/*-src.fif", "source_space_glob"),
        ("bem/*fiducials.fif", "fiducials_glob"),
    ):
        matches = sorted(str(p.relative_to(root)) for p in root.glob(pattern))
        inventory[key] = {"exists": bool(matches), "matches": matches}
    return inventory


def _signed_volume(verts: np.ndarray, tris: np.ndarray) -> float:
    """Closed-surface volume via the divergence theorem (in input units)."""
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    return float(np.abs(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum()) / 6.0)


def _surface_metrics(
    subjects_dir: str, fs_subject: str
) -> Dict[str, Dict[str, Any]]:
    """Compute per-surface volume and inter-surface min/max gaps (mm)."""
    metrics: Dict[str, Dict[str, Any]] = {}
    surfs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for name in _BEM_SURFACES:
        path = Path(subjects_dir) / fs_subject / "bem" / f"{name}.surf"
        if not path.exists():
            metrics[name] = {"exists": False}
            continue
        try:
            verts, tris = mne.read_surface(str(path))
            surfs[name] = (verts, tris)
            metrics[name] = {
                "exists": True,
                "n_vertices": int(verts.shape[0]),
                "n_triangles": int(tris.shape[0]),
                "volume_mm3": _signed_volume(verts, tris),
            }
        except Exception as e:
            metrics[name] = {"exists": True, "error": str(e)}

    # Inter-surface min/max distance (mm) using a kd-tree.
    try:
        from scipy.spatial import cKDTree
    except Exception:
        cKDTree = None  # type: ignore

    if cKDTree is not None:
        ordered = [n for n in _BEM_SURFACES if n in surfs]
        for inner, outer in zip(ordered, ordered[1:]):
            v_in = surfs[inner][0]
            v_out = surfs[outer][0]
            tree = cKDTree(v_out)
            dists, _ = tree.query(v_in, k=1)
            metrics[f"{inner}__to__{outer}_mm"] = {
                "min": float(dists.min()),
                "max": float(dists.max()),
                "mean": float(dists.mean()),
            }
    return metrics


def _overlay_bem_on_t1_with_nilearn(
    subjects_dir: str,
    fs_subject: str,
    out_path: Path,
    figsize: Tuple[float, float],
) -> Optional[Path]:
    """Plot T1 slices with BEM-surface contours overlaid via nilearn.

    Returns the saved path on success or None if nilearn / inputs unavailable.
    """
    try:
        import nibabel as nib
        from nilearn import plotting as nlp
    except ImportError:
        print("[_overlay_bem_on_t1_with_nilearn] nilearn/nibabel unavailable; skipping.")
        return None

    t1_path = Path(subjects_dir) / fs_subject / "mri" / "T1.mgz"
    if not t1_path.exists():
        print(f"[_overlay_bem_on_t1_with_nilearn] T1 missing: {t1_path}")
        return None

    fig = plt.figure(figsize=figsize)
    try:
        display = nlp.plot_anat(
            str(t1_path),
            display_mode="ortho",
            figure=fig,
            draw_cross=False,
            annotate=True,
            title=f"{fs_subject}: T1 + BEM surfaces",
        )
        # Build a label image per BEM surface and overlay as contours.
        t1_img = nib.load(str(t1_path))
        affine_inv = np.linalg.inv(t1_img.affine)
        shape = t1_img.shape
        for name, color in zip(
            _BEM_SURFACES, ("red", "yellow", "cyan")
        ):
            surf_path = Path(subjects_dir) / fs_subject / "bem" / f"{name}.surf"
            if not surf_path.exists():
                continue
            verts, _ = mne.read_surface(str(surf_path))
            # FreeSurfer surfaces are in MRI surface RAS (mm); convert to voxel
            # space using the T1 affine.
            verts_h = np.hstack([verts, np.ones((verts.shape[0], 1))])
            vox = (affine_inv @ verts_h.T).T[:, :3]
            mask = np.zeros(shape, dtype=np.uint8)
            ix = np.clip(np.round(vox[:, 0]).astype(int), 0, shape[0] - 1)
            iy = np.clip(np.round(vox[:, 1]).astype(int), 0, shape[1] - 1)
            iz = np.clip(np.round(vox[:, 2]).astype(int), 0, shape[2] - 1)
            mask[ix, iy, iz] = 1
            label_img = nib.Nifti1Image(mask, t1_img.affine)
            display.add_contours(label_img, levels=[0.5], colors=color, linewidths=0.5)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    except Exception as e:
        print(f"[_overlay_bem_on_t1_with_nilearn] failed: {e}")
        plt.close(fig)
        return None
    plt.close(fig)
    return out_path


def run_bem_diagnostics(
    cfg: SimpleNamespace,
    fs_subject: str,
    fs_subjects_dir: str,
    out_dir: Path,
    basename: str,
) -> Dict[str, Any]:
    """Render BEM slice plots and compute geometry metrics."""
    print("\n[run_bem_diagnostics] Rendering BEM slice plots...")
    figsize = tuple(getattr(cfg, "_coreg_diag_figsize", (10, 10)))

    plot_paths: List[str] = []
    for orientation in ("coronal", "sagittal", "axial"):
        try:
            fig = mne.viz.plot_bem(
                subject=fs_subject,
                subjects_dir=fs_subjects_dir,
                brain_surfaces="white",
                orientation=orientation,
                show=False,
            )
            if not isinstance(fig, plt.Figure):
                # Some MNE versions return a list of figures.
                fig = fig[0] if isinstance(fig, (list, tuple)) else fig
            fig.set_size_inches(*figsize)
            written = _save_fig(
                fig, f"{basename}_desc-bem-{orientation}", cfg, out_dir, kind="mpl"
            )
            plot_paths.extend(str(p) for p in written)
        except Exception as e:
            print(f"[run_bem_diagnostics] {orientation} failed: {e}")

    inventory = _inventory_freesurfer(fs_subjects_dir, fs_subject)
    metrics = _surface_metrics(fs_subjects_dir, fs_subject)

    nilearn_path: Optional[str] = None
    if getattr(cfg, "_coreg_diag_use_nilearn", True):
        out_path = out_dir / f"{basename}_desc-bem-nilearn-overlay.png"
        result = _overlay_bem_on_t1_with_nilearn(
            fs_subjects_dir, fs_subject, out_path, figsize
        )
        nilearn_path = str(result) if result is not None else None

    return {
        "plot_bem": plot_paths,
        "inventory": inventory,
        "metrics": metrics,
        "nilearn_overlay": nilearn_path,
    }


# --------------------------------------------------------------------------------------
# Alignment diagnostics
# --------------------------------------------------------------------------------------


def _available_surfaces(fs_subjects_dir: str, fs_subject: str) -> Dict[str, float]:
    """Build the ``surfaces`` kwarg for ``plot_alignment`` from on-disk files."""
    bem_dir = Path(fs_subjects_dir) / fs_subject / "bem"
    surf_dir = Path(fs_subjects_dir) / fs_subject / "surf"

    candidate: Dict[str, Tuple[List[Path], float]] = {
        "head-dense": ([bem_dir / "outer_skin.surf",
                        bem_dir / f"{fs_subject}-head-dense.fif"], 0.4),
        "head": ([bem_dir / "outer_skin.surf",
                  bem_dir / f"{fs_subject}-head.fif"], 0.4),
        "inner_skull": ([bem_dir / "inner_skull.surf"], 0.5),
        "brain": ([surf_dir / "lh.pial", surf_dir / "rh.pial"], 0.6),
    }
    surfaces: Dict[str, float] = {}
    # Prefer head-dense over head if both are present.
    for name in ("head-dense", "head"):
        paths, alpha = candidate[name]
        if any(p.exists() for p in paths):
            surfaces[name] = alpha
            break
    for name in ("inner_skull", "brain"):
        paths, alpha = candidate[name]
        if all(p.exists() for p in paths):
            surfaces[name] = alpha
    return surfaces


def _render_rotation_gif(fig: Any, out_path: Path, n_frames: int = 36) -> Optional[Path]:
    """Render a simple 360° rotating-azimuth GIF of an alignment figure."""
    try:
        plotter = fig.plotter
        plotter.open_gif(str(out_path))
        for i in range(n_frames):
            azimuth = (360.0 / n_frames) * i
            mne.viz.set_3d_view(figure=fig, azimuth=azimuth, elevation=90, roll=0)
            plotter.write_frame()
        plotter.close()
        return out_path
    except Exception as e:
        print(f"[_render_rotation_gif] failed: {e}")
        return None


def run_alignment_diagnostics(
    cfg: SimpleNamespace,
    info: mne.Info,
    trans: Optional[mne.transforms.Transform],
    fwd: Optional[mne.Forward],
    fs_subject: str,
    fs_subjects_dir: str,
    out_dir: Path,
    basename: str,
) -> Dict[str, Any]:
    """Render multi-view ``plot_alignment`` screenshots."""
    print("\n[run_alignment_diagnostics] Rendering alignment views...")

    if trans is None:
        print("[run_alignment_diagnostics] skipped: trans=None (no head-MRI transform available)")
        return {"skipped": "no trans"}
    
    print('trans:', trans)

    surfaces = _available_surfaces(fs_subjects_dir, fs_subject)
    if not surfaces:
        surfaces = {"head": 0.4}
    print(f"[run_alignment_diagnostics] surfaces={surfaces}")

    views = list(getattr(cfg, "_coreg_diag_alignment_views", _DEFAULT_VIEWS))
    plot_paths: List[str] = []
    gif_path: Optional[str] = None

    try:
        print('plot_algnment kwargs: surfaces=', surfaces)
        print('subject=', fs_subject, 'subjects_dir=', fs_subjects_dir)
        fig = mne.viz.plot_alignment(
            info=info,
            trans=trans,
            subject=fs_subject,
            subjects_dir=fs_subjects_dir,
            # surfaces=surfaces,
            surfaces="head-dense",
            meg='sensors',
            dig=True,
            show_axes=True,
        )
        print('plot_alignment rendered successfully')
    except Exception as e:
        print(f"[run_alignment_diagnostics] plot_alignment failed: {e}")
        return {"error": str(e), "surfaces": surfaces, "views": views}

    for view in views:
        if view not in _VIEWS:
            print(f"[run_alignment_diagnostics] unknown view {view!r}; skipping")
            continue
        try:
            print(f"[run_alignment_diagnostics] Rendering view: {view}")
            mne.viz.set_3d_view(figure=fig, **_VIEWS[view])
            path = out_dir / f"{basename}_desc-align-{view}.png"
            try:
                fig.plotter.screenshot(str(path))
                plot_paths.append(str(path))
            except Exception as e:
                print(f"[run_alignment_diagnostics] screenshot {view} failed: {e}")
        except Exception as e:
            print(f"[run_alignment_diagnostics] set_3d_view {view} failed: {e}")

    if getattr(cfg, "_coreg_diag_make_gif", False):
        gp = _render_rotation_gif(
            fig, out_dir / f"{basename}_desc-align-rotation.gif"
        )
        gif_path = str(gp) if gp is not None else None

    try:
        fig.plotter.close()
    except Exception:
        pass

    return {
        "plot_alignment": plot_paths,
        "gif": gif_path,
        "surfaces": surfaces,
        "views": views,
    }


# --------------------------------------------------------------------------------------
# Head-point distance diagnostic
# --------------------------------------------------------------------------------------


def run_headpoint_distance_diagnostic(
    cfg: SimpleNamespace,
    info: mne.Info,
    trans: Optional[mne.transforms.Transform],
    fs_subject: str,
    fs_subjects_dir: str,
    out_dir: Path,
    basename: str,
) -> Dict[str, Any]:
    """Histogram + summary of digitization → scalp distances (mm)."""
    print("\n[run_headpoint_distance_diagnostic] Computing dig → scalp distances...")

    if trans is None:
        return {"skipped": "no trans"}
    if not info.get("dig"):
        return {"skipped": "no dig points in info"}

    try:
        from mne.surface import _DistanceQuery, _get_head_surface

        # Get head-shape digitization points (extra + HPI) in head coords
        from mne.io.constants import FIFF as _FIFF
        head_pts = np.array([
            d["r"] for d in info["dig"]
            if d["kind"] in (
                _FIFF.FIFFV_POINT_EXTRA,
                _FIFF.FIFFV_POINT_HPI,
            )
        ])
        if head_pts.size == 0:
            return {"skipped": "no head-shape digitization points"}

        # Transform dig points from head to MRI surface coords
        head_mri_t = mne.transforms._get_trans(trans, "head", "mri")[0]
        mri_pts = mne.transforms.apply_trans(head_mri_t, head_pts)

        # Load dense scalp surface in MRI coords and compute nearest distances
        scalp = _get_head_surface(
            subject=fs_subject,
            source=["head-dense", "head"],
            subjects_dir=fs_subjects_dir,
            on_defects="warn",
        )
        dists_m, _ = _DistanceQuery(scalp["rr"]).query(mri_pts)
    except Exception as e:
        print(f"[run_headpoint_distance_diagnostic] failed: {e}")
        return {"error": str(e)}

    dists_mm = dists_m * 1e3
    figsize = tuple(getattr(cfg, "_coreg_diag_figsize", (10, 10)))
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(dists_mm, bins=40, color="steelblue", edgecolor="black")
    ax.axvline(np.median(dists_mm), color="red", linestyle="--", label="median")
    ax.set_xlabel("Digitization → scalp distance (mm)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"{fs_subject}: dig→scalp distance "
        f"(n={len(dists_mm)}, mean={dists_mm.mean():.2f} mm)"
    )
    ax.legend()
    written = _save_fig(
        fig, f"{basename}_desc-dig-distance-hist", cfg, out_dir, kind="mpl"
    )

    return {
        "n_points": int(dists_mm.size),
        "mean_mm": float(dists_mm.mean()),
        "median_mm": float(np.median(dists_mm)),
        "p95_mm": float(np.percentile(dists_mm, 95)),
        "max_mm": float(dists_mm.max()),
        "image": [str(p) for p in written],
    }


# --------------------------------------------------------------------------------------
# Sensitivity map
# --------------------------------------------------------------------------------------


def run_sensitivity_diagnostics(
    cfg: SimpleNamespace,
    forward: Optional[mne.Forward],
    fs_subject: str,
    fs_subjects_dir: str,
    out_dir: Path,
    basename: str,
) -> Dict[str, Any]:
    """Compute and visualise sensitivity maps from the forward solution."""
    print("\n[run_sensitivity_diagnostics] Computing sensitivity maps...")
    if forward is None:
        return {"skipped": "no forward"}

    ch_types = list(getattr(cfg, "ch_types", ["mag"]))
    modes = list(getattr(cfg, "_coreg_diag_sensitivity_modes", ["free", "radiality"]))

    results: Dict[str, Any] = {}
    hemi_views = [
        ("lh", "lateral"),
        ("lh", "medial"),
        ("rh", "lateral"),
        ("rh", "medial"),
    ]

    for ch_type in ch_types:
        for mode in modes:
            key = f"{ch_type}_{mode}"
            entry: Dict[str, Any] = {"images": [], "stc": None}
            try:
                stc = mne.sensitivity_map(forward, ch_type=ch_type, mode=mode)
            except Exception as e:
                print(
                    f"[run_sensitivity_diagnostics] sensitivity_map "
                    f"({ch_type}, {mode}) failed: {e}"
                )
                results[key] = {"error": str(e)}
                continue

            stc_stem = (
                out_dir / f"{basename}_desc-sensitivity-{ch_type}-{mode}"
            )
            try:
                stc.save(str(stc_stem), ftype="h5", overwrite=True)
                entry["stc"] = str(stc_stem) + "-stc.h5"
            except Exception as e:
                print(f"[run_sensitivity_diagnostics] STC save failed: {e}")

            for hemi, view in hemi_views:
                try:
                    brain = stc.plot(
                        subjects_dir=fs_subjects_dir,
                        subject=fs_subject,
                        hemi=hemi,
                        views=view,
                        surface="inflated",
                        colormap="magma",
                        clim=dict(kind="percent", lims=[5, 50, 95]),
                        background="white",
                        size=(800, 800),
                        time_viewer=False,
                        show_traces=False,
                    )
                    fig_name = (
                        f"{basename}_desc-sensitivity-{ch_type}-{mode}"
                        f"-{hemi}-{view}"
                    )
                    written = _save_fig(brain, fig_name, cfg, out_dir, kind="brain")
                    entry["images"].extend(str(p) for p in written)
                except Exception as e:
                    print(
                        f"[run_sensitivity_diagnostics] brain plot "
                        f"{ch_type}/{mode}/{hemi}/{view} failed: {e}"
                    )
            results[key] = entry

    return results


# --------------------------------------------------------------------------------------
# JSON report
# --------------------------------------------------------------------------------------


def _json_safe(obj: Any) -> Any:
    """Recursively coerce numpy / Path objects to JSON-friendly types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def build_json_report(
    cfg: SimpleNamespace, results: Dict[str, Any], paths: Dict[str, Any]
) -> Path:
    """Write ``coreg_diagnostics_report.json`` summarising the run."""
    out_path = paths["out_dir"] / "coreg_diagnostics_report.json"
    payload = {
        "version": getattr(cfg, "_version", None),
        "subject": paths["subject_clean"],
        "session": paths["session_clean"],
        "task": cfg.task,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "results": _json_safe(results),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[build_json_report] Wrote {out_path}")
    return out_path


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Coregistration & forward-model diagnostics for OPM-MEG data"
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    return p.parse_args()


def main() -> None:
    """Main entry point for coregistration diagnostics."""
    args = parse_args()

    # Load configuration (mirror run_beamformer.py).
    cfg = _import_config(config_path=args.config)
    _update_config_from_path(config=cfg, config_path=args.config)
    cfg.data_type = "meg"
    cfg.datatype = "meg"

    if not getattr(cfg, "_run_coreg_diagnostics", True):
        print(
            "\n[main] Coreg diagnostics disabled in configuration "
            "(_run_coreg_diagnostics=False). Exiting."
        )
        return

    _setup_3d_backend()

    data = load_diagnostic_data(cfg)
    paths = _diag_paths(cfg)
    results: Dict[str, Any] = {}

    sections = (
        ("bem", getattr(cfg, "_coreg_diag_run_bem", True),
         lambda: run_bem_diagnostics(
             cfg, data["fs_subject"], data["fs_subjects_dir"],
             paths["out_dir"], paths["basename"])),
        ("alignment", getattr(cfg, "_coreg_diag_run_alignment", True),
         lambda: run_alignment_diagnostics(
             cfg, data["info"], data["trans"], data["fs_subject"],
             data["fs_subjects_dir"], paths["out_dir"], paths["basename"])),
        ("headpoint", getattr(cfg, "_coreg_diag_run_headpoint", True),
         lambda: run_headpoint_distance_diagnostic(
             cfg, data["info"], data["trans"], data["fs_subject"],
             data["fs_subjects_dir"], paths["out_dir"], paths["basename"])),
        ("sensitivity", getattr(cfg, "_coreg_diag_run_sensitivity", True),
         lambda: run_sensitivity_diagnostics(
             cfg, data["forward"], data["fs_subject"], data["fs_subjects_dir"],
             paths["out_dir"], paths["basename"])),
    )

    for name, enabled, runner in sections:
        if not enabled:
            results[name] = {"skipped": "disabled in config"}
            continue
        print("\n" + "=" * 80)
        print(f"COREG DIAGNOSTICS: {name.upper()}")
        print("=" * 80)
        try:
            results[name] = runner()
        except Exception as e:
            print(f"[main] section {name!r} failed: {e}")
            results[name] = {"error": str(e)}

    build_json_report(cfg, results, paths)

    print("\n" + "=" * 80)
    print("COREG DIAGNOSTICS COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
