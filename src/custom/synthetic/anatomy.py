"""Synthetic anatomy: a FreeSurfer subject directory built from analytic shapes.

The pipeline's source steps never ask FreeSurfer for anything they cannot get
from a handful of files, so a usable "recon" is a small, well-defined set:

===============================  ==================================================
File                             Consumer
===============================  ==================================================
``mri/T1.mgz``                   ``coreg``, ``get_head_mri_trans``, report slices
``mri/transforms/talairach.xfm`` volume morphing to the group template
``surf/{lh,rh}.white``           ``mne.setup_source_space``
``surf/{lh,rh}.sphere``          ``mne.setup_source_space`` (spacing lookup)
``surf/{lh,rh}.sphere.reg``      surface morphing to the group template
``surf/{lh,rh}.pial``            source-estimate plotting
``surf/{lh,rh}.{curv,sulc}``     surface plotting background
``bem/inner_skull.surf``         ``mne.make_bem_model`` (single-shell, OPM)
``bem/{outer_skull,outer_skin}`` three-layer BEM, if ever requested
``bem/<subject>-head.fif``       ``mne.coreg.Coregistration``, ``plot_alignment``
``bem/<subject>-fiducials.fif``  ``mne.coreg.Coregistration(fiducials="auto")``
===============================  ==================================================

Nothing here needs FreeSurfer to be installed.

Coordinates
-----------
All surfaces are written in **MRI surface RAS** ("tkrRAS"), in millimetres,
which is what FreeSurfer surface files hold.  The T1 is a conformed
256 x 256 x 256 1 mm LIA volume whose scanner-RAS affine is identical to its
tkrRAS affine, so the two frames coincide and landmark round-trips through
``mne_bids.get_anat_landmarks`` / ``mne_bids.get_head_mri_trans`` are exact.

Author: Harrison Ritz (2025)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._geometry import ellipsoid, folding_field, icosphere


__all__ = ["HeadModel", "build_head_model", "write_freesurfer_subject"]


# Conformed FreeSurfer volume: 256^3, 1 mm, LIA.  With c_ras = 0 the scanner
# affine equals vox2ras_tkr, so surface RAS and scanner RAS are the same frame.
CONFORMED_SHAPE = (256, 256, 256)
CONFORMED_AFFINE = np.array(
    [
        [-1.0, 0.0, 0.0, 128.0],
        [0.0, 0.0, 1.0, -128.0],
        [0.0, -1.0, 0.0, 128.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Tissue intensities, chosen to look like a (very) stylised conformed T1.
_INTENSITY = dict(background=0, scalp=85, skull=25, csf=45, gray=105, white=160)

# Surface resolutions.  ico5 (10242 vertices) is the smallest icosahedral mesh
# that supports a ``spacing="oct6"`` source space; the BEM shells only ever get
# downsampled to ico4 by ``mne.make_bem_model``, so writing them any finer is
# wasted bytes in the repository.
CORTEX_GRADE = 5
BEM_GRADE = 4


@dataclass
class HeadModel:
    """Analytic head geometry, in metres, in MRI surface RAS.

    Attributes
    ----------
    center : ndarray, shape (3,)
        Centre of the nested tissue ellipsoids.
    scalp_axes, outer_skull_axes, inner_skull_axes : ndarray, shape (3,)
        Semi-axes of the three BEM shells.
    hemi_axes : ndarray, shape (3,)
        Semi-axes of one cortical hemisphere.
    hemi_offset : float
        Lateral (+/- x) displacement of each hemisphere from ``center``.
    fold_amplitude, fold_lobes : float
        Corrugation of the cortical surface (see ``_geometry.folding_field``).
    seed : int
        Seed the model was generated from; recorded for provenance.
    """

    center: np.ndarray
    scalp_axes: np.ndarray
    outer_skull_axes: np.ndarray
    inner_skull_axes: np.ndarray
    hemi_axes: np.ndarray
    hemi_offset: float
    # Deep enough that a useful fraction of the surface normals are close to
    # tangential.  This is not cosmetic: on a near-spherical conductor a
    # radially oriented dipole is magnetically almost silent, so a cortex whose
    # normals are all radial cannot host a source any MEG analysis could find.
    # At 8 mm / 16 lobes roughly a third of vertices have |n . radial| < 0.5.
    fold_amplitude: float = 8.0e-3
    fold_lobes: float = 16.0
    cortical_thickness: float = 2.5e-3
    seed: int = 0
    _fiducials: dict = field(default_factory=dict, repr=False)

    # -- landmarks ---------------------------------------------------------

    def project_to_scalp(self, direction) -> np.ndarray:
        """Project a direction from ``center`` onto the scalp ellipsoid."""
        d = np.asarray(direction, float)
        t = 1.0 / np.sqrt(np.sum((d / self.scalp_axes) ** 2))
        return self.center + d * t

    @property
    def fiducials(self) -> dict[str, np.ndarray]:
        """Nasion / LPA / RPA on the scalp, in MRI surface RAS metres.

        The landmarks are deliberately *not* coplanar with the ellipsoid axes:
        the pre-auricular points sit slightly posterior and inferior and the
        nasion slightly superior, as they do on a real head.  That makes the
        MRI -> head transform a genuine rotation plus translation rather than a
        pure translation, so coregistration code paths get exercised properly.
        """
        if not self._fiducials:
            self._fiducials.update(
                lpa=self.project_to_scalp([-1.0, -0.06, -0.13]),
                rpa=self.project_to_scalp([1.0, -0.06, -0.13]),
                nasion=self.project_to_scalp([0.0, 1.0, 0.09]),
            )
        return self._fiducials

    @property
    def mri_head_t(self):
        """MRI surface RAS -> head transform implied by the fiducials."""
        import mne

        fids = self.fiducials
        matrix = mne.transforms.get_ras_to_neuromag_trans(
            nasion=fids["nasion"], lpa=fids["lpa"], rpa=fids["rpa"]
        )
        return mne.transforms.Transform("mri", "head", matrix)

    # -- surfaces ----------------------------------------------------------

    def shell(self, which: str, grade: int = BEM_GRADE) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(rr, tris)`` in metres for a BEM shell."""
        axes = {
            "inner_skull": self.inner_skull_axes,
            "outer_skull": self.outer_skull_axes,
            "outer_skin": self.scalp_axes,
        }[which]
        unit_rr, tris = icosphere(grade)
        return ellipsoid(unit_rr, axes, self.center), tris

    def cortex(self, hemi: str) -> dict[str, np.ndarray]:
        """Return white / pial / sphere geometry and curvature for a hemisphere."""
        unit_rr, tris = icosphere(CORTEX_GRADE)
        sign = -1.0 if hemi == "lh" else 1.0
        center = self.center + np.array([sign * self.hemi_offset, 0.0, 0.0])

        base = ellipsoid(unit_rr, self.hemi_axes, center)
        normals = unit_rr / np.linalg.norm(unit_rr, axis=1, keepdims=True)
        disp = folding_field(
            unit_rr,
            amplitude=self.fold_amplitude,
            n_lobes=self.fold_lobes,
            phase=0.0 if hemi == "lh" else np.pi / 3.0,
        )

        # Sulci cut *inward* from the ellipsoid, which is therefore an envelope
        # rather than a mean surface.  Folding symmetrically about the ellipsoid
        # would push gyral crowns a full fold amplitude outward, and with folds
        # deep enough to give tangential normals that is enough to poke the
        # cortex through the inner skull.
        inward = disp - self.fold_amplitude
        white = base + normals * inward[:, None]
        pial = base + normals * (inward + self.cortical_thickness)[:, None]
        return dict(
            white=white,
            pial=pial,
            sphere=unit_rr * 100.0,  # FreeSurfer spheres have radius 100 mm
            tris=tris,
            # FreeSurfer sign convention: positive curv = sulcus (inward fold)
            curv=(-disp / max(self.fold_amplitude, 1e-12)).astype(np.float32),
            sulc=(-disp * 1e3).astype(np.float32),
        )


def build_head_model(seed: int = 0, jitter: float = 0.0) -> HeadModel:
    """Build a head phantom.

    Parameters
    ----------
    seed : int
        Seed for the per-subject geometric jitter.
    jitter : float
        Relative spread of the random head-size variation.  ``0`` reproduces
        the canonical head, which is what the committed template subject and
        the group template use; ``0.06`` gives a plausible spread of head
        shapes across a synthetic cohort.

    Returns
    -------
    head : HeadModel
    """
    rng = np.random.default_rng(seed)
    scale = 1.0 + jitter * rng.standard_normal(3) if jitter else np.ones(3)
    scale = np.clip(scale, 0.85, 1.15)

    scalp = np.array([0.082, 0.101, 0.088]) * scale
    # Proportional, not fixed, shell thicknesses: subtracting a constant would
    # make a jittered small head proportionally tighter and eventually let the
    # cortex touch the inner skull.
    return HeadModel(
        center=np.array([0.0, 0.012, 0.006]),
        scalp_axes=scalp,
        outer_skull_axes=scalp * 0.94,
        inner_skull_axes=scalp * 0.90,
        hemi_axes=np.array([0.030, 0.072, 0.056]) * scale,
        hemi_offset=0.032 * scale[0],
        fold_lobes=16.0 + (rng.uniform(-1.5, 1.5) if jitter else 0.0),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Writing the FreeSurfer subject
# ---------------------------------------------------------------------------


def _volume_info(shape=CONFORMED_SHAPE) -> dict:
    """FreeSurfer ``volume geometry`` block appended to surface files."""
    return dict(
        head=np.array([2, 0, 20], dtype=np.int32),
        valid="1  # volume info valid",
        filename="T1.mgz",
        volume=np.array(shape, dtype=np.int32),
        voxelsize=np.array([1.0, 1.0, 1.0]),
        xras=np.array([-1.0, 0.0, 0.0]),
        yras=np.array([0.0, 0.0, -1.0]),
        zras=np.array([0.0, 1.0, 0.0]),
        cras=np.array([0.0, 0.0, 0.0]),
    )


def _write_t1(head: HeadModel, path: Path) -> None:
    """Write a conformed T1 whose tissue boundaries match ``head``."""
    import nibabel as nib

    grid = np.arange(CONFORMED_SHAPE[0], dtype=np.float32)
    # Voxel -> tkrRAS for the conformed LIA affine above, in metres.
    ras_x = (-grid + 128.0) * 1e-3  # from voxel axis 0
    ras_z = (-grid + 128.0) * 1e-3  # from voxel axis 1
    ras_y = (grid - 128.0) * 1e-3  # from voxel axis 2

    vol = np.zeros(CONFORMED_SHAPE, dtype=np.uint8)
    cx, cy, cz = head.center
    yy, zz = np.meshgrid(ras_y - cy, ras_z - cz, indexing="ij")

    def radius2(dx, axes):
        return (dx / axes[0]) ** 2 + (yy / axes[1]) ** 2 + (zz / axes[2]) ** 2

    # Cortical ribbon lives inside two laterally offset ellipsoids; the gyral
    # corrugation is reproduced here so the T1 and the surfaces agree.
    for i, x in enumerate(ras_x):
        dx = x - cx
        scalp = radius2(dx, head.scalp_axes)
        skull = radius2(dx, head.outer_skull_axes)
        inner = radius2(dx, head.inner_skull_axes)

        slice_ = np.zeros(scalp.shape, dtype=np.uint8)
        slice_[scalp <= 1.0] = _INTENSITY["scalp"]
        slice_[skull <= 1.0] = _INTENSITY["skull"]
        slice_[inner <= 1.0] = _INTENSITY["csf"]

        for sign in (-1.0, 1.0):
            hx = dx - sign * head.hemi_offset
            hemi = (
                (hx / head.hemi_axes[0]) ** 2
                + (yy / head.hemi_axes[1]) ** 2
                + (zz / head.hemi_axes[2]) ** 2
            )
            fold = 0.10 * np.sin(head.fold_lobes * 20.0 * hx) * np.cos(
                head.fold_lobes * 15.0 * yy
            )
            slice_[hemi <= 1.0 + fold] = _INTENSITY["gray"]
            slice_[hemi <= 0.62] = _INTENSITY["white"]

        vol[i] = slice_

    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.MGHImage(vol, CONFORMED_AFFINE), str(path))


def _write_talairach_xfm(path: Path) -> None:
    """Write an identity talairach.xfm so volume morphing has something to read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "MNI Transform File\n"
        "% Synthetic phantom: MRI RAS is already treated as MNI305.\n"
        "\n"
        "Transform_Type = Linear;\n"
        "Linear_Transform =\n"
        "1.0 0.0 0.0 0.0\n"
        "0.0 1.0 0.0 0.0\n"
        "0.0 0.0 1.0 0.0;\n"
    )


def _bem_surface_dict(rr_m: np.ndarray, tris: np.ndarray, surf_id: int, sigma: float):
    """Build the surface dict that ``mne.write_bem_surfaces`` expects."""
    import mne
    from mne.io.constants import FIFF

    surf = dict(
        id=surf_id,
        sigma=sigma,
        np=len(rr_m),
        ntri=len(tris),
        coord_frame=FIFF.FIFFV_COORD_MRI,
        rr=np.asarray(rr_m, float),
        tris=np.asarray(tris, np.int64),
    )
    return mne.surface.complete_surface_info(surf, copy=False, do_neighbor_vert=False)


def write_freesurfer_subject(
    head: HeadModel,
    subjects_dir: Path | str,
    fs_subject: str,
    *,
    write_t1: bool = True,
) -> Path:
    """Write a minimal but complete FreeSurfer subject for ``head``.

    Parameters
    ----------
    head : HeadModel
        Geometry to render.
    subjects_dir : path-like
        FreeSurfer ``SUBJECTS_DIR``.
    fs_subject : str
        Subject directory name, e.g. ``"sub-001_ses-01"``.
    write_t1 : bool
        Write ``mri/T1.mgz``.  Only ever disabled to speed up tests that do
        not touch the volume.

    Returns
    -------
    subject_dir : Path
        The directory that was written.
    """
    import mne
    import nibabel as nib
    from mne.io.constants import FIFF

    subjects_dir = Path(subjects_dir)
    subject_dir = subjects_dir / fs_subject
    for sub in ("surf", "bem", "mri/transforms"):
        (subject_dir / sub).mkdir(parents=True, exist_ok=True)

    vol_info = _volume_info()

    # -- cortical surfaces --------------------------------------------------
    for hemi in ("lh", "rh"):
        cx = head.cortex(hemi)
        tris = cx["tris"]
        for name in ("white", "pial", "sphere"):
            mne.write_surface(
                subject_dir / "surf" / f"{hemi}.{name}",
                cx[name] * 1e3 if name != "sphere" else cx[name],
                tris,
                volume_info=vol_info,
                overwrite=True,
            )
        # sphere.reg is the registration to the group template.  Our template
        # shares this exact tessellation, so the registration is the identity.
        mne.write_surface(
            subject_dir / "surf" / f"{hemi}.sphere.reg",
            cx["sphere"],
            tris,
            volume_info=vol_info,
            overwrite=True,
        )
        nib.freesurfer.io.write_morph_data(
            subject_dir / "surf" / f"{hemi}.curv", cx["curv"], fnum=len(tris)
        )
        nib.freesurfer.io.write_morph_data(
            subject_dir / "surf" / f"{hemi}.sulc", cx["sulc"], fnum=len(tris)
        )

    # -- BEM shells ---------------------------------------------------------
    for name in ("inner_skull", "outer_skull", "outer_skin"):
        rr, tris = head.shell(name)
        mne.write_surface(
            subject_dir / "bem" / f"{name}.surf",
            rr * 1e3,
            tris,
            volume_info=vol_info,
            overwrite=True,
        )

    # -- head surface, for coregistration and alignment plots ---------------
    scalp_rr, scalp_tris = head.shell("outer_skin")
    mne.write_bem_surfaces(
        subject_dir / "bem" / f"{fs_subject}-head.fif",
        _bem_surface_dict(scalp_rr, scalp_tris, FIFF.FIFFV_BEM_SURF_ID_HEAD, 1.0),
        overwrite=True,
    )

    # -- fiducials, so Coregistration(fiducials="auto") works ---------------
    fids = head.fiducials
    dig = [
        dict(
            kind=FIFF.FIFFV_POINT_CARDINAL,
            ident=ident,
            r=np.asarray(fids[key], float),
            coord_frame=FIFF.FIFFV_COORD_MRI,
        )
        for key, ident in (
            ("lpa", FIFF.FIFFV_POINT_LPA),
            ("nasion", FIFF.FIFFV_POINT_NASION),
            ("rpa", FIFF.FIFFV_POINT_RPA),
        )
    ]
    mne.io.write_fiducials(
        subject_dir / "bem" / f"{fs_subject}-fiducials.fif",
        dig,
        FIFF.FIFFV_COORD_MRI,
        overwrite=True,
    )

    # -- volume -------------------------------------------------------------
    if write_t1:
        _write_t1(head, subject_dir / "mri" / "T1.mgz")
    _write_talairach_xfm(subject_dir / "mri" / "transforms" / "talairach.xfm")

    return subject_dir
