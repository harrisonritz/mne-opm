"""Synthetic Cerca-style triaxial OPM array.

The channel metadata mirrors what ``src/dev/cMEG_mne.py`` produces when it
reads a real Cerca ``.cMEG`` recording, because that is what the rest of the
pipeline has been written against:

* magnetometers with ``coil_type = FIFFV_COIL_QUSPIN_ZFOPM_MAG2``, ``unit = T``,
  ``cal = 1.0``, stored in the **device** coordinate frame;
* ``loc = [r0, ex, ey, ez]`` with ``ez`` the sensing axis (so homogeneous field
  correction and Maxwell filtering see the right geometry);
* names of the form ``"<helmet slot> <sensor id> <axis>"``, e.g. ``"C6 2B Y"``;
* a ``dev_head_t`` that is a real rigid transform, not the identity;
* digitisation with three cardinal fiducials plus a scalp point cloud.

Sensor count defaults to 48 slots x 3 axes = 144 magnetometers.  That is not
arbitrary: Maxwell filtering with ``mf_int_order = 10`` and ``mf_ext_order = 2``
needs 120 + 8 = 128 basis vectors, so anything under ~130 good channels makes
``mne.preprocessing.maxwell_filter`` ill-posed.

Author: Harrison Ritz (2025)
"""

from __future__ import annotations

import numpy as np

from ._geometry import fibonacci_directions, tangent_basis


__all__ = [
    "OPM_AXES",
    "EYE_CHANNELS",
    "TRIGGER_CHANNELS",
    "build_helmet",
    "build_info",
    "head_shape_points",
]


OPM_AXES = ("X", "Y", "Z")
TRIGGER_CHANNELS = tuple(f"Trigger {i}" for i in range(1, 9))

# Auxiliary channels.  In a real recording these arrive from the eye tracker and
# are merged in by ``format_bids.process_eyetracking``; the synthetic subject
# ships them directly in the MEG file so that ICA's EOG detection and the
# ``_regress_preds`` head-position regressors have something to work with
# without also having to fake an Eyelink ``.asc``.
EYE_CHANNELS: dict[str, str] = {
    "eye_nmf1": "eog",
    "eye_nmf2": "eog",
    "eye_nmf3": "eog",
    "xpos_right": "misc",
    "ypos_right": "misc",
    "pupil_right": "misc",
    "x_head": "misc",
    "y_head": "misc",
    "distance": "misc",
}

# 10-20 style row/column grid used to name helmet slots.
_ROWS = ("Fp", "AF", "F", "FC", "C", "CP", "P", "PO", "O")
_COLS = ("7", "5", "3", "1", "z", "2", "4", "6", "8")


def _slot_labels(unit_dirs: np.ndarray) -> list[str]:
    """Assign unique 10-20-like helmet-slot labels to outward directions.

    Rows run anterior (``Fp``) to posterior (``O``) by the elevation angle in
    the sagittal plane; within a row, slots are ordered left to right and take
    consecutive column labels centred on the midline (``z``).
    """
    unit_dirs = np.asarray(unit_dirs, float)
    # +pi/2 at the forehead, 0 at the vertex, -pi/2 at the occiput.
    elevation = np.arctan2(unit_dirs[:, 1], unit_dirs[:, 2])
    row_idx = np.clip(
        np.round((np.pi / 2 - elevation) / np.pi * (len(_ROWS) - 1)).astype(int),
        0,
        len(_ROWS) - 1,
    )

    labels = [""] * len(unit_dirs)
    mid = len(_COLS) // 2
    for row in np.unique(row_idx):
        members = np.flatnonzero(row_idx == row)
        members = members[np.argsort(unit_dirs[members, 0])]
        # Centre the run of columns on the midline label.
        start = mid - len(members) // 2
        for offset, slot in enumerate(members):
            col = _COLS[int(np.clip(start + offset, 0, len(_COLS) - 1))]
            label = f"{_ROWS[row]}{col}"
            suffix = 0
            while label in labels:
                suffix += 1
                label = f"{_ROWS[row]}{col}{suffix}"
            labels[slot] = label
    return labels


def _farthest_point_sample(points: np.ndarray, n: int) -> np.ndarray:
    """Greedy farthest-point subsampling, for an even helmet layout."""
    chosen = [int(np.argmax(points[:, 2]))]  # start at the vertex
    dist = np.linalg.norm(points - points[chosen[0]], axis=1)
    while len(chosen) < n:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(points - points[nxt], axis=1))
    return np.asarray(chosen)


def build_helmet(head, n_slots: int = 48, standoff: float = 0.008) -> dict:
    """Lay out OPM sensor slots over the scalp.

    Parameters
    ----------
    head : HeadModel
        Anatomy the helmet is fitted to.
    n_slots : int
        Number of sensor positions.  Each carries three orthogonal channels.
    standoff : float
        Distance from the scalp to the vapour cell, in metres.

    Returns
    -------
    helmet : dict
        ``positions`` (n_slots, 3) and ``normals`` (n_slots, 3) in MRI surface
        RAS metres, plus the assigned ``labels``.
    """
    axes = np.asarray(head.scalp_axes) + standoff

    # Candidate directions over the whole sphere, then keep the part of the
    # head a helmet actually covers: above the ear line, not over the face.
    dirs = fibonacci_directions(2048)
    keep = (dirs[:, 2] > -0.15) & ~((dirs[:, 1] > 0.55) & (dirs[:, 2] < 0.25))
    dirs = dirs[keep]

    positions = head.center + dirs * axes
    idx = _farthest_point_sample(positions, n_slots)
    # Deterministic, readable ordering: front-to-back, then left-to-right.
    idx = idx[np.lexsort((positions[idx, 0], -positions[idx, 1]))]

    positions = positions[idx]
    # Outward normal of the ellipsoid, i.e. the gradient of its implicit form.
    normals = (positions - head.center) / axes**2
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    labels = _slot_labels((positions - head.center) / axes)
    return dict(positions=positions, normals=normals, labels=labels)


def head_shape_points(head, n_points: int = 180, seed: int = 0) -> np.ndarray:
    """Sample a scalp point cloud, as a digitiser would.

    Points are drawn from the scalp above the ear line (a real digitisation
    never covers the underside of the head) and given sub-millimetre noise so
    that ICP coregistration has a realistic residual to minimise.
    """
    rng = np.random.default_rng(seed + 7717)
    dirs = fibonacci_directions(n_points * 4)
    keep = (dirs[:, 2] > -0.25) & ~((dirs[:, 1] > 0.6) & (dirs[:, 2] < 0.0))
    dirs = dirs[keep][:n_points]
    pts = head.center + dirs * head.scalp_axes
    return pts + rng.normal(scale=5e-4, size=pts.shape)


def _device_head_trans(seed: int = 0):
    """A plausible, non-identity helmet-to-head transform."""
    import mne

    rng = np.random.default_rng(seed + 991)
    pitch, roll, yaw = np.deg2rad(rng.uniform(-6.0, 6.0, size=3))
    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ]
    )
    ry = np.array(
        [[np.cos(roll), 0, np.sin(roll)], [0, 1, 0], [-np.sin(roll), 0, np.cos(roll)]]
    )
    rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rz @ ry @ rx
    matrix[:3, 3] = rng.uniform(-0.005, 0.005, size=3)
    return mne.transforms.Transform("meg", "head", matrix)


def build_info(
    head,
    sfreq: float,
    *,
    n_slots: int = 48,
    line_freq: float = 60.0,
    seed: int = 0,
    include_triggers: bool = True,
    include_eye: bool = True,
):
    """Build the :class:`mne.Info` for a synthetic OPM recording.

    Parameters
    ----------
    head : HeadModel
        Anatomy the helmet is fitted to.
    sfreq : float
        Sampling frequency, Hz.
    n_slots : int
        Sensor slots; the array has ``3 * n_slots`` magnetometers.
    line_freq : float
        Power-line frequency written to ``info["line_freq"]``.
    seed : int
        Seed for the device-to-head transform and digitiser noise.
    include_triggers : bool
        Add the eight ``"Trigger N"`` stim channels a Cerca system records.
        These only exist before BIDS conversion: ``format_bids`` folds them
        into annotations and drops them.
    include_eye : bool
        Add the eye-tracking / head-position auxiliary channels.

    Returns
    -------
    info : mne.Info
        Measurement info in the device frame, with ``dev_head_t`` and ``dig`` set.
    helmet : dict
        The layout returned by :func:`build_helmet`, for downstream use.
    """
    import mne
    from mne.io.constants import FIFF
    from mne.transforms import apply_trans, invert_transform

    helmet = build_helmet(head, n_slots=n_slots)
    mri_head_t = head.mri_head_t
    dev_head_t = _device_head_trans(seed)
    head_dev_t = invert_transform(dev_head_t)

    # MRI surface RAS -> head -> device
    pos_head = apply_trans(mri_head_t, helmet["positions"])
    nrm_head = apply_trans(mri_head_t, helmet["normals"], move=False)
    pos_dev = apply_trans(head_dev_t, pos_head)
    nrm_dev = apply_trans(head_dev_t, nrm_head, move=False)

    ch_names: list[str] = []
    ch_types: list[str] = []
    locs: list[np.ndarray] = []
    for slot, (label, r0, normal) in enumerate(
        zip(helmet["labels"], pos_dev, nrm_dev, strict=True)
    ):
        sensor_id = f"{slot // 10}{chr(ord('A') + slot % 10)}"
        radial = normal / np.linalg.norm(normal)
        tan1, tan2 = tangent_basis(radial)
        for axis, sensing in zip(OPM_AXES, (tan1, tan2, radial), strict=True):
            ex, ey = tangent_basis(sensing)
            ch_names.append(f"{label} {sensor_id} {axis}")
            ch_types.append("mag")
            locs.append(np.concatenate([r0, ex, ey, sensing]))

    n_meg = len(ch_names)
    if include_eye:
        ch_names += list(EYE_CHANNELS)
        ch_types += list(EYE_CHANNELS.values())
    if include_triggers:
        ch_names += list(TRIGGER_CHANNELS)
        ch_types += ["stim"] * len(TRIGGER_CHANNELS)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    info["line_freq"] = line_freq
    info["device_info"] = dict(
        type="Cerca_synthetic", model="cMEG", site="mne-opm synthetic dataset"
    )

    with info._unlock():
        for idx in range(n_meg):
            info["chs"][idx].update(
                loc=locs[idx],
                coord_frame=FIFF.FIFFV_COORD_DEVICE,
                kind=FIFF.FIFFV_MEG_CH,
                unit=FIFF.FIFF_UNIT_T,
                coil_type=FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2,
                cal=1.0,
                range=1.0,
            )
        info["dev_head_t"] = dev_head_t

    # Digitisation, in head coordinates.
    fids = head.fiducials
    hsp = apply_trans(mri_head_t, head_shape_points(head, seed=seed))
    montage = mne.channels.make_dig_montage(
        nasion=apply_trans(mri_head_t, fids["nasion"]),
        lpa=apply_trans(mri_head_t, fids["lpa"]),
        rpa=apply_trans(mri_head_t, fids["rpa"]),
        hsp=hsp,
        coord_frame="head",
    )
    info.set_montage(montage, on_missing="ignore")

    return info, helmet
