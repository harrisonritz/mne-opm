"""Geometry primitives for the synthetic head phantom.

Everything here is analytic and deterministic: given the same
:class:`~custom.synthetic.anatomy.HeadModel` you get bit-identical surfaces.
That is what lets the committed subject be regenerated and diffed.

Surfaces are built on icosahedral meshes so that MNE's BEM machinery
(:func:`mne.make_bem_model`, which downsamples to ``ico=4``) recognises the
tessellation.  MNE's own ``_get_ico_surface`` is used when available so the
vertex ordering matches what FreeSurfer's watershed algorithm would produce;
a self-contained subdivision is used as a fallback.

Author: Harrison Ritz (2025)
"""

from __future__ import annotations

import numpy as np


__all__ = [
    "icosphere",
    "ellipsoid",
    "folding_field",
    "tangent_basis",
    "fibonacci_directions",
]


# ---------------------------------------------------------------------------
# Icosahedral meshes
# ---------------------------------------------------------------------------


def _icosphere_fallback(grade: int) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide a regular icosahedron ``grade`` times onto the unit sphere."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    rr = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    rr /= np.linalg.norm(rr, axis=1, keepdims=True)
    tris = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )

    for _ in range(grade):
        cache: dict[tuple[int, int], int] = {}
        verts = list(rr)
        new_tris = []

        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key not in cache:
                mid = verts[a] + verts[b]
                mid /= np.linalg.norm(mid)
                verts.append(mid)
                cache[key] = len(verts) - 1
            return cache[key]

        for i, j, k in tris:
            ij, jk, ki = midpoint(i, j), midpoint(j, k), midpoint(k, i)
            new_tris += [[i, ij, ki], [ij, j, jk], [ki, jk, k], [ij, jk, ki]]

        rr = np.asarray(verts)
        tris = np.asarray(new_tris, dtype=np.int64)

    return rr, tris


def icosphere(grade: int) -> tuple[np.ndarray, np.ndarray]:
    """Return unit-sphere vertices and triangles for icosahedral ``grade``.

    Parameters
    ----------
    grade : int
        Subdivision level.  ``4`` gives 2562 vertices / 5120 triangles (the
        resolution MNE's BEM code expects), ``5`` gives 10242 / 20480 (enough
        for a ``spacing="oct6"`` source space).

    Returns
    -------
    rr : ndarray, shape (n_vertices, 3)
        Unit-norm vertex positions.
    tris : ndarray, shape (n_triangles, 3)
        Triangle vertex indices.
    """
    try:  # prefer MNE's ordering so make_bem_model's ico downsampling matches
        from mne.surface import _get_ico_surface

        surf = _get_ico_surface(grade)
        rr = np.array(surf["rr"], dtype=np.float64)
        tris = np.array(surf["tris"], dtype=np.int64)
        rr /= np.linalg.norm(rr, axis=1, keepdims=True)
        return rr, tris
    except Exception:  # pragma: no cover - only if MNE internals move
        return _icosphere_fallback(grade)


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def ellipsoid(
    unit_rr: np.ndarray, axes: np.ndarray | tuple, center: np.ndarray | tuple
) -> np.ndarray:
    """Map unit-sphere points onto an axis-aligned ellipsoid."""
    return np.asarray(unit_rr) * np.asarray(axes, float) + np.asarray(center, float)


def folding_field(
    unit_rr: np.ndarray, amplitude: float, n_lobes: float, phase: float = 0.0
) -> np.ndarray:
    """Smooth pseudo-gyral displacement field over the unit sphere.

    Produces the sulcal/gyral corrugation of the synthetic cortex.  This is
    what gives the source space non-degenerate surface normals, which matters
    for a ``pick_ori="max-power"`` beamformer: on a perfect ellipsoid every
    normal is radial and the orientation sign convention becomes trivial.

    Parameters
    ----------
    unit_rr : ndarray, shape (n_vertices, 3)
        Unit-sphere vertex positions.
    amplitude : float
        Peak displacement, in the same units as the surface (metres).
    n_lobes : float
        Angular frequency of the corrugation.  Higher = finer folds.
    phase : float
        Phase offset, used to decorrelate hemispheres and subjects.

    Returns
    -------
    disp : ndarray, shape (n_vertices,)
        Signed radial displacement per vertex.
    """
    x, y, z = np.asarray(unit_rr).T
    return amplitude * (
        np.sin(n_lobes * x + phase) * np.cos(n_lobes * y)
        + np.sin(n_lobes * z + 0.7 * phase) * np.cos(n_lobes * x)
    ) / 2.0


def tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors spanning the plane orthogonal to ``normal``.

    Mirrors the ``calc_tangent`` helper used by the Cerca reader so that
    synthetic sensor ``loc`` arrays follow the same convention as real data.
    """
    normal = np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(seed, normal)) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    ex = np.cross(seed, normal)
    ex /= np.linalg.norm(ex)
    ey = np.cross(normal, ex)
    ey /= np.linalg.norm(ey)
    return ex, ey


def fibonacci_directions(n: int) -> np.ndarray:
    """``n`` near-uniformly spaced unit vectors (spherical Fibonacci lattice)."""
    idx = np.arange(n, dtype=float) + 0.5
    z = 1.0 - 2.0 * idx / n
    r = np.sqrt(np.clip(1.0 - z**2, 0.0, None))
    theta = np.pi * (1.0 + np.sqrt(5.0)) * idx
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
