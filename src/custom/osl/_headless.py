"""Headless 3D rendering setup for the osl-ephys pipeline.

The source stage draws a 3D coregistration scene
(:func:`custom.osl.fs_bridge.fs_coregister`) on compute nodes with no display
of their own, where ``osl_ephys.source_recon`` breaks Qt on import: it pulls in
``cv2``, and the ``opencv-python`` wheel points ``QT_QPA_PLATFORM_PLUGIN_PATH``
at the Qt plugins bundled in ``cv2/qt/plugins``.  Those are built against a
different Qt than the one PyQt provides, so the first Qt application in the
process finds the ``xcb`` plugin and then fails to load it::

    qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in
    ".../cv2/qt/plugins" even though it was found.

MNE's 3D backend is Qt-based, so that application gets built as soon as a
figure is created.  Qt reacts by calling ``qFatal``, which raises ``SIGABRT``
rather than a Python exception -- the ``try``/``except`` around the plot cannot
catch it, and the stage dies with exit status 134 with the beamforming work
still ahead of it.

Functions
---------
setup_headless_3d
    Undo the ``cv2`` Qt breakage and render 3D figures off-screen.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import os


def setup_headless_3d() -> None:
    """Undo the ``cv2`` Qt breakage and render 3D figures off-screen.

    Safe to call more than once, and safe to call when a display *is*
    available.

    Notes
    -----
    Call this **after** ``osl_ephys.source_recon`` (or anything else that pulls
    in ``cv2``) has been imported: ``cv2`` sets the plugin path when it is
    imported, so clearing it any earlier does nothing.

    A display is still required -- MNE raises "Cannot connect to a valid
    display" before it reaches any of this when ``DISPLAY`` is unset, which is
    why the batch scripts wrap the pipeline in ``xvfb-run``.  Forcing
    ``QT_QPA_PLATFORM=offscreen`` does *not* lift that requirement, and under
    ``xvfb-run`` it actively breaks rendering (VTK still talks to the X server
    and hits ``BadWindow``), so this deliberately leaves that variable alone.
    """
    # Drop cv2's plugin path so Qt falls back to the plugins PyQt ships with.
    # Only ours: an explicit path set by the user or the cluster stays.
    plugin_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
    if "cv2" in plugin_path:
        del os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]
        print("[osl:3d] cleared the cv2 Qt plugin path")

    try:
        import pyvista
    except ImportError as exc:  # pragma: no cover - pyvista is an mne 3D dep
        print(f"[osl:3d] pyvista unavailable ({exc}); 3D figures will be skipped.")
        return

    # MNE reads pyvista.OFF_SCREEN when it builds a figure; without it the
    # plotter opens a real window on the (virtual) display and renders slower.
    pyvista.OFF_SCREEN = True

    try:
        import mne

        # Software rendering has no multisampling, and depth peeling does not
        # work under osmesa either -- both make VTK fall back to X11 paths.
        mne.viz.set_3d_options(depth_peeling=False, antialias=False)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[osl:3d] could not set 3D options ({exc}).")
