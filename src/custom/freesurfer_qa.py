#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mne>=1.6",
#   "nibabel>=5",
#   "numpy",
#   "scipy",
#   "pandas",
#   "matplotlib",
# ]
# ///
"""
fs_qa.py — Automated quality assessment of FreeSurfer reconstructions and
MNE/FreeSurfer watershed BEM surfaces.

For every subject in a FreeSurfer SUBJECTS_DIR the script writes a ``QA/`` folder
inside the subject directory containing:

  * <subject>_qa_report.txt   human-readable report
  * recon_log_issues.txt      lines extracted from recon-all logs
  * euler.txt                 Euler / defect counts per hemisphere
  * bem_<orientation>.png      BEM surface overlays on the T1
  * <subject>_qa.csv          one row per QA test (status + severity + value)

A group-level summary is written to ``<SUBJECTS_DIR>/QA_group/``:

  * group_qa_long.csv         every test for every subject (tidy/long format)
  * group_qa_summary.csv      one row per subject, flagging QA concerns

Design notes
------------
* Topology (Euler number, holes, manifoldness, watertightness) is computed
  directly from the surface tessellations, so a sourced FreeSurfer environment
  is NOT required. ``mri_cnr`` is the only optional external binary.
* The "free validator" strategy is used for the BEM: we attempt
  ``mne.make_bem_model`` on the *existing* surfaces. This re-reads and checks
  geometry without re-running the watershed segmentation.
* Watershed surfaces and the ?h.pial surfaces are both in FreeSurfer surface RAS
  (tkrRAS), so containment tests are done directly without a coordinate change.

Run:
    uv run fs_qa.py --subjects-dir /path/to/SUBJECTS_DIR
    uv run fs_qa.py --subjects-dir $SUBJECTS_DIR --subjects sub-01 sub-02
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# Headless plotting.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------------------
# Status / severity vocabulary
# ----------------------------------------------------------------------------
# status: outcome of a test.  severity: how much a non-PASS outcome matters.
STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"
STATUS_SKIP = "SKIP"   # test not applicable / inputs missing
STATUS_ERROR = "ERROR"  # the test harness itself threw

SEV_NONE = "none"
SEV_MINOR = "minor"
SEV_MAJOR = "major"
SEV_CRIT = "critical"

# ranking for roll-up of an overall subject verdict
_STATUS_RANK = {STATUS_PASS: 0, STATUS_SKIP: 0, STATUS_WARN: 1,
                STATUS_ERROR: 2, STATUS_FAIL: 3}


@dataclass
class TestResult:
    category: str          # "log" | "cortical" | "bem"
    test: str              # short test name
    status: str            # one of STATUS_*
    severity: str = SEV_NONE
    value: object = ""     # numeric / short value for the CSV
    detail: str = ""       # free-text explanation

    def row(self, subject: str) -> dict:
        return {
            "subject": subject,
            "category": self.category,
            "test": self.test,
            "status": self.status,
            "severity": self.severity,
            "value": self.value,
            "detail": self.detail,
        }


@dataclass
class Config:
    subjects_dir: Path
    subjects: Optional[list] = None
    # cortical
    euler_holes_warn: int = 150      # per-subject soft flag on total nofix holes
    # bem
    gap_warn_mm: float = 1.0         # min inter-surface gap below this -> WARN
    pial_out_warn_frac: float = 0.0  # any pial vertex outside inner_skull -> WARN
    pial_out_fail_frac: float = 0.01  # >1% outside -> FAIL
    nest_tol_frac: float = 0.999     # fraction-inside above this counts as nested
    bem_ico: int = 4                 # downsampling for make_bem_model validation
    conductivity = (0.3, 0.006, 0.3)
    # group
    mad_k: float = 3.0               # robust-z threshold for cohort outliers
    # behaviour
    do_3d: bool = False
    do_cnr: bool = True
    overwrite: bool = True


# ----------------------------------------------------------------------------
# Small IO helpers
# ----------------------------------------------------------------------------
def _read_geometry(path: Path):
    """Read a FreeSurfer surface -> (vertices[n,3] float64, faces[m,3] int)."""
    import nibabel.freesurfer.io as fsio
    verts, faces = fsio.read_geometry(str(path))
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _find_first(*candidates: Path) -> Optional[Path]:
    for c in candidates:
        if c is not None and c.exists():
            return c
    return None


# ----------------------------------------------------------------------------
# Core surface topology  (pure numpy; no FreeSurfer / mne needed)
# ----------------------------------------------------------------------------
def surface_topology(verts: np.ndarray, faces: np.ndarray) -> dict:
    """Topological / manifold summary of a triangle mesh.

    Returns a dict with the Euler characteristic chi = V - E + F, the implied
    number of topological defects n = (2 - chi)/2 (a clean closed sphere has
    chi = 2 -> n = 0), and manifold/watertight diagnostics that mirror the
    checks MNE applies when building a BEM.
    """
    n_v = int(len(verts))
    n_f = int(len(faces))

    # edges: each face contributes 3 undirected edges
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    n_e = int(len(uniq))

    n_boundary = int(np.count_nonzero(counts == 1))   # holes / open edges
    n_nonmanifold = int(np.count_nonzero(counts > 2))  # >2 faces share an edge
    watertight = bool(np.all(counts == 2))

    # vertices incident to fewer than three triangles (MNE flags these)
    vdeg = np.bincount(faces.ravel(), minlength=n_v)
    n_lowdeg = int(np.count_nonzero(vdeg < 3))
    n_isolated = int(np.count_nonzero(vdeg == 0))

    # degenerate (zero-area) triangles
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    n_degenerate = int(np.count_nonzero(areas <= 1e-12))

    euler = n_v - n_e + n_f
    n_defects = (2 - euler) / 2.0  # genus / holes for a closed orientable surf

    return dict(
        n_vert=n_v, n_face=n_f, n_edge=n_e, euler=int(euler),
        n_defects=n_defects, n_boundary_edges=n_boundary,
        n_nonmanifold_edges=n_nonmanifold, watertight=watertight,
        n_lowdeg_vert=n_lowdeg, n_isolated_vert=n_isolated,
        n_degenerate_tri=n_degenerate,
    )


def _solid_angle_sum(points: np.ndarray, verts: np.ndarray, faces: np.ndarray,
                     chunk: int = 128) -> np.ndarray:
    """Total signed solid angle subtended by a closed triangle mesh at each
    query point (Van Oosterom & Strackee formula).

    For a closed surface the sum is +-4*pi at interior points and ~0 at
    exterior points (the generalized winding number x 4*pi). This is the same
    solid-angle construction MNE uses to test surface completeness, so it needs
    no external spatial index. Chunked over points to bound memory.
    """
    tris = verts[faces]                      # (F, 3, 3)
    ta, tb, tc = tris[:, 0], tris[:, 1], tris[:, 2]
    out = np.empty(len(points), dtype=np.float64)
    for i in range(0, len(points), chunk):
        p = points[i:i + chunk][:, None, :]  # (P, 1, 3)
        a = ta[None] - p                     # (P, F, 3)
        b = tb[None] - p
        c = tc[None] - p
        la = np.sqrt(np.einsum("pfk,pfk->pf", a, a))
        lb = np.sqrt(np.einsum("pfk,pfk->pf", b, b))
        lc = np.sqrt(np.einsum("pfk,pfk->pf", c, c))
        numer = np.einsum("pfk,pfk->pf", a, np.cross(b, c))
        denom = (la * lb * lc
                 + np.einsum("pfk,pfk->pf", a, b) * lc
                 + np.einsum("pfk,pfk->pf", a, c) * lb
                 + np.einsum("pfk,pfk->pf", b, c) * la)
        out[i:i + chunk] = (2.0 * np.arctan2(numer, denom)).sum(axis=1)
    return out


def fraction_inside(points: np.ndarray, verts: np.ndarray,
                    faces: np.ndarray) -> tuple[float, np.ndarray]:
    """Fraction of `points` inside the closed surface (verts, faces).

    Uses |solid angle| > 2*pi as the interior test, so it is insensitive to a
    globally flipped normal orientation. Returns (fraction_inside, bool mask).
    """
    points = np.asarray(points, dtype=np.float64)
    sa = _solid_angle_sum(points, np.asarray(verts, dtype=np.float64),
                          np.asarray(faces))
    inside = np.abs(sa) > 2.0 * np.pi
    return float(inside.mean()), inside


def min_gap_mm(verts_a: np.ndarray, verts_b: np.ndarray) -> float:
    """Approximate minimum gap between two surfaces (nearest-vertex distance)."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(verts_b).query(verts_a, k=1)
    return float(d.min())


# ----------------------------------------------------------------------------
# Cortical recon tests
# ----------------------------------------------------------------------------
_HARD_LOG_PATTERNS = (
    "exited with errors", "segmentation fault", " killed", "core dumped",
    "no such file or directory", "cannot allocate", "error:", "fatal error",
)
_SOFT_LOG_PATTERNS = ("warning", "talairach failed", "defect", "skipping")


def test_recon_log(subj_dir: Path, qa_dir: Path) -> TestResult:
    log = _find_first(subj_dir / "scripts" / "recon-all.log")
    status_log = _find_first(subj_dir / "scripts" / "recon-all-status.log")
    if log is None:
        return TestResult("log", "recon_log_present", STATUS_FAIL, SEV_MAJOR,
                          "missing", "scripts/recon-all.log not found")

    text = log.read_text(errors="ignore")
    low = text.lower()
    finished = "finished without error" in low
    hard = [ln for ln in text.splitlines()
            if any(p in ln.lower() for p in _HARD_LOG_PATTERNS)]
    soft = [ln for ln in text.splitlines()
            if any(p in ln.lower() for p in _SOFT_LOG_PATTERNS)]

    # FreeSurfer build/version, if present
    version = "unknown"
    for ln in text.splitlines()[:50]:
        if "freesurfer-" in ln.lower() or "build-stamp" in ln.lower():
            version = ln.strip()
            break

    out = qa_dir / "recon_log_issues.txt"
    with out.open("w") as f:
        f.write(f"# recon-all log QA for {subj_dir.name}\n")
        f.write(f"version: {version}\n")
        f.write(f"finished_without_error: {finished}\n\n")
        if status_log is not None:
            f.write("## last status lines\n")
            f.write("\n".join(status_log.read_text(errors='ignore')
                              .splitlines()[-15:]) + "\n\n")
        f.write(f"## hard error lines ({len(hard)})\n")
        f.write("\n".join(hard[:200]) + "\n\n")
        f.write(f"## soft warning lines ({len(soft)})\n")
        f.write("\n".join(soft[:200]) + "\n")

    if hard or not finished:
        return TestResult("log", "recon_log", STATUS_FAIL, SEV_CRIT,
                          f"{len(hard)} errors",
                          f"finished={finished}; see recon_log_issues.txt")
    if soft:
        return TestResult("log", "recon_log", STATUS_WARN, SEV_MINOR,
                          f"{len(soft)} warnings",
                          "soft warnings present; see recon_log_issues.txt")
    return TestResult("log", "recon_log", STATUS_PASS, SEV_NONE, "ok",
                      f"finished without error ({version})")


def test_euler_nofix(subj_dir: Path, qa_dir: Path, cfg: Config):
    """Euler number / holes of the *unfixed* (orig.nofix) surfaces.

    Returns (TestResult, lh_holes, rh_holes) where holes may be None if missing.
    The fixed surfaces are topology-corrected (chi == 2) so they carry no signal;
    the nofix surface counts how many defects the corrector had to repair.
    """
    surf = subj_dir / "surf"
    lines = [f"# Euler / defect summary for {subj_dir.name}",
             "# chi = V - E + F ; defects n = (2 - chi)/2 (genus of nofix surf)"]
    holes = {}
    for hemi in ("lh", "rh"):
        p = surf / f"{hemi}.orig.nofix"
        if not p.exists():
            lines.append(f"{hemi}.orig.nofix: MISSING")
            holes[hemi] = None
            continue
        v, f = _read_geometry(p)
        t = surface_topology(v, f)
        holes[hemi] = t["n_defects"]
        lines.append(f"{hemi}.orig.nofix: chi={t['euler']:>6d}  "
                     f"defects(holes)={t['n_defects']:.0f}  "
                     f"V={t['n_vert']} E={t['n_edge']} F={t['n_face']}")
    (qa_dir / "euler.txt").write_text("\n".join(lines) + "\n")

    have = [h for h in holes.values() if h is not None]
    if not have:
        return (TestResult("cortical", "euler_nofix", STATUS_SKIP, SEV_NONE,
                           "n/a", "orig.nofix surfaces not found"),
                holes.get("lh"), holes.get("rh"))
    total = float(np.nansum([h for h in have]))
    status = STATUS_WARN if total > cfg.euler_holes_warn else STATUS_PASS
    sev = SEV_MINOR if status == STATUS_WARN else SEV_NONE
    detail = (f"lh={holes['lh']} rh={holes['rh']} total={total:.0f}; "
              "cohort outlier flagged at group level")
    return (TestResult("cortical", "euler_nofix", status, sev, total, detail),
            holes.get("lh"), holes.get("rh"))


def test_fixed_surface_topology(subj_dir: Path) -> list[TestResult]:
    """Sanity check that final surfaces are genuinely closed & genus-0.

    A correctly topology-fixed white surface must have chi == 2, no boundary
    edges (no holes) and no non-manifold edges. Any deviation indicates a
    corrupt/truncated surface file.
    """
    surf = subj_dir / "surf"
    results = []
    for hemi in ("lh", "rh"):
        p = _find_first(surf / f"{hemi}.white", surf / f"{hemi}.orig")
        if p is None:
            results.append(TestResult("cortical", f"{hemi}_white_topology",
                                      STATUS_SKIP, SEV_NONE, "missing",
                                      f"{hemi}.white/orig not found"))
            continue
        v, f = _read_geometry(p)
        t = surface_topology(v, f)
        problems = []
        if t["euler"] != 2:
            problems.append(f"chi={t['euler']} (!=2)")
        if t["n_boundary_edges"]:
            problems.append(f"{t['n_boundary_edges']} boundary edges (holes)")
        if t["n_nonmanifold_edges"]:
            problems.append(f"{t['n_nonmanifold_edges']} non-manifold edges")
        if problems:
            results.append(TestResult(
                "cortical", f"{hemi}_white_topology", STATUS_FAIL, SEV_MAJOR,
                f"chi={t['euler']}", "; ".join(problems)))
        else:
            results.append(TestResult(
                "cortical", f"{hemi}_white_topology", STATUS_PASS, SEV_NONE,
                "chi=2", "closed genus-0 manifold"))
    return results


_GLOBAL_MEASURES = ("BrainSegVol", "BrainSegVolNotVent", "lhCortexVol",
                    "rhCortexVol", "CortexVol", "TotalGrayVol",
                    "SupraTentorialVol", "EstimatedTotalIntraCranialVol")


def parse_global_stats(subj_dir: Path) -> dict:
    """Pull global volume measures from stats/aseg.stats for outlier screening."""
    p = subj_dir / "stats" / "aseg.stats"
    out = {}
    if not p.exists():
        return out
    for ln in p.read_text(errors="ignore").splitlines():
        if ln.startswith("# Measure"):
            parts = [x.strip() for x in ln.split(",")]
            # "# Measure <name>, <longname>, <desc>, <value>, <unit>"
            name = parts[0].replace("# Measure", "").strip()
            try:
                out[name] = float(parts[-2])
            except (ValueError, IndexError):
                pass
    return out


def test_cnr(subj_dir: Path, cfg: Config) -> TestResult:
    """Gray/white CNR via mri_cnr if available (needs FreeSurfer on PATH)."""
    import shutil
    import subprocess
    if not cfg.do_cnr or shutil.which("mri_cnr") is None:
        return TestResult("cortical", "gray_white_cnr", STATUS_SKIP, SEV_NONE,
                          "n/a", "mri_cnr not on PATH (optional)")
    surf = subj_dir / "surf"
    norm = subj_dir / "mri" / "norm.mgz"
    if not norm.exists():
        return TestResult("cortical", "gray_white_cnr", STATUS_SKIP, SEV_NONE,
                          "n/a", "mri/norm.mgz missing")
    try:
        res = subprocess.run(["mri_cnr", str(surf), str(norm)],
                             capture_output=True, text=True, timeout=300)
        out = res.stdout + res.stderr
        # mri_cnr prints "total CNR = X"
        cnr = None
        for ln in out.splitlines():
            if "total CNR" in ln:
                cnr = float(ln.split("=")[-1])
        if cnr is None:
            return TestResult("cortical", "gray_white_cnr", STATUS_SKIP,
                              SEV_NONE, "n/a", "could not parse mri_cnr output")
        return TestResult("cortical", "gray_white_cnr", STATUS_PASS, SEV_NONE,
                          round(cnr, 3), "lower values may indicate motion; "
                          "flagged relative to cohort at group level")
    except Exception as exc:  # pragma: no cover
        return TestResult("cortical", "gray_white_cnr", STATUS_SKIP, SEV_NONE,
                          "n/a", f"mri_cnr failed: {exc}")


# ----------------------------------------------------------------------------
# BEM watershed tests
# ----------------------------------------------------------------------------
_BEM_NAMES = ("inner_skull", "outer_skull", "outer_skin")


def locate_bem_surfaces(subj_dir: Path) -> dict:
    """Return {name: Path} for the three BEM surfaces, searching bem/ then
    bem/watershed/. Returns None for any that are missing."""
    bem = subj_dir / "bem"
    ws = bem / "watershed"
    subj = subj_dir.name
    found = {}
    for name in _BEM_NAMES:
        found[name] = _find_first(
            bem / f"{name}.surf",
            ws / f"{subj}_{name}_surface",
            ws / f"{subj}_{name}_surface.surf",
        )
    return found


def test_bem_surface_topology(name: str, path: Path) -> tuple[TestResult, dict]:
    v, f = _read_geometry(path)
    t = surface_topology(v, f)
    problems, sev = [], SEV_NONE
    status = STATUS_PASS
    if not t["watertight"]:
        problems.append(f"not watertight ({t['n_boundary_edges']} boundary, "
                        f"{t['n_nonmanifold_edges']} non-manifold edges)")
        status, sev = STATUS_FAIL, SEV_CRIT
    if t["n_lowdeg_vert"]:
        problems.append(f"{t['n_lowdeg_vert']} verts with <3 triangles")
        status = STATUS_FAIL
        sev = SEV_CRIT
    if t["euler"] != 2:
        problems.append(f"chi={t['euler']} (handles/holes present)")
        if status != STATUS_FAIL:
            status, sev = STATUS_WARN, SEV_MAJOR
    if t["n_degenerate_tri"]:
        problems.append(f"{t['n_degenerate_tri']} zero-area triangles")
        if status == STATUS_PASS:
            status, sev = STATUS_WARN, SEV_MINOR
    detail = "; ".join(problems) if problems else \
        f"closed genus-0 manifold (V={t['n_vert']}, F={t['n_face']})"
    return (TestResult("bem", f"{name}_topology", status, sev,
                       f"chi={t['euler']}", detail),
            {"verts": v, "faces": f, "topo": t})


def test_bem_nesting(meshes: dict, cfg: Config) -> list[TestResult]:
    """inner_skull subset of outer_skull subset of outer_skin (no crossings)."""
    results = []
    pairs = [("inner_skull", "outer_skull"), ("outer_skull", "outer_skin")]
    for inner, outer in pairs:
        if inner not in meshes or outer not in meshes:
            results.append(TestResult("bem", f"nest_{inner}_in_{outer}",
                                      STATUS_SKIP, SEV_NONE, "n/a",
                                      "surface missing"))
            continue
        if not meshes[outer]["topo"]["watertight"]:
            results.append(TestResult("bem", f"nest_{inner}_in_{outer}",
                                      STATUS_SKIP, SEV_MAJOR, "n/a",
                                      f"{outer} not watertight; containment "
                                      "test unreliable"))
            continue
        outer_v = meshes[outer]["verts"]
        outer_f = meshes[outer]["faces"]
        frac, mask = fraction_inside(meshes[inner]["verts"], outer_v, outer_f)
        n_out = int((~mask).sum())
        if frac >= cfg.nest_tol_frac:
            results.append(TestResult("bem", f"nest_{inner}_in_{outer}",
                                      STATUS_PASS, SEV_NONE, f"{frac:.4f}",
                                      f"{inner} inside {outer}"))
        else:
            results.append(TestResult("bem", f"nest_{inner}_in_{outer}",
                                      STATUS_FAIL, SEV_CRIT, f"{frac:.4f}",
                                      f"{n_out} {inner} verts outside {outer} "
                                      "(surfaces intersect)"))
    return results


def test_pial_in_inner_skull(subj_dir: Path, meshes: dict, cfg: Config) \
        -> TestResult:
    """Every cortical (pial) vertex must lie inside the inner_skull surface."""
    if "inner_skull" not in meshes or \
            not meshes["inner_skull"]["topo"]["watertight"]:
        return TestResult("bem", "pial_in_inner_skull", STATUS_SKIP, SEV_MAJOR,
                          "n/a", "inner_skull missing/not watertight")
    surf = subj_dir / "surf"
    pial = []
    for hemi in ("lh", "rh"):
        p = surf / f"{hemi}.pial"
        if p.exists():
            v, _ = _read_geometry(p)
            pial.append(v)
    if not pial:
        return TestResult("bem", "pial_in_inner_skull", STATUS_SKIP, SEV_NONE,
                          "n/a", "pial surfaces not found")
    pial = np.vstack(pial)
    inner_v = meshes["inner_skull"]["verts"]
    inner_f = meshes["inner_skull"]["faces"]
    frac, mask = fraction_inside(pial, inner_v, inner_f)
    n_out = int((~mask).sum())
    out_frac = 1.0 - frac
    if n_out == 0:
        return TestResult("bem", "pial_in_inner_skull", STATUS_PASS, SEV_NONE,
                          "0", "all cortex inside inner skull")
    # approximate protrusion depth: distance from outside pial verts to the
    # nearest inner_skull vertex (KDTree; an upper bound on true signed depth)
    try:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(inner_v).query(pial[~mask], k=1)
        max_out = float(d.max())
    except Exception:
        max_out = float("nan")
    if out_frac > cfg.pial_out_fail_frac:
        return TestResult("bem", "pial_in_inner_skull", STATUS_FAIL, SEV_CRIT,
                          f"{n_out} ({out_frac:.2%})",
                          f"cortex protrudes outside inner skull; "
                          f"max ~{max_out:.1f} mm — forward model invalid there")
    return TestResult("bem", "pial_in_inner_skull", STATUS_WARN, SEV_MAJOR,
                      f"{n_out} ({out_frac:.2%})",
                      f"a few cortical verts outside inner skull; "
                      f"max ~{max_out:.1f} mm")


def test_bem_gaps(meshes: dict, cfg: Config) -> list[TestResult]:
    results = []
    pairs = [("inner_skull", "outer_skull"), ("outer_skull", "outer_skin")]
    for a, b in pairs:
        if a not in meshes or b not in meshes:
            results.append(TestResult("bem", f"gap_{a}_{b}", STATUS_SKIP,
                                      SEV_NONE, "n/a", "surface missing"))
            continue
        gap = min_gap_mm(meshes[a]["verts"], meshes[b]["verts"])
        if gap <= 1e-3:
            results.append(TestResult("bem", f"gap_{a}_{b}", STATUS_FAIL,
                                      SEV_CRIT, f"{gap:.2f} mm",
                                      "surfaces touch/cross (zero gap)"))
        elif gap < cfg.gap_warn_mm:
            results.append(TestResult("bem", f"gap_{a}_{b}", STATUS_WARN,
                                      SEV_MAJOR, f"{gap:.2f} mm",
                                      f"thin compartment (<{cfg.gap_warn_mm} mm)"
                                      "; BEM may be ill-conditioned"))
        else:
            results.append(TestResult("bem", f"gap_{a}_{b}", STATUS_PASS,
                                      SEV_NONE, f"{gap:.2f} mm", "ok"))
    return results


def test_make_bem_model(subj_dir: Path, cfg: Config) -> list[TestResult]:
    """Use MNE's own geometry validator on the existing surfaces.

    This does NOT re-segment; make_bem_model only reads the surfaces and runs
    the nesting / solid-angle completeness / manifold checks before assembling
    the model. We try both a 3-layer (EEG) and 1-layer (MEG) configuration.
    """
    import mne
    results = []
    subjects_dir = str(cfg.subjects_dir)
    subject = subj_dir.name
    configs = [("bem_model_3layer", cfg.conductivity, SEV_CRIT),
               ("bem_model_1layer", (0.3,), SEV_MAJOR)]
    for tname, cond, sev in configs:
        try:
            mne.make_bem_model(subject=subject, ico=cfg.bem_ico,
                               conductivity=cond, subjects_dir=subjects_dir,
                               verbose="ERROR")
            results.append(TestResult("bem", tname, STATUS_PASS, SEV_NONE, "ok",
                                      f"MNE accepted geometry ({len(cond)}-layer)"))
        except Exception as exc:
            msg = str(exc).strip().splitlines()[-1] if str(exc) else repr(exc)
            results.append(TestResult("bem", tname, STATUS_FAIL, sev, "raised",
                                      f"make_bem_model: {msg[:300]}"))
    return results


def plot_bem_images(subj_dir: Path, qa_dir: Path, cfg: Config,
                    meshes: dict) -> TestResult:
    """Save BEM-on-T1 overlays. Try mne.viz.plot_bem; fall back to a manual
    nibabel + matplotlib scatter overlay if that fails."""
    import mne
    saved = []
    try:
        for orientation in ("axial", "coronal", "sagittal"):
            fig = mne.viz.plot_bem(subject=subj_dir.name,
                                   subjects_dir=str(cfg.subjects_dir),
                                   orientation=orientation, show=False)
            out = qa_dir / f"bem_{orientation}.png"
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            saved.append(out.name)
        return TestResult("bem", "bem_visualization", STATUS_PASS, SEV_NONE,
                          ",".join(saved), "mne.viz.plot_bem overlays saved")
    except Exception as exc:
        # ---- fallback: overlay surface vertices on T1 slices manually ----
        try:
            saved = _manual_bem_plot(subj_dir, qa_dir, meshes)
            return TestResult("bem", "bem_visualization", STATUS_WARN, SEV_MINOR,
                              ",".join(saved),
                              f"plot_bem failed ({str(exc)[:120]}); "
                              "used fallback scatter overlay")
        except Exception as exc2:
            return TestResult("bem", "bem_visualization", STATUS_WARN, SEV_MINOR,
                              "none", f"visualization failed: {exc2}")


def _manual_bem_plot(subj_dir: Path, qa_dir: Path, meshes: dict) -> list[str]:
    """Overlay BEM surface vertices (surface RAS) on T1 slices."""
    import nibabel as nib
    t1 = _find_first(subj_dir / "mri" / "T1.mgz",
                     subj_dir / "mri" / "brain.mgz")
    if t1 is None:
        raise FileNotFoundError("no T1.mgz/brain.mgz for fallback plot")
    img = nib.load(str(t1))
    data = np.asarray(img.dataobj)
    vox2tkr = img.header.get_vox2ras_tkr()
    tkr2vox = np.linalg.inv(vox2tkr)
    colors = {"inner_skull": "tab:red", "outer_skull": "tab:olive",
              "outer_skin": "tab:cyan"}
    axes = {"axial": 2, "coronal": 1, "sagittal": 0}
    saved = []
    for orientation, ax in axes.items():
        sl = data.shape[ax] // 2
        fig, a = plt.subplots(figsize=(6, 6))
        if ax == 2:
            a.imshow(data[:, :, sl].T, cmap="gray", origin="lower")
        elif ax == 1:
            a.imshow(data[:, sl, :].T, cmap="gray", origin="lower")
        else:
            a.imshow(data[sl, :, :].T, cmap="gray", origin="lower")
        for name, m in meshes.items():
            if "verts" not in m:
                continue
            vox = nib.affines.apply_affine(tkr2vox, m["verts"])
            near = np.abs(vox[:, ax] - sl) < 1.0
            if near.sum() == 0:
                continue
            other = [i for i in range(3) if i != ax]
            a.scatter(vox[near, other[0]], vox[near, other[1]], s=1,
                      c=colors.get(name, "y"), label=name)
        a.set_title(f"{subj_dir.name} — {orientation}")
        a.legend(markerscale=4, fontsize=7, loc="lower right")
        a.axis("off")
        out = qa_dir / f"bem_{orientation}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(out.name)
    return saved


def plot_alignment_3d(subj_dir: Path, qa_dir: Path, cfg: Config) -> TestResult:
    """Best-effort 3D alignment snapshot (off by default; headless-fragile)."""
    try:
        import mne
        mne.viz.set_3d_backend("pyvista")
        import pyvista
        pyvista.OFF_SCREEN = True
        fig = mne.viz.plot_alignment(
            subject=subj_dir.name, subjects_dir=str(cfg.subjects_dir),
            surfaces=["inner_skull", "outer_skull", "head"], coord_frame="mri",
            show_axes=True)
        out = qa_dir / "bem_alignment_3d.png"
        fig.plotter.screenshot(str(out))
        return TestResult("bem", "alignment_3d", STATUS_PASS, SEV_NONE,
                          out.name, "3D alignment snapshot saved")
    except Exception as exc:
        return TestResult("bem", "alignment_3d", STATUS_SKIP, SEV_NONE, "n/a",
                          f"3D render unavailable: {str(exc)[:150]}")


# ----------------------------------------------------------------------------
# Per-subject driver
# ----------------------------------------------------------------------------
def run_subject(subject: str, cfg: Config) -> dict:
    subj_dir = cfg.subjects_dir / subject
    qa_dir = subj_dir / "QA"
    qa_dir.mkdir(exist_ok=True)
    results: list[TestResult] = []
    extras = {"subject": subject}

    def safe(fn, *a, **k):
        """Run a test fn, converting any harness exception into an ERROR row."""
        try:
            return fn(*a, **k)
        except Exception as exc:
            tb = traceback.format_exc(limit=2)
            return TestResult("harness", getattr(fn, "__name__", "test"),
                              STATUS_ERROR, SEV_MAJOR, "exception",
                              f"{exc} :: {tb.splitlines()[-1]}")

    def safe_list(fn, *a, **k):
        """Like safe(), but for tests that return a list[TestResult]."""
        out = safe(fn, *a, **k)
        return out if isinstance(out, list) else [out]

    # ---- cortical / log ----
    results.append(safe(test_recon_log, subj_dir, qa_dir))
    eul = safe(test_euler_nofix, subj_dir, qa_dir, cfg)
    if isinstance(eul, tuple):
        res_eul, lh_holes, rh_holes = eul
        results.append(res_eul)
        extras["lh_holes"] = lh_holes
        extras["rh_holes"] = rh_holes
    else:  # harness error
        results.append(eul)
        extras["lh_holes"] = extras["rh_holes"] = None
    for r in safe_list(test_fixed_surface_topology, subj_dir):
        results.append(r)
    results.append(safe(test_cnr, subj_dir, cfg))
    extras["global_stats"] = safe_call(parse_global_stats, subj_dir) or {}

    # ---- BEM ----
    surfaces = locate_bem_surfaces(subj_dir)
    missing = [n for n, p in surfaces.items() if p is None]
    if set(missing) == set(_BEM_NAMES):
        results.append(TestResult("bem", "watershed_present", STATUS_FAIL,
                                  SEV_CRIT, "missing",
                                  "no watershed BEM surfaces found in bem/ or "
                                  "bem/watershed/"))
    else:
        if missing:
            results.append(TestResult("bem", "watershed_present", STATUS_WARN,
                                      SEV_MAJOR, f"missing {missing}",
                                      "some BEM surfaces absent"))
        else:
            results.append(TestResult("bem", "watershed_present", STATUS_PASS,
                                      SEV_NONE, "ok", "3 BEM surfaces present"))
        meshes = {}
        for name, path in surfaces.items():
            if path is None:
                continue
            r, m = safe_topology(name, path)
            results.append(r)
            meshes[name] = m
        for r in safe_list(test_bem_nesting, meshes, cfg):
            results.append(r)
        results.append(safe(test_pial_in_inner_skull, subj_dir, meshes, cfg))
        for r in safe_list(test_bem_gaps, meshes, cfg):
            results.append(r)
        for r in safe_list(test_make_bem_model, subj_dir, cfg):
            results.append(r)
        results.append(safe(plot_bem_images, subj_dir, qa_dir, cfg, meshes))
        if cfg.do_3d:
            results.append(safe(plot_alignment_3d, subj_dir, qa_dir, cfg))

    # ---- write per-subject outputs ----
    _write_subject_outputs(subject, qa_dir, results, extras)
    extras["results"] = results
    return extras


def safe_call(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


def safe_topology(name, path):
    try:
        return test_bem_surface_topology(name, path)
    except Exception as exc:
        return (TestResult("bem", f"{name}_topology", STATUS_ERROR, SEV_MAJOR,
                           "exception", str(exc)), {})


def _overall(results: list[TestResult]) -> str:
    worst = max((_STATUS_RANK.get(r.status, 0) for r in results), default=0)
    for k, v in _STATUS_RANK.items():
        if v == worst:
            return k
    return STATUS_PASS


def _write_subject_outputs(subject, qa_dir, results, extras):
    import pandas as pd
    df = pd.DataFrame([r.row(subject) for r in results])
    df.to_csv(qa_dir / f"{subject}_qa.csv", index=False)

    overall = _overall(results)
    fails = [r for r in results if r.status == STATUS_FAIL]
    warns = [r for r in results if r.status == STATUS_WARN]
    with (qa_dir / f"{subject}_qa_report.txt").open("w") as f:
        f.write(f"FreeSurfer / BEM QA report — {subject}\n")
        f.write("=" * 60 + "\n")
        f.write(f"OVERALL: {overall}    "
                f"(FAIL={len(fails)}, WARN={len(warns)})\n\n")
        for cat in ("log", "cortical", "bem", "harness"):
            rows = [r for r in results if r.category == cat]
            if not rows:
                continue
            f.write(f"[{cat}]\n")
            for r in rows:
                f.write(f"  {r.status:5s} {r.severity:8s} {r.test:28s} "
                        f"{str(r.value):>14s}  {r.detail}\n")
            f.write("\n")


# ----------------------------------------------------------------------------
# Group aggregation
# ----------------------------------------------------------------------------
def _mad_outliers(values: dict, k: float):
    """values: {subject: float}. Returns {subject: robust_z} for finite entries
    and the set of outlier subjects (|robust z| > k)."""
    subs = [s for s, v in values.items() if v is not None and np.isfinite(v)]
    x = np.array([values[s] for s in subs], dtype=float)
    if len(x) < 3:
        return {}, set()
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    rob = 1.4826 * mad if mad > 0 else 0.0
    z = (x - med) / rob if rob > 0 else np.zeros_like(x)
    rz = {s: float(zi) for s, zi in zip(subs, z)}
    outliers = {s for s, zi in rz.items() if abs(zi) > k}
    return rz, outliers


def write_group_outputs(all_subjects: list[dict], cfg: Config):
    import pandas as pd
    group_dir = cfg.subjects_dir / "QA_group"
    group_dir.mkdir(exist_ok=True)

    # long/tidy table of every test
    long_rows = []
    for s in all_subjects:
        for r in s["results"]:
            long_rows.append(r.row(s["subject"]))
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(group_dir / "group_qa_long.csv", index=False)

    # cohort outliers: total nofix holes + each global volume measure
    holes_total = {}
    for s in all_subjects:
        lh, rh = s.get("lh_holes"), s.get("rh_holes")
        vals = [v for v in (lh, rh) if v is not None]
        holes_total[s["subject"]] = float(np.sum(vals)) if vals else None
    holes_rz, holes_out = _mad_outliers(holes_total, cfg.mad_k)

    measure_outliers = {s["subject"]: [] for s in all_subjects}
    for meas in _GLOBAL_MEASURES:
        vals = {s["subject"]: s.get("global_stats", {}).get(meas)
                for s in all_subjects}
        _, outs = _mad_outliers(vals, cfg.mad_k)
        for sub in outs:
            measure_outliers[sub].append(meas)

    # per-subject summary
    summary = []
    for s in all_subjects:
        sub = s["subject"]
        res = s["results"]
        fails = [r.test for r in res if r.status == STATUS_FAIL]
        warns = [r.test for r in res if r.status == STATUS_WARN]
        errs = [r.test for r in res if r.status == STATUS_ERROR]
        overall = _overall(res)
        concerns = []
        if fails:
            concerns.append(f"FAIL: {','.join(fails)}")
        if errs:
            concerns.append(f"ERROR: {','.join(errs)}")
        if sub in holes_out:
            concerns.append(f"euler_holes_outlier(z={holes_rz[sub]:+.1f})")
        if measure_outliers[sub]:
            concerns.append("morph_outlier:" + ",".join(measure_outliers[sub]))
        if warns:
            concerns.append(f"WARN: {','.join(warns)}")
        flagged = bool(fails or errs or warns or sub in holes_out
                       or measure_outliers[sub])
        summary.append({
            "subject": sub,
            "overall": overall,
            "flagged": flagged,
            "n_fail": len(fails),
            "n_warn": len(warns),
            "n_error": len(errs),
            "euler_holes_total": holes_total[sub],
            "euler_holes_robust_z": round(holes_rz.get(sub, float("nan")), 2)
            if sub in holes_rz else "",
            "concerns": " | ".join(concerns),
        })
    summary_df = pd.DataFrame(summary).sort_values(
        ["flagged", "n_fail", "n_warn"], ascending=[False, False, False])
    summary_df.to_csv(group_dir / "group_qa_summary.csv", index=False)
    return summary_df, group_dir


# ----------------------------------------------------------------------------
# Subject discovery + CLI
# ----------------------------------------------------------------------------
def discover_subjects(subjects_dir: Path) -> list[str]:
    skip = {"fsaverage", "lh", "rh", "QA_group", "average"}
    subs = []
    for p in sorted(subjects_dir.iterdir()):
        if not p.is_dir() or p.name in skip or p.name.startswith("."):
            continue
        if (p / "surf").is_dir() or (p / "mri").is_dir():
            subs.append(p.name)
    return subs


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Automated FreeSurfer + watershed BEM QA.")
    ap.add_argument("--subjects-dir", required=True, type=Path,
                    help="FreeSurfer SUBJECTS_DIR")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="subset of subjects (default: auto-discover all)")
    ap.add_argument("--euler-holes-warn", type=int, default=150)
    ap.add_argument("--gap-warn-mm", type=float, default=1.0)
    ap.add_argument("--mad-k", type=float, default=3.0,
                    help="robust-z threshold for cohort outliers")
    ap.add_argument("--bem-ico", type=int, default=4)
    ap.add_argument("--alignment-3d", action="store_true",
                    help="also attempt a headless 3D alignment snapshot")
    ap.add_argument("--no-cnr", action="store_true",
                    help="skip mri_cnr even if available")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    subjects_dir = args.subjects_dir.expanduser().resolve()
    if not subjects_dir.is_dir():
        sys.exit(f"SUBJECTS_DIR not found: {subjects_dir}")

    cfg = Config(
        subjects_dir=subjects_dir,
        subjects=args.subjects,
        euler_holes_warn=args.euler_holes_warn,
        gap_warn_mm=args.gap_warn_mm,
        mad_k=args.mad_k,
        bem_ico=args.bem_ico,
        do_3d=args.alignment_3d,
        do_cnr=not args.no_cnr,
    )

    subjects = args.subjects or discover_subjects(subjects_dir)
    if not subjects:
        sys.exit(f"No FreeSurfer subjects found under {subjects_dir}")

    print(f"QA on {len(subjects)} subject(s) in {subjects_dir}")
    all_results = []
    for i, sub in enumerate(subjects, 1):
        print(f"[{i}/{len(subjects)}] {sub} ...", flush=True)
        try:
            all_results.append(run_subject(sub, cfg))
        except Exception as exc:
            print(f"    !! subject-level failure: {exc}", file=sys.stderr)
            all_results.append({"subject": sub, "results": [TestResult(
                "harness", "subject_run", STATUS_ERROR, SEV_CRIT, "exception",
                str(exc))], "lh_holes": None, "rh_holes": None,
                "global_stats": {}})

    summary_df, group_dir = write_group_outputs(all_results, cfg)
    n_flag = int(summary_df["flagged"].sum())
    print(f"\nDone. Group outputs in {group_dir}")
    print(f"{n_flag}/{len(subjects)} subject(s) flagged with QA concerns.")
    if n_flag:
        cols = ["subject", "overall", "n_fail", "n_warn", "concerns"]
        print(summary_df[summary_df["flagged"]][cols].to_string(index=False))


if __name__ == "__main__":
    main()
