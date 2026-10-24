"""Tests of adaptive multipole models (AMM) against SPM, and of the apply_amm step.

AMM is implemented in MNE (``mne.preprocessing.compute_proj_amm`` and
``apply_amm``), whose own tests check the physics and behaviour.  Here the
same synthetic arrays and data also go through SPM's ``spm_opm_amm.m``
(unmodified, downloaded at a pinned commit and run in MATLAB or Octave on a minimal
stand-in for SPM's data object, ``tests/spm_amm/``).  With MNE in SPM's
configuration (``center_basis=True``, point coils, the device frame) and the
one term SPM drops from the internal harmonics' radial derivative set to zero
here too, bases, spheroid, projector and filtered data agree.  Skipped when
neither MATLAB nor Octave is installed, or the download is unavailable.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
import pytest
import scipy.io
from mne._fiff.proj import make_projector
from mne.preprocessing import hfc

SPM_COMMIT = "54dc957191112dcad116e3eaa457e881fa8260de"
SPM_FILES = (
    "spm_opm_amm.m",
    "spm_ipharm.m",
    "spm_epharm.m",
    "spm_slm.m",
    "spm_match_str.m",
)
HARNESS = Path(__file__).parent / "spm_amm"
SFREQ = 200.0
HEAD_CENTER = np.array([0.002, 0.01, 0.04])


# ---------------------------------------------------------------------------
# Synthetic arrays and sources
# ---------------------------------------------------------------------------


def opm_array(n_pos=40, radii=(0.075, 0.095, 0.085), offset=0.015, seed=0):
    """Triaxial sensors 15 mm outside the upper part of a head-like ellipsoid.

    Returns ``(positions, orientations)`` of ``3 n_pos`` channels, in metres.
    """
    rng = np.random.default_rng(seed)
    radii = np.asarray(radii)
    k = np.arange(4 * n_pos) + 0.5  # Fibonacci points on the sphere
    z = 1 - 2 * k / len(k)
    phi = np.pi * (1 + 5**0.5) * k
    unit = np.column_stack(
        [np.sqrt(1 - z**2) * np.cos(phi), np.sqrt(1 - z**2) * np.sin(phi), z]
    )
    unit = unit[unit[:, 2] > -0.25]
    unit = unit[rng.choice(len(unit), n_pos, replace=False)]
    surface = unit * radii
    normal = surface / radii**2
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    pos = surface + offset * normal + HEAD_CENTER
    t1 = np.cross(normal, [0.3, 0.2, 1.0])
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(normal, t1)
    return np.repeat(pos, 3, axis=0), np.stack([normal, t1, t2], axis=1).reshape(-1, 3)


def make_info(pos, ori, bads=()):
    """Point magnetometers with a full coil frame, as MNE's coil handling needs."""
    names = [f"OPM{i:03d}" for i in range(len(pos))]
    info = mne.create_info(names, SFREQ, "mag")
    for ch, p, ez in zip(info["chs"], pos, ori):
        ex = np.cross(ez, [0.3, 0.2, 1.0])
        ex /= np.linalg.norm(ex)
        ch["loc"][:] = np.r_[p, ex, np.cross(ez, ex), ez]
        ch["coil_type"] = mne.io.constants.FIFF.FIFFV_COIL_POINT_MAGNETOMETER
    info["bads"] = [names[i] for i in bads]
    return info


SPM_CONFIG = dict(coord_frame="meg", center_basis=True, accuracy="point")


def dipole_field(pos, ori, r0, moment):
    """Field of a magnetic dipole at ``r0`` along each sensor's orientation (arbitrary units)."""
    r = pos - r0
    dist = np.linalg.norm(r, axis=1, keepdims=True)
    rhat = r / dist
    b = (3 * rhat * (rhat @ moment)[:, None] - moment) / dist**3
    return np.sum(b * ori, axis=1)


def current_dipole_field(pos, ori, r0, q, origin=HEAD_CENTER):
    """Field of a current dipole in a spherical conductor, along each sensor's orientation.

    Sarvas, J. (1987). Basic mathematical and electromagnetic concepts of the
    biomagnetic inverse problem. Physics in Medicine and Biology, 32(1), 11-22
    (eq. 25; mu_0 / 4 pi omitted).
    """
    r, r0 = pos - origin, np.asarray(r0) - origin
    a = r - r0
    a_n, r_n = np.linalg.norm(a, axis=1), np.linalg.norm(r, axis=1)
    a_r = np.sum(a * r, axis=1)
    f = a_n * (r_n * a_n + r_n**2 - r @ r0)
    grad_f = (a_n**2 / r_n + a_r / a_n + 2 * a_n + 2 * r_n)[:, None] * r - (
        a_n + 2 * r_n + a_r / a_n
    )[:, None] * r0
    q_r0 = np.cross(q, r0)
    b = (f[:, None] * q_r0 - (r @ q_r0)[:, None] * grad_f) / f[:, None] ** 2
    return np.sum(b * ori, axis=1)


def spm_mode(monkeypatch):
    """Drop the term SPM drops: the derivative of Q's hypergeometric factor."""
    real = hfc._hypergeometric_ratio
    monkeypatch.setattr(
        hfc,
        "_hypergeometric_ratio",
        lambda mu, a, c, deg, m: (real(mu, a, c, deg, m)[0], 0 * mu),
    )


def spm_normalise(fields):
    """Centre and scale every basis vector across channels, as SPM does."""
    fields = fields - fields.mean(axis=0)
    return fields / fields.std(axis=0, ddof=1)


def sign_aligned_error(mine, ref):
    sign = np.sign(np.sum(mine * ref, axis=0))
    return np.max(np.abs(mine * sign - ref)) / np.max(np.abs(ref))


def synthetic_data(pos, ori, n_times, seed=1):
    """Brain + far interference + a near interference source + sensor noise.

    The brain (two current dipoles, 11 and 23 Hz) has a peak SNR of 3 per
    channel.  Far interference (uniform field random walk, a gradient, a
    dipole 1.5 m away) is what the external harmonics model; the near source
    (a magnetic dipole 4 cm in front of the array, 7 Hz) is too close for
    them, so it leaks into the internal and intermediate spaces.  Returns
    ``(data, parts)``, ``parts`` holding the noise-free field of each kind.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_times) / SFREQ
    brain = sum(
        np.outer(
            current_dipole_field(pos, ori, HEAD_CENTER + r, q),
            np.sin(2 * np.pi * f * t + ph),
        )
        for r, q, f, ph in [
            (np.array([0.03, 0.02, 0.03]), np.array([1.0, 0.0, 0.0]), 11.0, 0.3),
            (np.array([-0.035, -0.01, 0.02]), np.array([0.0, 0.6, -0.4]), 23.0, 1.1),
        ]
    )
    scale = np.abs(brain).max()
    walk = np.cumsum(rng.normal(size=(3, n_times)), axis=1)
    uniform = ori @ walk * (30 * scale / np.abs(walk).max())
    gradient = np.outer(
        pos[:, 0] * ori[:, 2] + pos[:, 2] * ori[:, 0], np.sin(2 * np.pi * 0.7 * t)
    )
    gradient *= 20 * scale / np.abs(gradient).max()
    distant = np.outer(
        dipole_field(pos, ori, np.array([1.2, -0.8, 0.5]), np.array([3.0, 1.0, -2.0])),
        np.sin(2 * np.pi * 1.3 * t),
    )
    distant *= 20 * scale / np.abs(distant).max()
    near = np.outer(
        dipole_field(
            pos,
            ori,
            HEAD_CENTER + np.array([0.0, 0.14, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ),
        np.sin(2 * np.pi * 7.0 * t),
    )
    near *= 50 * scale / np.abs(near).max()
    noise = rng.normal(scale=scale / 3, size=brain.shape)
    parts = dict(brain=brain, external=uniform + gradient + distant, near=near)
    return brain + uniform + gradient + distant + near + noise, parts


def test_apply_amm_step(capsys):
    """The step applies MNE's AMM to the task and the task's projectors to the noise."""
    from custom.preprocessing.apply_amm import ApplyAMMAnalysis, run

    pos, ori = opm_array()
    info = make_info(pos, ori, bads=[2])
    data, _ = synthetic_data(pos, ori, 600)
    task = mne.io.RawArray(data * 1e-15, info, verbose=False)
    noise = mne.io.RawArray(
        np.random.default_rng(5).normal(size=data.shape) * 1e-15, info, verbose=False
    )
    cfg = SimpleNamespace(
        task="T",
        _do_AMM=True,
        _amm_li=7,
        _amm_corr_lim=0.98,
        _amm_window=1.0,
        _amm_coord_frame="meg",
    )
    results = ApplyAMMAnalysis(cfg).run({"T": task, "noise": noise})
    projs = mne.preprocessing.compute_proj_amm(info, int_order=7, coord_frame="meg")
    good = mne.pick_types(info, meg=True, exclude="bads")
    proj = make_projector(projs, [info["ch_names"][i] for i in good])[0]
    for raw in (results["T"], results["noise"]):
        assert len(raw.info["projs"]) == len(projs) == len(good) - 63
        np.testing.assert_allclose(
            proj @ raw.get_data(good), raw.get_data(good), atol=1e-25
        )
    out = capsys.readouterr().out
    assert "keeps 63 of 119 dimensions; temporal extension corr_lim=0.98" in out
    run(SimpleNamespace(_do_AMM=False))
    assert "Disabled" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Agreement with SPM
# ---------------------------------------------------------------------------


def _spm_command():
    """Prefer MATLAB, then Octave's CLI, then its general executable."""
    for executable, args in (
        ("matlab", ["-batch"]),
        ("octave-cli", ["--no-gui", "--quiet", "--eval"]),
        ("octave", ["--no-gui", "--quiet", "--eval"]),
    ):
        path = shutil.which(executable)
        if path is not None:
            return [path, *args]
    return None


@pytest.mark.parametrize(
    "available, expected",
    [
        (("matlab", "octave-cli", "octave"), ["/tools/matlab", "-batch"]),
        (("matlab",), ["/tools/matlab", "-batch"]),
        (
            ("octave-cli", "octave"),
            ["/tools/octave-cli", "--no-gui", "--quiet", "--eval"],
        ),
        (("octave",), ["/tools/octave", "--no-gui", "--quiet", "--eval"]),
        ((), None),
    ],
)
def test_spm_runtime_selection(monkeypatch, available, expected):
    """Select the preferred installed runtime without requiring either one."""
    monkeypatch.setattr(
        shutil, "which", lambda name: f"/tools/{name}" if name in available else None
    )
    assert _spm_command() == expected


@pytest.fixture(scope="module")
def spm_runtime():
    command = _spm_command()
    if command is None:
        pytest.skip("neither MATLAB nor Octave is installed")
    return command


@pytest.fixture(scope="module")
def spm_dir(tmp_path_factory, spm_runtime):
    cache = Path(os.environ.get("SPM_AMM_CACHE", tmp_path_factory.mktemp("spm")))
    cache.mkdir(parents=True, exist_ok=True)
    for name in SPM_FILES:
        target = cache / name
        if not target.exists():
            url = f"https://raw.githubusercontent.com/spm/spm/{SPM_COMMIT}/{name}"
            try:
                urllib.request.urlretrieve(url, target)
            except OSError as e:
                pytest.skip(f"could not download SPM's {name}: {e}")
    return cache


def run_spm(spm_dir, spm_runtime, tmp_path, pos, ori, data, bad, corr_lim, window):
    """Run SPM's spm_opm_amm and basis functions in MATLAB or Octave."""
    info = make_info(pos, ori, bads=[bad])
    rotation, center, radii = hfc._fit_spheroid(np.delete(pos, bad, axis=0))
    v = pos @ rotation.T - center
    n = ori @ rotation.T
    a, b = radii.max(), radii.min()
    scipy.io.savemat(
        tmp_path / "in.mat",
        dict(
            pos=pos * 1000,
            ori=ori,
            labels=np.array(info["ch_names"]),
            data=data,
            fsample=SFREQ,
            bad=np.array([bad + 1.0]),
            li=9.0,
            le=2.0,
            corrLim=corr_lim,
            window=window,
            v=v * 1000,
            n=n,
            a=a * 1000,
            b=b * 1000,
        ),
    )
    spm_path, harness_path, infile, outfile = (
        str(path).replace("'", "''")
        for path in (spm_dir, HARNESS, tmp_path / "in.mat", tmp_path / "out.mat")
    )
    cmd = (
        f"addpath('{spm_path}'); addpath('{harness_path}'); "
        f"run_spm_amm('{infile}', '{outfile}')"
    )
    result = subprocess.run(
        [*spm_runtime, cmd],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    out = scipy.io.loadmat(tmp_path / "out.mat")
    return info, v, n, np.r_[center, a, b], out


@pytest.mark.parametrize(
    "radii", [(0.075, 0.095, 0.085), (0.095, 0.075, 0.085)], ids=["y-long", "x-long"]
)
def test_matches_spm(spm_dir, spm_runtime, tmp_path, monkeypatch, radii):
    pos, ori = opm_array(radii=radii)
    n_win = 1000  # long enough that rounding adds no dimension in SPM's orth
    data, _ = synthetic_data(pos, ori, 3 * n_win + 1)
    bad = 7
    info, v, n, spheroid, out = run_spm(
        spm_dir, spm_runtime, tmp_path, pos, ori, data, bad, 0.98, n_win / SFREQ
    )
    a, b = spheroid[3:]

    def basis(order, kind):
        return spm_normalise(hfc._prolate_fields(v, n, a, b, order, kind))

    assert sign_aligned_error(basis(2, "external"), out["external"]) < 1e-12
    assert sign_aligned_error(basis(9, "internal"), out["internal"]) > 0.05
    spm_mode(monkeypatch)  # from here on, without the term SPM drops
    assert sign_aligned_error(basis(9, "internal"), out["internal"]) < 1e-10

    # spheroid (SPM prints it with 6 significant digits) and projector
    balance = str(out["balance"][0])
    numbers = np.array(re.findall(r"[-+0-9.e]+", balance.split("[")[1]), float)
    np.testing.assert_allclose(numbers[2:], spheroid * 1000, rtol=1e-5, atol=1e-3)
    good = [i for i in range(len(pos)) if i != bad]
    names = [info["ch_names"][i] for i in good]
    proj = make_projector(
        mne.preprocessing.compute_proj_amm(info, **SPM_CONFIG), names
    )[0]
    np.testing.assert_allclose(proj, out["tra"][np.ix_(good, good)], atol=1e-10)

    # filtered data, over the windows both process identically
    raw = mne.io.RawArray(data, info, verbose=False)
    raw = mne.preprocessing.apply_amm(
        raw, st_duration=n_win / SFREQ, st_correlation=0.98, **SPM_CONFIG
    )
    ours = raw.get_data(good)[:, : 2 * n_win]
    theirs = out["data"][good, : 2 * n_win]
    spatial = proj @ data[good, : 2 * n_win]
    assert np.linalg.norm(ours - spatial) > 0.01 * np.linalg.norm(ours)
    np.testing.assert_allclose(ours, theirs, atol=1e-9 * np.abs(theirs).max())
    np.testing.assert_array_equal(out["data"][good, -1], data[good, -1])  # SPM skips it
