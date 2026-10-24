"""Tests for the adaptive multipole model (custom.preprocessing._amm).

Three kinds of evidence that this is AMM as published:

* SPM: the same synthetic arrays and data go through SPM's own
  ``spm_opm_amm.m`` (unmodified, downloaded at a pinned commit and run in
  Octave on a minimal stand-in for SPM's data object, ``tests/spm_amm/``).
  With the one term SPM drops from the internal harmonics' radial derivative
  set to zero here too, bases, spheroid, projector and filtered data agree.
  Skipped when ``octave-cli`` or the download is unavailable.
* Physics: every basis vector is the field of its harmonic potential (finite
  differences), and the potentials satisfy Laplace's equation.
* Behaviour: fields of sources inside the spheroid are kept and fields of
  sources outside the array are removed; the temporal extension removes a
  near interference source that the external model cannot represent.
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

from custom.preprocessing import _amm

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
    names = [f"OPM{i:03d}" for i in range(len(pos))]
    info = mne.create_info(names, SFREQ, "mag")
    for ch, p, o in zip(info["chs"], pos, ori):
        ch["loc"][:3] = p
        ch["loc"][9:12] = o
    info["bads"] = [names[i] for i in bads]
    return info


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


def random_brain_dipoles(n, seed=0):
    """``n`` tangential current dipoles 2-6 cm from the head centre."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        q = rng.normal(size=3)
        out.append(
            (
                HEAD_CENTER + d * rng.uniform(0.02, 0.06) * np.array([0.8, 1.0, 0.9]),
                q - (q @ d) * d,
            )
        )
    return out


def spm_mode(monkeypatch):
    """Drop the term SPM drops: the derivative of Q's hypergeometric factor."""
    real = _amm._hypergeometric_ratio
    monkeypatch.setattr(
        _amm,
        "_hypergeometric_ratio",
        lambda mu, a, c, l, m: (real(mu, a, c, l, m)[0], 0 * mu),
    )


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


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------


def _potential(v, a, c, order, kind):
    mu, nu, phi = _amm.prolate_coordinates(v, c)
    cols = []
    for l in range(1, order + 1):
        for m in range(-l, l + 1):
            r, _ = _amm._radial(kind, mu, a, c, l, abs(m))
            cols.append(r * _amm._angular(nu, phi, l, m)[0])
    return np.column_stack(cols)


@pytest.mark.parametrize("kind", ["internal", "external"])
def test_fields_are_gradients_of_harmonic_potentials(kind):
    pos, ori = opm_array(n_pos=15)
    sph = _amm.fit_spheroid(pos)
    v = pos @ sph.rotation.T - sph.center
    n = ori @ sph.rotation.T
    a, b = sph.a, sph.b
    c = np.sqrt(a**2 - b**2)
    order, h = 5, 1e-5
    fields = _amm._harmonic_fields(v, n, a, b, order, kind)
    grad = np.zeros((3,) + fields.shape)
    lap = np.zeros(fields.shape)
    phi0 = _potential(v, a, c, order, kind)
    for j in range(3):
        step = np.zeros(3)
        step[j] = h
        plus, minus = (
            _potential(v + step, a, c, order, kind),
            _potential(v - step, a, c, order, kind),
        )
        grad[j] = (plus - minus) / (2 * h)
        lap += (plus - 2 * phi0 + minus) / h**2
    fd = np.einsum("ji,jik->ik", n.T, grad)
    scale = np.abs(fields).max(axis=0)
    assert np.all(np.abs(fd - fields).max(axis=0) < 1e-6 * scale)
    # Laplace's equation, relative to the size of the second derivatives
    second = np.abs(grad).max(axis=0) / np.linalg.norm(v, axis=1)[:, None]
    assert np.all(np.abs(lap).max(axis=0) < 1e-3 * second.max(axis=0))


def test_spm_dropped_term_changes_the_internal_fields(monkeypatch):
    """Without the hypergeometric derivative, internal basis vectors are not potential fields."""
    pos, ori = opm_array(n_pos=15)
    sph = _amm.fit_spheroid(pos)
    v, n = pos @ sph.rotation.T - sph.center, ori @ sph.rotation.T
    full = _amm.internal_basis(v, n, sph.a, sph.b, 9)
    spm_mode(monkeypatch)
    dropped = _amm.internal_basis(v, n, sph.a, sph.b, 9)
    assert sign_aligned_error(dropped, full) > 0.05


# ---------------------------------------------------------------------------
# Spheroid
# ---------------------------------------------------------------------------


def test_fit_spheroid_recovers_a_prolate_spheroid():
    rng = np.random.default_rng(3)
    u = rng.normal(size=(300, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    radii, center = np.array([0.08, 0.11, 0.08]), np.array([0.01, -0.02, 0.03])
    sph = _amm.fit_spheroid(u * radii + center)
    assert sph.long_axis == 1
    np.testing.assert_allclose(sph.center, center, atol=1e-9)
    # points lie on the fitted surface, so one 0.5 % step puts them all outside
    np.testing.assert_allclose(sph.radii, radii - 0.005 * radii.max(), rtol=1e-9)


def test_fit_spheroid_long_axis_and_errors(monkeypatch):
    pos, ori = opm_array(radii=(0.095, 0.075, 0.085))  # x longest
    sph = _amm.fit_spheroid(pos)
    assert sph.long_axis == 0
    v = pos @ sph.rotation.T - sph.center
    assert np.all(np.sum((v / sph.radii) ** 2, axis=1) > 1)
    u = np.random.default_rng(1).normal(size=(300, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    with pytest.raises(ValueError, match="leaves every sensor outside"):
        _amm.fit_spheroid(
            np.vstack([u * np.array([0.08, 0.11, 0.08]), np.zeros(3)])
        )  # one at the centre
    # any ellipsoid has an axis whose fit is prolate, so force oblate fits
    monkeypatch.setattr(
        _amm,
        "_fit_axis",
        lambda v, axis: (np.zeros(3), np.where(np.arange(3) == axis, 1.0, 2.0), 0.0),
    )
    with pytest.raises(ValueError, match="prolate"):
        _amm.fit_spheroid(pos)


def test_projector_does_not_depend_on_which_axis_is_long():
    """Rotating the array by SPM's 90 degree rotation leaves P_in unchanged."""
    pos, ori = opm_array()
    rot = _amm._ROTATIONS[0].T  # maps the y-long array to an x-long one
    p_y = _amm.amm_model(make_info(pos, ori)).p_in
    p_x = _amm.amm_model(make_info(pos @ rot.T, ori @ rot.T)).p_in
    np.testing.assert_allclose(p_x, p_y, atol=1e-10)


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def test_amm_keeps_brain_fields_and_removes_interference():
    pos, ori = opm_array()
    model = _amm.amm_model(make_info(pos, ori))
    assert model.rank < len(pos)
    sph = model.spheroid
    v, n = pos @ sph.rotation.T - sph.center, ori @ sph.rotation.T
    # the uncentred internal harmonics represent brain fields almost exactly
    u = _amm._range_basis(_amm._harmonic_fields(v, n, sph.a, sph.b, 9, "internal"))
    fields = np.array(
        [current_dipole_field(pos, ori, r0, q) for r0, q in random_brain_dipoles(200)]
    )
    in_span = np.linalg.norm(fields - fields @ u @ u.T, axis=1) / np.linalg.norm(
        fields, axis=1
    )
    assert np.percentile(in_span, 95) < 0.01
    # what AMM keeps of them: the part not shared with the external space and
    # not the channel mean (SPM's centring); median correlation ~0.9 here
    lost = np.linalg.norm(fields - fields @ model.p_in, axis=1) / np.linalg.norm(
        fields, axis=1
    )
    corr = np.sqrt(1 - lost**2)
    assert np.median(corr) > 0.85 and np.percentile(corr, 5) > 0.65
    # uniform fields and linear gradients are external harmonics of order <= 2,
    # so they go entirely; a dipole 1.5 m away leaves its order >= 3 terms
    uniform = ori
    gradient = np.column_stack(
        [
            pos[:, 0] * ori[:, 2] + pos[:, 2] * ori[:, 0],
            pos[:, 1] * ori[:, 1] - pos[:, 2] * ori[:, 2],
        ]
    )
    far = np.column_stack(
        [dipole_field(pos, ori, np.array([1.2, -0.8, 0.5]), m) for m in np.eye(3)]
    )
    for field, limit in [(uniform, 1e-10), (gradient, 1e-10), (far, 0.02)]:
        left = np.linalg.norm(model.p_in @ field, axis=0) / np.linalg.norm(
            field, axis=0
        )
        assert np.all(left < limit)


def test_temporal_extension_removes_a_near_source():
    pos, ori = opm_array()
    model = _amm.amm_model(make_info(pos, ori))
    data, parts = synthetic_data(pos, ori, 2000)
    t = np.arange(2000) / SFREQ

    def amplitude(out, freq, phase, topo):
        """Least-squares amplitude of ``topo`` at ``freq`` in ``out``, relative to P_in topo."""
        s = np.sin(2 * np.pi * freq * t + phase)
        ref = model.p_in @ topo
        return (out @ s / (s @ s)) @ ref / (ref @ ref)

    near_topo = (
        parts["near"]
        @ np.sin(2 * np.pi * 7.0 * t)
        / np.sum(np.sin(2 * np.pi * 7.0 * t) ** 2)
    )
    brain_topo = current_dipole_field(
        pos, ori, HEAD_CENTER + np.array([0.03, 0.02, 0.03]), np.array([1.0, 0, 0])
    )
    spatial = _amm.temporal_extension(data, model, 1.0)
    temporal = _amm.temporal_extension(data, model, 0.98)
    np.testing.assert_allclose(
        spatial, model.p_in @ data, atol=1e-10 * np.abs(data).max()
    )
    assert amplitude(spatial, 7.0, 0, near_topo) > 0.95
    assert abs(amplitude(temporal, 7.0, 0, near_topo)) < 0.1
    assert 0.8 < amplitude(temporal, 11.0, 0.3, brain_topo) < 1.2


def test_window_bounds():
    assert _amm.window_bounds(10, 4) == [(0, 4), (4, 10)]
    assert _amm.window_bounds(8, 4) == [(0, 4), (4, 8)]
    assert _amm.window_bounds(3, 4) == [(0, 3)]


# ---------------------------------------------------------------------------
# MNE integration
# ---------------------------------------------------------------------------


def test_apply_amm_to_raw(tmp_path):
    pos, ori = opm_array()
    info = make_info(pos, ori, bads=[4])
    data, _ = synthetic_data(pos, ori, 1000)
    raw = mne.io.RawArray(data * 1e-15, info, verbose=False)
    noise = raw.copy()
    raw, model = _amm.apply_amm(raw)
    good = mne.pick_types(raw.info, meg=True, exclude="bads")
    scale = np.abs(data).max() * 1e-15
    np.testing.assert_allclose(
        raw.get_data(good), model.p_in @ data[good] * 1e-15, atol=1e-10 * scale
    )
    np.testing.assert_allclose(
        raw.get_data([4]), data[[4]] * 1e-15, rtol=1e-12
    )  # bad channel kept
    assert len(raw.info["projs"]) == len(good) - model.rank
    assert all(p["active"] for p in raw.info["projs"])
    proj, _, _ = mne._fiff.proj.make_projector(raw.info["projs"], list(model.ch_names))
    np.testing.assert_allclose(proj, model.p_in, atol=1e-10)
    assert mne.compute_rank(raw, rank="info")["mag"] == model.rank
    path = tmp_path / "amm_raw.fif"
    raw.save(path, verbose=False)
    assert len(mne.io.read_raw_fif(path, verbose=False).info["projs"]) == len(
        raw.info["projs"]
    )
    # another recording with the same channels takes the task's projection
    noise, same = _amm.apply_amm(noise, model=model, corr_lim=0.98, window=2.0)
    assert same is model
    assert np.allclose(model.p_in @ noise.get_data(good), noise.get_data(good))
    raw.drop_channels([model.ch_names[0]])
    with pytest.raises(ValueError, match="lacks AMM channels"):
        _amm.apply_amm(raw.copy(), model=model)


def test_apply_amm_step(tmp_path, capsys):
    """The pipeline step fits AMM to the task and applies its projection to the noise."""
    from custom.preprocessing.apply_amm import ApplyAMMAnalysis, run

    pos, ori = opm_array()
    info = make_info(pos, ori, bads=[2])
    data, _ = synthetic_data(pos, ori, 600)
    task = mne.io.RawArray(data * 1e-15, info, verbose=False)
    noise = mne.io.RawArray(
        np.random.default_rng(5).normal(size=data.shape) * 1e-15, info, verbose=False
    )
    cfg = SimpleNamespace(
        task="T", _do_AMM=True, _amm_li=7, _amm_corr_lim=0.98, _amm_window=1.0
    )
    results = ApplyAMMAnalysis(cfg).run({"T": task, "noise": noise})
    model = _amm.amm_model(info, li=7)
    good = mne.pick_types(info, meg=True, exclude="bads")
    for raw in (results["T"], results["noise"]):
        assert len(raw.info["projs"]) == len(good) - model.rank
        np.testing.assert_allclose(
            model.p_in @ raw.get_data(good), raw.get_data(good), atol=1e-25
        )
    assert (
        "keeps 63 of 119 dimensions; temporal extension corr_lim=0.98"
        in capsys.readouterr().out
    )
    run(SimpleNamespace(_do_AMM=False))
    assert "Disabled" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Agreement with SPM
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spm_dir(tmp_path_factory):
    if shutil.which("octave-cli") is None:
        pytest.skip("octave-cli is not installed")
    cache = Path(os.environ.get("SPM_AMM_CACHE", tmp_path_factory.mktemp("spm")))
    for name in SPM_FILES:
        target = cache / name
        if not target.exists():
            url = f"https://raw.githubusercontent.com/spm/spm/{SPM_COMMIT}/{name}"
            try:
                urllib.request.urlretrieve(url, target)
            except OSError as e:
                pytest.skip(f"could not download SPM's {name}: {e}")
    return cache


def run_spm(spm_dir, tmp_path, pos, ori, data, bad, corr_lim, window):
    """Run SPM's spm_opm_amm (and its basis functions on our spheroid) in Octave."""
    info = make_info(pos, ori, bads=[bad])
    sph = _amm.amm_model(info).spheroid
    v = pos @ sph.rotation.T - sph.center
    n = ori @ sph.rotation.T
    labels = np.array(info["ch_names"])
    scipy.io.savemat(
        tmp_path / "in.mat",
        dict(
            pos=pos * 1000,
            ori=ori,
            labels=labels,
            data=data,
            fsample=SFREQ,
            bad=np.array([bad + 1.0]),
            li=9.0,
            le=2.0,
            corrLim=corr_lim,
            window=window,
            v=v * 1000,
            n=n,
            a=sph.a * 1000,
            b=sph.b * 1000,
        ),
    )
    cmd = (
        f"addpath('{spm_dir}'); addpath('{HARNESS}'); "
        f"run_spm_amm('{tmp_path / 'in.mat'}', '{tmp_path / 'out.mat'}')"
    )
    result = subprocess.run(
        ["octave-cli", "--no-gui", "--quiet", "--eval", cmd],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    out = scipy.io.loadmat(tmp_path / "out.mat")
    return info, v, n, sph, out


@pytest.mark.parametrize(
    "radii", [(0.075, 0.095, 0.085), (0.095, 0.075, 0.085)], ids=["y-long", "x-long"]
)
def test_matches_spm(spm_dir, tmp_path, monkeypatch, radii):
    pos, ori = opm_array(radii=radii)
    n_win = 1000  # long enough that rounding adds no dimension in SPM's orth
    data, _ = synthetic_data(pos, ori, 3 * n_win + 1)
    bad = 7
    info, v, n, sph, out = run_spm(
        spm_dir, tmp_path, pos, ori, data, bad, 0.98, n_win / SFREQ
    )

    assert (
        sign_aligned_error(_amm.external_basis(v, n, sph.a, sph.b, 2), out["external"])
        < 1e-12
    )
    full = _amm.internal_basis(v, n, sph.a, sph.b, 9)
    assert sign_aligned_error(full, out["internal"]) > 0.05  # SPM drops a term
    spm_mode(monkeypatch)
    assert (
        sign_aligned_error(_amm.internal_basis(v, n, sph.a, sph.b, 9), out["internal"])
        < 1e-10
    )

    # spheroid (SPM prints it with 6 significant digits) and projector
    balance = str(out["balance"][0])
    numbers = np.array(re.findall(r"[-+0-9.e]+", balance.split("[")[1]), float)
    mine = np.r_[sph.center * 1000, sph.a * 1000, sph.b * 1000]
    np.testing.assert_allclose(numbers[2:], mine, rtol=1e-5, atol=1e-3)
    model = _amm.amm_model(info)
    good = [i for i in range(len(pos)) if i != bad]
    np.testing.assert_allclose(model.p_in, out["tra"][np.ix_(good, good)], atol=1e-10)

    # filtered data, over the windows both process identically
    ours = np.column_stack(
        [
            _amm.temporal_extension(data[good, s : s + n_win], model, 0.98)
            for s in (0, n_win)
        ]
    )
    theirs = out["data"][good, : 2 * n_win]
    assert np.linalg.norm(
        ours - model.p_in @ data[good, : 2 * n_win]
    ) > 0.01 * np.linalg.norm(ours)
    np.testing.assert_allclose(ours, theirs, atol=1e-9 * np.abs(theirs).max())
    np.testing.assert_array_equal(out["data"][good, -1], data[good, -1])  # SPM skips it
