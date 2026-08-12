"""Forward-model-based simulation of the synthetic OPM recording.

Task data is built from a real forward solution computed on the synthetic
anatomy, so a beamformer run on this subject localises to a position that is
known exactly.  The ground-truth vertices and their MRI coordinates are written
to ``ground_truth.json`` alongside the dataset.

Signal budget (sensor-space RMS over the whole recording, before cleaning):

===========================  ==========  ==================================
Component                    RMS         Why it is there
===========================  ==========  ==================================
Evoked cortical response     ~120 fT     the thing beamforming should find
Background cortical activity ~300 fT     realistic, spatially structured noise
Occipital alpha              ~180 fT     gives the PSD a believable peak
Ocular (blink/saccade)       ~800 fT     ICA EOG detection, EOG regression
Cardiac                      ~250 fT     ICA ECG detection
Uniform external field       ~1.2 pT     what HFC / Maxwell filtering removes
External field gradient      ~0.5 pT     what HFC order > 1 removes
Sensor noise                 ~120 fT     white, 12 fT/sqrt(Hz)
===========================  ==========  ==================================

The evoked response is deliberately the *smallest* brain component: it is
meant to emerge from averaging, not to be visible in single trials.

Planted defects (so the detection steps have real targets):

* two high-variance channels and one near-flat channel;
* two broadband bursts, for ``bad_segments``;
* a handful of large ocular transients, for ``bad_epochs`` / ``ptp_reject``.

Author: Harrison Ritz (2025)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


__all__ = [
    "SourceSpec",
    "build_forward",
    "default_sources",
    "simulate_empty_room",
    "simulate_task",
]


MU0_OVER_4PI = 1e-7  # T*m/A

# Sensor-space RMS targets, in tesla, measured over the whole recording.
#
# The relative sizes matter more than the absolute ones.  In particular the
# evoked response has to stay *small* next to ongoing activity and next to the
# ocular artifact: it only becomes visible after averaging.  An evoked response
# that dominates single trials is not just unrealistic, it actively breaks the
# pipeline — ICA sees a large, repeatable, non-Gaussian component and correctly
# concludes it is an artifact, so the automatic component rejection removes the
# very signal the beamformer is supposed to find.
_TARGET = dict(
    evoked=1.2e-13,
    background=3.0e-13,
    alpha=1.8e-13,
    ocular=8.0e-13,
    cardiac=2.5e-13,
    uniform=1.2e-12,
    gradient=5.0e-13,
)
_SENSOR_NOISE_DENSITY = 12e-15  # T/sqrt(Hz)


@dataclass
class SourceSpec:
    """A ground-truth cortical source."""

    name: str
    hemi: str
    #: Direction from the hemisphere's centre, in MRI surface RAS.  Scaled by
    #: the hemisphere's semi-axes it gives a point on the cortical envelope,
    #: which is then snapped to a real source-space vertex.  Specifying a
    #: direction rather than a coordinate keeps the ground truth anchored to
    #: the anatomy when the phantom's proportions change.
    direction: tuple[float, float, float]
    #: Peak latency (s) and width (s) of the evoked response.
    latency: float = 0.09
    width: float = 0.035
    #: Relative strength, before the whole evoked ensemble is scaled to its
    #: sensor-space RMS target.
    amplitude: float = 1.0
    #: Per-condition gain, keyed by the ``condition`` column of the metadata.
    condition_gain: dict[str, float] = field(default_factory=lambda: {"A": 1.0, "B": 1.0})
    #: When set, the response is locked to the response event rather than the
    #: stimulus, and only fires for this response hand.
    response_locked: str | None = None

    # Filled in by ``_resolve_sources`` once a source space exists.  Note that
    # ``position`` and ``normal`` come out in **head** coordinates, which is
    # the frame ``mne.make_forward_solution`` leaves its source space in.
    vertno: int | None = None
    hemi_idx: int | None = None
    position: tuple[float, float, float] | None = None
    normal: tuple[float, float, float] | None = None
    radiality: float | None = None
    visibility: float | None = None
    column: int | None = None


def default_sources() -> list[SourceSpec]:
    """The three-dipole ground truth shipped with the synthetic dataset.

    Latencies are spread far enough apart that each source is the dominant one
    in the source map at its own peak, which is what makes ``latency_s`` a
    usable place to check localisation.  The left temporal source is deeper and
    the array covers it less well, so it carries a larger amplitude to land in
    the same ballpark as the other two at the sensors.
    """
    return [
        SourceSpec(
            name="occipital_visual",
            hemi="lh",
            direction=(-0.15, -0.95, 0.28),
            latency=0.070,
            width=0.022,
            amplitude=1.0,
            condition_gain={"A": 0.9, "B": 0.9},
        ),
        SourceSpec(
            name="left_temporal",
            hemi="lh",
            direction=(-0.95, 0.05, -0.30),
            latency=0.140,
            width=0.030,
            amplitude=1.2,
            condition_gain={"A": 1.0, "B": 0.35},
        ),
        SourceSpec(
            name="right_parietal",
            hemi="rh",
            direction=(0.35, -0.50, 0.79),
            latency=0.230,
            width=0.040,
            amplitude=1.0,
            condition_gain={"A": 0.35, "B": 1.0},
        ),
    ]


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def _sensor_geometry(info) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return MEG picks and their positions / sensing axes in head coordinates."""
    import mne
    from mne.transforms import apply_trans

    picks = mne.pick_types(info, meg=True, exclude=())
    pos = np.array([info["chs"][p]["loc"][:3] for p in picks])
    ori = np.array([info["chs"][p]["loc"][9:12] for p in picks])
    dev_head_t = info["dev_head_t"]
    return picks, apply_trans(dev_head_t, pos), apply_trans(dev_head_t, ori, move=False)


def _dipole_field(
    sensor_pos: np.ndarray,
    sensor_ori: np.ndarray,
    dipole_pos: np.ndarray,
    dipole_moment: np.ndarray,
) -> np.ndarray:
    """Field of a magnetic dipole, projected on each sensor's sensing axis.

    Used for the artifact sources (eyes, heart) that sit outside the head model
    and therefore cannot go through the BEM forward solution.
    """
    r = sensor_pos - np.asarray(dipole_pos, float)
    dist = np.linalg.norm(r, axis=1, keepdims=True)
    rhat = r / dist
    m = np.asarray(dipole_moment, float)
    field = MU0_OVER_4PI * (3.0 * rhat * (rhat @ m)[:, None] - m) / dist**3
    return np.einsum("ij,ij->i", field, sensor_ori)


def _pink_noise(rng, n_signals: int, n_times: int, exponent: float = 1.0) -> np.ndarray:
    """Zero-mean 1/f^exponent noise via spectral shaping."""
    spec = rng.standard_normal((n_signals, n_times // 2 + 1)) + 1j * rng.standard_normal(
        (n_signals, n_times // 2 + 1)
    )
    freqs = np.fft.rfftfreq(n_times, d=1.0)
    scale = np.ones_like(freqs)
    scale[1:] = freqs[1:] ** (-exponent / 2.0)
    scale[0] = 0.0
    out = np.fft.irfft(spec * scale, n=n_times)
    out -= out.mean(axis=1, keepdims=True)
    return out


def _scale_to_rms(gain: np.ndarray, source_ts: np.ndarray, target_rms: float):
    """Scale ``source_ts`` so ``gain @ source_ts`` has the requested sensor RMS."""
    sensor = gain @ source_ts
    rms = np.sqrt(np.mean(sensor**2))
    if rms <= 0:
        return sensor
    return sensor * (target_rms / rms)


def _bump(times: np.ndarray, centre: float, width: float) -> np.ndarray:
    """Evoked kernel: a Gaussian-windowed cosine, peaking at ``centre``.

    Peaking *at* the nominal latency matters — ``latency_s`` in
    ``ground_truth.json`` is what a test (or a person) evaluates localisation
    at, so a kernel whose extremum sits somewhere else makes the recorded
    ground truth quietly wrong.
    """
    u = (times - centre) / width
    return np.exp(-0.5 * u**2) * np.cos(np.pi * u / 2.5)


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------


def build_forward(info, head, subjects_dir, fs_subject, *, spacing: str = "oct4"):
    """Fixed-orientation cortical forward solution on the synthetic anatomy.

    A coarse (``oct4``) source space is plenty: it is only used to *generate*
    data, and keeping it small keeps dataset generation to a few seconds.
    The beamformer that later analyses the data builds its own, finer space.
    """
    import mne

    bem = mne.make_bem_solution(
        mne.make_bem_model(
            fs_subject, subjects_dir=str(subjects_dir), conductivity=(0.3,), ico=4
        )
    )
    src = mne.setup_source_space(
        fs_subject, spacing=spacing, subjects_dir=str(subjects_dir), add_dist=False
    )
    meg_info = mne.pick_info(info, mne.pick_types(info, meg=True, exclude=()))
    fwd = mne.make_forward_solution(
        meg_info, trans=head.mri_head_t, src=src, bem=bem, mindist=5.0, n_jobs=1
    )
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
    return fwd, bem


def _resolve_sources(
    fwd, sources: list[SourceSpec], head, *, search_radius: float = 0.025
) -> list[SourceSpec]:
    """Snap each requested position to a usable source-space vertex.

    Two things make this more than a nearest-neighbour lookup.

    First, ``mne.make_forward_solution`` returns its source space in **head**
    coordinates, while ``SourceSpec.target`` is written in MRI surface RAS
    (where one can actually reason about anatomy), so the target has to be
    transformed before anything is compared.

    Second, and more importantly: on a near-spherical conductor a dipole
    oriented along the radial direction produces almost no external magnetic
    field.  Since the simulation orients each source along the cortical normal,
    landing on a gyral crown would plant a source MEG physically cannot see —
    and no beamformer would find it.  So rather than the nearest vertex we take
    the *most visible* one within ``search_radius``: the vertex whose forward
    column has the largest norm.  That is the direct measure of how much field
    a unit dipole there produces, and it lands the source on a sulcal wall,
    which is where real MEG-visible sources live.
    """
    from mne.transforms import apply_trans

    src = fwd["src"]
    gain = fwd["sol"]["data"]
    offsets = np.cumsum([0] + [s["nuse"] for s in src])
    normals = fwd["source_nn"]
    centre = np.vstack([s["rr"][s["vertno"]] for s in src]).mean(axis=0)

    for spec in sources:
        hemi_idx = 0 if spec.hemi == "lh" else 1
        used = src[hemi_idx]["vertno"]
        rr = src[hemi_idx]["rr"][used]
        start = int(offsets[hemi_idx])
        nn = normals[start : offsets[hemi_idx + 1]]

        # Direction -> a point on the hemisphere's envelope, in MRI surface
        # RAS, then into the head frame the forward's source space lives in.
        unit = np.asarray(spec.direction, float)
        unit = unit / np.linalg.norm(unit)
        hemi_centre = head.center + np.array(
            [(-1.0 if spec.hemi == "lh" else 1.0) * head.hemi_offset, 0.0, 0.0]
        )
        target_head = apply_trans(
            head.mri_head_t, hemi_centre + unit * head.hemi_axes
        )
        distance = np.linalg.norm(rr - target_head, axis=1)
        candidates = np.flatnonzero(distance <= search_radius)
        if candidates.size == 0:
            candidates = np.array([int(np.argmin(distance))])

        visibility = np.linalg.norm(gain[:, start + candidates], axis=0)
        chosen = int(candidates[int(np.argmax(visibility))])

        radial = rr[chosen] - centre
        radial /= np.linalg.norm(radial)

        spec.hemi_idx = hemi_idx
        spec.vertno = int(used[chosen])
        spec.position = tuple(float(v) for v in rr[chosen])
        spec.normal = tuple(float(v) for v in nn[chosen])
        spec.radiality = float(abs(nn[chosen] @ radial))
        spec.column = start + chosen
        spec.visibility = float(np.linalg.norm(gain[:, spec.column]))
    return sources


# ---------------------------------------------------------------------------
# Component generators
# ---------------------------------------------------------------------------


def _evoked_sensor_data(fwd, sources, schedule, times, sfreq, rng) -> np.ndarray:
    """Trial-locked cortical responses at the ground-truth vertices.

    Responses vary trial to trial in amplitude and latency.  Beyond being
    realistic, the jitter keeps the corresponding independent component from
    looking like a perfectly repeating stereotype, which is what an automatic
    ICA classifier flags as an artifact.
    """
    gain = fwd["sol"]["data"]
    n_times = len(times)
    source_ts = np.zeros((gain.shape[1], n_times))

    meta = schedule.metadata
    for spec in sources:
        # Divide out how well the array sees this location, so that
        # ``SourceSpec.amplitude`` means "relative size at the sensors" rather
        # than "relative dipole moment".  Without this a deep or unluckily
        # oriented source contributes almost nothing and never becomes the peak
        # of the source map at its own latency.
        visibility = spec.visibility or np.linalg.norm(gain[:, spec.column])
        scale = spec.amplitude / max(visibility, 1e-30)
        trace = np.zeros(n_times)
        for _, row in meta.iterrows():
            if spec.response_locked is not None:
                if row["response"] != spec.response_locked or not row["responded"]:
                    continue
                anchor = row["response_onset"]
            else:
                anchor = row["stim_onset"]
            gain_c = spec.condition_gain.get(row["condition"], 1.0)
            jitter = float(rng.normal(0.0, 0.015))
            weight = float(rng.lognormal(0.0, 0.3))
            lo = max(int((anchor - 0.2) * sfreq), 0)
            hi = min(int((anchor + 0.6) * sfreq), n_times)
            if hi <= lo:
                continue
            trace[lo:hi] += (
                scale
                * gain_c
                * weight
                * _bump(times[lo:hi] - anchor, spec.latency + jitter, spec.width)
            )
        source_ts[spec.column] += trace

    return _scale_to_rms(gain, source_ts, _TARGET["evoked"])


def _background_sensor_data(fwd, rng, n_times, sfreq) -> np.ndarray:
    """Ongoing cortical activity: 1/f everywhere plus posterior alpha."""
    gain = fwd["sol"]["data"]
    n_src = gain.shape[1]

    pink = _pink_noise(rng, n_src, n_times, exponent=1.2)
    background = _scale_to_rms(gain, pink, _TARGET["background"])

    # Alpha in the most posterior tenth of the source space.
    rr = np.vstack([s["rr"][s["vertno"]] for s in fwd["src"]])
    posterior = np.argsort(rr[:, 1])[: max(n_src // 10, 1)]
    times = np.arange(n_times) / sfreq
    envelope = 1.0 + 0.8 * np.sin(2 * np.pi * 0.13 * times + rng.uniform(0, 2 * np.pi))
    alpha_ts = np.zeros((n_src, n_times))
    for idx in posterior:
        phase = rng.uniform(0, 2 * np.pi)
        alpha_ts[idx] = np.sin(2 * np.pi * 10.2 * times + phase) * envelope
    alpha = _scale_to_rms(gain, alpha_ts, _TARGET["alpha"])

    return background + alpha


def _ocular(rng, sensor_pos, sensor_ori, head, times, sfreq):
    """Blinks and saccades: eyeball dipoles plus the eye-tracker traces."""
    n_times = len(times)
    duration = times[-1] if n_times else 0.0

    blink_times = np.sort(rng.uniform(0.5, max(duration - 0.5, 1.0), size=int(duration / 4.5)))
    sacc_times = np.sort(rng.uniform(0.5, max(duration - 0.5, 1.0), size=int(duration / 1.6)))

    blink_trace = np.zeros(n_times)
    for t0 in blink_times:
        width = rng.uniform(0.05, 0.09)
        blink_trace += np.exp(-0.5 * ((times - t0) / width) ** 2)

    sacc_trace = np.zeros(n_times)
    for t0 in sacc_times:
        idx = int(t0 * sfreq)
        if 0 <= idx < n_times:
            sacc_trace[idx] = rng.choice([-1.0, 1.0]) * rng.uniform(0.4, 1.0)
    kernel = np.hanning(max(int(0.04 * sfreq), 3))
    sacc_trace = np.convolve(sacc_trace, kernel / kernel.sum(), mode="same")

    # Eyeballs sit anterior and inferior to the head centre.
    cx, cy, cz = head.center
    eye_offset = np.array([0.032, head.scalp_axes[1] * 0.80, -head.scalp_axes[2] * 0.35])
    topo = np.zeros(len(sensor_pos))
    for sign in (-1.0, 1.0):
        eye = np.array([cx + sign * eye_offset[0], cy + eye_offset[1], cz + eye_offset[2]])
        topo += _dipole_field(sensor_pos, sensor_ori, eye, np.array([0.0, 0.0, 1.0]))
    sacc_topo = np.zeros(len(sensor_pos))
    for sign in (-1.0, 1.0):
        eye = np.array([cx + sign * eye_offset[0], cy + eye_offset[1], cz + eye_offset[2]])
        sacc_topo += _dipole_field(sensor_pos, sensor_ori, eye, np.array([1.0, 0.0, 0.0]))

    sensor = np.outer(topo, blink_trace) + 0.5 * np.outer(sacc_topo, sacc_trace)
    rms = np.sqrt(np.mean(sensor**2))
    if rms > 0:
        sensor *= _TARGET["ocular"] / rms

    return sensor, blink_trace, sacc_trace, blink_times


def _cardiac(rng, sensor_pos, sensor_ori, head, times, sfreq):
    """A heartbeat from a distant dipole below and in front of the head."""
    n_times = len(times)
    duration = times[-1] if n_times else 0.0
    rate = rng.uniform(0.95, 1.25)  # Hz
    beats = np.arange(0.4, duration, 1.0 / rate)
    beats = beats + rng.normal(0.0, 0.02, size=beats.shape)  # heart-rate jitter

    trace = np.zeros(n_times)
    for t0 in beats:
        # Crude QRS: a sharp positive spike flanked by slower opposite lobes.
        trace += _bump(times - t0, 0.0, 0.012) - 0.25 * _bump(times - t0, 0.06, 0.045)

    heart = head.center + np.array([-0.02, 0.03, -0.30])
    topo = _dipole_field(sensor_pos, sensor_ori, heart, np.array([0.4, 0.6, -0.7]))
    sensor = np.outer(topo, trace)
    rms = np.sqrt(np.mean(sensor**2))
    if rms > 0:
        sensor *= _TARGET["cardiac"] / rms
    return sensor, trace


def _external_interference(rng, sensor_pos, sensor_ori, times, sfreq, line_freq):
    """Environmental field: a uniform term plus a first-order gradient.

    The uniform term is exactly what a first-order homogeneous field
    correction removes, and the gradient is what pushes ``_hfc_order`` above 1;
    both live in the external Maxwell basis.
    """
    n_times = len(times)

    drift = _pink_noise(rng, 3, n_times, exponent=1.6)
    drift /= np.abs(drift).max(axis=1, keepdims=True)
    line = np.vstack(
        [
            np.sin(2 * np.pi * line_freq * times + rng.uniform(0, 2 * np.pi))
            + 0.3 * np.sin(2 * np.pi * 2 * line_freq * times + rng.uniform(0, 2 * np.pi))
            for _ in range(3)
        ]
    )
    uniform_ts = drift + 0.35 * line
    uniform = sensor_ori @ uniform_ts
    uniform *= _TARGET["uniform"] / np.sqrt(np.mean(uniform**2))

    grad_ts = _pink_noise(rng, 9, n_times, exponent=1.6).reshape(3, 3, n_times)
    grad_ts = 0.5 * (grad_ts + grad_ts.transpose(1, 0, 2))
    trace = np.einsum("iit->t", grad_ts) / 3.0
    grad_ts -= np.eye(3)[:, :, None] * trace  # keep it divergence free
    centre = sensor_pos.mean(axis=0)
    rel = sensor_pos - centre
    gradient = np.einsum("ci,ijt,cj->ct", rel, grad_ts, sensor_ori)
    gradient *= _TARGET["gradient"] / np.sqrt(np.mean(gradient**2))

    return uniform + gradient


def _plant_defects(rng, data, picks, n_times, sfreq, seed_offset=0):
    """Add the bad channels and bad segments the detection steps look for.

    Returns the channel indices (into ``picks``) and time windows that were
    corrupted, so they can be recorded as ground truth.
    """
    local = np.random.default_rng(seed_offset + 31337)
    n_meg = len(picks)

    noisy = local.choice(n_meg, size=2, replace=False)
    flat = int(local.choice(np.setdiff1d(np.arange(n_meg), noisy)))
    for ch in noisy:
        data[picks[ch]] += local.standard_normal(n_times) * 8e-12

    segments = []
    for k in range(2):
        start = int((0.30 + 0.35 * k) * n_times)
        stop = min(start + int(0.8 * sfreq), n_times)
        burst = local.standard_normal((n_meg, stop - start)) * 6e-12
        data[picks, start:stop] += burst
        segments.append((start / sfreq, stop / sfreq))

    # A dead sensor last, so the bursts above do not put signal back into it:
    # it reports its own noise floor and nothing else, two orders of magnitude
    # below every other channel.
    data[picks[flat]] = local.standard_normal(n_times) * 2e-15

    return dict(
        noisy_channels=[int(n) for n in noisy],
        flat_channel=flat,
        bad_segments=segments,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def simulate_task(
    info,
    head,
    schedule,
    subjects_dir,
    fs_subject,
    *,
    seed: int = 0,
    sources: list[SourceSpec] | None = None,
    forward=None,
):
    """Simulate the task recording.

    Parameters
    ----------
    info : mne.Info
        Sensor definition from :func:`custom.synthetic.sensors.build_info`,
        including the trigger and eye-tracking channels.
    head : HeadModel
        Anatomy the forward model is computed on.
    schedule : TrialSchedule
        Trial timing and metadata.
    subjects_dir, fs_subject : path-like, str
        Location of the written FreeSurfer subject.
    seed : int
        Master seed for every stochastic component.
    sources : list of SourceSpec, optional
        Ground-truth dipoles.  Defaults to :func:`default_sources`.
    forward : mne.Forward, optional
        Reuse a previously computed forward (used when generating a cohort).

    Returns
    -------
    raw : mne.io.Raw
    ground_truth : dict
        Serialisable description of the sources and planted defects.
    """
    import mne
    from .events import trigger_annotations, trigger_waveforms
    from .sensors import EYE_CHANNELS, TRIGGER_CHANNELS

    rng = np.random.default_rng(seed + 1000)
    sfreq = info["sfreq"]
    n_times = int(round(schedule.duration * sfreq))
    times = np.arange(n_times) / sfreq

    if forward is None:
        forward, _ = build_forward(info, head, subjects_dir, fs_subject)
    sources = _resolve_sources(forward, sources or default_sources(), head)

    picks, sensor_pos, sensor_ori = _sensor_geometry(info)

    brain = _evoked_sensor_data(forward, sources, schedule, times, sfreq, rng)
    brain += _background_sensor_data(forward, rng, n_times, sfreq)
    ocular, blink_trace, sacc_trace, blink_times = _ocular(
        rng, sensor_pos, sensor_ori, head, times, sfreq
    )
    cardiac, cardiac_trace = _cardiac(rng, sensor_pos, sensor_ori, head, times, sfreq)
    interference = _external_interference(
        rng, sensor_pos, sensor_ori, times, sfreq, info["line_freq"]
    )
    noise = rng.standard_normal((len(picks), n_times)) * _SENSOR_NOISE_DENSITY * np.sqrt(
        sfreq / 2.0
    )

    data = np.zeros((len(info["ch_names"]), n_times))
    data[picks] = brain + ocular + cardiac + interference + noise
    defects = _plant_defects(rng, data, picks, n_times, sfreq, seed_offset=seed)

    # -- auxiliary channels -------------------------------------------------
    aux = _eye_channel_data(rng, blink_trace, sacc_trace, times, sfreq)
    for name, values in aux.items():
        data[info["ch_names"].index(name)] = values

    trig = trigger_waveforms(schedule, sfreq, n_times)
    for i, name in enumerate(TRIGGER_CHANNELS):
        if name in info["ch_names"]:
            data[info["ch_names"].index(name)] = trig[i]

    raw = mne.io.RawArray(data, info.copy(), verbose="error")
    raw.set_annotations(trigger_annotations(trig, sfreq))

    from mne.transforms import apply_trans, invert_transform

    head_mri_t = invert_transform(head.mri_head_t)
    meg_names = [info["ch_names"][picks[i]] for i in range(len(picks))]
    ground_truth = dict(
        sources=[
            dict(
                name=s.name,
                hemi=s.hemi,
                vertno=s.vertno,
                # Both frames, because it is genuinely easy to mix them up:
                # a forward solution's source space is in head coordinates,
                # while one read from bem/*-src.fif is in MRI surface RAS.
                position_head_m=list(s.position),
                position_mri_m=[
                    float(v) for v in apply_trans(head_mri_t, np.asarray(s.position))
                ],
                normal_head=list(s.normal),
                radiality=s.radiality,
                visibility=s.visibility,
                latency_s=s.latency,
                width_s=s.width,
                amplitude=s.amplitude,
                condition_gain=s.condition_gain,
            )
            for s in sources
        ],
        noisy_channels=[meg_names[i] for i in defects["noisy_channels"]],
        flat_channel=meg_names[defects["flat_channel"]],
        bad_segments_s=defects["bad_segments"],
        n_trials=int(schedule.n_trials),
        sfreq=float(sfreq),
        duration_s=float(schedule.duration),
        eye_channels=list(EYE_CHANNELS),
    )
    return raw, ground_truth, forward


def _eye_channel_data(rng, blink_trace, sacc_trace, times, sfreq) -> dict[str, np.ndarray]:
    """Eye-tracker derived traces, in the units ``format_bids`` would produce."""
    n_times = len(times)
    missing = (blink_trace > 0.35).astype(float)

    gaze_x = np.cumsum(sacc_trace) * 0.02
    gaze_x -= gaze_x.mean()
    gaze_y = _pink_noise(rng, 1, n_times, exponent=1.5)[0]
    gaze_y *= 0.02 / max(np.std(gaze_y), 1e-12)
    pupil = 1.0 + 0.15 * _pink_noise(rng, 1, n_times, exponent=1.5)[0]
    pupil = pupil / max(np.std(pupil), 1e-12) * 0.1 + 1.0
    pupil = pupil * (1.0 - missing)

    head_drift = _pink_noise(rng, 3, n_times, exponent=1.8)
    head_drift /= np.abs(head_drift).max(axis=1, keepdims=True)

    return {
        # format_bids scales its NMF eye features by 1e-5; match that so any
        # threshold tuned on real data behaves the same here.
        "eye_nmf1": blink_trace * 1e-5,
        "eye_nmf2": np.abs(sacc_trace) * 1e-5,
        "eye_nmf3": missing * 1e-5,
        "xpos_right": gaze_x,
        "ypos_right": gaze_y,
        "pupil_right": pupil,
        "x_head": head_drift[0] * 0.005,
        "y_head": head_drift[1] * 0.005,
        "distance": 0.9 + head_drift[2] * 0.01,
    }


def simulate_empty_room(info, head, duration: float, *, seed: int = 0):
    """Simulate an empty-room recording with the same array and environment.

    Contains the environmental interference and sensor noise of the task
    recording but no brain, ocular or cardiac activity, which is what makes it
    a valid noise covariance for ``noise_cov = "emptyroom"``.
    """
    import mne

    rng = np.random.default_rng(seed + 2000)
    sfreq = info["sfreq"]
    n_times = int(round(duration * sfreq))
    times = np.arange(n_times) / sfreq

    picks, sensor_pos, sensor_ori = _sensor_geometry(info)
    interference = _external_interference(
        rng, sensor_pos, sensor_ori, times, sfreq, info["line_freq"]
    )
    noise = rng.standard_normal((len(picks), n_times)) * _SENSOR_NOISE_DENSITY * np.sqrt(
        sfreq / 2.0
    )

    data = np.zeros((len(info["ch_names"]), n_times))
    data[picks] = interference + noise
    _plant_defects(rng, data, picks, n_times, sfreq, seed_offset=seed)

    return mne.io.RawArray(data, info.copy(), verbose="error")
