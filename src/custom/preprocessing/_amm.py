"""Adaptive multipole models (AMM) for OPM-MEG interference suppression.

AMM (Tierney et al., 2024) models the field an OPM array measures as the sum
of two sets of prolate spheroidal harmonics, both defined on a spheroid fitted
to the array: *internal* harmonics, the fields of sources inside the spheroid
(the brain), and *external* harmonics, the fields of sources outside the array
(interference).  Data are projected orthogonally onto the internal space after
removing the external space (Tierney et al., 2024, eq. 18).  An optional
temporal extension then removes the time courses that the internal estimate
shares with the intermediate space (the part of the data neither model
explains; eqs. 21-22), as tSSS does (Taulu & Simola, 2006).

The steps follow SPM's reference implementation by Tim Tierney
(``spm_opm_amm.m``, ``spm_ipharm.m``, ``spm_epharm.m`` and ``spm_slm.m``;
SPM commit 54dc957):

1. :func:`fit_spheroid`: least-squares fits of an axis-aligned spheroid with
   each coordinate axis as its long axis; the prolate fit with the smallest
   spread of the fitted surface equation is kept, the coordinates are rotated
   so that the long axis is the second one, and all radii are then shrunk in
   steps of 0.5 % of the largest radius until every sensor lies outside.
2. :func:`internal_basis` / :func:`external_basis`: the field component along
   each sensor's orientation of the potentials ``Q_l^m(mu/c) S_l^m`` and
   ``P_l^m(mu/c) S_l^m`` (eqs. 6-9; ``P``/``Q``: associated Legendre
   functions of the first and second kind, each scaled by its value on the
   spheroid; ``S``: real spherical harmonics of the spheroidal angles), for
   ``l = 1..order``.  As in SPM, every basis vector is centred and scaled
   across channels.  The paper's equations do not centre; centring removes
   the channel-mean direction from both spaces, so the output has zero mean
   across channels.
3. :class:`AmmModel`: ``P_out`` projects onto the external space and
   ``P_in`` onto ``(I - P_out) H_in``, the internal space after removing the
   external space; ``P_in`` is what is applied to the data.
4. :func:`temporal_extension`: per window, the canonical correlations between
   the time courses of the internal estimate and of the intermediate space;
   intermediate time courses correlated above ``corr_lim`` are projected out
   of the internal estimate (the paper used ``corr_lim = 0.98``; its eq. 23
   bounds the correlation brain signal alone can reach).

Attribution and licence
-----------------------
This is an independent implementation (BSD-3-Clause, as the rest of this
package), written from the mathematics above and not translated from SPM's
code, which is GPL-2.0-or-later.  It is checked against SPM itself:
``tests/test_amm.py`` runs the unmodified SPM functions in Octave and
compares bases, spheroid, projector and filtered data.  The credit for the
method and for the reference implementation belongs to their authors:

* The method, the choice of prolate spheroidal harmonics and the orthogonal
  projection: Tierney et al. (2024), and Tim Tierney's SPM implementation.
* The spheroid fit: SPM's ``spheroid_fit`` (Tim Tierney), the equal-radii
  case of the linear least-squares parameterisation of Yury Petrov's
  ``ellipsoid_fit`` (MATLAB File Exchange 24693).
* The series for the Legendre functions outside the focal segment
  (eqs. 7-8): Abramowitz & Stegun (1965) and Winch (1967), as cited by
  Tierney et al. (2024).
* The final-window convention of the temporal extension: MNE-Python's tSSS
  (``mne.preprocessing.maxwell_filter``).

Differences from SPM, all on purpose:

* The radial derivative of the internal harmonics includes the derivative of
  the hypergeometric factor of ``Q_l^m``.  In ``spm_ipharm.m`` the loop that
  sums that term starts with a zero term, so it never runs and the term is
  dropped; the basis vectors are then not exactly the fields of the internal
  potentials.
* The temporal extension takes the time courses from the coordinates of the
  data in the internal and intermediate spaces, so their dimension is at most
  that of each space.  SPM takes them from the channel-space data, where
  rounding can add a spurious dimension when the window is short.
* The temporal extension processes every sample, and a final window shorter
  than ``window`` is merged into the previous one.  SPM uses a short final
  window and leaves the last sample unprocessed.
* SPM's ``skip`` option (leave the first seconds unprocessed) is not
  implemented.

Sensors are treated as point magnetometers at the channel position, measuring
along the channel's z axis (``info["chs"][i]["loc"][9:12]``), as in SPM.

References
----------
Tierney, T. M., Seedat, Z., St Pier, K., Mellor, S., & Barnes, G. R. (2024).
Adaptive multipole models of optically pumped magnetometer data. Human Brain
Mapping, 45(4), e26596. https://doi.org/10.1002/hbm.26596

SPM: https://github.com/spm/spm (spm_opm_amm.m and the functions it calls).

Petrov, Y. Ellipsoid fit. MATLAB Central File Exchange 24693.
https://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit

Abramowitz, M., & Stegun, I. A. (1965). Handbook of mathematical functions.
Dover.

Winch, D. E. (1967). An application of oblate spheroidal harmonic functions
to the determination of geomagnetic potential. Journal of Geomagnetism and
Geoelectricity, 19(1), 49-61.

Taulu, S., & Simola, J. (2006). Spatiotemporal signal space separation method
for rejecting nearby interference in MEG measurements. Physics in Medicine and
Biology, 51(7), 1759-1768. https://doi.org/10.1088/0031-9155/51/7/008

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
from scipy.linalg import null_space
from scipy.special import gammaln, hyp2f1, lpmv

# SPM's rotations that bring the long axis of the spheroid to the second axis.
_ROTATIONS = {
    0: np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    1: np.eye(3),
    2: np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
}
_SHRINK_STEP = 0.005  # fraction of the largest radius


# ---------------------------------------------------------------------------
# 1. The spheroid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spheroid:
    """A prolate spheroid in a rotated frame whose second axis is the long axis.

    ``rotation`` maps original coordinates to that frame (``v @ rotation.T``);
    ``center`` and ``radii`` are in it, so ``radii = (b, a, b)`` with ``a``
    the semi-major and ``b`` the semi-minor axis.
    """

    rotation: np.ndarray
    center: np.ndarray
    radii: np.ndarray
    long_axis: int  # the long axis in the original coordinates

    @property
    def a(self):
        return float(np.max(self.radii))

    @property
    def b(self):
        return float(np.min(self.radii))


def _fit_axis(v, axis):
    """Least-squares axis-aligned spheroid whose axis ``axis`` differs from the others.

    The model is linear in its five parameters: ``|v|^2 = beta_0 (sum of the
    other two squared coordinates - 2 v_axis^2) + 2 beta_{1..3} . v + beta_4``.
    Returns ``(center, radii, spread)``, with ``spread`` the standard
    deviation over sensors of ``sum(((v - center) / radii)^2)``.
    """
    sq = v**2
    others = [i for i in range(3) if i != axis]
    shape = sq[:, others].sum(axis=1) - 2 * sq[:, axis]
    design = np.column_stack([shape, 2 * v, np.ones(len(v))])
    beta = np.linalg.pinv(design) @ sq.sum(axis=1)
    # quadric sum_j A_j v_j^2 + 2 sum_j L_j v_j + K = 0
    quad = np.full(3, beta[0] - 1.0)
    quad[axis] = -2 * beta[0] - 1.0
    lin, const = beta[1:4], beta[4]
    center = -lin / quad
    const_c = const - np.sum(lin**2 / quad)  # the constant once centred
    scaled = -quad / const_c
    radii = np.sign(scaled) / np.sqrt(np.abs(scaled))
    spread = np.std(np.sum(((v - center) / radii) ** 2, axis=1), ddof=1)
    return center, radii, spread


def fit_spheroid(positions):
    """The prolate spheroid AMM builds its harmonics on (see the module docstring).

    ``positions``: ``(n_sensors, 3)``, in any unit.  Raises ``ValueError``
    when no axis gives a prolate fit, or the sensors do not enclose a spheroid.
    """
    v = np.asarray(positions, float)
    fits = [_fit_axis(v, axis) for axis in range(3)]
    # prolate: the distinct axis is longer than the two equal ones
    prolate = [
        axis
        for axis, (_, radii, _) in enumerate(fits)
        if radii[axis] > radii[1 if axis != 1 else 0]
    ]
    if not prolate:
        raise ValueError("the sensor array is not well modelled by a prolate spheroid")
    long_axis = min(prolate, key=lambda axis: fits[axis][2])
    rotation = _ROTATIONS[long_axis]
    vr = v @ rotation.T
    center, radii, _ = _fit_axis(vr, 1)

    d2 = (vr - center) ** 2
    step = _SHRINK_STEP * np.max(radii)
    for n in range(int(np.ceil(1 / _SHRINK_STEP)) + 1):
        shrunk = radii - n * step
        if np.any(shrunk <= 0):
            break
        if np.all(np.sum(d2 / shrunk**2, axis=1) > 1):
            return Spheroid(rotation, center, shrunk, long_axis)
    raise ValueError("no spheroid inside the sensor array leaves every sensor outside")


# ---------------------------------------------------------------------------
# 2. Prolate spheroidal harmonics
# ---------------------------------------------------------------------------


def prolate_coordinates(v, c):
    """``(mu, nu, phi)`` of points ``v`` (long axis second) for focal distance ``c``.

    ``mu`` is the semi-major axis of the confocal spheroid through the point,
    ``nu`` the angle from the long axis and ``phi`` the azimuth in the plane
    of the first and third axes: ``v = (s sin nu cos phi, mu cos nu,
    s sin nu sin phi)`` with ``s = sqrt(mu^2 - c^2)``.
    """
    t = np.sum(v**2, axis=1) + c**2
    mu = np.sqrt((t + np.sqrt(np.maximum(t**2 - 4 * v[:, 1] ** 2 * c**2, 0))) / 2)
    nu = np.arccos(np.clip(v[:, 1] / mu, -1, 1))
    phi = np.arctan2(v[:, 2], v[:, 0])
    return mu, nu, phi


def _legendre_p_series(w, deg, m):
    """``q(w) = sum_k C_k w^k`` and ``q'(w)``, with ``P_l^m(xi) ∝ (xi^2-1)^{m/2} xi^{l-m} q(1/xi^2)``.

    ``C_k = (-1)^k (2l-2k)! / (k! (l-k)! (l-2k-m)!)``, scaled so ``C_0 = 1``.
    """
    k = np.arange((deg - m) // 2 + 1)
    log_c = (
        gammaln(2 * deg - 2 * k + 1)
        - gammaln(k + 1)
        - gammaln(deg - k + 1)
        - gammaln(deg - 2 * k - m + 1)
    )
    coef = (-1.0) ** k * np.exp(log_c - log_c[0])
    powers = np.asarray(w)[..., None] ** k
    value = powers @ coef
    deriv = (powers[..., :-1] @ (coef[1:] * k[1:])) if len(k) > 1 else 0 * value
    return value, deriv


def _hypergeometric_ratio(mu, a, c, deg, m):
    """``h = F(c^2/mu^2) / F(c^2/a^2)`` and ``dh/dmu`` for ``Q_l^m``.

    ``Q_l^m(xi) ∝ (xi^2-1)^{m/2} xi^{-l-m-1} F(1/xi^2)`` with
    ``F = 2F1((l+m+1)/2, (l+m+2)/2; l+3/2; .)``.
    """
    alpha, beta, gamma = (deg + m + 1) / 2, (deg + m + 2) / 2, deg + 1.5
    w = (c / mu) ** 2
    ref = hyp2f1(alpha, beta, gamma, (c / a) ** 2)
    h = hyp2f1(alpha, beta, gamma, w) / ref
    dfdw = alpha * beta / gamma * hyp2f1(alpha + 1, beta + 1, gamma + 1, w)
    return h, dfdw * (-2 * c**2 / mu**3) / ref


def _radial(kind, mu, a, c, deg, m):
    """Radial function ``R(mu) / R(a)`` and its derivative.

    ``kind="internal"``: ``Q_l^m(mu/c)``, decaying away from the spheroid;
    ``kind="external"``: ``P_l^m(mu/c)``, regular inside it.  Both are scaled
    to 1 on the spheroid ``mu = a``.  Written as ``g(mu) h(mu)`` with
    ``g = ((mu^2-c^2)/(a^2-c^2))^{m/2} (mu/a)^p`` and ``h`` the series ratio.
    """
    p = (deg - m) if kind == "external" else (-deg - m - 1)
    g = ((mu**2 - c**2) / (a**2 - c**2)) ** (m / 2) * (mu / a) ** p
    dg = g * (m * mu / (mu**2 - c**2) + p / mu)
    if kind == "external":
        q, dq = _legendre_p_series((c / mu) ** 2, deg, m)
        q_ref, _ = _legendre_p_series((c / a) ** 2, deg, m)
        h, dh = q / q_ref, dq * (-2 * c**2 / mu**3) / q_ref
    else:
        h, dh = _hypergeometric_ratio(mu, a, c, deg, m)
    return g * h, dg * h + g * dh


def _angular(nu, phi, deg, m):
    """Real spherical harmonic of ``(nu, phi)`` and its two angular derivatives.

    ``m > 0``: ``P_l^m(cos nu) cos(m phi)``; ``m < 0``: ``P_l^|m| sin(|m| phi)``.
    The normalisation does not matter: every basis vector is rescaled.
    """
    am = abs(m)
    x = np.cos(nu)
    p = lpmv(am, deg, x)  # with the Condon-Shortley phase
    if am == 0:
        dp = lpmv(1, deg, x)
    else:
        dp = -0.5 * (
            (deg + am) * (deg - am + 1) * lpmv(am - 1, deg, x) - lpmv(am + 1, deg, x)
        )
    if m > 0:
        trig, dtrig = np.cos(am * phi), -am * np.sin(am * phi)
    elif m < 0:
        trig, dtrig = np.sin(am * phi), am * np.cos(am * phi)
    else:
        trig, dtrig = np.ones_like(phi), np.zeros_like(phi)
    return p * trig, p * dtrig, dp * trig


def _harmonic_fields(v, n, a, b, order, kind):
    """Field along ``n`` of every potential ``R_l^m(mu) S_l^m(nu, phi)``, uncentred.

    ``v``/``n``: positions relative to the spheroid centre and orientations,
    in the frame whose second axis is the long axis.  Columns are ordered
    ``l = 1..order``, ``m = -l..l``.
    """
    c = np.sqrt(a**2 - b**2)
    mu, nu, phi = prolate_coordinates(v, c)
    s = np.sqrt(mu**2 - c**2)
    d = np.sqrt(mu**2 - c**2 * np.cos(nu) ** 2)
    sin_nu, cos_nu = np.sin(nu), np.cos(nu)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    # orientation . unit vectors, divided by the scale factors
    n_mu = (
        mu * sin_nu * (cos_phi * n[:, 0] + sin_phi * n[:, 2]) + s * cos_nu * n[:, 1]
    ) / d
    n_nu = (
        s * cos_nu * (cos_phi * n[:, 0] + sin_phi * n[:, 2]) - mu * sin_nu * n[:, 1]
    ) / d
    n_phi = cos_phi * n[:, 2] - sin_phi * n[:, 0]
    h_mu, h_nu, h_phi = d / s, d, s * sin_nu
    with np.errstate(divide="ignore", invalid="ignore"):
        w_phi = np.where(h_phi > 0, n_phi / h_phi, 0.0)  # on the axis: no phi term
    columns = []
    for deg in range(1, order + 1):
        for m in range(-deg, deg + 1):
            r, dr = _radial(kind, mu, a, c, deg, abs(m))
            ang, dang_phi, dang_nu = _angular(nu, phi, deg, m)
            columns.append(
                n_mu / h_mu * dr * ang
                + n_nu / h_nu * r * dang_nu
                + w_phi * r * dang_phi
            )
    return np.column_stack(columns)


def _normalise(fields):
    """Centre and scale every basis vector across channels, as SPM does."""
    fields = fields - fields.mean(axis=0)
    return fields / fields.std(axis=0, ddof=1)


def internal_basis(v, n, a, b, order):
    """``(n_sensors, order^2 + 2 order)`` internal prolate spheroidal harmonics."""
    return _normalise(_harmonic_fields(v, n, a, b, order, "internal"))


def external_basis(v, n, a, b, order):
    """``(n_sensors, order^2 + 2 order)`` external prolate spheroidal harmonics."""
    return _normalise(_harmonic_fields(v, n, a, b, order, "external"))


# ---------------------------------------------------------------------------
# 3. The projectors
# ---------------------------------------------------------------------------


def _range_basis(matrix):
    """Orthonormal basis of the column space, with MATLAB's ``orth``/``pinv`` tolerance."""
    if matrix.size == 0:
        return np.zeros((matrix.shape[0], 0))
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    tol = max(matrix.shape) * np.finfo(float).eps * (s[0] if len(s) else 0.0)
    return u[:, s > tol]


@dataclass(frozen=True)
class AmmModel:
    """The AMM subspaces for one channel set (see :func:`amm_model`).

    ``u_in``, ``u_out`` and ``u_res`` are orthonormal bases of the internal
    space (after removing the external space), of the external space and of
    the rest, which together span the channel space.
    """

    ch_names: tuple
    spheroid: Spheroid
    li: int
    le: int
    internal: np.ndarray  # (n_channels, n_internal), after any rank reduction
    external: np.ndarray  # (n_channels, n_external)
    u_in: np.ndarray
    u_out: np.ndarray
    u_res: np.ndarray

    @property
    def p_in(self):
        """Projector onto the internal space: what AMM applies to the data."""
        return self.u_in @ self.u_in.T

    @property
    def p_out(self):
        """Projector onto the external space."""
        return self.u_out @ self.u_out.T

    @property
    def rank(self):
        """The dimension of the space AMM keeps."""
        return self.u_in.shape[1]


def sensor_geometry(info, picks="meg"):
    """``(ch_names, positions, orientations)`` of the good channels in ``picks``."""
    picks = mne.pick_types(info, meg=True, exclude="bads") if picks == "meg" else picks
    chs = [info["chs"][i] for i in picks]
    pos = np.array([ch["loc"][:3] for ch in chs])
    ori = np.array([ch["loc"][9:12] for ch in chs])
    norms = np.linalg.norm(ori, axis=1)
    if not len(chs) or not np.all(np.isfinite(pos)) or np.any(norms == 0):
        raise ValueError("AMM needs the position and orientation of every channel")
    return tuple(info["ch_names"][i] for i in picks), pos, ori / norms[:, None]


def amm_model(info, li=9, le=2, reducerank=0.0, picks="meg"):
    """Fit the spheroid and build the AMM projectors for the good MEG channels of ``info``.

    ``li``/``le``: internal/external harmonic order (SPM's defaults 9 and 2).
    ``reducerank``: if in (0, 1), keep the internal basis directions whose
    cumulative share of the singular values stays below it (SPM's
    ``reducerank``); 0 keeps the whole basis.
    """
    ch_names, pos, ori = sensor_geometry(info, picks)
    spheroid = fit_spheroid(pos)
    v = pos @ spheroid.rotation.T - spheroid.center
    n = ori @ spheroid.rotation.T
    external = external_basis(v, n, spheroid.a, spheroid.b, le)
    internal = internal_basis(v, n, spheroid.a, spheroid.b, li)
    if reducerank:
        u, s, _ = np.linalg.svd(internal, full_matrices=False)
        internal = u[:, np.cumsum(s) / np.sum(s) < reducerank]
    u_out = _range_basis(external)
    u_in = _range_basis(internal - u_out @ (u_out.T @ internal))
    return AmmModel(
        ch_names=ch_names,
        spheroid=spheroid,
        li=li,
        le=le,
        internal=internal,
        external=external,
        u_in=u_in,
        u_out=u_out,
        u_res=null_space(np.column_stack([u_in, u_out]).T),
    )


def amm_projs(model):
    """SSP projectors that remove everything outside AMM's internal space.

    Applying them is applying ``P_in`` (one projector per removed direction),
    and keeping them in ``info`` makes MNE apply the same projection to the
    forward model, as SPM does by updating the sensors' ``tra``.
    """
    removed = np.column_stack([model.u_out, model.u_res])
    n = len(model.ch_names)
    return [
        mne.Projection(
            active=False,
            desc=f"AMM li={model.li} le={model.le} #{i + 1}",
            kind=1,  # FIFFV_PROJ_ITEM_FIELD
            explained_var=None,
            data=dict(
                nrow=1,
                ncol=n,
                row_names=None,
                col_names=list(model.ch_names),
                data=removed[:, i][None],
            ),
        )
        for i in range(removed.shape[1])
    ]


# ---------------------------------------------------------------------------
# 4. Temporal extension and application to Raw
# ---------------------------------------------------------------------------


def temporal_extension(data, model, corr_lim):
    """``P_in data`` minus its time courses shared with the residual (one window).

    ``data``: ``(n_channels, n_times)``.  The residual is what neither the
    internal nor the external model explains; its time courses whose
    canonical correlation with those of ``P_in data`` exceeds ``corr_lim``
    are projected out.  The time courses are taken from the coordinates of
    the data in each space (``u.T @ data``; see the module docstring).
    """
    coef = model.u_in.T @ data
    if corr_lim < 1:
        t_in = _range_basis(coef.T)
        t_res = _range_basis((model.u_res.T @ data).T)
        if t_in.shape[1] and t_res.shape[1]:
            _, corr, zt = np.linalg.svd(t_in.T @ t_res)
            shared = (t_res @ zt.T)[:, : int(np.sum(corr > corr_lim))]
            coef = coef - (coef @ shared) @ shared.T
    return model.u_in @ coef


def window_bounds(n_times, n_window):
    """``[(start, stop)]`` windows of ``n_window`` samples; a shorter rest joins the last."""
    starts = list(range(0, n_times, max(n_window, 1)))
    if len(starts) > 1 and n_times - starts[-1] < n_window:
        starts.pop()
    return list(zip(starts, starts[1:] + [n_times]))


def apply_amm(raw, model=None, li=9, le=2, corr_lim=1.0, window=10.0, reducerank=0.0):
    """Apply AMM to ``raw`` in place and return ``(raw, model)``.

    The spatial projection is added to ``raw.info["projs"]`` as SSP
    projectors (see :func:`amm_projs`) and applied; with ``corr_lim < 1`` the
    temporal extension then runs on windows of ``window`` seconds.  Pass the
    ``model`` of another recording (e.g. the task) to apply its projection,
    as for an empty-room recording; otherwise it is built from ``raw.info``.
    ``raw`` must be preloaded.
    """
    if model is None:
        model = amm_model(raw.info, li=li, le=le, reducerank=reducerank)
    missing = [ch for ch in model.ch_names if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"the recording lacks AMM channels {missing[:5]}")
    picks = [raw.ch_names.index(ch) for ch in model.ch_names]
    if corr_lim < 1:
        # the temporal extension needs the data before projection
        data = raw.get_data(picks)
        n_window = int(round(window * raw.info["sfreq"]))
        for start, stop in window_bounds(data.shape[1], n_window):
            data[:, start:stop] = temporal_extension(
                data[:, start:stop], model, corr_lim
            )
    raw.add_proj(amm_projs(model))
    raw.apply_proj()
    if corr_lim < 1:
        raw._data[picks] = data
    return raw, model
