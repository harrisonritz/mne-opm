"""Modular auxiliary preprocessing for OPM-MEG data.

Provided analyses (CLI --analysis):
    bad_segments   -> detect & annotate bad raw segments
    bad_channels   -> statistical detection of bad channels
    manual_channel -> interactive marking of bad channels
    apply_hfc      -> apply homogeneous field correction (HFC) projections
    bad_epochs     -> drop bad epochs post-epoching
    manual_ica     -> interactive ICA component review (+ optional ref ICA)
    regress_ref    -> regress out reference channel signals

Internal normalized keys remove underscores (e.g. bad_segments -> badsegments).

Outputs are written back into the BIDS structure using mne-bids utilities,
re-using existing derivative locations produced by mne-bids-pipeline.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt

import numpy as np

import argparse
import os
from types import SimpleNamespace
from typing import Dict, Any

import mne
import mne_bids
import pandas as pd
from scipy.linalg import qr
from scipy import stats
# from pygam import LinearGAM, s, te, l, terms


from mne._fiff.pick import _picks_to_idx


try:  # optional dependency for nicer Qt browser; skip if unavailable
    import mne_qt_browser

    _HAVE_QT_BROWSER = True
except Exception:  # pragma: no cover
    _HAVE_QT_BROWSER = False

from mne_bids_pipeline._config_import import (
    _update_config_from_path,
    _update_with_user_config,
    _import_config,
)

from osl_ephys.preprocessing.osl_wrappers import (
    bad_segments as osl_bad_segments,
    bad_channels as osl_bad_channels,
    drop_bad_epochs as osl_drop_bad_epochs,
    gesd as osl_gesd,
)


# %% GLOBAL VARIABLES

try:  # ensure a GUI backend for interactive steps
    mne.viz.set_browser_backend("qt")
except Exception:  # pragma: no cover - fallback if qt not available
    print("error: (qt not available)")
    pass

SEGMENT_LEN_SEC = 1.0  # length for segment-based detection


# --------------------------------------------------------------------------------------
# Utility / Loading
# --------------------------------------------------------------------------------------


# %% functions
def load_data(cfg: SimpleNamespace, analysis: str) -> Dict[str, Any]:
    """Load only the data needed for the selected analysis.

    Returns a dict with keys:
      bad_segments / bad_channels / manual_channel -> {task[, noise]}
      bad_epochs -> {task}
      manual_ica -> {task, manualica}
    """
    print(f"\n[load_data] analysis={analysis}")
    out: Dict[str, Any] = {}

    if analysis in {
        "badsegments",
        "badchannels",
        "manualchannel",
        "regressref",
        "applyhfc",
    }:
        if getattr(cfg, "_skip_on_deriv", False):
            deriv_path = os.path.join(cfg.deriv_root, f"sub-{cfg.subjects[0]}")
            if os.path.exists(deriv_path):
                print(f"\n[load_data] derivatives exist at {deriv_path}; exiting.")
                raise SystemExit(0)
        tasks = [cfg.task]
        if getattr(cfg, "process_empty_room", False):
            tasks.insert(0, "noise")
        for task in tasks:
            bids_path = mne_bids.find_matching_paths(
                root=cfg.bids_root,
                subjects=cfg.subjects,
                tasks=task,
                sessions=cfg.sessions,
                datatypes="meg",
                ignore_nosub=True,
                extensions=".fif",
            )[0]
            out[task] = mne_bids.read_raw_bids(
                bids_path, extra_params={"preload": True}
            )
            print(f"[load_data] loaded raw task={task}")

    elif analysis == "badepochs":
        ep_path = mne_bids.find_matching_paths(
            root=cfg.deriv_root,
            subjects=cfg.subjects,
            tasks=cfg.task,
            sessions=cfg.sessions,
            datatypes="meg",
            suffixes="epo",
            processings="clean",
            extensions=".fif",
        )[0]
        out[cfg.task] = mne.read_epochs(ep_path, preload=True)
        print("[load_data] loaded epochs")

    elif analysis == "manualica":
        raw_path = mne_bids.find_matching_paths(
            root=cfg.deriv_root,
            subjects=cfg.subjects,
            tasks=cfg.task,
            sessions=cfg.sessions,
            datatypes="meg",
            suffixes="raw",
            processings="clean",
            extensions=".fif",
        )[0]
        out[cfg.task] = mne_bids.read_raw_bids(raw_path, extra_params={"preload": True})
        ica_path = mne_bids.find_matching_paths(
            root=cfg.deriv_root,
            subjects=cfg.subjects,
            tasks=cfg.task,
            sessions=cfg.sessions,
            datatypes="meg",
            suffixes="ica",
            processings="ica",
            extensions=".fif",
        )[0]
        out["manualica"] = mne.preprocessing.read_ica(ica_path)
        print("[load_data] loaded raw + ICA")
    else:  # pragma: no cover
        raise ValueError(f"Unknown analysis {analysis}")
    return out


# --------------------------------------------------------------------------------------
# Individual Analysis Functions
# --------------------------------------------------------------------------------------


def regress_reference(
    raw: mne.io.BaseRaw, cfg: SimpleNamespace, is_noise: bool = False
) -> mne.io.BaseRaw:
    """Regress out reference channels from MEG channels."""

    print("\n[regress_reference] regressing-out reference channels")

    # remove breaks
    print(raw)
    if getattr(cfg, "find_breaks", False) and not is_noise:
        mne.preprocessing.annotate_break(
            raw,
            min_break_duration=cfg.min_break_duration,
            t_start_after_previous=cfg.t_break_annot_start_after_previous_event,
            t_stop_before_next=cfg.t_break_annot_stop_before_next_event,
        )

    # estimate weights
    raw_filt = raw.copy().filter(
        l_freq=cfg.l_freq, h_freq=cfg.h_freq, method="iir", picks=["ref_meg"]
    )  # filter data for estimating regression weights

    if getattr(cfg, "_regress_ref_timevarying", False):
        # time-varying regression (sliding window)
        print("\n[regress_reference] using time-varying regression")

        if getattr(cfg, "_regress_ref_method", "window") == "window":
            raw_data = raw.get_data()
            filt_data = raw_filt.get_data(picks="ref_meg")
            info = raw.info

            mag_idx = _picks_to_idx(info, cfg.ch_types[0])
            n_channels, n_times = raw_data.shape
            n_ref, _ = filt_data.shape
            sfreq = info["sfreq"]

            # ------- sliding window regression ----------
            window_size = int(
                sfreq * getattr(cfg, "mf_st_duration", 60.0)
            )  # window size
            step_size = int(window_size // 2)  # step size
            n_windows = (n_times - window_size) // step_size + 1

            # prior = np.sqrt(1e-3) * np.eye(2*n_ref)  # ridge prior for numerical stability
            prior = np.diag(
                np.hstack(
                    [
                        np.repeat([np.sqrt(1e-4)], n_ref),  # linear terms
                        np.repeat([np.sqrt(1e-4)], n_ref),  # quadratic terms
                    ]
                )
            )

            print(
                f"[regress_reference] processing {n_windows} windows of {window_size} samples ({window_size / sfreq:.1f} sec) in steps of {step_size} samples ({step_size / sfreq:.1f} sec)"
            )
            for w in range(n_windows):
                start = w * step_size
                end = start + window_size

                # get qr decomposition of reference channels in this window
                data_win = filt_data[:, start:end] - np.mean(
                    filt_data[:, start:end], axis=1, keepdims=True
                )
                data_x = np.hstack(
                    [
                        stats.zscore(data_win, axis=1).T,  # linear terms
                        stats.zscore(data_win**2, axis=1).T,  # quadratic terms
                    ]
                )
                X = np.vstack(
                    [
                        data_x,
                        prior,  # ridge prior
                    ]
                )

                Q, _, _ = qr(X, pivoting=True, mode="economic")
                Qd = Q[:window_size, :]  # get data rows

                raw_data[mag_idx, start:end] -= (
                    raw_data[mag_idx, start:end] @ Qd
                ) @ Qd.T

                if w % np.max((n_windows // 10, 1)) == 0:
                    print(f"[regress_reference] processed {w}/{n_windows} windows")

            raw_clean = mne.io.RawArray(raw_data, info)
            del raw_data, filt_data

        elif getattr(cfg, "_regress_ref_method", "window") == "gam":
            print("\n[regress_reference] using GAM regression")

            raw_data = raw.copy().get_data()
            n_times = raw.n_times
            sfreq = raw.info["sfreq"]
            info = raw.info
            ch_names = raw.ch_names
            # prepend raw.times to filt_data for GAM
            # filt_ref = np.vstack([raw.times, raw_filt.get_data(picks='ref_meg')])

            # svd of reference channels
            refs = raw.copy().get_data(picks="ref_meg")
            refs -= np.mean(refs, axis=1, keepdims=True)
            U, S, _ = np.linalg.svd(refs, full_matrices=False)
            print("reference SVs: ", S / np.min(S))

            # use SVD components as references
            ref_data = np.vstack([raw.times, U.T @ refs])
            ref_data -= np.mean(ref_data, axis=1, keepdims=True)
            n_refs = ref_data.shape[0] - 1
            del raw, refs, U

            n_splines = np.max(
                [4, int(n_times / (sfreq * 60.0))]
            )  # one spline per minute, min 4
            max_iter = 100
            fit_intercept = True

            print(f"refs: {n_refs}, times: {n_times}, splines: {n_splines}")

            if n_refs == 6:
                gam = LinearGAM(
                    terms=te(0, 1, spline_order=[3, 1], n_splines=[n_splines, 2])
                    + te(0, 2, spline_order=[3, 1], n_splines=[n_splines, 2])
                    + te(0, 3, spline_order=[3, 1], n_splines=[n_splines, 2])
                    + te(0, 4, spline_order=[3, 1], n_splines=[n_splines, 2])
                    + te(0, 5, spline_order=[3, 1], n_splines=[n_splines, 2])
                    + te(0, 6, spline_order=[3, 1], n_splines=[n_splines, 2]),
                    max_iter=max_iter,
                    fit_intercept=fit_intercept,
                    callbacks=None,
                    verbose=False,
                    lam=0.2,
                )
            else:
                raise ValueError(
                    f"GAM regression only implemented for 6 reference channels, got {n_refs}"
                )

            print(
                f"\n----- Fitting {len(_picks_to_idx(info, cfg.ch_types[0]))} channels -----\n"
            )
            loop_start = time.perf_counter()
            c = 1
            for mag in _picks_to_idx(info, cfg.ch_types[0]):
                print(f"\nfitting channel {c}: {ch_names[mag]} --------------")
                c += 1

                start_time = time.perf_counter()
                ref_dec = ref_data[:, ::12].T
                raw_dec = raw_data[mag, ::12].T
                gam.fit(ref_dec, raw_dec)
                # gam.gridsearch(ref_dec, raw_dec)
                # print('residualize')
                raw_data[mag, :] = gam.deviance_residuals(
                    ref_data.T, raw_data[mag, :].T
                ).T
                print(f"R2: {gam.statistics_['pseudo_r2']['explained_deviance']:.2f}")

                # Print the execution time
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"Execution time: {elapsed_time:.2f} seconds")

                # # print summary
                # print(gam.summary())

                # ## plotting
                # plt.switch_backend('qt5agg')
                # fig, axs = plt.subplots(1,3, figsize=(15,5), subplot_kw={"projection": "3d"})
                # titles = ['time', 'ref1', 'ref2']
                # for i, ax in enumerate(axs):
                #     ax.set_title(titles[i])

                #     # XX = gam.generate_X_grid(term=i)
                #     # ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
                #     # ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')

                #     XX = gam.generate_X_grid(term=i, meshgrid=True)
                #     Z = gam.partial_dependence(term=i, X=XX, meshgrid=True)
                #     ax.plot_surface(XX[0], XX[1], Z, cmap='viridis')
                # plt.show()
            loop_elapsed = time.perf_counter() - loop_start
            print(
                f"\n------------- Total Execution time: {loop_elapsed:.2f} seconds ------------- \n"
            )

            raw_clean = mne.io.RawArray(raw_data, info)

    else:
        print("\n[regress_reference] using standard regression")
        weights = mne.preprocessing.EOGRegression(
            picks=cfg.ch_types[0], picks_artifact="ref_meg"
        ).fit(raw_filt)
        raw_clean = weights.apply(raw, copy=True)
        del weights

    del raw_filt
    print("\nFinished reference regression!\n---------------\n")

    return raw_clean


def detect_bad_segments(
    raw: mne.io.BaseRaw, cfg: SimpleNamespace, is_noise: bool = False
) -> mne.io.BaseRaw:
    """Detect bad segments (single pass for noise, two passes otherwise)."""
    if is_noise:
        return osl_bad_segments(
            raw,
            picks=cfg.ch_types[0],
            ref_meg=False,
            metric="std",
            detect_zeros=False,
            channel_wise=True,
            segment_len=round(raw.info["sfreq"] * SEGMENT_LEN_SEC),
            channel_threshold=0.50,
        )
    if getattr(cfg, "find_breaks", False):
        mne.preprocessing.annotate_break(
            raw,
            min_break_duration=cfg.min_break_duration,
            t_start_after_previous=cfg.t_break_annot_start_after_previous_event,
            t_stop_before_next=cfg.t_break_annot_stop_before_next_event,
        )
    first = osl_bad_segments(
        raw,
        picks=cfg.ch_types[0],
        ref_meg=False,
        metric="std",
        detect_zeros=False,
        channel_wise=True,
        segment_len=round(raw.info["sfreq"] * SEGMENT_LEN_SEC),
        channel_threshold=0.05,
    )
    second = osl_bad_segments(
        first,
        picks=cfg.ch_types[0],
        ref_meg=False,
        metric="std",
        detect_zeros=False,
        channel_wise=True,
        segment_len=round(first.info["sfreq"] * SEGMENT_LEN_SEC * 0.66),
        channel_threshold=0.05,
    )
    return second


def detect_bad_channels(raw: mne.io.BaseRaw, cfg: SimpleNamespace) -> list[str]:
    """
    Find bad channel using GESD.
    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw MEG data to process.
    cfg : SimpleNamespace
        Configuration object containing:
        - l_freq : float
            Low cutoff frequency for bandpass filtering.
        - h_freq : float
            High cutoff frequency for bandpass filtering.
        - ch_types : list
            Channel types to process (e.g., ['mag']).
    Returns
    -------
    list[str]
        List of detected bad channel names.
    """
    filt = raw.copy().filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, method="iir")
    detected = osl_bad_channels(
        filt,
        picks=cfg.ch_types[0],
        ref_meg=None,
        significance_level=0.05,
    )
    # print bad channels
    print(
        f"\n[detect_bad_channels] detected {len(detected.info['bads'])} bad channels: {detected.info['bads']}"
    )
    return list(detected.info["bads"])


def drop_bad_epochs(epochs: mne.Epochs, cfg: SimpleNamespace) -> mne.Epochs:
    """
    Drop bad epochs using GESD.
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs to process.
    cfg : SimpleNamespace
        Configuration object containing:
        - ch_types : list
            Channel types to process (e.g., ['mag']).
    Returns
    -------
    mne.Epochs
        The cleaned epochs with bad epochs removed.
    """

    clean_epochs = osl_drop_bad_epochs(
        epochs,
        picks=cfg.ch_types[0],
        ref_meg=None,
        metric="std",
    )

    # print dropped epochs
    n_dropped = len(epochs) - len(clean_epochs)
    print(f"\n[drop_bad_epochs] dropped {n_dropped} epochs")

    return clean_epochs


def apply_hfc(
    raw: mne.io.BaseRaw, cfg: SimpleNamespace, noise: mne.io.BaseRaw | None = None
) -> tuple[mne.io.BaseRaw, mne.io.BaseRaw | None]:
    """Apply Homogeneous Field Correction (HFC) projections to MEG data.

    HFC projections remove spatial gradients in the magnetic field that are
    uniform across the sensor array, typically caused by distant sources or
    movements of the head relative to the sensors.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw MEG data to apply HFC projections to.
    cfg : SimpleNamespace
        Configuration object containing:
        - _do_HFC : bool
            Whether to apply HFC projections. If False, returns data unchanged.
        - _hfc_order : int
            Order of the HFC projections (typically 1-3).
        - ch_types : list
            Channel types to apply projections to (e.g., ['mag']).
    noise : mne.io.BaseRaw | None, optional
        Optional empty-room noise data to apply the same projections to.

    Returns
    -------
    raw : mne.io.BaseRaw
        Raw data with HFC projections applied (if enabled).
    noise : mne.io.BaseRaw | None
        Noise data with HFC projections applied (if provided and HFC enabled).

    Notes
    -----
    HFC projections are computed based on the sensor positions and applied
    as SSP (Signal Space Projection) projectors. The projections remove
    components that vary smoothly across the sensor array.
    """
    if not getattr(cfg, "_do_HFC", False):
        print("\n[apply_hfc] HFC disabled in configuration; skipping")
        return raw, noise

    print("\n[apply_hfc] Computing and applying HFC projections")
    print(f"[apply_hfc] Using HFC order: {cfg._hfc_order}")

    projs = mne.preprocessing.compute_proj_hfc(
        raw.info,
        order=cfg._hfc_order,
        picks=cfg.ch_types[0],
    )

    print(f"[apply_hfc] Computed {len(projs)} HFC projection(s)")
    raw.add_proj(projs=projs).apply_proj()

    if noise is not None:
        print("[apply_hfc] Applying same projections to noise data")
        noise.add_proj(projs=projs).apply_proj()

    print("[apply_hfc] HFC projections applied successfully")
    return raw, noise


def manual_channel_selection(
    raw: mne.io.BaseRaw, cfg: SimpleNamespace, noise: mne.io.BaseRaw | None = None
) -> tuple[mne.io.BaseRaw, list[str], mne.io.BaseRaw | None]:
    """Interactive manual selection of bad channels.

    Opens an interactive plot for visual inspection of the data, allowing
    the user to mark bad channels by clicking on them. The same bad channels
    are also marked in the noise data if provided.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw MEG data for channel inspection.
    cfg : SimpleNamespace
        Configuration object containing filtering parameters:
        - l_freq : float
            High-pass filter frequency for display.
        - h_freq : float
            Low-pass filter frequency for display.
    noise : mne.io.BaseRaw | None, optional
        Optional empty-room noise data that will receive the same bad channel markings.

    Returns
    -------
    raw : mne.io.BaseRaw
        Raw data with bad channels marked.
    bads : list[str]
        List of bad channel names selected by the user.
    noise : mne.io.BaseRaw | None
        Noise data with the same bad channels marked (if provided).

    Notes
    -----
    If Qt browser is not available, the interactive plot is skipped but the
    function will still process any existing bad channel markings in the data.
    """
    if not _HAVE_QT_BROWSER:
        print(
            "\n[manual_channel_selection] Qt browser not available; skipping interactive plot (set SKIP_MANUAL=1 to omit this step entirely)."
        )
    else:
        print(
            "\n[manual_channel_selection] Opening interactive plot for channel inspection"
        )
        raw.plot(
            precompute=True,
            n_channels=64,
            show_options=True,
            show=True,
            block=True,
            highpass=cfg.l_freq,
            lowpass=cfg.h_freq,
            decim=4,
            scalings=dict(mag=1e-11, eyegaze=0.01, pupil=0.01),
        )

    # Process bad channel markings
    bads: list[str] = []
    for ch in raw.info["bads"]:
        bads.append(ch if isinstance(ch, str) else ch.item())
    raw.info["bads"] = bads

    if noise is not None:
        print(
            f"[manual_channel_selection] Marking {len(bads)} bad channels in noise data"
        )
        noise.info["bads"] = bads.copy()

    print(f"[manual_channel_selection] Marked {len(bads)} bad channels: {bads}")
    return raw, bads, noise


def manual_ica_review(
    ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw, cfg: SimpleNamespace
) -> mne.preprocessing.ICA:
    # label bad components based on reference channels
    if getattr(cfg, "ref_bads", True):
        print(
            "\n[manual_ica_review] identifying bad components based on reference sensors -------\n"
        )

        ref_raw = raw.copy().pick("ref_meg").filter(l_freq=1, h_freq=None)
        ref_ica = mne.preprocessing.ICA(
            n_components=0.99,
            method="picard",
            max_iter=256,
            allow_ref_meg=True,
        )
        ref_ica.fit(ref_raw, decim=2, reject_by_annotation=True)
        ref_src = ref_ica.get_sources(ref_raw)
        ref_src.rename_channels(lambda x: f"REF_{x}")
        raw.add_channels([ref_src], force_update_info=True)
        ref_idx, _ = ica.find_bads_ref(inst=raw, method="separate")
        print(
            f"\n[manual_ica_review] marking {len(ref_idx)} components based on reference sensors: {ref_idx}\n"
        )
        ica.exclude.extend(ref_idx)
        del ref_raw, ref_ica, ref_src

    # label bad components based on osl's gesd
    if getattr(cfg, "gesd_bads", True):
        print(
            "\n[manual_ica_review] identifying bad components based on GESD -------\n"
        )
        sources = ica.get_sources(raw).get_data()
        kurtosis_scores = stats.kurtosis(sources, axis=1)
        std_scores = np.std(sources, axis=1, ddof=1)

        if (sources.shape[0] - len(ica.exclude)) < 5:
            print(
                f"\n[manual_ica_review] too few components remaining for GESD ({n_comps - len(ica.exclude)}); skipping\n"
            )
        else:
            # plot histogram of ic_scores
            # plt.figure()
            # plt.hist(ic_score[~np.isnan(ic_score)], bins=64, color='gray', edgecolor='black')
            # plt.xlabel("ICA Component Score (kurtosis)")
            # plt.ylabel("Count")
            # plt.title("Histogram of ICA Component Scores for GESD")
            # plt.show()

            # loop over both scores, include their names for plotting
            print("ica.exclude before: ", ica.exclude)
            for score, name in zip([kurtosis_scores, std_scores], ["kurtosis", "std"]):
                gesd_idx, _ = osl_gesd(score, p_out=1.0, outlier_side=1)

                if len(gesd_idx) == 0:
                    print(f"\n[manual_ica_review] {name} GESD found no outliers\n")
                else:
                    ica.exclude.extend(np.where(gesd_idx)[0].tolist())
                print(
                    f"\n[manual_ica_review] marking {len(np.where(gesd_idx)[0])} components based on {name} GESD: {np.where(gesd_idx)[0]}"
                )
            print("ica.exclude after: ", ica.exclude)

    ica.plot_components(inst=raw, nrows=5)
    ica.plot_sources(inst=raw, show_scrollbars=True, block=True)
    return ica


# --------------------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------------------


def run_analysis(
    cfg: SimpleNamespace, analysis: str, data: Dict[str, Any]
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"bads": []}

    if analysis == "badsegments":
        for task, raw in data.items():
            cleaned = detect_bad_segments(raw, cfg, is_noise=(task == "noise"))
            results[task] = cleaned

    elif analysis == "badchannels":
        dedup = set()
        for task, raw in data.items():
            bads = detect_bad_channels(raw, cfg)
            results[task] = raw
            for b in bads:
                if b not in dedup:
                    dedup.add(b)
        results["bads"] = sorted(dedup)

    elif analysis == "manualchannel":
        noise = data.get("noise") if getattr(cfg, "process_empty_room", False) else None
        raw, bads, noise = manual_channel_selection(data[cfg.task], cfg, noise)
        results[cfg.task] = raw
        results["bads"] = bads
        if noise is not None:
            results["noise"] = noise

    elif analysis == "badepochs":
        results[cfg.task] = drop_bad_epochs(data[cfg.task], cfg)

    elif analysis == "manualica":
        ica = manual_ica_review(data["manualica"], data[cfg.task], cfg)
        results[cfg.task] = data[cfg.task]
        results["ica"] = ica

    elif analysis == "regressref":
        for task, raw in data.items():
            print(f"[run_analysis] regressing reference channels from {task} data")
            results[task] = regress_reference(raw, cfg, is_noise=(task == "noise"))
        # if getattr(cfg, "process_empty_room", False):
        #     print("\n[run_analysis] regressing reference channels from noise data")
        #     results["noise"] = regress_reference(data.get("noise"), cfg)
        # results[cfg.task] = regress_reference(data[cfg.task], cfg)

    elif analysis == "applyhfc":
        noise = data.get("noise") if getattr(cfg, "process_empty_room", False) else None
        raw, noise = apply_hfc(data[cfg.task], cfg, noise)
        results[cfg.task] = raw
        if noise is not None:
            results["noise"] = noise

    else:  # pragma: no cover
        raise ValueError(f"Unknown analysis {analysis}")
    return results


def save_results(cfg: SimpleNamespace, analysis: str, results: Dict[str, Any]):
    tasks = {k: v for k, v in results.items() if k not in {"bads", "ica"}}
    print(f"\n[save_results] saving results for analysis={analysis}")
    if analysis in {
        "badsegments",
        "badchannels",
        "manualchannel",
        "regressref",
        "applyhfc",
    }:
        unique_bads = (
            sorted(set(results.get("bads", []))) if results.get("bads") else []
        )
        for task, raw in tasks.items():
            print(f"[save_results] writing task={task}")
            bp = mne_bids.find_matching_paths(
                root=cfg.bids_root,
                subjects=cfg.subjects,
                tasks=task,
                sessions=cfg.sessions,
                datatypes="meg",
                ignore_nosub=True,
                splits=None,
                extensions=".fif",
            )[0]
            bp.split = None
            if analysis in {"badchannels", "manualchannel"} and unique_bads:
                if analysis == "badchannels":
                    # merge existing and newly detected, keep unique
                    merged = sorted({*raw.info.get("bads", []), *unique_bads})
                    raw.info["bads"] = merged
                else:
                    raw.info["bads"] = list(unique_bads)
                mne_bids.mark_channels(
                    bp,
                    ch_names=unique_bads,
                    status="bad",
                    descriptions="osl",
                )
            if "noise" in tasks and task != "noise":
                er_bp = mne_bids.find_matching_paths(
                    root=cfg.bids_root,
                    subjects=cfg.subjects,
                    tasks="noise",
                    sessions=cfg.sessions,
                    datatypes="meg",
                    ignore_nosub=True,
                    splits=None,
                    extensions=".fif",
                )[0]
                mne_bids.write_raw_bids(
                    raw,
                    bp,
                    allow_preload=True,
                    overwrite=True,
                    format="FIF",
                    empty_room=er_bp,
                )
            else:
                mne_bids.write_raw_bids(
                    raw,
                    bp,
                    allow_preload=True,
                    overwrite=True,
                    format="FIF",
                )
    elif analysis == "badepochs":
        ep_bp = mne_bids.find_matching_paths(
            root=cfg.deriv_root,
            subjects=cfg.subjects,
            tasks=cfg.task,
            sessions=cfg.sessions,
            suffixes="epo",
            processings="clean",
            extensions=".fif",
        )[0]
        ep_bp.split = None
        tasks[cfg.task].save(ep_bp, split_naming="bids", overwrite=True)
    if analysis == "manualica":
        # Update TSV + save ICA object
        tsv_path = mne_bids.find_matching_paths(
            root=cfg.deriv_root,
            subjects=cfg.subjects,
            tasks=cfg.task,
            sessions=cfg.sessions,
            suffixes="components",
            processings="ica",
            extensions=".tsv",
        )[0]
        df = pd.read_csv(tsv_path, sep="\t")
        ica = results["ica"]
        for comp in ica.exclude:
            mask = df["component"].astype(str) == str(comp)
            if mask.any():
                df.loc[mask, "status"] = "bad"
                df.loc[mask, "status_description"] = "manual"
        df.to_csv(tsv_path, sep="\t", index=False)
        ica_path = mne_bids.find_matching_paths(
            root=cfg.deriv_root,
            subjects=cfg.subjects,
            tasks=cfg.task,
            sessions=cfg.sessions,
            suffixes="ica",
            processings="ica",
            extensions=".fif",
        )[0]
        ica.save(ica_path, overwrite=True)


def parse_args():
    p = argparse.ArgumentParser(description="Modular OPM auxiliary preprocessing")
    p.add_argument(
        "--analysis",
        required=True,
        choices=[
            "bad_segments",
            "bad_channels",
            "bad_epochs",
            "manual_channel",
            "manual_ica",
            "regress_ref",
            "apply_hfc",
        ],
    )
    p.add_argument("--config", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = _import_config(config_path=args.config)
    _update_config_from_path(config=cfg, config_path=args.config)
    analysis_key = args.analysis.replace("_", "")
    # cfg = SimpleNamespace()
    # _update_with_user_config(config=cfg, config_path=args.config)

    if analysis_key == "manualchannel" and not getattr(cfg, "_manual_bads", False):
        print("\n[main] manual channel selection disabled; exiting")
        return
    if analysis_key == "manualica":
        if (
            not getattr(cfg, "_manual_ica", False)
            or getattr(cfg, "spatial_filter", None) != "ica"
        ):
            print("\n[main] manual ICA disabled or spatial_filter != ica; exiting")
            return
    if analysis_key == "regressref" and not getattr(cfg, "_regress_ref", False):
        print("\n[main] reference regression disabled; exiting")
        return
    if analysis_key == "applyhfc" and not getattr(cfg, "_do_HFC", False):
        print("\n[main] HFC disabled; exiting")
        return

    data = load_data(cfg, analysis_key)
    results = run_analysis(cfg, analysis_key, data)
    save_results(cfg, analysis_key, results)


if __name__ == "__main__":
    main()
