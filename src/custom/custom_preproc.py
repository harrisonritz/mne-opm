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

import argparse
import os
from types import SimpleNamespace
from typing import Dict, Any

from click import pause
import mne
import mne_bids
import pandas as pd
import numpy as np
from scipy.linalg import qr
from scipy import stats

from mne._fiff.pick import _picks_to_idx


try:  # optional dependency for nicer Qt browser; skip if unavailable
    import mne_qt_browser  # noqa: F401
    _HAVE_QT_BROWSER = True
except Exception:  # pragma: no cover
    _HAVE_QT_BROWSER = False
from mne_bids_pipeline._config_import import _update_config_from_path
from osl_ephys.preprocessing.osl_wrappers import (
    bad_segments as osl_bad_segments,
    bad_channels as osl_bad_channels,
    drop_bad_epochs as osl_drop_bad_epochs,
    gesd as osl_gesd
)



# %% GLOBAL VARIABLES

try:  # ensure a GUI backend for interactive steps
    mne.viz.set_browser_backend("qt")
except Exception:  # pragma: no cover - fallback if qt not available
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
    print(f"\n\n[load_data] analysis={analysis}")
    out: Dict[str, Any] = {}

    if analysis in {"badsegments", "badchannels", "manualchannel", "regressref", "applyhfc"}:
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
            out[task] = mne_bids.read_raw_bids(bids_path, extra_params={"preload": True})
            print(f"\n[load_data] loaded raw task={task}")

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
        print("\n[load_data] loaded epochs")

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
        print("\n[load_data] loaded raw + ICA")
    else:  # pragma: no cover
        raise ValueError(f"Unknown analysis {analysis}")
    return out


# --------------------------------------------------------------------------------------
# Individual Analysis Functions
# --------------------------------------------------------------------------------------

def detect_bad_segments(raw: mne.io.BaseRaw, cfg: SimpleNamespace, is_noise: bool = False) -> mne.io.BaseRaw:
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
    

def regress_reference(raw: mne.io.BaseRaw, cfg: SimpleNamespace) -> mne.io.BaseRaw:
    """Regress out reference channels from MEG channels."""

    print("\n[regress_reference] regressing-out reference channels")

    # remove breaks
    if getattr(cfg, "find_breaks", False):
            mne.preprocessing.annotate_break(
                raw,
                min_break_duration=cfg.min_break_duration,
                t_start_after_previous=cfg.t_break_annot_start_after_previous_event,
                t_stop_before_next=cfg.t_break_annot_stop_before_next_event,
            )

    # estimate weights
    raw_filt = raw.copy().filter(
        l_freq=cfg.l_freq, 
        h_freq=cfg.h_freq, 
        method="iir", 
        picks=['ref_meg']
        ) # filter data for estimating regression weights
    # raw_filt = raw.copy()
    
    if getattr(cfg, "_regress_ref_timevarying", False):
        # time-varying regression (sliding window)
        print("\n[regress_reference] using time-varying regression")
        
        sfreq = raw.info['sfreq']
        raw_data = raw.get_data()
        filt_data = raw_filt.get_data(picks="ref_meg")

        mag_idx = _picks_to_idx(raw.info, cfg.ch_types[0])
        # ref_idx = _picks_to_idx(raw.info, 'ref_meg')

        n_channels, n_times = raw_data.shape
        n_ref, _ = filt_data.shape

        # n_ref = len(ref_idx)
        window_size = int(sfreq * getattr(cfg, "mf_st_duration", 100.0)) # window size
        step_size = int(window_size//2) # step size
        n_windows = (n_times - window_size) // step_size + 1

        prior = np.sqrt(1e-6) * np.eye(n_ref)  # ridge prior for numerical stability

        print(f"[regress_reference] processing {n_windows} windows")
        for w in range(n_windows):

            start = w * step_size
            end = start + window_size

            # get qr decomposition of reference channels in this window
            X = np.vstack([
                    stats.zscore(filt_data[:, start:end], axis=1).T,    # linear terms
                    prior,                                              # ridge prior
                ])
            # data_win = filt_data[:, start:end] - np.mean(filt_data[:, start:end], axis=1, keepdims=True)
            # X = np.vstack([
            #         np.hstack([
            #             stats.zscore(data_win, axis=1).T,       # linear terms
            #             stats.zscore(data_win**2, axis=1).T,    # quadratic terms
            #             ]),
            #         prior,                                      # ridge prior
            #     ])
            Q, _, _ =  qr(X, pivoting=True, mode='economic')
            Qd = Q[:window_size, :]  # get data rows
            
            raw_data[mag_idx, start:end] -= (raw_data[mag_idx, start:end] @ Qd) @ Qd.T

            if w % (n_windows // 10) == 0:
                print(f"[regress_reference] processed {w}/{n_windows} windows")

        raw_clean = mne.io.RawArray(raw_data, raw.info)
        # raw._data = raw_data
        del raw_data, filt_data
        
    else:
        print("\n[regress_reference] using standard regression")
        weights = mne.preprocessing.EOGRegression(picks=cfg.ch_types[0], picks_artifact="ref_meg").fit(raw_filt)
        raw_clean = weights.apply(raw, copy=True)
        del weights
        
    del raw_filt
    return raw_clean
    

def detect_bad_channels(raw: mne.io.BaseRaw, cfg: SimpleNamespace) -> list[str]:
    """Return list of detected bad channel names."""
    filt = raw.copy().filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, method="iir")
    detected = osl_bad_channels(
        filt,
        picks=cfg.ch_types[0],
        ref_meg=None,
        significance_level=0.05,
    )
    return list(detected.info["bads"])


def drop_bad_epochs(epochs: mne.Epochs, cfg: SimpleNamespace) -> mne.Epochs:
    return osl_drop_bad_epochs(
        epochs,
        picks=cfg.ch_types[0],
        ref_meg=None,
        metric="std",
    )


def apply_hfc(raw: mne.io.BaseRaw, cfg: SimpleNamespace, noise: mne.io.BaseRaw | None = None) -> tuple[mne.io.BaseRaw, mne.io.BaseRaw | None]:
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


def manual_channel_selection(raw: mne.io.BaseRaw, cfg: SimpleNamespace, noise: mne.io.BaseRaw | None = None) -> tuple[mne.io.BaseRaw, list[str], mne.io.BaseRaw | None]:
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
        print("\n[manual_channel_selection] Qt browser not available; skipping interactive plot (set SKIP_MANUAL=1 to omit this step entirely).")
    else:
        print("\n[manual_channel_selection] Opening interactive plot for channel inspection")
        raw.plot(
            precompute=True,
            n_channels=64,
            show_options=True,
            show=True,
            block=True,
            highpass=cfg.l_freq,
            lowpass=cfg.h_freq,
            decim=4,
            scalings=dict(mag=1e-11, eyegaze=.01, pupil=.01),
        )
    
    # Process bad channel markings
    bads: list[str] = []
    for ch in raw.info["bads"]:
        bads.append(ch if isinstance(ch, str) else ch.item())
    raw.info["bads"] = bads
    
    if noise is not None:
        print(f"[manual_channel_selection] Marking {len(bads)} bad channels in noise data")
        noise.info["bads"] = bads.copy()
    
    print(f"[manual_channel_selection] Marked {len(bads)} bad channels: {bads}")
    return raw, bads, noise


def manual_ica_review(ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw, cfg: SimpleNamespace) -> mne.preprocessing.ICA:
    
    # label bad components based on reference channels
    if getattr(cfg, "ref_bads", True):

        print("\n[manual_ica_review] identifying bad components based on reference sensors -------\n")

        ref_raw = raw.copy().pick("ref_meg").filter(l_freq=1, h_freq=None)
        ref_ica = mne.preprocessing.ICA(
            n_components=.99,
            method="picard",
            max_iter=256,
            allow_ref_meg=True,
        )
        ref_ica.fit(ref_raw, decim=2, reject_by_annotation=True)
        ref_src = ref_ica.get_sources(ref_raw)
        ref_src.rename_channels(lambda x: f"REF_{x}")
        raw.add_channels([ref_src], force_update_info=True)
        ref_idx, _ = ica.find_bads_ref(inst=raw, method="separate")
        print(f"\n[manual_ica_review] marking {len(ref_idx)} components based on reference sensors: {ref_idx}\n")
        ica.exclude.extend(ref_idx)
        del ref_raw, ref_ica, ref_src

    # label bad components based on osl's gesd
    if getattr(cfg, "gesd_bads", True):
        
        picks = ica.ch_names
        picks = _picks_to_idx(raw.info, picks, exclude=())
        def ica_metric(x): return np.log(np.std(x))
        # def ica_metric(x): return stats.kurtosis(x, axis=None)

        ic_score = np.full(np.max([ica.n_components_, 32]), np.nan)
        n_comps = len(ic_score)

        if (n_comps - len(ica.exclude)) < 5:

            print(f"\n[manual_ica_review] too few components remaining for GESD ({n_comps - len(ica.exclude)}); skipping\n")

        else:

            data = raw.get_data(picks=picks, reject_by_annotation="omit")

            print(f"\n\n[manual_ica_review] computing {n_comps - len(ica.exclude)}/{n_comps} component scores for GESD -------\n")
            for ii in range(n_comps):

                if ii in ica.exclude:
                    continue

                ic_score[ii] = ica_metric(ica._pick_sources(
                                    data=data, 
                                    include=None, 
                                    exclude=[ii], 
                                    n_pca_components=cfg.ica_n_components,
                                    ))
            
                print(f"[manual_ica_review] component {ii}: score={ic_score[ii]:.4f}\n")
        
            # plot histogram of ic_scores
            import matplotlib.pyplot as plt
            plt.figure()
            plt.hist(ic_score[~np.isnan(ic_score)], bins=64, color='gray', edgecolor='black')
            plt.xlabel("ICA Component Score (mean log std)")
            plt.ylabel("Count")
            plt.title("Histogram of ICA Component Scores for GESD")
            plt.show()

            gesd_idx,_ = osl_gesd(ic_score, p_out=1.0, outlier_side=-1)
            
            print('ica.exclude before: ', ica.exclude)
            if len(gesd_idx) == 0:
                print("\n[manual_ica_review] GESD found no outliers\n")
            else:
                ica.exclude.extend(np.where(gesd_idx)[0].tolist())
            print('ica.exclude after: ', ica.exclude)
            print(f"\n[manual_ica_review] marking {len(np.where(gesd_idx)[0])} components based on GESD: {np.where(gesd_idx)[0]}")

        

        

    ica.plot_components(inst=raw, nrows=5)
    ica.plot_sources(inst=raw, show_scrollbars=True, block=True)
    return ica


# --------------------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------------------


def run_analysis(cfg: SimpleNamespace, analysis: str, data: Dict[str, Any]) -> Dict[str, Any]:
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
        raw = regress_reference(data[cfg.task], cfg)
        results[cfg.task] = raw

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
    if analysis in {"badsegments", "badchannels", "manualchannel", "regressref", "applyhfc"}:
        unique_bads = sorted(set(results.get("bads", []))) if results.get("bads") else []
        for task, raw in tasks.items():
            print(f"\n\n[save_results] writing task={task}")
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
    p.add_argument("--analysis", required=True, choices=[
        "bad_segments", "bad_channels", "bad_epochs", "manual_channel", "manual_ica", "regress_ref", "apply_hfc"
    ])
    p.add_argument("--config", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    analysis_key = args.analysis.replace("_", "")
    cfg = SimpleNamespace()
    _update_config_from_path(config=cfg, config_path=args.config)

    if analysis_key == "manualchannel" and not getattr(cfg, "_manual_bads", False):
        print("\n[main] manual channel selection disabled; exiting")
        return
    if analysis_key == "manualica":
        if not getattr(cfg, "_manual_ica", False) or getattr(cfg, "spatial_filter", None) != "ica":
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
