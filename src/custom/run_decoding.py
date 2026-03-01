"""Multivariate decoding (MVPA) analysis for OPM-MEG data.

This script performs time-resolved, temporal-generalization, and full-epoch
decoding on preprocessed MEG data.  It supports within-condition (LOGO CV)
and cross-condition (train-on-A / test-on-B) analyses.

Usage:
    python run_decoding.py --config=/path/to/config.py

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from mne.decoding import (
    GeneralizingEstimator,
    LinearModel,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
    get_coef,
)
from mne_bids import BIDSPath
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from custom.transformers import FlexPCA, MultivariateNoiseNormalizer

# Add mne-bids-pipeline to path for importing utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mne-bids-pipeline"))

from mne_bids_pipeline._config_import import _update_config_from_path, _import_config
from mne_bids_pipeline._config_utils import sanitize_cond_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_fig(fig, path_stem, formats, dpi=150):
    """Save a figure in each requested format."""
    for fmt in formats:
        p = Path(f"{path_stem}.{fmt}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {p}")


def _resolve_n_components(setting, rank):
    """Resolve a high-level n_components setting to a PCA-compatible value.

    Parameters
    ----------
    setting : "rank" | int | float
        * ``"rank"`` — use *rank* (the data rank estimated from epochs.info).
        * ``int``    — fixed number of components (FlexPCA will cap if needed).
        * ``float``  — variance ratio passed through to PCA (e.g. 0.99).
    rank : int
        Data rank obtained from ``get_data_rank``.
    """
    if isinstance(setting, str) and setting.lower() == "rank":
        return rank
    return setting


def get_data_rank(epochs):
    """Estimate the numerical rank of the data from epochs.info.

    Uses ``mne.compute_rank`` which accounts for SSP projectors, ICA,
    Maxwell filtering, etc.  Returns the sum of ranks across channel types.
    """
    rank_dict = mne.compute_rank(epochs, rank="info")
    total = sum(rank_dict.values())
    print(f"      Data rank: {rank_dict} -> total = {total}")
    return total


def _prep_contrast(epochs, contrast, decim=1, group_column="run"):
    """Subset, equalize, and prepare data arrays for one binary contrast.

    Returns (ep_all, X, y, times, groups, n1, n2)
    or None if either condition is empty.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data with metadata.
    contrast : dict
        Must have 'name' and 'conditions' (list of exactly 2 metadata queries).
    decim : int
        Decimation factor applied to the concatenated epochs before data
        extraction (1 = no decimation).
    group_column : str
        Metadata column used for LOGO CV grouping.
    """
    cond_1, cond_2 = contrast["conditions"]
    ep1 = epochs[cond_1]
    ep2 = epochs[cond_2]
    n1, n2 = len(ep1), len(ep2)
    print(f"      n_cond1={n1}, n_cond2={n2} -- equalizing ...")

    if n1 == 0 or n2 == 0:
        print(f"      WARNING: empty condition -- skipping {contrast['name']}")
        return None

    mne.epochs.equalize_epoch_counts([ep1, ep2])
    n1, n2 = len(ep1), len(ep2)
    print(f"      After equalization: n_cond1={n1}, n_cond2={n2}")

    ep_all = mne.concatenate_epochs([ep1, ep2], verbose="error")
    ep_all.pick(
        mne.pick_types(ep_all.info, meg=True, eeg=True, ref_meg=False, exclude="bads")
    )

    if decim > 1:
        ep_all.decimate(decim)
        print(f"      Decimated x{decim}: {len(ep_all.times)} time points remaining")

    X = ep_all.get_data()
    y = np.concatenate([np.ones(n1), np.zeros(n2)])
    times = ep_all.times
    groups = ep_all.metadata[group_column].values

    return ep_all, X, y, times, groups, n1, n2


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------

def _make_time_clf(n_components):
    """Fresh per-time-step pipeline (for SlidingEstimator / GeneralizingEstimator).

    MultivariateNoiseNormalizer -> FlexPCA(n_components) -> LinearModel(LinearSVC)

    FlexPCA will silently cap n_components to min(n_samples, n_features)
    if an integer value exceeds the CV-fold sample count.
    """
    return make_pipeline(
        MultivariateNoiseNormalizer(),
        FlexPCA(n_components=n_components, whiten=False),
        LinearModel(LinearSVC(C=1, max_iter=10000)),
    )


def _make_epoch_clf(n_components):
    """Fresh full-epoch pipeline.

    MultivariateNoiseNormalizer (3-D) -> Vectorizer -> FlexPCA(n_components) -> LinearSVC
    """
    return make_pipeline(
        MultivariateNoiseNormalizer(),
        Vectorizer(),
        FlexPCA(n_components=n_components, whiten=False),
        LinearSVC(C=1, max_iter=10000),
    )


# ---------------------------------------------------------------------------
# Decoding functions
# ---------------------------------------------------------------------------

def run_subject_time_decoding(epochs, contrast, scoring="roc_auc",
                               n_jobs=1, decim=1, group_column="run",
                               time_n_components="rank"):
    """Sliding-window decoding for one subject x contrast (LOGO cross-validation).

    Returns
    -------
    dict with keys: times, scores, cv_scores, patterns, filters, info, n_cond1, n_cond2
    or None if a condition has no epochs.
    """
    prepped = _prep_contrast(epochs, contrast, decim=decim, group_column=group_column)
    if prepped is None:
        return None
    ep_all, X, y, times, groups, n1, n2 = prepped

    rank = get_data_rank(ep_all)
    n_comp = _resolve_n_components(time_n_components, rank)
    print(f"      PCA n_components: {time_n_components!r} -> {n_comp}")

    sliding = SlidingEstimator(
        _make_time_clf(n_comp), scoring=scoring, n_jobs=n_jobs, verbose=False
    )

    print(f"      Running time decoding (LOGO CV): {contrast['name']} ...")
    cv_scores = cross_val_multiscore(
        sliding, X=X, y=y,
        cv=LeaveOneGroupOut(), groups=groups,
        n_jobs=1, verbose=False,
    )
    mean_scores = cv_scores.mean(axis=0)

    print(f"      Fitting on all data for patterns/filters ...")
    sliding.fit(X, y)
    patterns = get_coef(sliding, "patterns_", inverse_transform=True)
    filters_ = get_coef(sliding, "filters_", inverse_transform=True)

    print(
        f"      {contrast['name']}: peak score = {mean_scores.max():.3f} "
        f"at t = {times[mean_scores.argmax()]:.3f}s"
    )
    return dict(
        times=times, scores=mean_scores, cv_scores=cv_scores,
        patterns=patterns, filters=filters_,
        info=ep_all.info, n_cond1=n1, n_cond2=n2,
    )


def run_subject_temporal_gen(epochs, contrast, scoring="roc_auc",
                              n_jobs=1, decim=1, group_column="run",
                              time_n_components="rank"):
    """Temporal-generalization decoding for one subject x contrast (LOGO CV).

    Returns
    -------
    dict with keys: times, cv_scores, scores_mean, n_cond1, n_cond2
    or None if a condition has no epochs.
    """
    prepped = _prep_contrast(epochs, contrast, decim=decim, group_column=group_column)
    if prepped is None:
        return None
    ep_all, X, y, times, groups, n1, n2 = prepped

    rank = get_data_rank(ep_all)
    n_comp = _resolve_n_components(time_n_components, rank)
    print(f"      PCA n_components: {time_n_components!r} -> {n_comp}")

    tg = GeneralizingEstimator(
        _make_time_clf(n_comp), scoring=scoring, n_jobs=n_jobs, verbose=False
    )

    print(f"      Running temporal generalization (LOGO CV): {contrast['name']} ...")
    cv_scores = cross_val_multiscore(
        tg, X=X, y=y,
        cv=LeaveOneGroupOut(), groups=groups,
        n_jobs=1, verbose=False,
    )

    print(
        f"      {contrast['name']}: TG peak = {cv_scores.mean(axis=0).max():.3f}"
    )
    return dict(
        times=times, cv_scores=cv_scores, scores_mean=cv_scores.mean(axis=0),
        n_cond1=n1, n_cond2=n2,
    )


def run_subject_epoch_decoding(epochs, contrast, scoring="roc_auc",
                                group_column="run",
                                epoch_n_components=0.99):
    """Full-epoch decoding for one subject x contrast (LOGO cross-validation).

    Returns
    -------
    dict with keys: score, cv_scores, n_cond1, n_cond2
    or None if a condition has no epochs.
    """
    prepped = _prep_contrast(epochs, contrast, group_column=group_column)
    if prepped is None:
        return None
    ep_all, X, y, times, groups, n1, n2 = prepped

    rank = get_data_rank(ep_all)
    n_comp = _resolve_n_components(epoch_n_components, rank)
    print(f"      PCA n_components (epoch): {epoch_n_components!r} -> {n_comp}")

    clf = _make_epoch_clf(n_comp)

    print(f"      Running epoch decoding (LOGO CV): {contrast['name']} ...")
    cv_scores = cross_val_score(
        clf, X, y,
        cv=LeaveOneGroupOut(), groups=groups,
        scoring=scoring, n_jobs=1,
    )
    mean_score = cv_scores.mean()

    print(
        f"      {contrast['name']}: mean = {mean_score:.3f} "
        f"+/- {cv_scores.std():.3f}"
    )
    return dict(score=mean_score, cv_scores=cv_scores, n_cond1=n1, n_cond2=n2)


def run_subject_cross_decoding(epochs, cross_contrast, scoring="roc_auc",
                                n_jobs=1, decim=1, group_column="run",
                                time_n_components="rank",
                                epoch_n_components=0.99):
    """Cross-condition decoding: fit on condition A, evaluate on condition B.

    No CV is used -- training and test conditions are independent datasets.

    Returns
    -------
    dict with keys: name, train_name, test_name, and optional
    time/tg/epoch sub-dicts.  Returns None if either condition is empty.
    """
    analyses = cross_contrast.get("analyses", [])
    if not analyses:
        return None

    train_prepped = _prep_contrast(
        epochs, cross_contrast["train"], decim=decim, group_column=group_column
    )
    test_prepped = _prep_contrast(
        epochs, cross_contrast["test"], decim=decim, group_column=group_column
    )

    if train_prepped is None or test_prepped is None:
        return None

    ep_train, X_train, y_train, times, _, n1_tr, n2_tr = train_prepped
    _, X_test, y_test, _, _, n1_te, n2_te = test_prepped

    rank = get_data_rank(ep_train)
    n_comp_time = _resolve_n_components(time_n_components, rank)
    n_comp_epoch = _resolve_n_components(epoch_n_components, rank)
    counts = dict(n_train1=n1_tr, n_train2=n2_tr, n_test1=n1_te, n_test2=n2_te)

    out = dict(
        name=cross_contrast["name"],
        train_name=cross_contrast["train"]["name"],
        test_name=cross_contrast["test"]["name"],
        train_conditions=cross_contrast["train"]["conditions"],
        test_conditions=cross_contrast["test"]["conditions"],
    )

    if "time" in analyses:
        print(f"      [cross time] {cross_contrast['name']} ...")
        sliding = SlidingEstimator(
            _make_time_clf(n_comp_time), scoring=scoring, n_jobs=n_jobs, verbose=False
        )
        sliding.fit(X_train, y_train)
        scores = sliding.score(X_test, y_test)
        patterns = get_coef(sliding, "patterns_", inverse_transform=True)
        filters_ = get_coef(sliding, "filters_", inverse_transform=True)
        print(f"      peak = {scores.max():.3f} at t={times[scores.argmax()]:.3f}s")
        out["time"] = dict(
            times=times, scores=scores, patterns=patterns, filters=filters_,
            info=ep_train.info, **counts,
        )

    if "tg" in analyses:
        print(f"      [cross TG] {cross_contrast['name']} ...")
        tg = GeneralizingEstimator(
            _make_time_clf(n_comp_time), scoring=scoring, n_jobs=n_jobs, verbose=False
        )
        tg.fit(X_train, y_train)
        scores_mat = tg.score(X_test, y_test)
        print(f"      TG peak = {scores_mat.max():.3f}")
        out["tg"] = dict(times=times, scores_mean=scores_mat, **counts)

    if "epoch" in analyses:
        print(f"      [cross epoch] {cross_contrast['name']} ...")
        clf = _make_epoch_clf(n_comp_epoch)
        clf.fit(X_train, y_train)
        score = roc_auc_score(y_test, clf.decision_function(X_test))
        print(f"      score = {score:.3f}")
        out["epoch"] = dict(score=score, **counts)

    return out


# ---------------------------------------------------------------------------
# BIDS save functions
# ---------------------------------------------------------------------------

def save_time_results(bids_path, contrast, result, scoring, out_dir):
    """Save time-by-time decoding results as TSV and patterns as NPZ."""
    cond_san = sanitize_cond_name(contrast["name"])
    cond_1, cond_2 = contrast["conditions"]
    times = result["times"]

    tsv_name = bids_path.copy().update(
        processing=cond_san,
        suffix=f"decode-time+{scoring}",
        extension=".tsv",
    ).fpath.name
    tsv_save_path = out_dir / tsv_name
    pd.DataFrame({
        "cond_1": [cond_1] * len(times),
        "cond_2": [cond_2] * len(times),
        "time": times,
        "mean_crossval_score": result["scores"],
        "metric": [scoring] * len(times),
    }).to_csv(tsv_save_path, sep="\t", index=False)
    print(f"      Saved: {tsv_save_path}")

    npz_name = bids_path.copy().update(
        processing=cond_san,
        suffix="decode-patterns",
        extension=".npz",
    ).fpath.name
    npz_save_path = out_dir / npz_name
    np.savez(
        npz_save_path,
        patterns=result["patterns"],
        filters=result["filters"],
        times=times,
        ch_names=result["info"]["ch_names"],
    )
    print(f"      Saved: {npz_save_path}")


def save_epoch_results(bids_path, contrast, result, scoring, out_dir):
    """Save full-epoch decoding results as TSV."""
    cond_san = sanitize_cond_name(contrast["name"])
    cond_1, cond_2 = contrast["conditions"]

    tsv_name = bids_path.copy().update(
        processing=cond_san,
        suffix=f"decode-epoch+{scoring}",
        extension=".tsv",
    ).fpath.name
    tsv_save_path = out_dir / tsv_name
    pd.DataFrame({
        "cond_1": [cond_1],
        "cond_2": [cond_2],
        "mean_crossval_score": [result["score"]],
        "metric": [scoring],
    }).to_csv(tsv_save_path, sep="\t", index=False)
    print(f"      Saved: {tsv_save_path}")


def save_tg_results(bids_path, contrast, result, out_dir):
    """Save temporal generalization results as NPZ."""
    cond_san = sanitize_cond_name(contrast["name"])

    npz_name = bids_path.copy().update(
        processing=cond_san,
        suffix="decode-tg",
        extension=".npz",
    ).fpath.name
    npz_save_path = out_dir / npz_name
    np.savez(
        npz_save_path,
        cv_scores=result["cv_scores"],
        scores_mean=result["scores_mean"],
        times=result["times"],
    )
    print(f"      Saved: {npz_save_path}")


def save_cross_time_results(bids_path, cross_contrast, result, scoring, out_dir):
    """Save cross-condition time decoding results."""
    cc_san = sanitize_cond_name(cross_contrast["name"])
    times = result["times"]

    tsv_name = bids_path.copy().update(
        processing=cc_san,
        suffix=f"decode-crosstime+{scoring}",
        extension=".tsv",
    ).fpath.name
    tsv_save_path = out_dir / tsv_name
    pd.DataFrame({
        "train_cond_1": [cross_contrast["train"]["conditions"][0]] * len(times),
        "train_cond_2": [cross_contrast["train"]["conditions"][1]] * len(times),
        "test_cond_1": [cross_contrast["test"]["conditions"][0]] * len(times),
        "test_cond_2": [cross_contrast["test"]["conditions"][1]] * len(times),
        "time": times,
        "score": result["scores"],
        "metric": [scoring] * len(times),
    }).to_csv(tsv_save_path, sep="\t", index=False)
    print(f"      Saved: {tsv_save_path}")

    npz_name = bids_path.copy().update(
        processing=cc_san,
        suffix="decode-crosspatterns",
        extension=".npz",
    ).fpath.name
    npz_save_path = out_dir / npz_name
    np.savez(
        npz_save_path,
        patterns=result["patterns"],
        filters=result["filters"],
        times=times,
        ch_names=result["info"]["ch_names"],
    )
    print(f"      Saved: {npz_save_path}")


def save_cross_epoch_results(bids_path, cross_contrast, result, scoring, out_dir):
    """Save cross-condition epoch decoding results."""
    cc_san = sanitize_cond_name(cross_contrast["name"])

    tsv_name = bids_path.copy().update(
        processing=cc_san,
        suffix=f"decode-crossepoch+{scoring}",
        extension=".tsv",
    ).fpath.name
    tsv_save_path = out_dir / tsv_name
    pd.DataFrame({
        "train_cond_1": [cross_contrast["train"]["conditions"][0]],
        "train_cond_2": [cross_contrast["train"]["conditions"][1]],
        "test_cond_1": [cross_contrast["test"]["conditions"][0]],
        "test_cond_2": [cross_contrast["test"]["conditions"][1]],
        "score": [result["score"]],
        "metric": [scoring],
    }).to_csv(tsv_save_path, sep="\t", index=False)
    print(f"      Saved: {tsv_save_path}")


def save_cross_tg_results(bids_path, cross_contrast, cc_res, result, out_dir):
    """Save cross-condition temporal generalization results."""
    cc_san = sanitize_cond_name(cross_contrast["name"])

    npz_name = bids_path.copy().update(
        processing=cc_san,
        suffix="decode-crosstg",
        extension=".npz",
    ).fpath.name
    npz_save_path = out_dir / npz_name
    np.savez(
        npz_save_path,
        scores_mean=result["scores_mean"],
        times=result["times"],
        train_name=cc_res["train_name"],
        test_name=cc_res["test_name"],
    )
    print(f"      Saved: {npz_save_path}")


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_subject_time_ribbon(time_results, subject, out_dir, save_formats,
                              chance=0.5):
    """Ribbon plot of time-resolved decoding for a single subject."""
    contrast_names = sorted(time_results.keys())
    n_c = len(contrast_names)
    if n_c == 0:
        return

    n_cols = min(4, n_c)
    n_rows = int(np.ceil(n_c / n_cols))
    palette = sns.color_palette("deep", n_c)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False, sharex=True, sharey=True,
    )

    for ci, cname in enumerate(contrast_names):
        ax = axes[ci // n_cols, ci % n_cols]
        res = time_results[cname]
        times = res["times"]
        cv_scores = res["cv_scores"]
        mean_sc = cv_scores.mean(axis=0)
        std_sc = cv_scores.std(axis=0)

        for fi in range(cv_scores.shape[0]):
            ax.plot(times, cv_scores[fi, :], color="gray", alpha=0.25,
                    linewidth=0.6, zorder=1)

        ax.fill_between(
            times, mean_sc - std_sc, mean_sc + std_sc,
            alpha=0.35, color=palette[ci], zorder=2,
        )
        ax.plot(
            times, mean_sc, color=palette[ci], linewidth=2,
            label=f"Mean (n_folds={cv_scores.shape[0]})", zorder=3,
        )

        ax.axhline(chance, color="red", linestyle="--", linewidth=1, zorder=0)
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8, zorder=0)
        ax.set_title(cname, fontsize=9)
        if ci % n_cols == 0:
            ax.set_ylabel("ROC-AUC")
        if ci // n_cols == n_rows - 1:
            ax.set_xlabel("Time (s)")
        ax.legend(fontsize=6, loc="upper right")

    for ci in range(n_c, n_rows * n_cols):
        axes[ci // n_cols, ci % n_cols].set_visible(False)

    fig.suptitle(
        f"Time-Resolved Decoding -- {subject}\n"
        f"(ribbon = +/-1 SD across folds, gray = individual folds)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / f"{subject}__time_ribbon", save_formats)
    plt.close(fig)


def plot_subject_epoch_bar(epoch_results, subject, out_dir, save_formats,
                            chance=0.5):
    """Bar plot of full-epoch decoding for a single subject."""
    contrast_names = sorted(epoch_results.keys())
    if not contrast_names:
        return

    records = []
    for cname, res in epoch_results.items():
        for sc in res["cv_scores"]:
            records.append({"contrast": cname, "score": float(sc)})
    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(max(6, len(contrast_names) * 1.2), 5))
    sns.barplot(
        data=df, x="contrast", y="score",
        errorbar=("ci", 95), capsize=0.15,
        color="steelblue", edgecolor="black", ax=ax,
    )
    sns.stripplot(
        data=df, x="contrast", y="score",
        color="black", alpha=0.6, size=5, jitter=True, ax=ax,
    )
    ax.axhline(chance, color="red", linestyle="--", linewidth=1,
               label="chance (0.5)")
    ax.set_xlabel("Contrast")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(
        f"Epoch Decoding -- {subject}\n"
        f"(bar = mean +/- 95% CI, dots = individual folds)"
    )
    ax.legend(fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir / f"{subject}__epoch_bar", save_formats)
    plt.close(fig)


def plot_subject_tg_heatmap(tg_results, subject, out_dir, save_formats):
    """Grid of temporal-generalization heatmaps for a single subject."""
    contrast_names = sorted(tg_results.keys())
    n_c = len(contrast_names)
    if n_c == 0:
        return

    n_cols = min(4, n_c)
    n_rows = int(np.ceil(n_c / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 5 * n_rows),
        squeeze=False,
    )

    for ci, cname in enumerate(contrast_names):
        ax = axes[ci // n_cols, ci % n_cols]
        res = tg_results[cname]
        times = res["times"]
        mat = res["scores_mean"]
        n_train, n_test = mat.shape
        dt = times[1] - times[0] if len(times) > 1 else 0.01

        mat_flip = mat[::-1, :]
        half_range = max(abs(mat.max() - 0.5), abs(mat.min() - 0.5), 0.05)

        tick_step = max(1, int(round(0.1 / abs(dt))))
        tick_pos_set = set(range(0, n_test, tick_step))
        col_labels = [f"{times[j]:.1f}" if j in tick_pos_set else ""
                      for j in range(n_test)]
        row_labels = [f"{times[n_train - 1 - i]:.1f}" if i in tick_pos_set else ""
                      for i in range(n_train)]

        sns.heatmap(
            mat_flip,
            cmap="coolwarm", center=0.5,
            vmin=0.5 - half_range, vmax=0.5 + half_range,
            xticklabels=col_labels, yticklabels=row_labels,
            cbar_kws={"label": "ROC-AUC", "shrink": 0.8},
            ax=ax,
        )

        ax.plot([0, n_test], [n_train, 0], "k--", linewidth=0.8, alpha=0.5)

        t0_idx = int(np.argmin(np.abs(times)))
        ax.axvline(t0_idx + 0.5, color="gray", linewidth=0.6, linestyle=":")
        ax.axhline(n_train - t0_idx - 0.5, color="gray", linewidth=0.6,
                    linestyle=":")

        ax.set_title(cname, fontsize=9)
        ax.set_xlabel("Testing time (s)", fontsize=8)
        ax.set_ylabel("Training time (s)", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    for ci in range(n_c, n_rows * n_cols):
        axes[ci // n_cols, ci % n_cols].set_visible(False)

    fig.suptitle(
        f"Temporal Generalization -- {subject}\n"
        f"(coolwarm centered at 0.5, dashed = train=test diagonal)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / f"{subject}__tg_heatmaps", save_formats)
    plt.close(fig)


def plot_subject_cross_ribbon(cross_time_results, subject, out_dir,
                               save_formats, chance=0.5):
    """Time-score line plots for cross-condition decoding."""
    names = sorted(cross_time_results.keys())
    n_c = len(names)
    if n_c == 0:
        return

    n_cols = min(4, n_c)
    n_rows = int(np.ceil(n_c / n_cols))
    palette = sns.color_palette("deep", n_c)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False, sharex=True, sharey=True,
    )

    for ci, name in enumerate(names):
        ax = axes[ci // n_cols, ci % n_cols]
        res = cross_time_results[name]
        times = res["times"]
        scores = res["scores"]

        ax.plot(times, scores, color=palette[ci], linewidth=2, zorder=3)
        ax.axhline(chance, color="red", linestyle="--", linewidth=1, zorder=0)
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8, zorder=0)
        ax.set_title(
            f"{name}\ntrain={res.get('train_name', '')} -> "
            f"test={res.get('test_name', '')}",
            fontsize=8,
        )
        if ci % n_cols == 0:
            ax.set_ylabel("ROC-AUC")
        if ci // n_cols == n_rows - 1:
            ax.set_xlabel("Time (s)")

    for ci in range(n_c, n_rows * n_cols):
        axes[ci // n_cols, ci % n_cols].set_visible(False)

    fig.suptitle(
        f"Cross-Condition Time Decoding -- {subject}\n"
        f"(fit-all/score-all, no CV)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / f"{subject}__cross_time_ribbon", save_formats)
    plt.close(fig)


def plot_subject_cross_epoch_bar(cross_epoch_results, subject, out_dir,
                                  save_formats, chance=0.5):
    """Bar chart of cross-condition epoch scores."""
    names = sorted(cross_epoch_results.keys())
    if not names:
        return

    records = [
        {"name": n, "score": cross_epoch_results[n]["score"]}
        for n in names
    ]
    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(max(5, len(names) * 1.2), 5))
    sns.barplot(data=df, x="name", y="score", color="steelblue",
                edgecolor="black", ax=ax)
    ax.axhline(chance, color="red", linestyle="--", linewidth=1,
               label="chance (0.5)")
    ax.set_xlabel("Cross-contrast")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(
        f"Cross-Condition Epoch Decoding -- {subject}\n"
        f"(fit-all/score-all, no CV)"
    )
    ax.legend(fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir / f"{subject}__cross_epoch_bar", save_formats)
    plt.close(fig)


def plot_subject_cross_tg_heatmap(cross_tg_results, subject, out_dir,
                                    save_formats):
    """TG heatmaps for cross-condition decoding."""
    names = sorted(cross_tg_results.keys())
    n_c = len(names)
    if n_c == 0:
        return

    n_cols = min(4, n_c)
    n_rows = int(np.ceil(n_c / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 5 * n_rows),
        squeeze=False,
    )

    for ci, name in enumerate(names):
        ax = axes[ci // n_cols, ci % n_cols]
        res = cross_tg_results[name]
        times = res["times"]
        mat = res["scores_mean"]
        n_train, n_test = mat.shape
        dt = times[1] - times[0] if len(times) > 1 else 0.01

        mat_flip = mat[::-1, :]
        half_range = max(abs(mat.max() - 0.5), abs(mat.min() - 0.5), 0.05)

        tick_step = max(1, int(round(0.1 / abs(dt))))
        tick_pos_set = set(range(0, n_test, tick_step))
        col_labels = [f"{times[j]:.1f}" if j in tick_pos_set else ""
                      for j in range(n_test)]
        row_labels = [f"{times[n_train - 1 - i]:.1f}" if i in tick_pos_set else ""
                      for i in range(n_train)]

        sns.heatmap(
            mat_flip,
            cmap="coolwarm", center=0.5,
            vmin=0.5 - half_range, vmax=0.5 + half_range,
            xticklabels=col_labels, yticklabels=row_labels,
            cbar_kws={"label": "ROC-AUC", "shrink": 0.8},
            ax=ax,
        )

        t0_idx = int(np.argmin(np.abs(times)))
        ax.plot([0, n_test], [n_train, 0], "k--", linewidth=0.8, alpha=0.5)
        ax.axvline(t0_idx + 0.5, color="gray", linewidth=0.6, linestyle=":")
        ax.axhline(n_train - t0_idx - 0.5, color="gray", linewidth=0.6,
                    linestyle=":")

        ax.set_title(f"{name}", fontsize=9)
        ax.set_xlabel(
            f"Testing time -- {res.get('test_name', '')} (s)", fontsize=8
        )
        ax.set_ylabel(
            f"Training time -- {res.get('train_name', '')} (s)", fontsize=8
        )
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    for ci in range(n_c, n_rows * n_cols):
        axes[ci // n_cols, ci % n_cols].set_visible(False)

    fig.suptitle(
        f"Cross-Condition Temporal Generalization -- {subject}\n"
        f"(coolwarm centered at 0.5, dashed = train-time = test-time diagonal)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / f"{subject}__cross_tg_heatmaps", save_formats)
    plt.close(fig)


def plot_subject_time_patterns(results, subject, out_dir, save_formats,
                               pattern_times=None, file_prefix="time_patterns"):
    """Plot decoder patterns as a topographic joint plot (EvokedArray.plot_joint).

    One figure per contrast.  Patterns are mapped back to sensor space via
    ``get_coef(..., inverse_transform=True)`` (already done upstream) so the
    EvokedArray inherits the full channel info and montage.

    Parameters
    ----------
    results : dict
        Mapping contrast_name -> result dict containing at least
        ``patterns`` (n_channels x n_times), ``info``, and ``times``.
    subject : str
        Subject label used in the figure title and filename.
    out_dir : Path
        Directory where figures are written.
    save_formats : list of str
        File format extensions (e.g. ["png", "pdf"]).
    pattern_times : array-like or None
        Time points (in seconds) shown as topomaps in plot_joint.
        None / not set → MNE auto-selects representative times.
    file_prefix : str
        Filename prefix distinguishing within- vs cross-condition patterns.
    """
    times_arg = pattern_times if pattern_times is not None else "auto"
    joint_kwargs = dict(ts_args=dict(time_unit="s"), topomap_args=dict(time_unit="s"))

    for cname, res in results.items():
        if "patterns" not in res or "info" not in res:
            continue
        patterns = res["patterns"]   # (n_channels, n_times)
        info = res["info"]
        times = res["times"]

        evoked = mne.EvokedArray(patterns, info, tmin=times[0])
        try:
            fig = evoked.plot_joint(
                times=times_arg,
                title=f"Patterns: {cname} -- {subject}",
                show=False,
                **joint_kwargs,
            )
        except Exception as e:
            print(f"      WARNING: pattern plot_joint failed for {cname}: {e}")
            plt.close("all")
            continue

        cname_san = sanitize_cond_name(cname)
        save_fig(fig, out_dir / f"{subject}__{file_prefix}_{cname_san}", save_formats)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Per-subject orchestration
# ---------------------------------------------------------------------------

def process_subject(cfg):
    """Load epochs, run all decoding analyses, save results and plots.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration with all _decoder_* attributes and BIDS paths.
    """
    subject = cfg.subjects[0]
    session = cfg.sessions[0]

    print(f"\n{'=' * 60}")
    print(f"  Processing decoding: {subject}")
    print(f"{'=' * 60}")

    # Construct base BIDS path
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        root=cfg.deriv_root,
        datatype="meg",
        check=False,
    )

    # Load clean epochs
    epochs_path = bids_path.copy().update(
        suffix="epo", processing="clean", extension=".fif"
    )
    if not epochs_path.fpath.exists():
        raise FileNotFoundError(
            f"Clean epochs not found at {epochs_path.fpath}\n"
            f"Run preprocessing first."
        )
    print(f"  Loading epochs: {epochs_path.fpath}")
    epochs = mne.read_epochs(epochs_path, preload=True)

    if cfg._decoder_baseline is not None:
        epochs.apply_baseline(cfg._decoder_baseline)

    # Extract config params
    contrasts = cfg._decoder_contrasts
    cross_contrasts = cfg._decoder_cross_contrasts
    scoring = cfg._decoder_scoring
    n_jobs = cfg._decoder_n_jobs_inner
    decim = cfg._decoder_decim
    run_tg = cfg._decoder_run_temporal_gen
    save_formats = cfg._decoder_save_formats
    chance = cfg._decoder_chance
    group_column = cfg._decoder_group_column
    time_n_components = getattr(cfg, "_decoder_time_n_components", "rank")
    epoch_n_components = getattr(cfg, "_decoder_epoch_n_components", 0.99)
    pattern_times = getattr(cfg, "_decoder_pattern_times", None)

    # Output directory for all decoding results (data files + figures)
    decoding_dir = epochs_path.fpath.parent / "decoding"
    decoding_dir.mkdir(parents=True, exist_ok=True)

    # Result accumulators (for plotting)
    time_results = {}
    epoch_results = {}
    tg_results = {}
    cross_time_results = {}
    cross_epoch_results = {}
    cross_tg_results = {}

    # ---- Within-condition contrasts ----
    for contrast in contrasts:
        cname = contrast["name"]
        cond_1, cond_2 = contrast["conditions"]
        print(f"\n  {subject} -- {cname}  ({cond_1}  vs  {cond_2})")

        # Time-resolved decoding
        print(f"  [time decoding]")
        t_res = run_subject_time_decoding(
            epochs, contrast,
            scoring=scoring, n_jobs=n_jobs, decim=decim,
            group_column=group_column,
            time_n_components=time_n_components,
        )
        if t_res is not None:
            save_time_results(bids_path, contrast, t_res, scoring, decoding_dir)
            time_results[cname] = t_res

        # Temporal generalization
        if run_tg:
            print(f"  [temporal generalization]")
            tg_res = run_subject_temporal_gen(
                epochs, contrast,
                scoring=scoring, n_jobs=n_jobs, decim=decim,
                group_column=group_column,
                time_n_components=time_n_components,
            )
            if tg_res is not None:
                save_tg_results(bids_path, contrast, tg_res, decoding_dir)
                tg_results[cname] = tg_res

        # Full-epoch decoding
        print(f"  [epoch decoding]")
        e_res = run_subject_epoch_decoding(
            epochs, contrast,
            scoring=scoring,
            group_column=group_column,
            epoch_n_components=epoch_n_components,
        )
        if e_res is not None:
            save_epoch_results(bids_path, contrast, e_res, scoring, decoding_dir)
            epoch_results[cname] = e_res

    # ---- Cross-condition contrasts ----
    for cc in cross_contrasts:
        ccname = cc["name"]
        print(f"\n  {subject} -- cross: {ccname}")
        cc_res = run_subject_cross_decoding(
            epochs, cc,
            scoring=scoring, n_jobs=n_jobs, decim=decim,
            group_column=group_column,
            time_n_components=time_n_components,
            epoch_n_components=epoch_n_components,
        )
        if cc_res is None:
            continue

        if "time" in cc_res:
            save_cross_time_results(bids_path, cc, cc_res["time"], scoring, decoding_dir)
            # Add metadata for plot labels
            cc_res["time"]["train_name"] = cc_res["train_name"]
            cc_res["time"]["test_name"] = cc_res["test_name"]
            cross_time_results[ccname] = cc_res["time"]

        if "epoch" in cc_res:
            save_cross_epoch_results(bids_path, cc, cc_res["epoch"], scoring, decoding_dir)
            cross_epoch_results[ccname] = cc_res["epoch"]

        if "tg" in cc_res:
            save_cross_tg_results(bids_path, cc, cc_res, cc_res["tg"], decoding_dir)
            # Add metadata for plot labels
            cc_res["tg"]["train_name"] = cc_res["train_name"]
            cc_res["tg"]["test_name"] = cc_res["test_name"]
            cross_tg_results[ccname] = cc_res["tg"]

    # ---- Per-subject plots ----
    print(f"\n  Plotting individual results for {subject} ...")
    if time_results:
        plot_subject_time_ribbon(time_results, subject, decoding_dir,
                                  save_formats, chance)
        plot_subject_time_patterns(time_results, subject, decoding_dir,
                                   save_formats, pattern_times)
    if epoch_results:
        plot_subject_epoch_bar(epoch_results, subject, decoding_dir,
                                save_formats, chance)
    if tg_results:
        plot_subject_tg_heatmap(tg_results, subject, decoding_dir, save_formats)
    if cross_time_results:
        plot_subject_cross_ribbon(cross_time_results, subject, decoding_dir,
                                   save_formats, chance)
        plot_subject_time_patterns(cross_time_results, subject, decoding_dir,
                                   save_formats, pattern_times,
                                   file_prefix="cross_time_patterns")
    if cross_epoch_results:
        plot_subject_cross_epoch_bar(cross_epoch_results, subject, decoding_dir,
                                      save_formats, chance)
    if cross_tg_results:
        plot_subject_cross_tg_heatmap(cross_tg_results, subject, decoding_dir,
                                        save_formats)

    del epochs
    print(f"\n  Done: {subject}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Multivariate decoding (MVPA) analysis for OPM-MEG data"
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    return p.parse_args()


def main():
    """Main entry point for decoding analysis."""

    args = parse_args()

    # Load configuration (matching run_beamformer.py pattern)
    cfg = _import_config(config_path=args.config)
    _update_config_from_path(config=cfg, config_path=args.config)
    cfg.data_type = "meg"
    cfg.datatype = "meg"

    # Check master switch
    if not getattr(cfg, "_run_decoding", False):
        print("\n[main] Decoding disabled in configuration (_run_decoding=False)")
        print("[main] Exiting without running analysis.")
        return

    # Run decoding
    process_subject(cfg)

    print("\n" + "=" * 60)
    print("DECODING ANALYSIS COMPLETE")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
