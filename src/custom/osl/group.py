"""Group-level stage of the osl-ephys OPM pipeline.

Runs after the per-subject array finishes:

1. Pick a template subject and sign-flip every subject's parcel time courses
   against it (:mod:`custom.osl.sign_flip`), optionally in parallel with Dask.
2. Stack the sign-flipped parcel epochs into a
   ``(subjects, parcels, times)`` array per condition.
3. Compute the contrasts declared in the config, and write summary figures.

Sign flipping comes first because a beamformer resolves each dipole's
orientation only up to a sign, independently per subject: averaging parcel time
courses across subjects without fixing that cancels the signal.

Outputs land in ``{outdir}/group/``:

===============================  ============================================
``template_subject.txt``         The template used, and the flip counts
``group_parcel_evoked.npz``      Per-condition ``(subjects, parcels, times)``
``group_contrasts.npz``          Per-contrast ``(subjects, parcels, times)``
``sign_flip_summary.png``        Flips per subject, template match metric
``contrast_<name>.png``          Group-mean contrast, per parcel
===============================  ============================================

Functions
---------
run
    Run the group stage.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np

from . import sign_flip
from ._paths import resolve_paths


DEFAULT_SIGN_FLIP: dict = {
    "n_embeddings": 15,
    "standardize": True,
    "n_init": 3,
    "n_iter": 2500,
    "max_flips": 20,
}
"""Sign-flipping defaults, matching the osl-ephys examples."""


def run(cfg: SimpleNamespace) -> bool:
    """Run the group stage.

    Parameters
    ----------
    cfg : SimpleNamespace
        Config from :func:`custom.osl._config.load_config`.  Reads the
        ``group`` section; see the module docstring for what it produces.

    Returns
    -------
    success : bool
        True if the group outputs were written.

    Raises
    ------
    ValueError
        If the config has no ``group`` section, or fewer than two subjects have
        parcellated data.
    """
    group_cfg = dict(cfg.group or {})
    if not group_cfg:
        raise ValueError(
            f"No 'group' section in {cfg.path}; cannot run the group stage."
        )

    pipeline = cfg.pipeline
    paths = resolve_paths(pipeline)
    outdir = paths.outdir
    groupdir = outdir / "group"
    os.makedirs(groupdir, exist_ok=True)

    epoched = pipeline.source_input == "epochs"
    if not epoched:
        # Checked up front rather than after sign flipping, which works fine on
        # continuous data and would otherwise burn an hour before the condition
        # averaging discovered it had no epochs to average.
        raise ValueError(
            f"The group stage averages epochs by condition, so it needs "
            f"pipeline.source_input: epochs, not {pipeline.source_input!r}."
        )

    source_method = group_cfg.get("source_method", "lcmv")

    subjects = _resolve_subjects(group_cfg, pipeline, outdir, epoched, source_method)
    print(f"[osl:group] {len(subjects)} subject(s) with parcellated data")

    # --- 1. Sign flipping ---
    results = _sign_flip_all(
        outdir, subjects, group_cfg, epoched, source_method, groupdir
    )
    flipped = [r["subject"] for r in results if r["error"] is None]
    if len(flipped) < 2:
        raise ValueError(
            f"Sign flipping succeeded for only {len(flipped)} subject(s); "
            f"cannot run a group analysis."
        )

    # --- 2 & 3. Stack, contrast, plot ---
    conditions = group_cfg.get("conditions")
    stacked, times, parcel_names = _stack_conditions(
        outdir, flipped, conditions, epoched, source_method, group_cfg
    )

    np.savez_compressed(
        groupdir / "group_parcel_evoked.npz",
        subjects=np.array(flipped),
        times=times,
        parcels=np.array(parcel_names),
        **stacked,
    )
    print(f"[osl:group] wrote {groupdir / 'group_parcel_evoked.npz'}")

    contrasts = _compute_contrasts(stacked, group_cfg.get("contrasts") or [])
    if contrasts:
        np.savez_compressed(
            groupdir / "group_contrasts.npz",
            subjects=np.array(flipped),
            times=times,
            parcels=np.array(parcel_names),
            **contrasts,
        )
        print(f"[osl:group] wrote {groupdir / 'group_contrasts.npz'}")

    _plot(groupdir, results, stacked, contrasts, times)

    return True


# ---------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------


def _resolve_subjects(
    group_cfg: dict,
    pipeline: SimpleNamespace,
    outdir: Path,
    epoched: bool,
    source_method: str,
) -> list[str]:
    """Return the subjects to include, checking each has parcellated data."""
    subjects = group_cfg.get("subjects")

    if subjects:
        labels = [
            pipeline.subject_label.format(
                subject=str(s), session=pipeline.session, task=pipeline.task
            )
            if "{" in pipeline.subject_label
            else str(s)
            for s in subjects
        ]
    else:
        # Discover from the output tree: any subject directory with parcel data.
        labels = sorted(
            child.name for child in outdir.iterdir() if (child / "parc").is_dir()
        )
        print(f"[osl:group] discovered {len(labels)} subject director(ies)")

    found = sign_flip.available_subjects(outdir, labels, epoched, source_method)
    if len(found) < 2:
        raise ValueError(
            f"Sign flipping needs two or more subjects with parcellated data "
            f"under {outdir}, found {len(found)}. Has the source stage run?"
        )
    return found


# ---------------------------------------------------------------------------
# Sign flipping
# ---------------------------------------------------------------------------


def _sign_flip_all(
    outdir: Path,
    subjects: Sequence[str],
    group_cfg: dict,
    epoched: bool,
    source_method: str,
    groupdir: Path,
) -> list[dict]:
    """Find the template and flip every subject against it."""
    options = {**DEFAULT_SIGN_FLIP, **(group_cfg.get("sign_flip") or {})}

    template = group_cfg.get("template")
    if template:
        print(f"[osl:group] using the configured template subject: {template}")
    else:
        template = sign_flip.find_template(
            outdir,
            subjects,
            n_embeddings=options["n_embeddings"],
            standardize=options["standardize"],
            epoched=epoched,
            source_method=source_method,
        )
        print(f"[osl:group] template subject: {template}")

    work = dict(
        outdir=str(outdir),
        template=template,
        epoched=epoched,
        source_method=source_method,
        **options,
    )

    n_workers = int(group_cfg.get("n_workers", 1) or 1)
    if n_workers > 1:
        results = _flip_with_dask(subjects, work, n_workers)
    else:
        results = [
            sign_flip.flip_subject(subject=s, **work) for s in subjects
        ]

    failed = [r for r in results if r["error"] is not None]
    for result in failed:
        print(f"[osl:group] sign flipping FAILED for {result['subject']}: "
              f"{result['error']}")

    summary = {
        "template": template,
        "options": options,
        "epoched": epoched,
        "source_method": source_method,
        "subjects": {
            r["subject"]: {"n_flipped": r["n_flipped"], "error": r["error"]}
            for r in results
        },
    }
    with open(groupdir / "template_subject.txt", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[osl:group] sign flipped {len(results) - len(failed)}/{len(results)} "
        f"subject(s)"
    )
    return results


def _flip_with_dask(subjects: Sequence[str], work: dict, n_workers: int) -> list[dict]:
    """Run the sign-flip search across a local Dask cluster.

    Each subject's search is independent and CPU-bound, so this is the same
    parallelisation osl-ephys' own examples use, kept inside one SLURM job.
    """
    from dask.distributed import Client

    print(f"[osl:group] sign flipping with {n_workers} dask worker(s)")
    with Client(n_workers=n_workers, threads_per_worker=1, dashboard_address=None):
        from dask import compute, delayed

        tasks = [
            delayed(sign_flip.flip_subject)(subject=s, **work) for s in subjects
        ]
        return list(compute(*tasks))


# ---------------------------------------------------------------------------
# Stacking and contrasts
# ---------------------------------------------------------------------------


def _stack_conditions(
    outdir: Path,
    subjects: Sequence[str],
    conditions: Optional[Sequence[str]],
    epoched: bool,
    source_method: str,
    group_cfg: dict,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Stack each subject's per-condition parcel average.

    Returns
    -------
    stacked : dict
        Condition name -> ``(subjects, parcels, times)``.  A subject missing a
        condition contributes NaN, so the array stays rectangular and the
        gap is visible rather than silently dropped.
    times : numpy.ndarray
        Epoch time axis, in seconds.
    parcel_names : list of str
        Parcel channel names.
    """
    import mne

    baseline = group_cfg.get("baseline")
    if baseline is not None:
        baseline = tuple(baseline)

    per_subject: dict[str, dict[str, np.ndarray]] = {}
    times = None
    parcel_names: list[str] = []

    for subject in subjects:
        path = sign_flip.sflip_file(outdir, subject, epoched, source_method)
        epochs = mne.read_epochs(path, preload=True, verbose=False)

        epochs = epochs.pick(sign_flip.parcel_channels(epochs))

        if baseline is not None:
            epochs.apply_baseline(baseline, verbose=False)

        if times is None:
            times = epochs.times
            parcel_names = list(epochs.ch_names)

        wanted = conditions or sorted(epochs.event_id)
        per_subject[subject] = {}
        for condition in wanted:
            try:
                per_subject[subject][condition] = (
                    epochs[condition].average(picks="all").data
                )
            except KeyError:
                print(
                    f"[osl:group] {subject}: no epochs for '{condition}', "
                    f"filling with NaN"
                )

        print(
            f"[osl:group] {subject}: {len(epochs)} epochs, "
            f"{len(per_subject[subject])}/{len(wanted)} condition(s)"
        )

    all_conditions = conditions or sorted(
        {c for subject in per_subject.values() for c in subject}
    )
    n_parcels, n_times = len(parcel_names), len(times)

    stacked = {}
    for condition in all_conditions:
        block = np.full((len(subjects), n_parcels, n_times), np.nan)
        for index, subject in enumerate(subjects):
            data = per_subject[subject].get(condition)
            if data is not None:
                block[index] = data
        stacked[_safe_key(condition)] = block

    return stacked, times, parcel_names


def _safe_key(name: str) -> str:
    """Make a condition name usable as an npz key."""
    return name.replace("/", "__")


def _compute_contrasts(
    stacked: dict[str, np.ndarray], contrasts: Sequence[dict]
) -> dict[str, np.ndarray]:
    """Form weighted combinations of the stacked condition averages."""
    out: dict[str, np.ndarray] = {}

    for contrast in contrasts:
        name = contrast["name"]
        conditions = [_safe_key(c) for c in contrast["conditions"]]
        weights = contrast["weights"]

        missing = [c for c in conditions if c not in stacked]
        if missing:
            print(
                f"[osl:group] contrast '{name}' skipped: no data for "
                f"{[m.replace('__', '/') for m in missing]}"
            )
            continue

        if len(conditions) != len(weights):
            print(
                f"[osl:group] contrast '{name}' skipped: {len(conditions)} "
                f"condition(s) but {len(weights)} weight(s)"
            )
            continue

        total = sum(
            weight * stacked[condition]
            for condition, weight in zip(conditions, weights, strict=True)
        )
        out[_safe_key(name)] = total
        print(f"[osl:group] contrast '{name}': {total.shape}")

    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot(
    groupdir: Path,
    results: Sequence[dict],
    stacked: dict[str, np.ndarray],
    contrasts: dict[str, np.ndarray],
    times: np.ndarray,
) -> None:
    """Write the sign-flip summary and per-contrast group figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        _plot_sign_flip(groupdir, results, plt)
        for name, data in {**stacked, **contrasts}.items():
            _plot_parcels(groupdir, name, data, times, plt)
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"[osl:group] could not render a figure: {exc}")


def _plot_sign_flip(groupdir: Path, results: Sequence[dict], plt) -> None:
    """Flips per subject, and the template match metric across initialisations."""
    ok = [r for r in results if r["error"] is None]
    if not ok:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].bar(range(len(ok)), [r["n_flipped"] for r in ok])
    axes[0].set_xticks(range(len(ok)))
    axes[0].set_xticklabels([r["subject"] for r in ok], rotation=90, fontsize=7)
    axes[0].set_ylabel("parcels flipped")
    axes[0].set_title("Sign flips per subject")

    for result in ok:
        if result["metrics"]:
            axes[1].plot(result["metrics"], alpha=0.6, marker="o", markersize=3)
    axes[1].set_xlabel("initialisation")
    axes[1].set_ylabel("covariance correlation with template")
    axes[1].set_title("Template match")

    fig.tight_layout()
    fig.savefig(groupdir / "sign_flip_summary.png", dpi=150)
    plt.close(fig)
    print(f"[osl:group] wrote {groupdir / 'sign_flip_summary.png'}")


def _plot_parcels(
    groupdir: Path, name: str, data: np.ndarray, times: np.ndarray, plt
) -> None:
    """Group mean per parcel, with the across-parcel mean +/- s.e.m. over subjects."""
    group_mean = np.nanmean(data, axis=0)  # parcels x times

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(times, group_mean.T, lw=0.6, alpha=0.5)
    axes[0].axvline(0, color="k", lw=0.8, ls="--")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("parcel amplitude")
    axes[0].set_title(f"{name.replace('__', '/')}: group mean, each parcel")

    across_parcels = np.nanmean(data, axis=1)  # subjects x times
    mean = np.nanmean(across_parcels, axis=0)
    n = np.sum(~np.isnan(across_parcels[:, 0]))
    sem = np.nanstd(across_parcels, axis=0) / max(np.sqrt(n), 1)

    axes[1].plot(times, mean, color="C0")
    axes[1].fill_between(times, mean - sem, mean + sem, alpha=0.3, color="C0")
    axes[1].axvline(0, color="k", lw=0.8, ls="--")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("amplitude")
    axes[1].set_title(f"mean over parcels (n={int(n)} subjects, +/- s.e.m.)")

    fig.tight_layout()
    path = groupdir / f"contrast_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
