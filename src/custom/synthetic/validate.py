"""Check that a beamformer recovers the dipoles planted in a synthetic subject.

The point of simulating through a forward model is that source analyses have a
correct answer.  This module makes that answer checkable: it fits an LCMV
beamformer to a subject's cleaned epochs and reports, per ground-truth dipole,
how far the peak of the source map sits from where the dipole actually is.

It is deliberately independent of ``run_beamformer.py`` -- it builds its own
filters from the derivatives -- so it can be used as a regression check on
coregistration, forward modelling and spatial filtering without also depending
on the beamformer configuration under test.

Usage::

    python -m custom.synthetic.validate \\
        --bids synthetic/datasets/synth/bids \\
        --deriv synthetic/datasets/synth/bids/derivatives/trial__<version>

Author: Harrison Ritz (2025)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


__all__ = ["localization_errors", "main"]


def _stc_positions(stc, src) -> np.ndarray:
    """Head-frame position of every row of ``stc.data``.

    Worth spelling out, because getting it wrong is silent and expensive:
    ``make_forward_solution(mindist=...)`` drops sources that sit too close to
    the inner skull, so a forward's source space holds *fewer* vertices than
    the ``bem/*-src.fif`` it was built from.  Indexing source-estimate rows
    into the unpruned source space therefore misaligns them -- on this phantom
    by up to 90 mm, which looks exactly like a beamformer that cannot localise.
    Always go through ``stc.vertices``.
    """
    positions = []
    for hemi, space in enumerate(src):
        vertices = stc.vertices[hemi]
        index = np.searchsorted(space["vertno"], vertices)
        if not np.array_equal(space["vertno"][index], vertices):
            raise ValueError(
                "Source estimate vertices are not a subset of the forward's "
                "source space; they do not describe the same source space."
            )
        positions.append(space["rr"][vertices])
    return np.vstack(positions)


def _load_epochs(deriv_root: Path, subject: str, session: str, task: str):
    """Load the cleanest epochs the pipeline produced for this subject."""
    import mne

    meg_dir = deriv_root / f"sub-{subject}" / f"ses-{session}" / "meg"
    for suffix in ("proc-clean_epo", "proc-ica_epo", "epo"):
        matches = sorted(meg_dir.glob(f"sub-{subject}_ses-{session}_task-{task}_{suffix}.fif"))
        if matches:
            return mne.read_epochs(matches[0], preload=True, verbose="error"), matches[0]
    raise FileNotFoundError(
        f"No epochs found under {meg_dir}. Run the preprocessing pipeline first."
    )


def localization_errors(
    bids_root: Path | str,
    deriv_root: Path | str,
    *,
    subject: str = "001",
    session: str = "01",
    task: str = "synth",
    reg: float = 0.05,
) -> dict[str, float]:
    """Distance from each ground-truth dipole to the beamformer's peak.

    Parameters
    ----------
    bids_root : path-like
        Synthetic BIDS root (the one containing ``ground_truth.json``).
    deriv_root : path-like
        Derivatives directory the pipeline wrote for this analysis.
    subject, session, task : str
        Which recording to check.
    reg : float
        LCMV regularisation.

    Returns
    -------
    errors : dict
        Source name -> distance in millimetres, evaluated at that source's
        ground-truth peak latency.

    Notes
    -----
    Everything is done in **head** coordinates: a forward solution's source
    space is in the head frame, and ``ground_truth.json`` records
    ``position_head_m`` for exactly this comparison.  Mixing that up with the
    MRI-surface-RAS positions costs about 15 mm on this phantom, which is the
    same order as the effect being measured.
    """
    import mne
    from mne.beamformer import apply_lcmv, make_lcmv

    bids_root, deriv_root = Path(bids_root), Path(deriv_root)
    truth = json.loads((bids_root / "ground_truth.json").read_text())
    sources = truth["subjects"][subject]["sources"]

    epochs, epochs_path = _load_epochs(deriv_root, subject, session, task)
    meg_dir = epochs_path.parent
    fwd = mne.read_forward_solution(
        meg_dir / f"sub-{subject}_ses-{session}_task-{task}_fwd.fif", verbose="error"
    )

    data_cov = mne.compute_covariance(
        epochs, tmin=0.0, tmax=None, method="shrunk", verbose="error"
    )
    noise_cov = mne.compute_covariance(
        epochs, tmin=None, tmax=0.0, method="shrunk", verbose="error"
    )
    filters = make_lcmv(
        epochs.info,
        fwd,
        data_cov,
        reg=reg,
        noise_cov=noise_cov,
        pick_ori="max-power",
        weight_norm="unit-noise-gain",
        rank="info",
        verbose="error",
    )

    stc = apply_lcmv(epochs.average(), filters, verbose="error")
    positions = _stc_positions(stc, fwd["src"])

    errors = {}
    for source in sources:
        index = int(np.argmin(np.abs(stc.times - source["latency_s"])))
        peak = int(np.argmax(np.abs(stc.data[:, index])))
        truth_pos = np.asarray(source["position_head_m"])
        errors[source["name"]] = float(
            np.linalg.norm(positions[peak] - truth_pos) * 1e3
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids", required=True, type=Path, help="Synthetic BIDS root.")
    parser.add_argument(
        "--deriv", required=True, type=Path, help="Pipeline derivatives directory."
    )
    parser.add_argument("--subject", default="001")
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="synth")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=20.0,
        help="Fail if any source is further than this (mm). Default: %(default)s.",
    )
    args = parser.parse_args(argv)

    errors = localization_errors(
        args.bids,
        args.deriv,
        subject=args.subject,
        session=args.session,
        task=args.task,
    )

    print("\nLCMV localisation vs ground truth")
    print("-" * 46)
    worst = 0.0
    for name, error in errors.items():
        flag = "ok " if error <= args.tolerance else "FAIL"
        worst = max(worst, error)
        print(f"  [{flag}] {name:20s} {error:6.1f} mm")
    print("-" * 46)
    print(f"  worst: {worst:.1f} mm (tolerance {args.tolerance:.0f} mm)\n")
    return 0 if worst <= args.tolerance else 1


if __name__ == "__main__":
    sys.exit(main())
