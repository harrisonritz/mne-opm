#!/usr/bin/env python
"""Generate a synthetic, BIDS-formatted OPM dataset for development.

The repository ships one subject under ``synthetic/datasets/synth`` so that a
fresh clone can run the pipeline without any real data.  This script is what
built it, and what you use to build more -- a larger cohort for group-level
work, a longer recording, a different sensor count.

Examples
--------
Regenerate the committed subject in place (deterministic; the result should be
identical modulo FIF timestamps)::

    python src/custom/make_synthetic.py --out synthetic/datasets/synth

A twelve-subject cohort for group-level development, somewhere scratch::

    python src/custom/make_synthetic.py --out /tmp/synth-cohort --n-subjects 12

Keep the pre-BIDS Cerca-style tree so ``format_bids.py`` / ``run_bids.sh`` can
be exercised against it::

    python src/custom/make_synthetic.py --out /tmp/synth --keep-raw

Then point the pipeline at the result::

    sh mne-opm.sh preproc --exp synth --sub 001 \\
        --data synthetic/datasets --config synthetic/config \\
        --analysis trial --session 01 \\
        --subjects-dir synthetic/datasets/synth/bids/derivatives/freesurfer/subjects

Author: Harrison Ritz (2025)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``custom.*`` importable when run as a script from the repository root.
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from custom.synthetic import DatasetSpec, make_dataset  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Dataset root; 'raw/' and 'bids/' are created underneath.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Explicit subject IDs (e.g. --subjects 001 002). Overrides --n-subjects.",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=1,
        help="Generate subjects 1..N. Default: 1.",
    )
    parser.add_argument("--task", default=DatasetSpec.task, help="BIDS task name.")
    parser.add_argument("--session", default=DatasetSpec.session, help="BIDS session.")
    parser.add_argument(
        "--duration",
        type=float,
        default=DatasetSpec.duration,
        help="Task recording length in seconds. Default: %(default)s.",
    )
    parser.add_argument(
        "--noise-duration",
        type=float,
        default=DatasetSpec.noise_duration,
        help="Empty-room length in seconds. Default: %(default)s.",
    )
    parser.add_argument(
        "--sfreq",
        type=float,
        default=DatasetSpec.sfreq,
        help="Sampling frequency in Hz. Default: %(default)s.",
    )
    parser.add_argument(
        "--n-slots",
        type=int,
        default=DatasetSpec.n_slots,
        help=(
            "Triaxial sensor slots; the array has 3x this many magnetometers. "
            "Maxwell filtering with mf_int_order=10 needs at least ~128 "
            "channels, i.e. 43 slots. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--line-freq",
        type=float,
        default=DatasetSpec.line_freq,
        help="Power-line frequency in Hz. Default: %(default)s.",
    )
    parser.add_argument(
        "--seed", type=int, default=DatasetSpec.seed, help="Master random seed."
    )
    parser.add_argument(
        "--head-jitter",
        type=float,
        default=DatasetSpec.head_jitter,
        help=(
            "Relative head-size variation across subjects. The first subject is "
            "always the un-jittered reference. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the pre-BIDS FIFs (regenerable, so discarded by default).",
    )
    parser.add_argument(
        "--no-template",
        action="store_true",
        help="Skip writing the synthetic 'fsaverage' group template.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)

    spec = DatasetSpec(
        task=args.task,
        session=args.session,
        sfreq=args.sfreq,
        duration=args.duration,
        noise_duration=args.noise_duration,
        n_slots=args.n_slots,
        line_freq=args.line_freq,
        seed=args.seed,
        head_jitter=args.head_jitter,
        keep_raw=args.keep_raw,
    )

    if spec.n_slots * 3 < 128:
        print(
            f"[make_synthetic] NOTE: {spec.n_slots * 3} magnetometers is below the "
            f"128 basis vectors that mf_int_order=10 / mf_ext_order=2 needs. "
            f"Maxwell filtering will fail on this dataset; use HFC, or raise "
            f"--n-slots to at least 43."
        )

    summary = make_dataset(
        args.out,
        subjects=args.subjects,
        spec=spec,
        n_subjects=args.n_subjects,
        write_template=not args.no_template,
    )

    print("\n" + "=" * 70)
    print(f"Synthetic dataset written to {args.out}")
    for label, gt in summary["subjects"].items():
        srcs = ", ".join(s["name"] for s in gt["sources"])
        print(f"  sub-{label}: {gt['n_trials']} trials | ground-truth sources: {srcs}")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
