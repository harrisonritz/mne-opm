"""Synthetic group template, named ``fsaverage``.

Group-level source analyses morph every subject onto ``fsaverage``.  Fetching
the real one costs a network round trip and a few hundred megabytes, which
defeats the point of a dataset meant to work from a clean clone with no
downloads, so the generator writes its own template *into the synthetic
subjects directory* under that name.

The shadowing is deliberate and strictly local: only code that passes
``subjects_dir=<synthetic>/derivatives/freesurfer/subjects`` sees it, and a
real ``fsaverage`` elsewhere on the machine is untouched.  Point
``subjects_dir`` at the real one if you want the real one.

Because every synthetic subject and the template share the same icosahedral
tessellation, ``?h.sphere.reg`` is the identity map and surface morphing is
exact -- a convenient property for testing morph code, and one worth
remembering when interpreting results (real morphs are not exact).

Author: Harrison Ritz (2025)
"""

from __future__ import annotations

from pathlib import Path


__all__ = ["write_group_template"]

TEMPLATE_NAME = "fsaverage"


def write_group_template(subjects_dir: Path | str, seed: int = 0) -> Path:
    """Write the ``fsaverage`` template subject.

    Parameters
    ----------
    subjects_dir : path-like
        FreeSurfer subjects directory to write into.
    seed : int
        Seed of the dataset; the template always uses the canonical
        (un-jittered) head so that it sits at the centre of any cohort.

    Returns
    -------
    template_dir : Path
    """
    import mne

    from .anatomy import build_head_model, write_freesurfer_subject

    subjects_dir = Path(subjects_dir)
    head = build_head_model(seed=seed, jitter=0.0)
    template_dir = write_freesurfer_subject(head, subjects_dir, TEMPLATE_NAME)

    # mne-bids-pipeline's ``use_template_mri="fsaverage"`` branch reads a
    # ready-made trans from the template's bem/ directory; provide one so that
    # path works too.  For the phantom the head frame is defined by the same
    # fiducials, so this is just the inverse of the MRI->head transform.
    mne.write_trans(
        template_dir / "bem" / f"{TEMPLATE_NAME}-trans.fif",
        mne.transforms.invert_transform(head.mri_head_t),
        overwrite=True,
    )

    # The real fsaverage ships a precomputed BEM solution.  We deliberately do
    # not: at ico4 the solution matrix is 2562 x 2562, which is 26 MB on disk —
    # more than the rest of the dataset combined — and `mne.make_bem_solution`
    # rebuilds it from the shipped `bem/*.surf` in a couple of seconds.
    return template_dir
