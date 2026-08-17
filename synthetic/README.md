# Synthetic OPM dataset

A complete, fully synthetic OPM-MEG subject in BIDS format, committed so that a
fresh clone can run the whole pipeline — preprocessing, source modelling,
beamforming — with no real data and no downloads.

There is **no human data here**. The anatomy is an analytic phantom and the
recordings are simulated through a BEM forward model computed on it.

```
synthetic/
├── config/synth/                  ready-to-run pipeline configuration
│   ├── config-trial.py            mne-bids-pipeline + mne-opm settings
│   └── bids/                      format_bids conversion configs
└── datasets/synth/
    ├── raw/synth_001/metadata/    per-trial behavioural CSV
    └── bids/
        ├── sub-001/ses-01/meg/    task + empty-room recordings
        ├── sub-001/ses-01/anat/   T1w with anatomical landmarks
        ├── ground_truth.json      simulated dipoles and planted artifacts
        └── derivatives/freesurfer/subjects/
            ├── sub-001_ses-01/    phantom "recon"
            └── fsaverage/         group template for morphing
```

## Running the pipeline

```bash
bash mne-opm.sh preproc \
    --exp synth --sub 001 --session 01 --analysis trial \
    --data synthetic/datasets --config synthetic/config
```

`SUBJECTS_DIR` may be left unset: `config-trial.py` defaults it to the
FreeSurfer subjects directory inside the dataset. Individual stages work the
same way — swap `preproc` for `coreg`, `sensor`, `source` or `beamformer`.

To drive `mne-bids-pipeline` directly:

```bash
mne_bids_pipeline --steps=preprocessing --config=synthetic/config/synth/config-trial.py
```

## What is in the recording

| | |
|---|---|
| Array | 48 Cerca-style triaxial slots = **144 magnetometers**, QuSpin coil type, `"C6 2B Y"`-style names |
| Sampling | 200 Hz, 100 s task + 40 s empty room |
| Aux channels | `eye_nmf1-3` (EOG), `xpos_right`/`ypos_right`/`pupil_right`, `x_head`/`y_head`/`distance` (misc) |
| Design | 38 trials, two conditions (`trial/cond_a`, `trial/cond_b`), left/right responses, `feedback`, `ITI`, and a 9 s rest between two blocks |
| Anatomy | ellipsoidal head/skull/scalp with a deeply folded cortical surface; `oct6` source space (4098 sources per hemisphere) |

Two of those numbers are sized for Maxwell filtering rather than chosen freely:

- **144 magnetometers** — `mf_int_order = 10` plus `mf_ext_order = 2` needs 128
  basis vectors, so an array below ~130 channels makes `maxwell_filter`
  ill-posed. 144 leaves headroom for bad channels.
- **40 s of empty room** — `process_empty_room` runs the same spatial filter on
  the noise recording, and tSSS refuses a `mf_st_duration` longer than the
  data. A covariance estimate alone would be happy with far less.

Both `HFC` (the default) and `maxwell` run cleanly on this subject; switch with
`SYNTH_SPATIAL_FILTER=maxwell`, which also changes the derivatives directory
name so the two do not overwrite each other.

### Signal content

Data is simulated through a real forward solution, so beamforming has a correct
answer to find. `ground_truth.json` records it:

- **three cortical dipoles** — a left temporal source (stronger in condition A),
  a right parietal source (stronger in condition B) and a shared occipital
  source, each with its own peak latency. They sit on sulcal walls: a dipole
  oriented radially in a near-spherical conductor is magnetically almost
  silent, so the generator picks, near each anatomical target, the vertex whose
  forward column has the largest norm;
- **ongoing cortical activity** (1/f) plus posterior **alpha**, so the PSD looks
  plausible;
- **ocular** artifacts from eyeball dipoles, mirrored into the EOG channels, and
  a **cardiac** artifact from a distant dipole — real targets for ICA;
- **environmental interference**: a uniform field plus a first-order gradient,
  with a 60 Hz line component. This is what HFC and Maxwell filtering remove,
  and it is present in the empty-room recording too, which is what makes
  `noise_cov = "emptyroom"` meaningful;
- **planted defects**: two high-variance channels, one dead channel and two
  broadband bursts, so the bad-channel and bad-segment steps find something.
  On the shipped config the bad-channel step recovers all three planted
  channels exactly.

The evoked response is deliberately the *smallest* brain component — it should
emerge from averaging, not dominate single trials. That is not just realism: an
evoked response large enough to be visible per-trial gets picked up by ICA as a
large, repeatable, non-Gaussian component and removed as an artifact, taking
the signal the beamformer is meant to find with it.

### Checking it works

```bash
python -m custom.synthetic.validate \
    --bids synthetic/datasets/synth/bids \
    --deriv synthetic/datasets/synth/bids/derivatives/trial__<version>
```

This fits its own LCMV beamformer to the pipeline's cleaned epochs and reports
how far each source map's peak is from the true dipole, at that dipole's
labelled latency. On the shipped config all three land within ~8 mm (all three
exactly, with `SYNTH_SPATIAL_FILTER=maxwell`); the pipeline's own
`run_beamformer.py` puts the surface reconstruction on the exact ground-truth
vertex and the volume one within one 8 mm grid step.

One trap worth knowing about, since it looks exactly like a broken beamformer:
`make_forward_solution(mindist=...)` prunes sources near the inner skull, so a
forward's source space has *fewer* vertices than the `bem/*-src.fif` it came
from. Index source-estimate rows through `stc.vertices`, never through the
unpruned source space. Ground-truth positions are recorded in both head and MRI
surface RAS for the same reason — a forward's source space is in head
coordinates, one read from `bem/*-src.fif` is in MRI surface RAS, and mixing
them up costs about 15 mm here.

### Coregistration

The head↔MRI transform is known exactly by construction. It is saved at
`derivatives/freesurfer/subjects/sub-001_ses-01/bem/sub-001_ses-01-trans.fif`,
and the anatomical landmarks in the T1w sidecar were written by the repository's
own `preprocessing/coreg.py`, so `source/make_forward` recovers it through
`mne_bids.get_head_mri_trans` the way it does for a real subject. The generator
asserts that round trip, so a broken landmark write fails loudly at generation
time rather than as a mislocalised source later.

Because there is no display in most development environments, the ICP path in
`coreg.py` is skipped in favour of the ground-truth transform. The fiducials
file is still written, so `mne.coreg.Coregistration(fiducials="auto")` works if
you want to exercise ICP.

## Generating more subjects

The committed subject is one call of the generator; everything about it is
reproducible and adjustable.

```bash
# regenerate the committed subject in place
python src/custom/make_synthetic.py --out synthetic/datasets/synth

# a 12-subject cohort for group-level work, somewhere scratch
python src/custom/make_synthetic.py --out /tmp/synth-cohort --n-subjects 12

# keep the pre-BIDS Cerca-style tree, to exercise format_bids.py / run_bids.sh
python src/custom/make_synthetic.py --out /tmp/synth --keep-raw
```

Head geometry is jittered per subject (`--head-jitter`) so a cohort has a
realistic spread of head sizes and coregistrations; the first subject is always
the un-jittered reference, matching the group template. See
`python src/custom/make_synthetic.py --help` for sensor count, duration,
sampling rate and seed.

## Notes and caveats

- **The pre-BIDS FIFs are not committed.** They are exactly regenerable
  (`--keep-raw`) and would double the size. The behavioural CSV *is* committed,
  under `raw/synth_001/metadata/`, because `config-trial.py` reads it from
  `RAW_DIR` the way the real configs do.
- **`fsaverage` here is synthetic.** The generator writes its own group template
  under that name inside the dataset's subjects directory so that morphing works
  offline. It shadows the real `fsaverage` only for code that passes this
  `subjects_dir`. Because every synthetic subject shares the template's
  tessellation, surface morphs are exact — convenient for testing, but not
  representative of a real morph.
- **The phantom is not a brain.** Two ellipsoidal hemispheres with sinusoidal
  folding. Source estimates are anatomically meaningless; what they are good for
  is checking that code runs, that orientations and signs are handled
  consistently, and that a beamformer recovers a dipole you planted.
- Decoding and coregistration diagnostics are switched off in the shipped config
  (~38 trials makes decoding meaningless and both roughly double the runtime).
  Turn them on with `decode`/`_run_decoding` and `_run_coreg_diagnostics`.
- **HTML reports are off by default** (`generate_reports = False`). Their 3D
  panels render through VTK and need a display, and without one
  `source/make_forward` aborts with *"Cannot connect to a valid display"* — the
  usual situation in a container. With Xvfb available you can turn them back on
  and run under `xvfb-run -a`.
