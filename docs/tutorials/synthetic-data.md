# Synthetic dataset

The repository ships a complete, fully synthetic OPM-MEG subject in BIDS format
under `synthetic/`. It exists so that the pipeline can be run end to end from a
clean clone — no real recordings, no FreeSurfer install, no downloads. That
makes it the natural target for developing pipeline changes, reproducing bugs,
and for coding agents, which cannot reach the real data at all.

There is no human data involved: the anatomy is an analytic phantom and the
recordings are simulated through a BEM forward model computed on it.

## Running the pipeline against it

```bash
bash mne-opm.sh preproc \
    --exp synth --sub 001 --session 01 --analysis trial \
    --data synthetic/datasets --config synthetic/config
```

`SUBJECTS_DIR` can be left unset — the shipped config defaults it to the
FreeSurfer subjects directory inside the dataset. Swap `preproc` for `coreg`,
`sensor`, `source` or `beamformer` to run a single stage, or drive
mne-bids-pipeline directly:

```bash
mne_bids_pipeline --steps=preprocessing \
    --config=synthetic/config/synth/config-trial.py
```

## What it contains

| | |
|---|---|
| Array | 48 Cerca-style triaxial slots = 144 magnetometers, QuSpin coil type |
| Sampling | 200 Hz, 100 s task + 40 s empty room |
| Aux channels | `eye_nmf1-3` (EOG), gaze/pupil and head-position misc channels |
| Design | ~38 trials, `trial/cond_a` vs `trial/cond_b`, left/right responses, `feedback`, `ITI`, one 9 s block break |
| Anatomy | phantom FreeSurfer subject supporting `spacing="oct6"`, plus a synthetic `fsaverage` group template |

The 144-channel array and the 40 s empty room are both sized for Maxwell
filtering: `mf_int_order = 10` plus `mf_ext_order = 2` needs 128 basis vectors,
and tSSS refuses an `mf_st_duration` longer than the noise recording it is
applied to. Both `HFC` (the default) and `maxwell` run cleanly here — switch
with `SYNTH_SPATIAL_FILTER=maxwell`, which also renames the derivatives
directory so the two do not collide.

## Ground truth

Because the data comes from a real forward solution, source analyses have a
correct answer. `synthetic/datasets/synth/bids/ground_truth.json` records it:
three cortical dipoles with distinct peak latencies and condition-dependent
gains, the two noisy channels and one dead channel planted in the recording,
and the two broadband bursts the bad-segment step should find.

Running the shipped config, the bad-channel step recovers all three planted
channels, and the LCMV beamformer puts the surface reconstruction on the exact
ground-truth vertex and the volume reconstruction within one 8 mm grid step.
That makes the dataset a usable regression check on changes to forward
modelling, coregistration, spatial filtering or the beamformer itself:

```bash
python -m custom.synthetic.validate \
    --bids synthetic/datasets/synth/bids \
    --deriv synthetic/datasets/synth/bids/derivatives/trial__<version>
```

## Generating more subjects

The committed subject is one invocation of the generator, and everything about
it is adjustable:

```bash
# regenerate the committed subject
python src/custom/make_synthetic.py --out synthetic/datasets/synth

# a cohort for group-level development
python src/custom/make_synthetic.py --out /tmp/synth-cohort --n-subjects 12

# keep the pre-BIDS Cerca-style tree, to exercise format_bids.py
python src/custom/make_synthetic.py --out /tmp/synth --keep-raw
```

Head geometry is jittered per subject so a cohort has a realistic spread of head
sizes and coregistrations; the first subject is always the un-jittered reference
and matches the group template. See `--help` for sensor count, duration,
sampling rate, line frequency and seed.

Group-level code that morphs to `fsaverage` works offline: the generator writes
its own template under that name inside the dataset's subjects directory, which
shadows the real `fsaverage` only for code passing that `subjects_dir`.

## Caveats

The phantom is not a brain — two ellipsoidal hemispheres with sinusoidal
folding — so source estimates are anatomically meaningless. What the dataset is
good for is checking that code runs, that coordinate frames, orientations and
signs are handled consistently, and that a beamformer recovers a dipole you
planted. It is not a substitute for validating on real data.

HTML reports are disabled in the shipped config because their 3D panels need a
display; see `synthetic/README.md` for the full list of caveats and for how to
turn them back on.
