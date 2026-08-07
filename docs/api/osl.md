# osl-ephys pipeline

`custom.osl` runs an [osl-ephys](https://osl-ephys.readthedocs.io) pipeline over
BIDS OPM data, from preprocessing through to LCMV beamforming and parcellation.
It is an alternative to the mne-bids-pipeline route in `custom.preprocessing`:
both read the same BIDS data written by `custom.format_bids`, and write to
separate derivative trees, so they can be run side by side and compared.

The pipeline runs **one subject per invocation**, which is what makes it usable
as the body of a SLURM array job.

```bash
# check the config before submitting anything
python src/custom/run_osl.py --stage=validate --config=<analysis>.yaml

# one subject, preprocessing through beamforming
python src/custom/run_osl.py --stage=all --config=<analysis>.yaml

# once the array finishes, build the group reports
python src/custom/run_osl.py --stage=collate --config=<analysis>.yaml
```

or through the CLI wrapper:

```bash
./mne-opm.sh osl --exp TSX --sub 007 --analysis trialResponse --stage all \
    --data /path/to/data --config /path/to/config
```

## Stages

| Stage | What it does |
|-------|--------------|
| `preproc` | Run the osl-ephys `preproc` chain over the subject's BIDS raw file |
| `source` | Surfaces, coregistration, forward model, LCMV beamforming, parcellation |
| `all` | `preproc` then `source`; stops if preprocessing fails |
| `collate` | Rebuild the group-level HTML reports across every subject |
| `validate` | Check the config without running anything |

`collate` is separate on purpose. osl-ephys rebuilds the shared
`subject_report.html` at the end of every chain, from inside its own
`try`/`except`. In an array job that means every task writing the same file, and
a rendering hiccup marking an otherwise-successful subject as failed. The
per-subject stages therefore write only their own subject's report data, and
`collate` renders the shared pages once.

## Source backends

osl-ephys' LCMV path is written against RHINO, and therefore needs FSL:
`beamforming.make_lcmv` reads the forward model from the RHINO file tree,
`transform_recon_timeseries` needs RHINO's transforms and shells out to `flirt`,
and even `parcellation.resample_parcellation` calls `flirt`. Its
`surface_extraction_method='freesurfer'` path avoids FSL but only reaches
minimum-norm estimates, not beamforming.

`pipeline.source_backend` selects between:

**`rhino`** (default) — osl-ephys' native path. RHINO extracts surfaces from the
T1 and fits its own coregistration, then beamforms onto a volumetric grid in MNI
space. Requires FSL. Does not use the FreeSurfer/MNE coregistration produced by
`mne-opm.sh coreg`.

**`freesurfer`** — the wrappers in `custom.osl.fs_bridge`, which reuse the
existing `recon-all` output and `-trans.fif`, beamform with
`mne.beamformer.make_lcmv`, and morph to MNI via FreeSurfer's `talairach.xfm`.
Parcel time courses are computed with osl-ephys' own maths, so output is
directly comparable with the RHINO backend. Needs no FSL.

Because the two backends need different steps, `source_recon` in the config may
be keyed by backend so that one file describes both.

## Events from annotations

osl-ephys builds epochs from `dataset['events']`, and its only built-in way to
populate them is `find_events`, which reads a stim channel. `format_bids`
deliberately converts the Cerca trigger channels to *annotations* and then drops
the stim channels, because leaving them in makes `write_raw_bids` re-extract
events with different parameters every time a derivative is re-saved.

`custom.osl.extra_funcs.events_from_annotations` bridges that gap, so the
osl-ephys pipeline reads exactly the same BIDS data as the mne-bids-pipeline
route with no change to `format_bids`. Use it in a config like any other step:

```yaml
preproc:
  - events_from_annotations: {}
  - epochs: {tmin: -0.5, tmax: 0.5}
```

Event codes come from the config's `meta.event_codes`, so numbering is identical
across subjects even when one is missing a condition. Descriptions with no
annotations are dropped from the mapping, because `mne.Epochs` raises on an
`event_id` entry that matches no event.

## Modules

### run_osl

```{eval-rst}
.. automodule:: custom.run_osl
   :members:
   :no-index:
```

### osl.\_config

```{eval-rst}
.. automodule:: custom.osl._config
   :members:
   :no-index:
```

### osl.\_paths

```{eval-rst}
.. automodule:: custom.osl._paths
   :members:
   :no-index:
```

### osl.extra_funcs

```{eval-rst}
.. automodule:: custom.osl.extra_funcs
   :members:
   :no-index:
```

### osl.preproc

```{eval-rst}
.. automodule:: custom.osl.preproc
   :members:
   :no-index:
```

### osl.source

```{eval-rst}
.. automodule:: custom.osl.source
   :members:
   :no-index:
```

### osl.fs_bridge

```{eval-rst}
.. automodule:: custom.osl.fs_bridge
   :members:
   :no-index:
```

### osl.collate

```{eval-rst}
.. automodule:: custom.osl.collate
   :members:
   :no-index:
```

### osl.validate

```{eval-rst}
.. automodule:: custom.osl.validate
   :members:
   :no-index:
```
