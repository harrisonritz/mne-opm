# CLAUDE.md

Guidance for Claude Code (and other AI assistants) when working in this
repository.

## Project overview

`mne-opm` is an OPM-MEG preprocessing pipeline built on MNE-Python,
mne-bids-pipeline, and OSL-ephys. The custom pipeline steps live in
`src/custom/` and are driven entirely by a user-supplied configuration file
(passed to the run scripts as `CONFIG_PATH`).

- `src/custom/` — custom preprocessing steps and OSL wrappers
- `src/run/` — shell wrappers for high-level pipeline stages
- `docs/config-sample.py` — the canonical, fully-annotated sample configuration

## Keep the sample config up to date (REQUIRED)

`docs/config-sample.py` is the single source of truth for every configuration
option the pipeline understands. It must always document the complete set of
options.

**Whenever you add, rename, or remove a configuration option anywhere in the
pipeline, you MUST update `docs/config-sample.py` in the same change:**

- Adding a new option (any `cfg.<name>` / `getattr(cfg, "<name>", ...)` read in
  `src/`): add it to the relevant section of `docs/config-sample.py` with a
  sensible default value and an explanatory comment.
- Renaming an option: rename it in the sample too, and remove the old name.
- Removing an option: delete it from the sample.

Custom options are prefixed with `_` (read by the mne-opm custom steps); options
without a prefix are standard `mne_bids_pipeline` parameters. Document both.

After editing, sanity-check that the sample still parses:

```bash
python -m py_compile docs/config-sample.py
```

Do not include experiment-specific values (subject-specific conditions,
contrasts, or metadata-derivation logic) in the sample — use neutral,
illustrative defaults so the file stays a generic reference.
