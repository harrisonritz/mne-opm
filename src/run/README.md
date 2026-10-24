# TSX_OPM
Run scripts


## general procedure
1. run_bids
2. run_freesurfer
3. run_coreg
4. run_preproc
5. run_sensors
6. run_source


## preprocessing stages (run_preproc)
`PREPROC_STAGE` splits `run_preproc.sh` so that several variants of a subject
share its first steps:

- `all` (default): every step.
- `trunk`: INIT, annotate_breaks, select_trial_response, bad_segments_1 and
  bad_channels, then `src/custom/trunk.py save-state` records the flags those
  steps exported in
  `<deriv_root>/sub-<S>/ses-<X>/meg/sub-<S>_ses-<X>_task-<T>_proc-<custom_proc>_desc-trunk_state.json`,
  and the script stops.
- `branch`: needs `MNE_OPM_TRUNK_DERIV`, the trunk config's `deriv_root`
  (absolute). `src/custom/trunk.py seed` clears this config's own
  `proc-<custom_proc>` files (as INIT does), copies the trunk's
  `proc-<custom_proc>` files and `participants.tsv`/`.json` into its
  `deriv_root`, and the script exports the recorded flags and runs
  manual_channel onwards. Both configs must use the same `custom_proc`.

`RUN_BEAMFORMER=0` skips the final BEAMFORMER step (default `1`).

The TSX comparison harness also sets `TSX_COMPARE_DRIVER` to its Python
preprocessing entry point. This hook runs preprocessing through epoch creation,
then custom ICA labelling and one ICA application, and covariance/forward
calculation without diagnostic rendering. When unset, the normal preprocessing
commands below are used. The harness manages comparison intermediate retention.

```bash
PREPROC_STAGE=trunk sh mne-opm.sh preproc --analysis trunk ...
PREPROC_STAGE=branch MNE_OPM_TRUNK_DERIV=/abs/bids/derivatives/<trunk> \
    RUN_BEAMFORMER=0 sh mne-opm.sh preproc --analysis variantA ...
```
