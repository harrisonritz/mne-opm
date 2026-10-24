# SPM reference for the AMM tests

`tests/test_amm.py` compares `custom.preprocessing._amm` with SPM's own
implementation of adaptive multipole models by Tim Tierney (Tierney et al.,
2024, https://doi.org/10.1002/hbm.26596). It runs SPM's functions unmodified
in Octave.

- SPM's files (`spm_opm_amm.m`, `spm_ipharm.m`, `spm_epharm.m`, `spm_slm.m`,
  `spm_match_str.m`) are downloaded at test time from
  https://github.com/spm/spm at the commit pinned in `test_amm.py`
  (`SPM_COMMIT`). They are GPL-2.0-or-later and are not part of this
  repository. Set `SPM_AMM_CACHE` to a folder to keep them between runs.
- This folder holds only what is needed to call them outside SPM:
  - `@fakemeeg/`: the few methods of SPM's `meeg` object that
    `spm_opm_amm` uses;
  - `run_spm_amm.m`: runs `spm_opm_amm`, `spm_ipharm` and `spm_epharm` on the
    inputs the test writes, and saves their outputs;
  - `spm.m`: `spm('time')`;
  - `std.m`: MATLAB's `'omitnan'` flag, which Octave 8's `std` lacks;
  - `figure.m`, `plot3.m`, `daspect.m`, `hold.m`, `ellipsoid.m`: no-ops, so
    the spheroid plot `spm_opm_amm` draws for rotated arrays needs no display.

The tests are skipped when `octave-cli` is not installed or the download fails.
