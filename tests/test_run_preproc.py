"""Control flow of src/run/run_preproc.sh for each PREPROC_STAGE.

``python`` and ``mne_bids_pipeline`` are replaced by stubs on ``PATH`` that log
their arguments and the trunk flags they were run with, so the test checks
which steps run, in what order, and with which exported flags.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

# Logs "<step> <flags>", with a 1/0 per trunk flag; a seed prints MNE log
# noise and two of the three flag names, as src/custom/trunk.py would.
_STUB = """#!/usr/bin/env bash
step=$1
for arg in "$@"; do
    case "$arg" in
        --analysis=*|--steps=*) step=${arg#*=}; step=${step%%,*} ;;
        *trunk.py|*run_beamformer.py) step=$(basename "$arg" .py) ;;
        save-state|seed) step="$step:$arg" ;;
    esac
done
echo "$step ${MNE_OPM_BREAKS_ALREADY_ANNOTATED:-0}${MNE_OPM_REQUIRE_RESPONSE_METADATA:-0}${MNE_OPM_STAGE1_BREAKS_SAVED:-0}" >> "$STUB_LOG"
if [ "$step" = "trunk:seed" ]; then
    [ "${STUB_SEED_FAIL:-0}" = "1" ] && exit 1
    echo "Using matplotlib as 2D backend."
    echo "MNE_OPM_BREAKS_ALREADY_ANNOTATED"
    echo "MNE_OPM_REQUIRE_RESPONSE_METADATA"
fi
exit 0
"""

TRUNK = [
    "init",
    "annotate_breaks",
    "select_trial_response",
    "bad_segments_1",
    "bad_channels",
]
BRANCH = [
    "manual_channel",
    "apply_hfc",
    "apply_amm",
    "apply_zca",
    "preprocessing",
    "ica_convergence",
    "bad_ICs",
    "manual_ica",
    "preprocessing/apply_ica",
    "bad_epochs",
    "sensor/make_evoked",
]


def _run(tmp_path, **env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("python", "mne_bids_pipeline"):
        (bin_dir / name).write_text(_STUB)
        (bin_dir / name).chmod(0o755)
    log = tmp_path / "steps.log"
    log.touch()
    full_env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "ROOT_DIR": str(ROOT_DIR),
        "CONFIG_DIR": str(tmp_path),
        # run_beamformer.sh runs under ``set -u`` when failing fast.
        **dict.fromkeys(
            ("EXPERIMENT", "ANALYSIS", "SUBJECT", "RAW_DIR", "BIDS_DIR"), "test"
        ),
        "SUBJECTS_DIR": "test",
        "STUB_LOG": str(log),
        # Left over from an earlier run in the same shell: must be cleared.
        "MNE_OPM_STAGE1_BREAKS_SAVED": "1",
        **env,
    }
    for name in (
        "PREPROC_STAGE",
        "RUN_BEAMFORMER",
        "FAIL_ON_FIRST_CRASH",
        "TSX_COMPARE_DRIVER",
    ):
        if name not in env:
            full_env.pop(name, None)
    proc = subprocess.run(
        ["bash", str(ROOT_DIR / "src" / "run" / "run_preproc.sh")],
        env=full_env,
        capture_output=True,
        text=True,
    )
    steps = dict(line.split() for line in log.read_text().splitlines())
    order = [line.split()[0] for line in log.read_text().splitlines()]
    return proc, order, steps


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, TRUNK + BRANCH + ["run_beamformer"]),
        ({"PREPROC_STAGE": "all", "RUN_BEAMFORMER": "0"}, TRUNK + BRANCH),
        ({"PREPROC_STAGE": "trunk"}, TRUNK + ["trunk:save-state"]),
        ({"PREPROC_STAGE": "branch"}, ["trunk:seed"] + BRANCH + ["run_beamformer"]),
        ({"PREPROC_STAGE": "branch", "RUN_BEAMFORMER": "0"}, ["trunk:seed"] + BRANCH),
    ],
)
@pytest.mark.parametrize("fail_fast", ["0", "1"])
def test_stage_steps_and_flags(tmp_path, env, expected, fail_fast):
    proc, order, steps = _run(tmp_path, FAIL_ON_FIRST_CRASH=fail_fast, **env)
    assert proc.returncode == 0, proc.stderr
    assert order == expected
    if "init" in steps:  # flags are exported as each trunk step succeeds
        assert [steps[s] for s in TRUNK] == ["000", "000", "100", "110", "111"]
    if "trunk:seed" in steps:  # a branch exports only the trunk's flags
        assert steps["trunk:seed"] == "000"
        assert steps["manual_channel"] == "110"
        assert "Using matplotlib as 2D backend." in proc.stdout
    elif "manual_channel" in steps:
        assert steps["manual_channel"] == "111"
    if "trunk:save-state" in steps:
        assert steps["trunk:save-state"] == "111"


@pytest.mark.parametrize(
    "env",
    [
        {"PREPROC_STAGE": "branch", "STUB_SEED_FAIL": "1"},
        {"PREPROC_STAGE": "bogus"},
    ],
)
def test_stage_errors_stop_the_script(tmp_path, env):
    proc, order, _ = _run(tmp_path, **env)
    assert proc.returncode == 1
    assert "[ERROR]" in proc.stderr
    assert order == (["trunk:seed"] if "STUB_SEED_FAIL" in env else [])


def test_comparison_driver_replaces_redundant_steps(tmp_path):
    proc, order, _ = _run(
        tmp_path,
        PREPROC_STAGE="branch",
        RUN_BEAMFORMER="0",
        TSX_COMPARE_DRIVER="/tmp/run_preprocessing.py",
        FAIL_ON_FIRST_CRASH="1",
    )
    assert proc.returncode == 0, proc.stderr
    assert order == [
        "trunk:seed",
        "manual_channel",
        "apply_hfc",
        "apply_amm",
        "apply_zca",
        "/tmp/run_preprocessing.py",
        "ica_convergence",
        "/tmp/run_preprocessing.py",
        "manual_ica",
        "/tmp/run_preprocessing.py",
        "preprocessing/apply_ssp",
        "bad_epochs",
        "/tmp/run_preprocessing.py",
    ]
    assert "preprocessing" not in order and "preprocessing/apply_ica" not in order
    assert "sensor/make_evoked" not in order
