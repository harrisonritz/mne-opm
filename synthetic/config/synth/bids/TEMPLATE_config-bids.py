"""BIDS-conversion configuration template for the synthetic dataset.

Consumed by ``src/custom/format_bids.py`` (via ``run_bids.sh``), which reads
``RAW_DIR``/``BIDS_DIR`` from the environment and everything else from here.

You only need this when regenerating BIDS from a pre-BIDS tree, i.e. after
running ``make_synthetic.py --keep-raw``.  The committed subject was already
converted with these exact settings.

Author: Harrison Ritz (2025)
"""

import os

# Session information
ids = int(os.environ.get("SUBJECT", "1"))
task = os.environ.get("EXPERIMENT", "synth")
session = os.environ.get("SESSION", "01")

# Trigger mapping.  Must match custom.synthetic.events.TRIGGER_DESC — the
# generator drives one of eight parallel-port lines per event, and
# format_bids.convert_triggers packs them back into these codes.
rename_annot = True

trigger_desc = {
    1: "ITI",
    2: "feedback",
    4: "trial/cond_a",
    8: "trial/cond_b",
    16: "response/left",
    32: "response/right",
}

# Responses arrive on the trigger lines rather than on BNC inputs, so there is
# nothing to rename here.
response_desc = {}

# Recording information
line_freq = 60.0
bads = []
crop = 0

device_info = dict(
    type="Cerca_synthetic", site="mne-opm synthetic dataset", model="cMEG"
)
