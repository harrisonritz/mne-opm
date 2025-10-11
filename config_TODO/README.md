# Minimal example configs for mne-opm

This folder contains templates you can copy into your own analysis directory and point to with `--config /path/to/config/<EXPERIMENT>` when invoking `mne-opm.sh`.

Copy and edit these files:

- `config-CSI.py` — main pipeline config used by preprocessing, sensor, and source stages
- `bids/sub-007_config-bids.py` — per-subject BIDS formatting config used by `run_bids.sh`

Example usage:

```
mkdir -p /path/to/config/TSXpilot/bids
cp config_TODO/config-CSI.py /path/to/config/TSXpilot/config-CSI.py
cp config_TODO/bids/sub-007_config-bids.py /path/to/config/TSXpilot/bids/sub-007_config-bids.py
```

Then run, for example:

```
./mne-opm.sh preproc \
	--exp TSXpilot \
	--sub 007 \
	--data /path/to/data \
	--config /path/to/config \
	--analysis CSI
```


