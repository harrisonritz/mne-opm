# !/bin/bash

# Parse arguments
PIPELINE=$1 # e.g., bids, coreg, freesurfer, preproc, sensor, source
ANALYSIS=""
CONFIG_BASE="/Users/hr0283/Projects/mne-opm/config"
DATA_BASE="/Users/hr0283/Projects/TSX_OPM/data"
while [[ $# -gt 0 ]]; do
    case $1 in
        bids|coreg|freesurfer|preproc|sensor|source)
            shift 1
            ;;
        -e|--exp|--experiment)
            EXPERIMENT=$2
            shift 2
            ;;
        -s|--sub|--subject)
            SUBJECT=$2
            shift 2
            ;;
        -a|--analysis)
            ANALYSIS=$2
            shift 2
            ;;
        -d|--data)
            DATA_BASE=$2
            shift 2
            ;;
        -c|--config)
            CONFIG_BASE=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 <pipeline> --exp <value> --sub <value> [--analysis <value>] [--data <value>] [--config <value>]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 <pipeline> --exp <value> --sub <value> [--analysis <value>] [--data <value>] [--config <value>]"
            exit 1
            ;;
    esac
done

if [ ! ${EXPERIMENT} ]; then
    echo "Experiment not set. Please provide it using -e or --experiment."
    exit 1
fi

if [ ! ${SUBJECT} ]; then
    echo "Subject not set. Please provide it using -s or --subject."
    exit 1
fi

# export variables
export EXPERIMENT
export SUBJECT
export ANALYSIS

export ROOT_DIR=$PWD

export CONFIG_DIR="$CONFIG_BASE/$EXPERIMENT"

export DATA_DIR="$DATA_BASE/$EXPERIMENT"
export RAW_DIR="$DATA_DIR/raw"
export BIDS_DIR="$DATA_DIR/bids"
export SUBJECTS_DIR="$DATA_DIR/bids/derivatives/freesurfer/subjects"


# run the analysis pipeline
echo "\nRunning '${PIPELINE}' pipeline on experiment '${EXPERIMENT}' for subject ${SUBJECT}...\n"
source "./src/run/run_$PIPELINE.sh"
echo "\nPipeline '${PIPELINE}' completed for experiment '${EXPERIMENT}' and subject ${SUBJECT}.\n"




