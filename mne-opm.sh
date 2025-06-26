# !/bin/bash

# Defaults
EXPERIMENT="TSXpilot"
ANALYSIS=""
SUBJECT=""
CONFIG_BASE="/Users/hr0283/Projects/TSX_OPM/config"
DATA_BASE="/Users/hr0283/Projects/TSX_OPM/data"

# Parse arguments
PIPELINE=$1 # e.g., bids, coreg, freesurfer, preproc, sensor, source

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
            echo "Usage: $0 <pipeline> --exp <value> --sub <value> [--analysis <value>] [--data <value>] [--config <value>] [--help]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 <pipeline> --exp <value> --sub <value> [--analysis <value>] [--data <value>] [--config <value>] [--help]"
            exit 1
            ;;
    esac
done

# Check if required variables are set
if [ ! ${PIPELINE} ]; then
    echo "Pipeline not set. Please provide it as the first argument (e.g., bids, coreg, freesurfer, preproc, sensor, source)."
    exit 1
fi

if [ ! ${EXPERIMENT} ]; then
    echo "Experiment not set. Please provide it using -e or --exp."
    exit 1
fi

if [ ! ${SUBJECT} ]; then
    echo "Subject not set. Please provide it using -s or --sub."
    exit 1
fi

if [ ! ${DATA_BASE} ]; then
    echo "data directory not set. Please provide it using -d or --data."
    exit 1
fi

if [ ! ${CONFIG_BASE} ]; then
    echo "Config directory not set. Please provide it using -c or --config."
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
echo "\nStarting '${PIPELINE}' pipeline on experiment '${EXPERIMENT}' for subject ${SUBJECT}\n"
source "./src/run/run_$PIPELINE.sh"
echo "\nPipeline '${PIPELINE}' completed for experiment '${EXPERIMENT}' and subject ${SUBJECT}.\n"




