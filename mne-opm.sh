# !/bin/bash

echo ""
echo ""
echo ""
echo ""
echo ""
echo ""
echo "      ~~~~~~~~~~~~~~         ~~~~~~~~~~~~~~"
echo "      ~~~~~~~~~~~~~~ MNE-OPM ~~~~~~~~~~~~~~"
echo "      ~~~~~~~~~~~~~~         ~~~~~~~~~~~~~~"                                                                                  
echo ""
echo ""
echo ""


# init
set -e                      # fail on first crash
source .venv/bin/activate   # activate virtual environment

# Defaults
PIPELINE="func" # default pipeline; e.g., bids, coreg, freesurfer, preproc, sensor, source
EXPERIMENT="TSXpilot"
ANALYSIS="trial"
SUBJECT="009"
SESSION="01"
CONFIG_BASE="/Users/hr0283/Projects/TSX_OPM/analysis/config"
FREESURFER_HOME=/Applications/freesurfer/8.1.0
FAIL_ON_FIRST_CRASH=1

DATA_BASE="/Users/hr0283/Brown Dropbox/Harrison Ritz/___Export_Folder/opm_data/data"
# SUBJECTS_DIR="$DATA_BASE/$EXPERIMENT/bids/derivatives/freesurfer/subjects"
SUBJECTS_DIR="/Users/hr0283/freesurfer/$EXPERIMENT"
T1W_PATH=$SUBJECTS_DIR/anat/sub-$SUBJECT/anat_ses-01_T1w_acq-0.8mm-MPR_t1w.nii.gz
T2W_PATH=$SUBJECTS_DIR/anat/sub-$SUBJECT/anat_ses-01_T2w_acq-0.8mm-MPR_t2w.nii.gz

MAX_WORKERS=10

usage="Usage: sh $0 <pipeline> --exp <experiment> --sub <subject number> --data <data directory> --config <configuration directory> [--analysis <analysis name>] [--session <session number>] [--fs <freesurfer directory>] [--t1w <T1w image path>] [--fail-on-first-crash] [--help]"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        nifti|coreg|freesurfer|bids|preproc|sensor|source|beamformer|all|func|anat)
            PIPELINE=$1
            shift 1
            ;;
        -e|--exp|--experiment)
            EXPERIMENT=$2
            shift 2
            ;;
        -s|--sub|--subj|--subject)
            SUBJECT=$2
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
        -a|--analysis)
            ANALYSIS=$2
            shift 2
            ;;
        --session)
            ANALYSIS=$2
            shift 2
            ;;
        --fs)
            FREESURFER_HOME=$2
            shift 2
            ;;
        --t1w)
            T1W_PATH=$2
            shift 2
            ;;
        --t2w)
            T2W_PATH=$2
            shift 2
            ;;
        -w|--workers)
            MAX_WORKERS=$2
            shift 2
            ;;
        --fail-on-first-crash)
            FAIL_ON_FIRST_CRASH=1
            shift 2
            ;;
        -h|--help)
            echo $usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo $usage
            exit 1
            ;;
    esac
done

# Check if required variables are set
if [ ! ${PIPELINE} ]; then
    echo "ERROR: Pipeline not set"
    echo $usage
    exit 1
fi

if [ ! ${EXPERIMENT} ]; then
    echo "ERROR: Experiment not set"
    echo $usage
    exit 1
fi

if [ ! ${SUBJECT} ]; then
    echo "ERROR: Subject not set"
    echo $usage
    exit 1
fi

if [ ! "${DATA_BASE}" ]; then
    echo "ERROR: Data directory not set"
    echo $usage
    exit 1
fi

if [ ! "${CONFIG_BASE}" ]; then
    echo "ERROR: Config directory not set"
    echo $usage
    exit 1
fi


# export variables
export EXPERIMENT
export SUBJECT
export SUBJECT_NUM=sub-$SUBJECT
export ANALYSIS
export SESSION

export ROOT_DIR=$PWD

export CONFIG_DIR="$CONFIG_BASE/$EXPERIMENT"

export FREESURFER_HOME
export SUBJECTS_DIR

export FAIL_ON_FIRST_CRASH=${FAIL_ON_FIRST_CRASH:-0}
export DATA_DIR="$DATA_BASE/$EXPERIMENT"
export RAW_DIR="$DATA_DIR/raw"
export BIDS_DIR="$DATA_DIR/bids"
export MAX_WORKERS


export T1W_PATH
if [ ! ${T1W_PATH} ]; then
    DEFAULT_T1W_PATH=$BIDS_DIR/${SUBJECT_NUM}/ses-01/anat/${SUBJECT_NUM}_ses-01_T1w.nii.gz
    if [ -f "$DEFAULT_T1W_PATH" ]; then
        export T1W_PATH=$DEFAULT_T1W_PATH
    else
        export T1W_PATH=""
    fi
fi


export T2W_PATH
if [ ! ${T2W_PATH} ]; then
    DEFAULT_T2W_PATH=$BIDS_DIR/${SUBJECT_NUM}/ses-01/anat/${SUBJECT_NUM}_ses-01_T2w.nii.gz
    if [ -f "$DEFAULT_T2W_PATH" ]; then
        export T2W_PATH=$DEFAULT_T2W_PATH
    else
        export T2W_PATH=""
    fi
fi







# run the analysis pipeline
echo "\nStarting '${PIPELINE}' pipeline on experiment '${EXPERIMENT}' for subject ${SUBJECT}\n--------------\n"
source "$ROOT_DIR/src/run/run_$PIPELINE.sh"




