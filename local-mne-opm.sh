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


# Defaults
PIPELINE="preproc" # default pipeline; e.g., bids, coreg, freesurfer, preproc, sensor, source
EXPERIMENT="TSXpilot"
ANALYSIS="CSI"
SUBJECT="007"
SESSION="01"
CONFIG_BASE="/Users/hr0283/Projects/TSX_OPM/analysis/config"
DATA_BASE="/Users/hr0283/Projects/TSX_OPM/data"
FREESURFER_HOME=/Applications/freesurfer/8.0.0
MAX_WORKERS=16

usage="Usage: sh $0 <pipeline> --exp <experiment> --sub <subject number> --data <data directory> --config <configuration directory> [--analysis <analysis name>] [--session <session number>] [--fs <freesurfer directory>] [--t1w <T1w image path>] [--help]"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        nifti|coreg|freesurfer|bids|preproc|sensor|source)
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

if [ ! ${DATA_BASE} ]; then
    echo "ERROR: Data directory not set"
    echo $usage
    exit 1
fi

if [ ! ${CONFIG_BASE} ]; then
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
export DATA_DIR="$DATA_BASE/$EXPERIMENT"
export RAW_DIR="$DATA_DIR/raw"
export BIDS_DIR="$DATA_DIR/bids"
export SUBJECTS_DIR="$DATA_DIR/bids/derivatives/freesurfer/subjects"
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




