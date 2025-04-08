## Convert Cerca OPM data to BIDS format.
# Harrison Ritz (2025)




# %% import -------------------------------------------------------------------

import mne
import mne_bids
import os
import numpy as np
import datetime
import yaml
import sys


# %% import parameters

def set_bids_params(config_path=""):

    # set-up configuration ==========================================================================================================
    print("\n\n\nloading configuration ---------------------------------------------------\n")

    # baseline configuration (set for oddball example)
    base_config = """
    dirs:
        data_dir: /Users/hr0283/Projects/mne-opm/data/raw
        emptyroom_dir: /Users/hr0283/Projects/mne-opm/data/raw
        anat_path: /Users/hr0283/Projects/mne-opm/data/raw/sub-004/mri/t1w.nii
        bids_dir: /Users/hr0283/Projects/mne-opm/data/bids

    session:
        ids: 4
        task: "AVpilot"
        session: "01"
        
        run_prefix: 
        - "20250221_155344"
        - "20250221_160103"
        - "20250221_160425"
        - "20250221_160709"
        - "20250221_161209"
        
        emptyroom_prefix: "20250221_143401"


    trigger:
        find_events: false
        
        rename_annot: true 
        event_desc:
            "Trigger 2": "iti"
            "Trigger 3": "vis"
            "Trigger 4": "aud"
            "Trigger 5": "audvis" 
        

    recording_info:        
        line_freq: 60.0

    """

    # Load config file
    cfg = yaml.safe_load(base_config)
   
    print(f"\n\nloading config: {config_path}\n")
    if config_path:
        with open(config_path, 'r') as stream:
            proc = yaml.safe_load(stream)
        cfg.update(proc)

    return cfg




def bids_conversion(cfg):
    """
    Converts raw MEG data files to BIDS format using configuration parameters provided in the cfg dictionary.
    This function performs the following steps:
        1. Extracts necessary configuration parameters such as subject ID, session information, run prefixes,
           empty room prefix, task name, and anatomical scan indicator.
        2. Reads and processes the empty room file if specified in cfg["session"]["emptyroom_prefix"]. The
           empty room raw data is read, its line frequency updated based on cfg["recording_info"]["line_freq"],
           and then written to a BIDS-compatible directory structure.
        3. Iterates over each run specified in cfg["session"]["run_prefix"]:
            - Constructs the file path for the raw MEG data.
            - Reads the raw data and updates its metadata (line frequency and subject information).
            - Optionally finds and merges events if cfg["trigger"]["find_events"] is True, setting the resulting
              annotations on the raw data.
            - Appends the processed raw data for later concatenation.
        4. Concatenates the individual raw run data into a single raw object and prints the recording duration.
        5. Optionally renames annotation descriptions if cfg["trigger"]["rename_annot"] is True.
        6. Associates the empty room information with the concatenated raw object if applicable.
        7. Writes the concatenated raw data to the BIDS directory.
        8. If an anatomical scan is provided (cfg["dirs"]["anat_path"] is not None), writes the anatomical image
           to the BIDS structure.
    Parameters:
        cfg (dict): A configuration dictionary containing settings required for the conversion.
            Must include the following keys:
                - "session": A dictionary with keys:
                    - "ids": Subject identifier.
                    - "run_prefix": List of run prefixes.
                    - "emptyroom_prefix": Prefix for empty room data (evaluated as False if not provided).
                    - "task": Task name.
                    - "session": Session label.
                - "dirs": A dictionary with keys:
                    - "data_dir": Base directory for raw MEG data.
                    - "emptyroom_dir": Directory containing empty room files.
                    - "anat_path": Path to anatomical scans (evaluated as False if not provided).
                    - "bids_dir": Output directory for BIDS formatted data.
                - "recording_info": A dictionary with key:
                    - "line_freq": The line frequency value to be set in the raw data info.
                - "trigger": A dictionary with keys:
                    - "find_events": Boolean flag to determine if events should be located.
                    - "stim_id": List of stimulus channel identifiers per run.
                    - "old_trigger_id": List of trigger IDs to be replaced.
                    - "new_trigger_id": List of new trigger IDs to use.
                    - "event_desc": Dictionary mapping event codes to descriptions.
                    - "rename_annot": Boolean flag to determine if annotations should be renamed.
    Returns:
        None
    Raises:
        Any exceptions raised from file I/O operations, MNE functions, or issues during raw concatenation are
        propagated to the caller.
    Note:
        This function requires the mne and mne_bids libraries to be imported in the calling environment.
    """

    # %% convert to BIDS ---------------------------------------------------------

    subj = cfg["session"]["ids"]
    runs = cfg["session"]["run_prefix"]
    emptyroom = cfg["session"]["emptyroom_prefix"]
    task = cfg["session"]["task"]
    anat_path = cfg["dirs"]["anat_path"]

    raw_list = list()
    print("\nparticipant: ", subj,
          "\nruns: ", runs,
          "\nemptyroom: ", emptyroom,
          "\ntask: ", task,
          "\nanat path: ", anat_path,
          "\n--------\n")
    
    # Process empty room data ------------------------------------------------
    if emptyroom:

        fn_empty_room = os.path.join(
            cfg['dirs']['emptyroom_dir'], 
            f"sub-{subj:03}", 
            f"{emptyroom}_cMEG_Data", 
            f"{emptyroom}_meg.fif"
            )
        
        raw_empty_room = mne.io.read_raw_fif(fn_empty_room)
        raw_empty_room.info["line_freq"] = cfg["recording_info"]["line_freq"]
        
        emptyroom_bids_path = mne_bids.BIDSPath(
            subject=f"{subj:03}",
            session=cfg["session"]["session"],
            task="noise",
            root=cfg['dirs']['bids_dir'],
        )
        
        mne_bids.write_raw_bids(
            raw_empty_room,
            emptyroom_bids_path,
            allow_preload=True,
            overwrite=True,
            events=None,
            format="FIF",
        )
    
    # Loop over runs for this subject -----------------------------------------
    for rr, (run) in enumerate(runs):
        print("\nrun: ", run, "--------\n")
        
        # Construct file path
        fn = os.path.join(cfg['dirs']['data_dir'], f"sub-{subj:03}", f"{run}_cMEG_Data", f"{run}_meg.fif")
        raw = mne.io.read_raw_fif(fn)
        
        
        raw.info["line_freq"] = cfg["recording_info"]["line_freq"]
        raw.info["subject_info"] = {
            "id": int(subj),
            "his_id": f"{subj:03}",
            }
        
        
        # Add events using triggers from cfg["trigger"]
        if cfg["trigger"]["find_events"]:

            print("\n\n\nFINDING EVENTS\n\n\n")
            event_list = list()
            for stim, old_trigger, new_trigger in zip(cfg["trigger"]["stim_id"][rr],
                                                    cfg["trigger"]["old_trigger_id"][rr],
                                                    cfg["trigger"]["new_trigger_id"][rr]):
    
                event = mne.find_events(raw, stim_channel=stim, min_duration=0.001)
                event_list.append(mne.merge_events(event, [int(old_trigger)], new_trigger))
            
            events = np.concatenate(event_list, axis=0)
            print("\nevents: ", events)
            print("number of events: ", len(events), "\n")
            
            annot = mne.annotations_from_events(
                events=events,
                sfreq=raw.info["sfreq"],
                event_desc=cfg["trigger"]["event_desc"]
            )
            raw.set_annotations(annot)
 
        raw_list.append(raw)


    # Concatenate raws for all runs of this subject
    all_raw = mne.concatenate_raws(raw_list, preload=True, on_mismatch="raise")

    recording_duration = all_raw.times[-1] - all_raw.times[0]
    print(f"Recording duration for subject {subj}: {recording_duration:.2f} seconds")

    # Rename annotations
    if cfg["trigger"]["rename_annot"]:
        all_raw.annotations.rename(cfg["trigger"]["event_desc"])
    
    # Write to BIDS -----------------------------------------------------------
    bids_path = mne_bids.BIDSPath(
        subject=f"{subj:03}",
        session=cfg["session"]["session"],
        task=task,
        run="01",
        root=cfg['dirs']['bids_dir'],
    )

    mne_bids.write_raw_bids(
        all_raw,
        bids_path,
        allow_preload=True,
        overwrite=True,
        format="FIF",
        empty_room=emptyroom_bids_path,
    )


    if anat_path:

        anat_bids_path = mne_bids.BIDSPath(
            subject=f"{subj:03}",
            session=cfg["session"]["session"],
            suffix="T1w",
            root=cfg['dirs']['bids_dir'],
        )

        mne_bids.write_anat(
            image=anat_path, 
            bids_path=anat_bids_path, 
            overwrite=True
            )
        
        print('saved to anat path: ', anat_path)
            
    
# %% main ---------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        print('config path: ', config_path)
    else:
        config_path = ""

    print('config path: ', config_path)
    cfg = set_bids_params(config_path)
    bids_conversion(cfg)

    print("\n\n\nDONE!\n\n\n")
