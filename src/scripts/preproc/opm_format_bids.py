## Convert Cerca OPM data to BIDS format.
# Harrison Ritz (2025)


# TODO
# - update tigger/annotation mapping



# %% import -------------------------------------------------------------------

from dotenv import load_dotenv, find_dotenv
import mne
import mne_bids
import os
import numpy as np
import yaml
import sys
import glob



# %% import parameters

def set_bids_params(config_path=""):

    # set-up configuration ==========================================================================================================
    print("\n\n\nloading configuration ---------------------------------------------------\n")

    # set up env
    load_dotenv(find_dotenv('TSX_env.env', usecwd=True), verbose=True, override=True)

    if os.environ.get('TSX_DATA'):
        DATA_DIR = f"{os.environ.get('TSX_DATA')}/raw"
        BIDS_DIR = f"{os.environ.get('TSX_BIDS')}"
        print('data & bids dirs from environment variables')
    else:
        DATA_DIR = "/Volumes/hritz/2025_TSX_Pilot/raw"
        BIDS_DIR = "/Volumes/hritz/2025_TSX_Pilot/bids"
        print('data & bids dir from local variables')


    # baseline configuration (set for oddball example)
    base_config = f"""
    
    dirs:

        data_dir: {DATA_DIR}
        bids_dir: {BIDS_DIR}

        
    session:

        ids: 3
        task: "TSXpilot"
        session: "01"

        
    trigger:

        rename_annot: true 

        trigger_desc:
            1: 'feedback'
            2: 'ITI'
            5: 'CSI'

            4: 'trial/read_noresp'
            8: 'trial/listen_noresp'
            16: 'trial/av_noresp'

            32: 'trial/unimodal_read'
            64: 'trial/bimodal_read'
            6: 'trial/unimodal_listen'
            10: 'trial/bimodal_listen'

            9: 'trial/read_read'
            17: 'trial/listen_listen'
            33: 'trial/read_listen'
            65: 'trial/listen_read'
            

        response_desc:
            BNC 2 Z: 'response/left'
            BNC 5 Z: 'response/right'
        
        
    recording_info:        

        line_freq: 60.0

        bads: 
          - 'C6 2B X'
          - 'C6 2B Y'
          - 'C6 2B Z'
    """

    # Load config file
    cfg = yaml.safe_load(base_config)
   
    print(f"\n\nloading config: {config_path}\n")
    if config_path:
        with open(config_path, 'r') as stream:
            proc = yaml.safe_load(stream)
        cfg.update(proc)

    print('\nconfig:"\n', cfg)

    return cfg



def convert_triggers(raw, cfg):


    print("\n\n\nconverting triggers ----------------------\n")
    # Extract data from trigger channels
    trigger_channels = [f'Trigger {i}' for i in range(1, 9)]

    trigger_data = []
    for ch_name in trigger_channels:
        trigger_data.append(raw.get_data(ch_name))

    # Stack the channels together (shape: n_channels x n_timepoints)
    stacked_triggers = np.vstack(trigger_data)

    # convert to binary
    stacked_triggers[stacked_triggers < 2] = 0
    stacked_triggers[stacked_triggers > 2] = 1

    # convert to binary pattern to integer at each time point
    powers_of_two = 2 ** np.arange(len(trigger_channels))[:, np.newaxis]  # Column vector
    integer_triggers = np.sum(stacked_triggers * powers_of_two, axis=0).astype(int)

    # add combined trigger to raw
    raw.add_channels([mne.io.RawArray(
        integer_triggers.reshape(1, -1), 
        mne.create_info(['Trigger Combined'], raw.info['sfreq'], ['stim'])
    )], force_update_info=True)


    # extract events
    events = mne.find_events(raw, 
                             stim_channel='Trigger Combined', 
                             min_duration=0.001,
                             consecutive=True)
    

    # convert to annotation
    new_annotations = mne.annotations_from_events(
        events,
        event_desc=cfg['trigger']['trigger_desc'],
        sfreq=raw.info['sfreq'],
        orig_time=raw.info['meas_date'],
    )

    # Remove annotations related to trigger channels
    old_annotations = raw.copy().annotations
    trigger_ch_mask = np.ones(len(old_annotations), dtype=bool)
    for i, description in enumerate(old_annotations.description):
        for ch_name in trigger_channels:
            if ch_name in description:
                trigger_ch_mask[i] = False
                break

    # Keep only annotations not related to trigger channels
    old_annotations = mne.Annotations(
        onset=old_annotations.onset[trigger_ch_mask],
        duration=old_annotations.duration[trigger_ch_mask],
        description=old_annotations.description[trigger_ch_mask],
        orig_time=old_annotations.orig_time
    )

    # Set the new annotations
    raw.set_annotations(old_annotations + new_annotations)


    # rename response annotations ----------------------------------
    raw.annotations.rename(cfg['trigger']['response_desc'])


    print("Trigger & Response conversion completed.\n----------\n")
    return raw
    










def bids_conversion(cfg):
    """
    Converts raw OPM MEG data files to BIDS format using configuration parameters provided in the cfg dictionary.
    This function performs the following steps:
        1. Extracts necessary configuration parameters such as subject ID, session information, and task name.
        2. Locates the empty room file, task file(s), and anatomical scan file using glob pattern matching.
        3. Reads and processes the empty room file if found. The empty room raw data is read, its line 
           frequency updated based on cfg["recording_info"]["line_freq"], bad channels set, and then 
           written to a BIDS-compatible directory structure.
        4. Processes task files:
            - Reads the raw data and updates its metadata (line frequency and subject information).
            - Appends the processed raw data for later concatenation.
        5. Concatenates the individual raw run data into a single raw object and prints the recording duration.
        6. Converts triggers and renames annotations if cfg["trigger"]["rename_annot"] is True using the 
           convert_triggers function.
        7. Sets bad channels from the configuration.
        8. Writes the concatenated raw data to the BIDS directory with empty room reference if available.
        9. If an anatomical scan is found, writes the anatomical image to the BIDS structure.
    
    Parameters:
        cfg (dict): A configuration dictionary containing settings required for the conversion.
            Must include the following keys:
                - "session": A dictionary with keys:
                    - "ids": Subject identifier.
                    - "task": Task name.
                    - "session": Session label.
                - "dirs": A dictionary with keys:
                    - "data_dir": Base directory for raw MEG data.
                    - "bids_dir": Output directory for BIDS formatted data.
                - "recording_info": A dictionary with key:
                    - "line_freq": The line frequency value to be set in the raw data info.
                    - "bads": List of bad channels to be marked.
                - "trigger": A dictionary with keys:
                    - "rename_annot": Boolean flag to determine if annotations should be renamed.
                    - "trigger_desc": Dictionary mapping event codes to descriptions.
                    - "response_desc": Dictionary mapping response channel descriptions to new descriptions.
    
    Returns:
        None
    
    Note:
        This function requires the mne, mne_bids, and glob libraries to be imported in the calling environment.
    """

    # %% convert to BIDS ---------------------------------------------------------

    subj = cfg["session"]["ids"]
    task = cfg["session"]["task"]

    emptyroom_path =  glob.glob(
        os.path.join(
            cfg['dirs']['data_dir'],
            f"*_{subj:03}",
            "*_noise",
            "*_meg.fif"
            ))  # Take the first match
    emptyroom_path = emptyroom_path[0] if emptyroom_path else False  # Take the first match or False if not found
    
    task_path =  glob.glob(
        os.path.join(
            cfg['dirs']['data_dir'],
            f"*_{subj:03}",
            "*_task",
            "*_meg.fif"
            ))  # Take the first match
    task_path = task_path if task_path else False

    anat_path = glob.glob(
        os.path.join(
            cfg['dirs']['data_dir'],
            f"*_{subj:03}",
            "*",
            "*_t1w.nii*"
            ))  # Take the first match
    anat_path = anat_path[0] if anat_path else False  # Take the first match or False if not found


    raw_list = list()
    print(  "\nparticipant: ", subj,
            "\ntask: ", task,
            "\ndata dir: ", cfg['dirs']['data_dir'],
            "\nbids dir: ", cfg['dirs']['bids_dir'],
            "\ntask path: ", task_path,
            "\nemptyroom path: ", emptyroom_path,
            "\nanat path: ", anat_path,
            "\n--------\n")
    

    # Process empty room data ------------------------------------------------

    if emptyroom_path:

        raw_empty_room = mne.io.read_raw_fif(emptyroom_path)
        raw_empty_room.info["line_freq"] = cfg["recording_info"]["line_freq"]
        
        # set bad channels
        if cfg['recording_info']['bads']:
            raw_empty_room.info['bads'] = cfg['recording_info']['bads']
        
        # make bids path
        emptyroom_bids_path = mne_bids.BIDSPath(
            subject=f"{subj:03}",
            session=cfg["session"]["session"],
            task="noise",
            root=cfg['dirs']['bids_dir'],
        )
        
        # Write empty room data to BIDS
        mne_bids.write_raw_bids(
            raw_empty_room,
            emptyroom_bids_path,
            allow_preload=True,
            overwrite=True,
            events=None,
            format="FIF",
        )
    
    # Concatenate data for this subject -----------------------------------------
    
    
    for rr, (fn) in enumerate(task_path):
        print("\nrun: ", rr, "---------------------------------------------------\n")
        
        # get path
        raw = mne.io.read_raw_fif(fn)
        
        raw.info["line_freq"] = cfg["recording_info"]["line_freq"]
        raw.info["subject_info"] = {
            "id": int(subj),
            "his_id": f"{subj:03}",
            }
        
        # append raw to list
        raw_list.append(raw)  


    # Concatenate raws for all runs of this subject
    all_raw = mne.concatenate_raws(raw_list, preload=True, on_mismatch="raise")

    recording_duration = all_raw.times[-1] - all_raw.times[0]
    print(f"Recording duration for subject {subj}: {(recording_duration/60):.2f} minutes")


    # Rename annotations
    if cfg["trigger"]["rename_annot"]:
        all_raw = convert_triggers(all_raw, cfg)


    # set bad channels
    if cfg['recording_info']['bads']:
        all_raw.info['bads'] = cfg['recording_info']['bads']
    


    # Write to BIDS -----------------------------------------------------------
    # set bids path
    bids_path = mne_bids.BIDSPath(
        subject=f"{subj:03}",
        session=cfg["session"]["session"],
        task=task,
        run="01",
        root=cfg['dirs']['bids_dir'],
    )

    # write raw data to BIDS
    mne_bids.write_raw_bids(
        all_raw,
        bids_path,
        allow_preload=True,
        overwrite=True,
        format="FIF",
        empty_room=emptyroom_bids_path,
    )


    # Write anatomical image to BIDS --------------------------------------
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
            overwrite=True,
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
