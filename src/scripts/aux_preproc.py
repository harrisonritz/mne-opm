# Run Auxiliary preprocessing steps on OPM data
#
# This script is designed to run auxiliary preprocessing steps on OPM data using MNE-Python and OSL.
# It includes functions to load configuration, load data from BIDS, run analyses (bad segments, bad channels, bad epochs),
# and save results in BIDS format.

# %%
from dotenv import load_dotenv, find_dotenv
import mne_bids
import mne
import os
import argparse
import importlib.util
import pandas as pd  # Add pandas for handling TSV files
from osl_ephys.preprocessing.osl_wrappers import bad_segments, bad_channels, drop_bad_epochs
from mne_bids_pipeline._config_import import _update_config_from_path
from types import SimpleNamespace



# %% GLOBAL VARIABLES

SEGMENT_LEN = 1.0




# %% functions
def load_config(config_path):
    """Load configuration from a Python file."""
    print(f"Loading configuration from: {config_path}")
    
    # Load the configuration file as a module
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Extract necessary variables
    config = {
        'subjects': getattr(config_module, 'subjects', []),
        'deriv_root': getattr(config_module, 'deriv_root', None),
        'bids_root': getattr(config_module, 'bids_root', None),
    }
    
    return config


def load_data(cfg, args):
    """Load data from BIDS directory based on configuration.
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary with subjects, bids_root, and deriv_root.
    args : argparse.Namespace
        Command line arguments containing analysis type and other parameters.
    
    Returns
    -------
    dict
        Dictionary with loaded data for each subject and task.
    """
 
    print(f"\n---------\nLoading data from BIDS directory: {cfg.bids_root} for subjects: {cfg.subjects}\n")

        
    # Define tasks to process
    data_dict = {}

    if args.analysis in ['badsegments', 'badchannels', 'manualchannel']:

        # For bad_segments and bad_channels, load raw data from BIDS
        tasks = []
        if cfg.process_empty_room:
            tasks.append("noise")
        tasks.append(cfg.task)  # Main task
        
        for task in tasks:
            print(f"Loading task: {task}")
            bids_path = mne_bids.find_matching_paths(
                root=cfg.bids_root,
                subjects=cfg.subjects,
                tasks=task,
                sessions=cfg.sessions,
                datatypes="meg",
                ignore_nosub=True,
                extensions=".fif"
            )[0]

            raw = mne_bids.read_raw_bids(bids_path, extra_params={'preload': True})
            data_dict[task] = raw
            print(f"Successfully loaded {task} data for subject {cfg.subjects[0]}")


    elif args.analysis in ['badepochs', 'manualica']:
    
        bids_path = mne_bids.find_matching_paths(
            root=cfg.deriv_root,
            subjects=cfg.subjects,
            tasks=cfg.task,
            sessions=cfg.sessions,
            datatypes="meg",
            suffixes="epo",
            processings='clean',
            extensions=".fif",
        )[0]
        print(f"Found epochs path: {bids_path}")
            
        epochs = mne.read_epochs(bids_path, preload=True)
        data_dict[cfg.task] = epochs
        print(f"Successfully loaded TSXpilot epochs for subject {cfg.subjects[0]}")


        if args.analysis == 'manualica':

            # Load ICA data
            bids_path = mne_bids.find_matching_paths(
                root=cfg.deriv_root,
                subjects=cfg.subjects,
                tasks=cfg.task,
                sessions=cfg.sessions,
                datatypes="meg",
                suffixes="ica",
                processings='ica',
                extensions=".fif",
            )[0]
            print(f"Found ICA path: {bids_path}")
                
            ica = mne.preprocessing.read_ica(bids_path)
            data_dict['manualica'] = ica
            print(f"Successfully loaded ICA data for subject {cfg.subjects[0]}")

    else:
        raise ValueError(f"Unknown analysis type: {args.analysis}. Unable to load data.")

    return data_dict


def run_analysis(cfg, args, data_dict):
    """Run the specified OSL analysis on the loaded data."""
    
    print(f"\n---------\nRunning {args.analysis} analysis\n")

    results_dict = {'bads': []}
    for task, data in data_dict.items():
        print(f"Processing SUBJECT: {cfg.subjects[0]} // TASK: {task} // DATA: {data}")

        if args.analysis == 'badsegments':
            # Detect bad segments in the data

            clean_data = bad_segments(
                data, 
                picks=cfg.ch_types[0],
                ref_meg=False,
                metric="kurtosis",
                detect_zeros=False,
                channel_wise=False,
                segment_len=round(data.info['sfreq']*SEGMENT_LEN),
                )
            results_dict[task] = clean_data


        elif args.analysis == 'badchannels':
            # Detect bad channels in the data

            # Filter the raw data
            data_filt = data.copy().filter(
                l_freq=cfg.l_freq,
                h_freq=cfg.h_freq,
                method='iir',
                )

            # Detect bad channels
            clean_data = bad_channels(
                data_filt,
                picks=cfg.ch_types[0], 
                ref_meg=None, 
                significance_level=0.05, 
                )
            del data_filt

            results_dict[task] = data
            results_dict['bads'].extend(clean_data.info['bads'])
            print(f"Bad channels in task {task}: [{clean_data.info['bads']}]")


        elif args.analysis == 'badepochs':
            # Drop bad epochs from the already loaded epochs object
            results_dict[task] = drop_bad_epochs(
                data,
                picks=cfg.ch_types[0],
                ref_meg=None,
                metric='std',
                )
            

    if args.analysis == 'manualchannel':
        
        # Drop bad epochs from the already loaded epochs object
        mne.viz.use_browser_backend('qt')
        data_dict[cfg.task].plot(
            n_channels=64,
            show_options=True,
            show=True,
            block=True,
            # highpass=cfg.l_freq,
            # lowpass=cfg.h_freq,
            decim=5,
            use_opengl=True,
            scalings=dict(mag=2e-12),
        )
        
        # update bads based on selections to the raw data
        results_dict['bads'] = []
        for item in data_dict[cfg.task].info['bads']:
            if isinstance(item, str):
                results_dict['bads'].append(item)
            else:
                results_dict['bads'].append(item.item())

        results_dict[cfg.task] = data_dict[cfg.task]
        results_dict[cfg.task].info['bads'] = results_dict['bads']
        if cfg.process_empty_room:
            results_dict['noise'] = data_dict['noise']
            results_dict['noise'].info['bads'] = results_dict['bads']



    elif args.analysis == 'manualica':

        # Run ICA on the data
        mne.viz.use_browser_backend('qt')
        ica = data_dict['manualica']
        data = data_dict[cfg.task]


        # ica.find_bads_ref(
        #     inst=data, 
        #     ch_name='ref_meg', 
        #     threshold=2.0)

        # plot components and sources
        ica.plot_components(
            inst=data,
            nrows=6,
        )

        ica.plot_sources(
            inst=data,
            show_scrollbars=True,
            block=True,
            use_opengl=True,
        )

        results_dict['ica'] = ica




    return results_dict


def save_results(cfg, args, results_dict):
    """Save the analysis results in BIDS format."""
    
    print(f"\n---------\nSaving results to BIDS directory: {cfg.bids_root}\n")

    task_dict = {task: data for task, data in results_dict.items() if task not in ['bads', 'ica']} 
    for task, data in task_dict.items():
        
        if data is None:
            raise ValueError(f"No data to save for task: {task}")
            
        print(f"Saving subject {cfg.subjects[0]}, task {task} data")

        if args.analysis in ['badsegments', 'badchannels', 'manualchannel']:

            # For bad segements, use mne_bids to save with OSL processing label
            bids_path = mne_bids.find_matching_paths(
                root=cfg.bids_root,
                subjects=cfg.subjects,
                tasks=task,
                sessions=cfg.sessions,
                datatypes="meg",
                ignore_nosub=True,
                splits=None,
                extensions=".fif"
            )[0]

            bids_path.split = None


            if args.analysis in ['badchannels', 'manualchannel']:

                if results_dict['bads'] == []:
                    print(f"No bad channels detected. No changes made to raw data.")

                else:

                    # Mark bad channels in the raw data 
                    if args.analysis == 'badchannels':
                        data.info['bads'].extend(results_dict['bads'])
                    elif args.analysis == 'manualchannel':
                        # update bads based on selections to the raw data
                        data.info['bads'] = results_dict['bads']
                    print(f"Bad channels marked in raw data: {data.info['bads']}")

                    # mark channels in the BIDS dataset
                    mne_bids.mark_channels(
                        bids_path,
                        ch_names=results_dict['bads'],
                        status='bad',
                        descriptions='osl',
                        )
                    print(f"Successfully updated bad channels")
                    

            print(f"Saving to: {bids_path}")
            data.save(
                bids_path,
                split_naming='bids',  
                overwrite=True)
            print(f"Successfully saved cleaned data")



        elif args.analysis == 'badepochs':

            # For bad epochs, overwrite the original preprocessed epochs file
            bids_path = mne_bids.find_matching_paths(
                root=cfg.deriv_root,
                subjects=cfg.subjects,
                tasks=cfg.task,
                sessions=cfg.sessions,
                suffixes="epo",
                processings='clean',
                extensions=".fif",
            )[0]
            bids_path.split = None

            print(f"Saving to: {bids_path}")
            data.save(
                bids_path,
                split_naming='bids', 
                overwrite=True)
            print(f"Successfully saved claned epochs")

        else:
            print(f"Unknown analysis type: {args.analysis}. Unable to save results.")
            return None
        
    if args.analysis == 'manualica':
        # save ica components to tsv and *_ica.fif
        # Find the path to the ICA components TSV file
        tsv_path = mne_bids.find_matching_paths(
            root=cfg.deriv_root,
            subjects=cfg.subjects,
            tasks=cfg.task,
            sessions=cfg.sessions,
            suffixes="components",
            processings='ica',
            extensions=".tsv",
        )[0]

        # load the tsv file
        print(f"Loading ICA components from: {tsv_path}")
        components_df = pd.read_csv(tsv_path, sep='\t')
        
        # Get the ICA object and its excluded components
        ica = results_dict['ica']
        bad_components = ica.exclude
        print(f"Excluded components: {bad_components}")
        
        if not bad_components:
            print("No bad ICA components detected.")
        else:
            print(f"Marking {len(bad_components)} components as bad: {bad_components}")
            
            # Update the status column for bad components
            for comp_idx in bad_components:
                # Find rows where the 'component' column equals the component index
                mask = components_df['component'].astype(str) == str(comp_idx)
                if mask.any():
                    components_df.loc[mask, 'status'] = 'bad'
                    components_df.loc[mask, 'status_description'] = 'manual'
                else:
                    print(f"Warning: Component {comp_idx} not found in components file")
            
            print(components_df[components_df['status'] == 'bad'])  # Print only bad components for verification
        
        # Save the updated components file
        components_df.to_csv(tsv_path, sep='\t', index=False)
        print(f"Successfully updated ICA components file: {tsv_path}")
        
        # Also save the ICA object with updated exclusions
        ica_path = mne_bids.find_matching_paths(
            root=cfg.deriv_root,
            subjects=cfg.subjects,
            tasks=cfg.task,
            sessions=cfg.sessions,
            suffixes="ica",
            processings='ica',
            extensions=".fif",
        )[0]
        
        print(f"Saving updated ICA object to: {ica_path}")
        ica.save(ica_path, overwrite=True)
        print(f"Successfully saved updated ICA object")
           
        
def main():

    """Main function to parse arguments and run the workflow."""
    parser = argparse.ArgumentParser(description='Run OSL preprocessing steps.')
    parser.add_argument('--analysis', choices=['bad_segments', 'bad_channels', 'bad_epochs', 'manual_channel', 'manual_ica'], 
                        required=True, 
                        help='Which analysis to run')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to mne_bids_pipeline configuration file')
    
    args = parser.parse_args()
    args.analysis = args.analysis.replace('_', '')  # Normalize analysis name for dictionary keys
    
    # Load configuration
    config_path = '/Users/hr0283/Projects/TSX_OPM/analysis/config/TSX-pilot/sub-004/config-preproc_sub-004.py'
    cfg = SimpleNamespace()
    _update_config_from_path(config=cfg, config_path=args.config)

    if (args.analysis == 'rawplot' and not cfg._manual_bads):
        print("Skipping raw plot analysis as it is not enabled in the configuration.")
        return None

    if args.analysis == 'manualica':
        if not cfg._manual_ica or (cfg.spatial_filter != 'ica'):
            print("Skipping ICA analysis as it is not enabled in the configuration.")
            return None

    # Load data
    data_dict = load_data(cfg, args)
    if not data_dict:
        raise ValueError("No data loaded. Check your configuration and BIDS directory.")
    
    # Run analysis
    results_dict = run_analysis(cfg, args, data_dict)
    if not results_dict:
        raise ValueError("Analysis failed. Check your data and analysis type.")
    
    # Save results
    save_results(cfg, args, results_dict)

if __name__ == "__main__":
    main()




