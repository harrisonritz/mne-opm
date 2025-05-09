# run ols_ephys

# %%
from dotenv import load_dotenv, find_dotenv
import mne_bids
import mne
import os
import argparse
from osl_ephys.preprocessing.osl_wrappers import detect_badsegments, detect_badchannels, drop_bad_epochs
from mne_bids_pipeline._config_import import _update_config_from_path
from types import SimpleNamespace

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
    analysis_type : str
        Type of analysis to perform ('badsegment', 'badchannels', 'badepochs').
    run_noise : bool
        Whether to process noise/emptyroom data as well.
    
    Returns
    -------
    dict
        Dictionary with loaded data for each subject and task.
    """
 
    print(f"Loading data from BIDS directory: {cfg.bids_dir} for subjects: {cfg.subjects}")
    
    data_dict = {}

        
    # Define tasks to process
    if analysis_type == 'badepochs':
    
        path = BIDSPath(
            subject=f"{subject_id}",
            task=cfg.task,
            root=deriv_dir,
            datatype="meg",
            suffix="epo",
            extension=".fif"
        )
        epochs_path = mne_bids.find_matching_path 
            
            
            os.path.join(deriv_dir, f"sub-{subject_id}", "ses-01", "meg", 
                                        f"sub-{subject_id}_ses-01_task-TSXpilot_proc-clean_epo.fif")
            
            if os.path.exists(epochs_path):
                print(f"Loading epochs from: {epochs_path}")
                epochs = mne.read_epochs(epochs_path)
                data_dict[subject_id]['TSXpilot'] = epochs
                print(f"Successfully loaded TSXpilot epochs for subject {subject_id}")
            else:
                print(f"Warning: Could not find preprocessed epochs at {epochs_path}")
                
            # Handle noise epochs if requested
            if run_noise:
                noise_epochs_path = os.path.join(deriv_dir, f"sub-{subject_id}", "ses-01", "meg", 
                                                f"sub-{subject_id}_ses-01_task-noise_proc-clean_epo.fif")
                if os.path.exists(noise_epochs_path):
                    print(f"Loading noise epochs from: {noise_epochs_path}")
                    noise_epochs = mne.read_epochs(noise_epochs_path)
                    data_dict[subject_id]['noise'] = noise_epochs
                    print(f"Successfully loaded noise epochs for subject {subject_id}")
                else:
                    print(f"Warning: Could not find preprocessed noise epochs")
            
        except Exception as e:
            print(f"Error loading epochs for subject {subject_id}: {e}")
            
    else:
        # For badsegment and badchannels, load raw data from BIDS
        tasks = []
        if run_noise:
            tasks.append("noise")
        tasks.append("TSXpilot")  # Main task
        
        for task in tasks:
            print(f"Loading task: {task}")
            bids_path = mne_bids.BIDSPath(
                subject=f"{subject_id}",
                task=task,
                root=bids_dir,
                datatype="meg",
                extension=".fif"
            )
            
            try:
                raw = mne_bids.read_raw_bids(bids_path)
                data_dict[subject_id][task] = raw
                print(f"Successfully loaded {task} data for subject {subject_id}")
            except Exception as e:
                print(f"Error loading {task} data for subject {subject_id}: {e}")
    
    return data_dict


def run_analysis(data_dict, analysis_type):
    """Run the specified OSL analysis on the loaded data."""
    print(f"Running {analysis_type} analysis")
    
    results_dict = {}
    
    for subject_id, subject_data in data_dict.items():
        results_dict[subject_id] = {}
        
        for task_name, data in subject_data.items():
            print(f"Processing subject {subject_id}, task: {task_name}")
            
            if analysis_type == 'badsegment':
                # Detect bad segments in the data
                clean_data = detect_badsegments(data)
                results_dict[subject_id][task_name] = clean_data
                
            elif analysis_type == 'badchannels':
                # Detect bad channels in the data
                clean_data = detect_badchannels(data)
                results_dict[subject_id][task_name] = clean_data
                    
            elif analysis_type == 'badepochs':
                # Drop bad epochs from the already loaded epochs object
                if isinstance(data, mne.Epochs):
                    epochs_clean = drop_bad_epochs(data)
                    results_dict[subject_id][task_name] = epochs_clean
                else:
                    print(f"Error: Expected Epochs for {task_name}, but got {type(data)}")
                    results_dict[subject_id][task_name] = None
            
            else:
                print(f"Unknown analysis type: {analysis_type}")
                return None
    
    return results_dict


def save_results(results_dict, config, analysis_type):
    """Save the analysis results in BIDS format."""
    bids_dir = config['bids_root']
    deriv_dir = config['deriv_root']
    
    print(f"Saving results to BIDS directory: {bids_dir}")
    
    for subject_id, subject_data in results_dict.items():
        for task_name, data in subject_data.items():
            if data is None:
                continue
                
            print(f"Saving subject {subject_id}, task {task_name} data")
            
            if analysis_type == 'badepochs':
                # For badepochs, overwrite the original preprocessed epochs file
                epochs_path = os.path.join(deriv_dir, f"sub-{subject_id}", "ses-01", "meg", 
                                         f"sub-{subject_id}_ses-01_task-{task_name}_proc-clean_epo.fif")
                
                try:
                    data.save(epochs_path, overwrite=True)
                    print(f"Successfully saved cleaned epochs to {epochs_path}")
                except Exception as e:
                    print(f"Error saving epochs for subject {subject_id}, task {task_name}: {e}")
            
            else:
                # For raw data, use mne_bids to save with OSL processing label
                bids_path = mne_bids.BIDSPath(
                    subject=f"{subject_id}",
                    task=task_name,
                    root=bids_dir,
                    processing="osl",
                    suffix=analysis_type,
                    datatype="meg",
                    extension=".fif"
                )
                
                try:
                    if isinstance(data, mne.io.Raw):
                        mne_bids.write_raw_bids(data, bids_path, overwrite=True)
                        print(f"Successfully saved raw data")
                    else:
                        print(f"Expected Raw, but got {type(data)}. Unable to save.")
                except Exception as e:
                    print(f"Error saving results for subject {subject_id}, task {task_name}: {e}")


def main():

    """Main function to parse arguments and run the workflow."""
    parser = argparse.ArgumentParser(description='Run OSL preprocessing steps.')
    parser.add_argument('--analysis', choices=['badsegment', 'badchannels', 'badepochs'], 
                        help='Which analysis to run')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to mne_bids_pipeline configuration file')
    
    args = parser.parse_args()
    
    # Load configuration


    config_path = '/Users/hr0283/Projects/TSX_OPM/analysis/config/TSX-pilot/sub-004/config-preproc_sub-004.py'
    cfg = SimpleNamespace()
    _update_config_from_path(config=cfg, config_path=args.config)
    
    print(cfg)
    
    
    # Load data
    data_dict = load_data(cfg, args)
    if not data_dict:
        raise ValueError("No data loaded. Check your configuration and BIDS directory.")
    
    # Run analysis
    results_dict = run_analysis(data_dict, args.analysis)
    if not results_dict:
        raise ValueError("Analysis failed. Check your data and analysis type.")
    
    # Save results
    save_results(results_dict, config, args.analysis)
    
    print("OSL preprocessing completed successfully!")


if __name__ == "__main__":
    main()




