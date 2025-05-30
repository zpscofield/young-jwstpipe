import os
import glob
import sys
from astropy.io import fits
import yaml

def get_observation_info(data_dir, dir_prefix="MAST_", combine=False, group_by_directory=False, name="Combined_Mosaic"):
    """
    Extract observation information from directories with a specified prefix.

    Args:
        data_dir (str): Data directory to search for uncalibrated data.
        dir_prefix (str): Prefix for directories to search (default: "MAST_").
        combine (bool): Whether to treat all observations as a single mosaic.
        group_by_directory (bool): Whether to group files by their parent directory.
        name (str): Custom name for the combined observation.

    Returns:
        tuple: Contains two elements:
            - obs_info (list of tuples): Each tuple contains (observation name, directory or list of files).
            - program_names (list): A unique list of all target names.
    """
    obs_info = []
    program_names = set()

    # Search for all directories matching the prefix
    search_pattern = os.path.join(data_dir, f"{dir_prefix}*", "**", "*_uncal.fits")
    uncal_files = glob.glob(search_pattern, recursive=True)

    if uncal_files:
        if group_by_directory:
            # Group files by their parent directory
            parent_dirs = set(os.path.dirname(file) for file in uncal_files)
            for parent_dir in parent_dirs:
                # Extract the first subdirectory under the working directory
                first_subdir = os.path.relpath(parent_dir, data_dir).split(os.sep)[0]
                obs_name = f"Output_{first_subdir}".replace(" ", "_").replace(".", "_")

                files_in_dir = [file for file in uncal_files if os.path.dirname(file) == parent_dir]
                obs_info.append((obs_name, ",".join(files_in_dir)))

                for file in files_in_dir:
                    with fits.open(file) as hdul:
                        program_id = hdul[0].header.get('PROGRAM', '00000').strip()
                        program_names.add(program_id)
        else:
            for uncal in uncal_files:
                with fits.open(uncal) as hdul:
                    parent_dir = os.path.dirname(uncal)
                    full_id = os.path.basename(uncal).split('_')[0]
                    observation_id = full_id[:11]
                    program_id = hdul[0].header.get('PROGRAM', '00000').strip()
                    updated_dir = os.path.join(parent_dir, observation_id + '*')

                if not combine and program_id not in program_names:
                    program_names.add(program_id)  # Add to the set of unique target names
                    obs_info.append((program_id, updated_dir)) # For individual mode, append each observation to obs_info

            if combine:
                # For combined mode, include all files instead of a single directory
                for uncal in uncal_files:
                    with fits.open(uncal) as hdul:
                        program_id = hdul[0].header.get('PROGRAM', '00000').strip()
                        program_names.add(program_id)

                obs_info.append((name, ",".join(uncal_files)))
    return obs_info, list(program_names)        

if __name__ == "__main__":
    pipeline_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    config_file = os.path.join(pipeline_dir, "config.yaml")
    # Read combine_observations from config
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    data_dir = config.get("data_directory")
    combine_observations = config.get("combine_observations", False)
    custom_name = config.get("custom_name", "Combined_Mosaic")
    group_directory = config.get("group_by_directory", False)
    prefix = config.get("dir_prefix", "MAST_") or "MAST_"

    obs_info, program_names = get_observation_info(data_dir, dir_prefix=prefix, combine=combine_observations, group_by_directory=group_directory, name=custom_name)

    # Print observation info for the shell script
    for target, dir_path_or_files in obs_info:
        print(f"OBS:{target}:{dir_path_or_files}")
    
    # Print the list of all unique target names
    print(f"TARGET_NAMES:{','.join(program_names)}")
