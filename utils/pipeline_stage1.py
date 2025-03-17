import os
import glob
import yaml
import numpy as np
from jwst.pipeline import Detector1Pipeline
from tqdm.auto import tqdm
import logging
import sys
import argparse
from multiprocessing import Pool, cpu_count
from contextlib import contextmanager

@contextmanager
def redirect_output_to_file(log_file):
    """Temporarily redirect stdout and stderr to a log file."""
    log_fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    stdout_fd = os.dup(1)  # Save the original stdout file descriptor
    stderr_fd = os.dup(2)  # Save the original stderr file descriptor
    try:
        os.dup2(log_fd, 1)  # Redirect stdout to log file
        os.dup2(log_fd, 2)  # Redirect stderr to log file
        yield
    finally:
        os.dup2(stdout_fd, 1)  # Restore original stdout
        os.dup2(stderr_fd, 2)  # Restore original stderr
        os.close(log_fd)

# Load configuration
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

os.environ['CRDS_PATH'] = config['crds_path']
os.environ['CRDS_SERVER_URL'] = config['crds_server_url']

def setup_logger(output_dir):
    """
    Setup a logger for the pipeline using the provided output directory.
    """
    parent_dir = os.path.dirname(output_dir)
    log_file_path = os.path.join(parent_dir, "logs/pipeline_stage1.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Ensure log directory exists

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    with open(log_file_path, 'a') as log_file:
        log_file.write("------------------\n")
        log_file.write("Stage 1 Processing\n")
        log_file.write("------------------\n\n")

    return log, log_file_path

def process_file(args):
    """Process a single file."""
    img, output_dir, log, log_file = args
    try:
        # Use the scoped redirection
        with redirect_output_to_file(log_file):
            Detector1Pipeline.call(
                img,
                steps={
                    'ramp_fit': {'maximum_cores': config['ramp_fit_cores']},
                    'jump': {'maximum_cores': config['jump_cores']}
                },
                output_dir=output_dir,
                save_results=True
            )
    except Exception as e:
        log.error(f"Failed to process {img}: {e}")

def main(combined_mode, input_dir, output_dir, nproc, log):
    if combined_mode:
        uncal_list = input_dir.split(',')
        uncal_list = np.sort(uncal_list)
    else:
        pattern = f"{input_dir}.fits"
        uncal_list = glob.glob(pattern)
        uncal_list = np.sort(uncal_list)

    log.info(f"Total files to process: {len(uncal_list)}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory created: {output_dir}")

    task_args = [(img, output_dir, log, log_file_path) for img in uncal_list]

    if nproc == 1:
        # Sequential processing for single-core execution
        with tqdm(total=len(task_args), file=sys.stderr) as pbar:
            for args in task_args:
                process_file(args) 
                pbar.update(1)

    else:
        num_files = len(uncal_list)
        effective_nproc = min(nproc, num_files)
        with Pool(processes=effective_nproc) as pool:
            with tqdm(total=len(task_args), file=sys.stdout) as pbar:
                for _ in pool.imap_unordered(process_file, task_args):
                    pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stage 1 of the JWST data reduction pipeline.')
    parser.add_argument('--combined_mode', action='store_true', help='Whether the final mosaic is a combination of multiple observations (multiple uncal.fits directories)')
    parser.add_argument('--input_dir', type=str, help='Directory containing input files.')
    parser.add_argument('--output_dir', type=str, help='Directory where output will be written')
    parser.add_argument('--nproc', type=int, default=cpu_count() // 2,
                        help='Number of parallel processes to use (default: half of available cores)')
    args = parser.parse_args()
    log, log_file_path = setup_logger(args.output_dir)

    main(args.combined_mode, args.input_dir, args.output_dir, args.nproc, log)
