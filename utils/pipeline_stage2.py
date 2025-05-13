import os
import yaml
import numpy as np
from jwst.pipeline import Image2Pipeline
from tqdm.auto import tqdm
import logging
import sys
import argparse
from multiprocessing import Pool, cpu_count

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
    log_file_path = os.path.join(parent_dir, "logs/pipeline_stage2.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Ensure log directory exists

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    with open(log_file_path, 'a') as log_file:
        log_file.write("\n------------------\n")
        log_file.write("Stage 2 Processing\n")
        log_file.write("------------------\n\n")

    return log, log_file_path

def redirect_output_to_log(log_file):
    """Redirect stdout and stderr to the specified log file."""
    sys.stdout.flush()
    sys.stderr.flush()
    log_fd = open(log_file, 'a')
    os.dup2(log_fd.fileno(), sys.stdout.fileno())
    os.dup2(log_fd.fileno(), sys.stderr.fileno())

def process_file(args):
    """Process a single file."""
    img, output_dir, log, log_file = args
    try:
        redirect_output_to_log(log_file)

        # Run Image2Pipeline
        Image2Pipeline.call(
            img, 
            steps={'resample': {'skip': config['skip_resample']}}, 
            output_dir=output_dir, 
            save_results=True
        )
        print(f"Successfully processed {img}")
    except Exception as e:
        log.error(f"Failed to process {img}: {e}")

def main(input_dir, output_dir, nproc, log, log_file_path):
    # Get the list of rate.fits files
    file_list = os.listdir(input_dir)
    rate_list = [file for file in file_list if file.endswith('rate.fits')]
    rate_list = np.sort(rate_list)

    log.info(f"Total files to process: {len(rate_list)}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory created: {output_dir}")

    task_args = [(os.path.join(input_dir, img), output_dir, log, log_file_path) for img in rate_list]
    effective_nproc = min(nproc, len(task_args))
    with Pool(processes=effective_nproc) as pool:
        with tqdm(total=len(task_args), file=sys.stdout) as pbar:
            for _ in pool.imap_unordered(process_file, task_args):
                pbar.update(1)

    log.info("Pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stage 2 of the JWST data reduction pipeline.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory where input rate.fits files are located')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory where output will be written')
    parser.add_argument('--nproc', type=int, default=cpu_count() // 2,
                        help='Number of parallel processes to use (default: half of available cores)')
    args = parser.parse_args()

    log, log_file_path = setup_logger(args.output_dir)
    main(args.input_dir, args.output_dir, args.nproc, log, log_file_path)
