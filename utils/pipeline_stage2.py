import os
import yaml
import numpy as np
from jwst.pipeline import Image2Pipeline
from tqdm.auto import tqdm
import logging
import sys
import argparse
from multiprocessing import Pool, cpu_count

from log_utils import archive_existing_log

# Load configuration
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

os.environ['CRDS_PATH'] = os.path.expanduser(str(config['crds_path']))
os.environ['CRDS_SERVER_URL'] = config['crds_server_url']

def setup_logger(output_dir):
    """
    Setup a logger for the pipeline using the provided output directory.
    """
    parent_dir = os.path.dirname(output_dir)
    log_file_path = os.path.join(parent_dir, "logs/pipeline_stage2.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    archive_existing_log(log_file_path)

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
        # Stage 2 resampling is skipped by default (the mosaic is built in
        # stage 3); the JWST default would otherwise run it. An explicit
        # skip_resample: false in config still re-enables it.
        steps = {'resample': {'skip': config.get('skip_resample', True)}}
        # Deep-merge any guided per-step overrides from the UI. Absent =>
        # unchanged behaviour.
        for _step, _params in (config.get('stage2_step_overrides') or {}).items():
            steps.setdefault(_step, {}).update(_params or {})
        Image2Pipeline.call(
            img,
            steps=steps,
            output_dir=output_dir,
            save_results=True
        )
        print(f"Successfully processed {img}")
    except Exception as e:
        log.error(f"Failed to process {img}: {e}")
        print(f"[Stage2] FAILED {os.path.basename(img)}: {e}", flush=True)
        return False
    return True

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
    if not task_args:
        msg = f"No *rate.fits files found in {input_dir}."
        log.error(msg)
        print(f"[Stage2] {msg}", flush=True)
        return 1

    results = []
    effective_nproc = max(1, min(nproc, len(task_args)))
    with Pool(processes=effective_nproc) as pool:
        with tqdm(total=len(task_args), file=sys.stdout) as pbar:
            for ok in pool.imap_unordered(process_file, task_args):
                results.append(ok)
                pbar.update(1)

    n_failed = sum(1 for ok in results if not ok)
    if n_failed:
        msg = f"{n_failed} of {len(results)} exposures failed in stage 2. See {log_file_path}."
        log.error(msg)
        print(f"[Stage2] {msg}", flush=True)
        return 1

    log.info("Pipeline completed successfully.")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stage 2 of the JWST data reduction pipeline.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory where input rate.fits files are located')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory where output will be written')
    parser.add_argument('--nproc', type=int, default=cpu_count() // 2,
                        help='Number of parallel processes to use (default: half of available cores)')
    args = parser.parse_args()

    log, log_file_path = setup_logger(args.output_dir)
    sys.exit(main(args.input_dir, args.output_dir, args.nproc, log, log_file_path))
