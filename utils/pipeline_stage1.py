import os
import yaml
import numpy as np
from jwst.pipeline import Detector1Pipeline
from tqdm.auto import tqdm
import logging
import sys
import argparse
from mpi4py import MPI

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

os.environ['CRDS_PATH'] = config['crds_path']
os.environ['CRDS_SERVER_URL'] = config['crds_server_url']

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

obs_path = config['target']
log_file_path = os.path.join(obs_path, "pipeline.log")

formatter = logging.Formatter(f'%(asctime)s - Rank {rank} - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

if rank == 0:
    with open(log_file_path, 'a') as log_file:
        log_file.write("------------------\n")
        log_file.write("Stage 1 Processing\n")
        log_file.write("------------------\n\n")

file_handler = logging.FileHandler(log_file_path, mode='a')  # Append mode
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

crds_log = logging.getLogger("CRDS")
crds_log.setLevel(logging.INFO)
crds_handler = logging.FileHandler(log_file_path, mode='a')
crds_handler.setFormatter(formatter)
crds_log.addHandler(crds_handler)

stpipe_log = logging.getLogger("stpipe")
stpipe_log.setLevel(logging.INFO)
stpipe_handler = logging.FileHandler(log_file_path, mode='a')
stpipe_handler.setFormatter(formatter)
stpipe_log.addHandler(stpipe_handler)

for logger in [log, crds_log, stpipe_log]:
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)

def split_jobs(img_list):
    # img_count = len(img_list)
    # img_per_process = img_count//size
    # start = rank * img_per_process
    # end = start + img_per_process if rank != size - 1 else img_count
    # img_set = img_list[start:end]
    # return img_set
    img_set = [img_list[i] for i in range(len(img_list)) if i % size == rank]
    return img_set

def main(img, output_dir):
    
    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    result = Detector1Pipeline.call(
        img, 
        steps={
            'ramp_fit': {'maximum_cores': config['ramp_fit_cores']}, 
            'jump': {'maximum_cores': config['jump_cores']}
        }, 
        output_dir=output_dir, 
        save_results=True
    )

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Stage 1 of the JWST data reduction pipeline.')
    parser.add_argument('--output_dir', type=str, help='Directory where output will be written')
    args = parser.parse_args()

    path = config['uncal_path']
    file_list1 = os.listdir(path)
    uncal_list = [file for file in file_list1 if file.endswith('uncal.fits')]
    uncal_list = np.sort(uncal_list)
    total_files = len(uncal_list)
    uncal_set = split_jobs(uncal_list)

    if rank == 0:
        set_length = len(uncal_set)
        print(f'Number of processes: {size}')
        print(f'Files to process for root process: {set_length}')

    failed_files = []

    for i, img in enumerate(tqdm(uncal_set, file=sys.stderr, disable=(rank != 0))):
        try:
            main(os.path.join(path, img), args.output_dir)
        except Exception as e:
            log.error(f"Rank {rank} failed to process {img}: {e}")
            failed_files.append(os.path.join(path, img)) 
    
    log.info(f"Rank {rank} has finished processing.")
    comm.Barrier()

    all_failed_files = comm.gather(failed_files, root=0)
    if rank == 0:
        log.info("Reprocessing failed files on Rank 0...")
        combined_failed_files = [file for sublist in all_failed_files for file in sublist]
        print(f"Number of files to retry: {len(combined_failed_files)}")

        for file in combined_failed_files:
            try:
                main(file, args.output_dir)
                log.info(f"Successfully reprocessed {file}")
            except Exception as e:
                log.error(f"Failed to reprocess {file}: {e}")

    comm.Barrier()
    MPI.Finalize()

    