import os
import subprocess
from multiprocessing import current_process
import json
import yaml
import shutil
import numpy as np
from jwst.pipeline import Image3Pipeline
from astropy.io import fits
from tqdm.auto import tqdm
import logging
import sys
import argparse
import concurrent.futures
import psutil
import time

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

os.environ['CRDS_PATH'] = config['crds_path']
os.environ['CRDS_SERVER_URL'] = config['crds_server_url']

MAX_WORKERS = 8  # Maximum concurrent filters
MEMORY_THRESHOLD = 80  # Percent memory usage to avoid exceeding

def setup_logger(output_dir):
    """
    Setup a logger for the pipeline using the provided output directory.
    """
    parent_dir = os.path.dirname(output_dir)
    log_file_path = os.path.join(parent_dir, "logs/pipeline_stage3.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Ensure log directory exists

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    # Redirect CRDS logs
    crds_log = logging.getLogger("CRDS")
    crds_log.setLevel(logging.INFO)
    crds_handler = logging.FileHandler(log_file_path, mode='a')
    crds_handler.setFormatter(formatter)
    crds_log.addHandler(crds_handler)

    # Redirect stpipe logs
    stpipe_log = logging.getLogger("stpipe")
    stpipe_log.setLevel(logging.INFO)
    stpipe_handler = logging.FileHandler(log_file_path, mode='a')
    stpipe_handler.setFormatter(formatter)
    stpipe_log.addHandler(stpipe_handler)

    # Remove any StreamHandlers to prevent terminal output
    for logger in [log, crds_log, stpipe_log]:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)

    with open(log_file_path, 'a') as log_file:
        log_file.write("\n------------------\n")
        log_file.write("Stage 3 Processing\n")
        log_file.write("------------------\n\n")

    return log, log_file_path

def memory_usage():
    """Check memory usage in percent."""
    return psutil.virtual_memory().percent

def estimate_filter_memory(filter_dir):
    total_size = sum(os.path.getsize(os.path.join(filter_dir, f)) for f in os.listdir(filter_dir))
    base_overhead = 2 * 1024 ** 3  # 2 GB base overhead
    return total_size + base_overhead

def wait_for_memory_estimate(log, required_memory):
    """Wait until enough free memory exists for this process."""
    while (psutil.virtual_memory().available < required_memory):
        log.warning(f"Not enough memory for next filter ({required_memory / 1e9:.2f} GB). Waiting...")
        time.sleep(30)

def wait_for_memory(log):
    """Wait until memory usage drops below safe threshold."""
    while memory_usage() > MEMORY_THRESHOLD:
        log.warning(f"High memory usage detected ({memory_usage()}%). Waiting...")
        time.sleep(30)

def process_filter(filter_dir, log, target, long_cat, long_params, config):
    log, _ = setup_logger(filter_dir + '/output_files/')
    try:
        required_memory = estimate_filter_memory(filter_dir)
        wait_for_memory_estimate(log, required_memory)
        stage3(filter_dir, log, target, reference_catalog=long_cat, resample_params=long_params, config=config)
    except Exception as e:
        log.error(f"Error processing filter {filter_dir}: {e}")

def process_filters_parallel(filter_dirs, target, long_cat, long_params, output_base_dir, config):
    num_workers = min(8, len(filter_dirs))

    log, _ = setup_logger(output_base_dir)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for dir in filter_dirs:
            future = executor.submit(process_filter, dir, log, target, long_cat, long_params, config)
            futures[future] = dir

        for future in concurrent.futures.as_completed(futures):
            dir = futures[future]
            if future.exception():
                log.error(f"Filter processing failed for {dir}: {future.exception()}")
            else:
                log.info(f"Completed processing for {dir}")

def get_filter_from_exposure(exp):
    header = fits.getheader(exp, ext=0)  # Directly get the header from the primary HDU
    return header['FILTER']

def organize_exposures_by_filter(input_dir, output_base_dir, log):
    files = [file for file in os.listdir(input_dir) if file.endswith('cal_final.fits')]
    if not files:
        log.info("No 'cal_final.fits' files found in the input directory. Exiting function.")
        return
    
    for filename in files:
        full_path = os.path.join(input_dir, filename)
        if os.path.isfile(full_path):
            filter = get_filter_from_exposure(full_path)
            filter_dir = os.path.join(output_base_dir, filter)
            if not os.path.exists(filter_dir):
                os.makedirs(filter_dir)
            new_filename = filename.replace('cal_final.fits', 'cal.fits')
            new_full_path = os.path.join(filter_dir, new_filename)
            shutil.move(full_path, new_full_path)

def create_custom_association(filter_dir, output_filename, program, target, instrument, filter, pupil="clear", subarray="full", exp_type="nrc_image"):
    """
    Creates a single custom association file with specified metadata, including all exposures in the directory.

    :param directory: The directory containing the FITS files.
    :param output_filename: The path to save the association JSON file.
    :param program: The program ID.
    :param target: The target ID.
    :param instrument: The instrument used.
    :param filter: The filter used.
    :param pupil: The pupil setting (default "clear").
    :param subarray: The subarray setting (default "full").
    :param exp_type: The exposure type (default "nrc_image").
    """
    members = []

    for file in os.listdir(filter_dir):
        if file.endswith('.fits'):
            # Assuming all files should be included as science exposures
            members.append({
                "expname": file,
                "exptype": "science",
                "exposerr": None,
                "asn_candidate": "(custom, observation)"
            })

    association = {
        "asn_type": "image3",
        "asn_rule": "candidate_Asn_Lv3Image",
        "version_id": None,
        "code_version": "1.16.0",
        "degraded_status": "No known degraded exposures in association.",
        "program": program,
        "constraints": f"DMSAttrConstraint('{{'name': 'program', 'sources': ['program'], 'value': '{program[1:]}'}})\n"
                       f"DMSAttrConstraint('{{'name': 'instrument', 'sources': ['instrume'], 'value': '{instrument}'}})\n"
                       f"DMSAttrConstraint('{{'name': 'opt_elem', 'sources': ['filter'], 'value': '{filter}'}})\n"
                       f"DMSAttrConstraint('{{'name': 'opt_elem2', 'sources': ['pupil'], 'value': '{pupil}'}})\n"
                       f"DMSAttrConstraint('{{'name': 'subarray', 'sources': ['subarray'], 'value': '{subarray}'}})\n"
                       f"Constraint_Image('{{'name': 'exp_type', 'sources': ['exp_type'], 'value': '{exp_type}'}})",
        "asn_id": filter + "_asn",  
        "target": target,
        "asn_pool": filter + "_pool",
        "products": [
            {
                "name": f"{target}_nircam_clear-{filter}",
                "members": members
            }
        ]
    }

    # Save the association to a JSON file
    with open(output_filename, 'w') as f:
        json.dump(association, f, indent=4)
    

def convert_catalog_to_tweakreg_format(folder_name, filter):
    folder_path = os.path.join(os.getcwd(), folder_name)
    
    awk_command = f"awk '{{print $4 \",\" $5}}' {folder_path}/*nircam_clear-{filter}_cat.ecsv > {folder_path}/{filter}.tmp"
    os.system(awk_command)
    os.system(f"tail -n +270 {folder_path}/{filter}.tmp > {folder_path}/{filter}.tmp2")
    os.system(f"echo 'RA,DEC' > {folder_path}/{filter}.csv")
    os.system(f"cat {folder_path}/{filter}.tmp2 >> {folder_path}/{filter}.csv")
    os.system(f"rm {folder_path}/{filter}.tmp {folder_path}/{filter}.tmp2")

def extract_resample_info(mosaic_img_path):
    with fits.open(mosaic_img_path) as mosaic_img:
        resample_info = {
            'naxis1': mosaic_img[1].header.get('NAXIS1', None),
            'naxis2': mosaic_img[1].header.get('NAXIS2', None),
            'crpix1': mosaic_img[1].header.get('CRPIX1', None),
            'crpix2': mosaic_img[1].header.get('CRPIX2', None),
            'crval1': mosaic_img[1].header.get('CRVAL1', None),
            'crval2': mosaic_img[1].header.get('CRVAL2', None),
        }
    return resample_info

def stage3(filter_dir, log, target, reference_catalog=None, resample_params=None, config=None):

    output_dir = filter_dir + '/output_files'

    processed_file = os.path.join(output_dir, f'{target}_nircam_clear-{os.path.basename(filter_dir)}_i2d.fits')
    if os.path.exists(processed_file):
        log.info(f"Skipping processing for {filter_dir} as output already exists.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tweakreg_config = {}
    if reference_catalog:
        tweakreg_config = {
            'tweakreg': {
                'starfinder': config['starfinder'],
                'snr_threshold': config['tweakreg_snr'],
                'abs_refcat': reference_catalog,
                'abs_fitgeometry': 'general',
                'fitgeometry': 'general'
            }
        }
    if reference_catalog == None:
        tweakreg_config = {
            'tweakreg': {
                'starfinder': config['starfinder'],
                'snr_threshold': config['tweakreg_snr'],
                'abs_refcat': 'GAIADR3',
                'fitgeometry': 'general'
            }
        }

    resample_config = {}
    if resample_params:
        resample_config = {
            'resample': {
                'kernel':config['res_kernel'],
                'pixel_scale':config['pixel_scale'],
                'pixfrac':config['pixfrac'],
                'rotation':config['rotation'],
                'crpix': [resample_params['crpix1']-1.0, resample_params['crpix2']-1.0],
                'crval': [resample_params['crval1'], resample_params['crval2']],
                'output_shape': [resample_params['naxis1'], resample_params['naxis2']],
                'in_memory':config['resample_in_memory']
            }
        }
    if resample_params == None:
        resample_config = {
            'resample': {
                'kernel':config['res_kernel'],
                'pixel_scale':config['pixel_scale'],
                'pixfrac':config['pixfrac'],
                'rotation':config['rotation'],
                'in_memory':config['resample_in_memory']
            }
        }

    outlier_config = {
        'outlier_detection': {
                'pixfrac':config['pixfrac'],
                'in_memory':config['outlier_in_memory']
            }
        }
    
    skymatch_config = {
        'skymatch': {
            'skymethod':config['skymethod'],
            'subtract':True
        }
    }

    step_config = {**tweakreg_config, **resample_config, **outlier_config, **skymatch_config}

    filter = os.path.basename(filter_dir)
    program = "00000"
    instrument = "nircam"
    output_filename = filter_dir + '/' + filter + '_asn.json'

    create_custom_association(filter_dir, output_filename, program, target, instrument, filter)

    asn_list = [os.path.join(filter_dir, file) for file in os.listdir(filter_dir) if file.endswith('asn.json')]
    asn = asn_list[0]

    result = Image3Pipeline.call(asn, steps=step_config, output_dir=output_dir, save_results=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stage 1 of the JWST data reduction pipeline.')
    parser.add_argument('--output_dir', type=str, help='Directory where output will be written')
    parser.add_argument('--input_dir', type=str, help='Directory where output will be written')
    parser.add_argument('--target', type=str, help='Target name')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_base_dir = args.output_dir
    target = args.target
    log, log_file_path = setup_logger(output_base_dir)

    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    filter_mapping = {
        'F070W': 0.7, 'F090W': 0.9, 'F115W': 1.15, 'F140M': 1.41, 'F150W': 1.5,
        'F162M': 1.63, 'F164N': 1.65, 'F150W2': 1.69, 'F182M': 1.85, 'F187N': 1.87, 'F200W': 2.0,
        'F210M': 2.1, 'F212N': 2.12, 'F250M': 2.5, 'F277W': 2.78, 'F300M': 3.0,
        'F323N': 3.24, 'F322W2': 3.25, 'F335M': 3.36, 'F356W': 3.57, 'F360M': 3.62,
        'F405N': 4.05, 'F410M': 4.08, 'F430M': 4.28, 'F444W': 4.40, 'F460M': 4.63,
        'F466N': 4.65, 'F470N': 4.71, 'F480M': 4.81,
    }

    #organize_exposures_by_filter(input_dir, output_base_dir, log)
    filter_dirs = [os.path.join(output_base_dir, d) for d in os.listdir(output_base_dir) if os.path.isdir(os.path.join(output_base_dir, d))]
    filter_names = [os.path.basename(d) for d in filter_dirs]
    sorted_filters = sorted(filter_names, key=lambda x: filter_mapping[x], reverse=True)
    sorted_filter_dirs = [os.path.join(output_base_dir, f) for f in sorted_filters]

    # Process first filter (longest wavelength) serially
    ref_cat = config.get("external_reference", None) or None
    if ref_cat is None:
        log.info('NO REFERENCE CATALOG PROVIDED, USING GAIADR3')

    #stage3(sorted_filter_dirs[0], log, target, reference_catalog=ref_cat, resample_params=None, config=config)

    path_longest = sorted_filter_dirs[0] + '/output_files/'
    #convert_catalog_to_tweakreg_format(path_longest, sorted_filters[0])
    long_cat = os.path.join(path_longest, f'{sorted_filters[0]}.csv')
    long_list = [file for file in os.listdir(path_longest) if file.endswith(f'nircam_clear-{sorted_filters[0]}_i2d.fits')]
    long_processed_file = os.path.join(path_longest, long_list[0])
    long_params = extract_resample_info(long_processed_file)

    # Process all other filters in parallel
    process_filters_parallel(sorted_filter_dirs[1:], target, long_cat, long_params, output_base_dir, config=config)
