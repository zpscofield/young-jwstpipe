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

def setup_filter_logger(filter_dir):
    filter_name = os.path.basename(filter_dir)
    log_file_path = os.path.join(filter_dir, f"stage3_{filter_name}.log")

    log_filter = logging.getLogger(f"filter_logger_{filter_name}")
    log_filter.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file_path, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    log_filter.addHandler(file_handler)

    # Redirect CRDS logs
    crds_log_filter = logging.getLogger("CRDS")
    crds_log_filter.setLevel(logging.INFO)
    crds_handler = logging.FileHandler(log_file_path, mode='a')
    crds_handler.setFormatter(formatter)
    crds_log_filter.addHandler(crds_handler)

    # Redirect stpipe logs
    stpipe_log_filter = logging.getLogger("stpipe")
    stpipe_log_filter.setLevel(logging.INFO)
    stpipe_handler = logging.FileHandler(log_file_path, mode='a')
    stpipe_handler.setFormatter(formatter)
    stpipe_log_filter.addHandler(stpipe_handler)

    # Remove any StreamHandlers to prevent terminal output
    for logger in [log_filter, crds_log_filter, stpipe_log_filter]:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)
    return log_filter

def process_filter(filter_dir, log, target, long_cat, long_params, extract_settings, config):
    log_filter = setup_filter_logger(filter_dir)
    try:
        stage3(filter_dir, log_filter, target, reference_catalog=long_cat, resample_params=long_params, config=config)
        extract_data(filter_dir, target, extract_settings, log_filter)
    except Exception as e:
        log.error(f"Error processing filter {filter_dir}: {e}")

def process_filters_parallel(filter_dirs, target, long_cat, long_params, extract_settings, config):
    num_workers = min(config['min_processes'], len(filter_dirs))

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=len(filter_dirs), file=sys.stdout, desc="Processing Filters") as pbar:
            futures = {}
            for dir in filter_dirs:
                future = executor.submit(process_filter, dir, log, target, long_cat, long_params, extract_settings, config)
                futures[future] = dir

            for future in concurrent.futures.as_completed(futures):
                dir = futures[future]
                if future.exception():
                    log.error(f"Filter processing failed for {dir}: {future.exception()}")
                else:
                    log.info(f"Completed processing for {dir}")
                    pbar.update(1)

def extract_data(filter_dir, target, extract_settings, log):
    suffixes = ['sci', 'err', 'con', 'wht', 'var_poisson', 'var_rnoise', 'var_flat']
    output_dir = filter_dir + '/output_files'
    processed_file = os.path.join(output_dir, f'{target}_nircam_clear-{os.path.basename(filter_dir)}_i2d.fits')
    log.info(f"Extracting requested data from {os.path.basename(filter_dir)} i2d file...")
    for i in range(len(extract_settings)):
        if extract_settings[i]:
            data = fits.getdata(processed_file, ext=(i+1))
            header = fits.getheader(processed_file, ext=(i+1))
            pri_header = fits.getheader(processed_file, ext=0)

            primary_hdu = fits.PrimaryHDU(data=None, header=pri_header)
            image_hdu = fits.ImageHDU(data=data, header=header)
            hdulist = fits.HDUList([primary_hdu, image_hdu])
            hdulist.writeto(f'{output_dir}/{target}_{os.path.basename(filter_dir)}_{suffixes[i]}.fits', overwrite=True)
            log.info(f"Extracted {suffixes[i]} data from {os.path.basename(filter_dir)} i2d file.")
    log.info(f"Data extraction complete for {os.path.basename(filter_dir)}.")

def get_filter_from_exposure(exp):
    header = fits.getheader(exp, ext=0)
    return header['FILTER']

def organize_exposures_by_filter(input_dir, output_base_dir, log, suffix="_cal"):
    """
    Organize exposures by filter, given a suffix like '_cfnoise', '_wisp', or '_cal'.
    """
    files = [file for file in os.listdir(input_dir) if file.endswith(f'{suffix}.fits')]
    
    if not files:
        log.info(f"No '*{suffix}.fits' files found in the input directory. Exiting function.")
        return
    
    for filename in files:
        full_path = os.path.join(input_dir, filename)
        if os.path.isfile(full_path):
            filter = get_filter_from_exposure(full_path)
            filter_dir = os.path.join(output_base_dir, filter)
            new_filename = filename.replace(f'{suffix}.fits', '.fits')
            if not os.path.exists(filter_dir):
                os.makedirs(filter_dir)
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
                'snr_threshold': config['snr_threshold'],
                'abs_refcat': reference_catalog,
                'abs_fitgeometry': config['abs_fitgeometry'],
                'fitgeometry': config['fitgeometry'],
            }
        }
    if reference_catalog == None:
        tweakreg_config = {
            'tweakreg': {
                'starfinder': config['starfinder'],
                'snr_threshold': config['snr_threshold'],
                'abs_refcat': '',
                'fitgeometry': config['fitgeometry'],
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
                'in_memory': config['resample_in_memory']
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

    source_cat_config = {
        'source_catalog': {
            'snr_threshold':config['snr_threshold']
        }
    }

    step_config = {**tweakreg_config, **resample_config, **outlier_config, **skymatch_config, **source_cat_config}

    filter = os.path.basename(filter_dir)
    if config.get('combine_observations') == True:
        program = "00000"
    else:
        fits_files = [f for f in os.listdir(filter_dir) if f.endswith('.fits')]
        header = fits.getheader(os.path.join(filter_dir, fits_files[0]), ext=0)
        program = header['PROGRAM']
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
    parser.add_argument('--input_suffix', type=str, default="_cal", help='Suffix of the input cal files')

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

    organize_exposures_by_filter(input_dir, output_base_dir, log, suffix=args.input_suffix)
    filter_dirs = [os.path.join(output_base_dir, d) for d in os.listdir(output_base_dir) if os.path.isdir(os.path.join(output_base_dir, d))]
    filter_names = [os.path.basename(d) for d in filter_dirs]
    sorted_filters = sorted(filter_names, key=lambda x: filter_mapping[x], reverse=True)
    sorted_filter_dirs = [os.path.join(output_base_dir, f) for f in sorted_filters]

    extract_settings = [
        config.get('extract_sci'), 
        config.get('extract_err'), 
        config.get('extract_con'), 
        config.get('extract_wht'), 
        config.get('extract_var_poisson'), 
        config.get('extract_var_rnoise'), 
        config.get('extract_var_flat')
        ]
    
    # Process first filter
    ref_cat = config.get("external_reference", None) or None
    if ref_cat is None:
        log.info('No reference catalog provided, no absolute astrometric fitting performed.')
    log.info('Running stage 3 for the longest wavelength first...')
    print('Running stage 3 for the longest wavelength first...')

    log_long_filter = setup_filter_logger(sorted_filter_dirs[0])
    stage3(sorted_filter_dirs[0], log_long_filter, target, reference_catalog=ref_cat, resample_params=None, config=config)
    extract_data(sorted_filter_dirs[0], target, extract_settings, log_long_filter)

    use_multiprocessing = config.get('stage3_use_multiprocessing', False)
    if use_multiprocessing == True:
        print('Finished. Starting stage 3 processing for remaining filters with multiprocessing.')
        log.info('Finished. Starting stage 3 processing for remaining filters with multiprocessing.')
    else:
        print('Finished. Starting stage 3 processing for remaining filters in series.')
        log.info('Finished. Starting stage 3 processing for remaining filters in series.')

    path_longest = sorted_filter_dirs[0] + '/output_files/'
    convert_catalog_to_tweakreg_format(path_longest, sorted_filters[0])
    long_cat = os.path.join(path_longest, f'{sorted_filters[0]}.csv')
    long_list = [file for file in os.listdir(path_longest) if file.endswith(f'nircam_clear-{sorted_filters[0]}_i2d.fits')]
    long_processed_file = os.path.join(path_longest, long_list[0])
    long_params = extract_resample_info(long_processed_file)

    if use_multiprocessing == True:
        log.info('Multiprocessing is being used. Beginning stage 3 for remaining filters...')
        process_filters_parallel(sorted_filter_dirs[1:], target, long_cat, long_params, extract_settings, config=config)

    else:
        for i,dir in enumerate(tqdm(sorted_filter_dirs[1:], file=sys.stderr)):
            log_filter = setup_filter_logger(dir)
            stage3(dir, log_filter, target, reference_catalog=long_cat, resample_params=long_params, config=config)
            extract_data(dir, target, extract_settings, log_filter)