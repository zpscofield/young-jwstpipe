import os
import copy as cp
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip, sigma_clip
from photutils.segmentation import detect_sources
from scipy.stats import median_abs_deviation, mode
from astropy.io import fits
import argparse
from glob import glob
from multiprocessing import Pool, cpu_count
import sys
from tqdm.auto import tqdm
import logging
import yaml
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)



with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

def setup_logger(output_dir):
    """
    Setup a logger for the pipeline using the provided output directory.
    """
    parent_dir = os.path.dirname(output_dir)
    log_file_path = os.path.join(parent_dir, "logs/pipeline_cfnoise.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Ensure log directory exists

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    with open(log_file_path, 'a') as log_file:
        log_file.write("\n----------------\n")
        log_file.write("Wisp subtraction\n")
        log_file.write("----------------\n\n")

    return log, log_file_path

def detect_sources(image, threshold):
    return image > threshold

def compute_x_derivatives(image):
    dx = np.diff(image, axis=1)
    dx = np.pad(dx, ((0, 0), (1, 0)), mode='constant', constant_values=0)  # 패딩 추가
    return dx

def compute_y_derivatives(image):
    dy = np.diff(image, axis=0)
    dy = np.pad(dy, ((1, 0), (0, 0)), mode='constant', constant_values=0)  # 패딩 추가
    return dy

def reconstruct_image_from_dx(dx, initial_values):
    reconstructed_image = np.cumsum(dx, axis=1)
    reconstructed_image += initial_values[:, np.newaxis]
    return reconstructed_image

def reconstruct_image_from_dy(dy, initial_values):
    reconstructed_image = np.cumsum(dy, axis=0)
    reconstructed_image += initial_values[:, np.newaxis]
    return reconstructed_image

def fnoise_reduction(image, output_dir, threshold1=70, threshold2=95):
    ### Step 1. dx estimation
    #### step 1-1. Load original image
    ori_data = fits.open(image)
    ori_imag = ori_data[1].data
    ori_imag_cache = cp.deepcopy(ori_imag)
    ori_imag_cache2 = cp.deepcopy(ori_imag)
    ori_imag[np.isnan(ori_imag)] = 0
    ori_imag_cache2[np.isnan(ori_imag_cache2)] = 0

    #### step 1-2. Set two mask
    # Broad mask (To conserve diffuse broad signals)
    pix_threshold1 = np.nanpercentile(ori_imag, threshold1)
    mask_map1 = detect_sources(ori_imag, pix_threshold1)
    # Narrow mask (To mask the dx image)
    pix_threshold2 = np.nanpercentile(ori_imag, threshold2)
    mask_map2 = detect_sources(ori_imag, pix_threshold2)
    
    pix_threshold3 = np.nanpercentile(ori_imag, 50)
    mask_map3 = detect_sources(ori_imag, pix_threshold3)


    #### step 1-3. calculate dx_ref, dx_med
    dx = compute_x_derivatives(ori_imag)
    dx_ref = cp.deepcopy(dx)
    dx[mask_map2] = np.nan
    dx_med = np.nanmedian(dx, axis = 0)
    dx_med = np.repeat(dx_med, 2048)
    dx_med = np.reshape(dx_med, (2048,2048))
    dx_med = dx_med.T


    ### Step 2. Reconstruct image without x_noise
    #### step 2-1. reduce dx_ref image
    dx_ref -= dx_med
    dx_ref_mask = cp.deepcopy(dx_ref)
    dx_ref_mask[np.isnan(ori_imag_cache)] = np.nan
    dx_ref_med = np.nanmedian(dx_ref_mask, axis = 0)
    dx_ref_med = np.repeat(dx_ref_med, 2048)
    dx_ref_med = np.reshape(dx_ref_med, (2048,2048))
    dx_ref_med.T
    dx_ref[np.isnan(dx_ref)] = dx_ref_med[np.isnan(dx_ref)]
    dx_ref -= np.nanmedian(dx_ref[mask_map2==False])

    #### step 2-2. reconstruct image from dx_ref
    x_ref = reconstruct_image_from_dx(dx_ref, initial_values=ori_imag_cache2[:,0])
    x_ref_mask = cp.deepcopy(x_ref)
    x_ref_mask[mask_map3] = np.nan
    lin_x_ref = np.nanmedian(x_ref_mask, axis = 0)
    lin_x_ref = np.repeat(lin_x_ref, 2048)
    lin_x_ref = np.reshape(lin_x_ref, (2048,2048))
    lin_x_ref = lin_x_ref.T
    x_ref -= lin_x_ref
    x_ref[ori_imag_cache2==0]=0

    ### Step3. dy estimation
    #### step 3-1. Save x_ref image
    x_ref_cache = cp.deepcopy(x_ref)
    x_ref[np.isnan(x_ref)]= 0
    #### step 3-2. set two mask
    # Broad mask (To conserve diffuse broad signals)
    pix_threshold1 = np.nanpercentile(x_ref, threshold1)
    mask_map1 = detect_sources(x_ref, pix_threshold1)
    # Narrow mask (To mask the dx image)
    pix_threshold2 = np.nanpercentile(x_ref, threshold2)
    mask_map2 = detect_sources(x_ref, pix_threshold2)
    pix_threshold3 = np.nanpercentile(x_ref, 50)
    mask_map3 = detect_sources(x_ref, pix_threshold3)

    #### step 3-3. calculate dy_ref, dy_med
    dy = compute_y_derivatives(x_ref)
    dy_ref = cp.deepcopy(dy)
    dy[mask_map2] = np.nan
    dy_med = np.nanmedian(dy, axis = 1)
    dy_med = np.repeat(dy_med, 2048)
    dy_med = np.reshape(dy_med, (2048, 2048))

    ### Step 4. Reconstruct image without y_noise
    #### step 4-1. reduce dy_ref image
    dy_ref -= dy_med
    dy_ref_mask = cp.deepcopy(dy_ref)
    dy_ref_mask[np.isnan(x_ref_cache)] = np.nan
    dy_ref_med = np.nanmedian(dy_ref_mask, axis = 1)
    dy_ref_med = np.repeat(dy_ref_med, 2048)
    dy_ref_med = np.reshape(dy_ref_med, (2048,2048))
    dy_ref[np.isnan(dy_ref)] = dy_ref_med[np.isnan(dy_ref)]
    dy_ref -= np.nanmedian(dy_ref[mask_map2==False])

    #### step 4-2. reconstruct image from dy_ref
    y_ref = reconstruct_image_from_dy(dy_ref, initial_values=x_ref_cache[0,:])
    # hdu = fits.PrimaryHDU(y_ref)
    # hdu.writeto(output_dir+'/test4-2-1.fits', overwrite = True)
    y_ref_mask = cp.deepcopy(y_ref)
    y_ref_mask[mask_map3] = np.nan
    lin_y_ref = np.nanmedian(y_ref_mask, axis = 1)
    lin_y_ref = np.repeat(lin_y_ref, 2048)
    lin_y_ref = np.reshape(lin_y_ref, (2048,2048))
    # hdu = fits.PrimaryHDU(lin_y_ref)
    # hdu.writeto(output_dir+'/test4-2-2.fits', overwrite = True)
    y_ref -= lin_y_ref
    y_ref[ori_imag_cache==0]=0
    
    y_ref[np.isnan(ori_imag_cache)]=np.nan

    return y_ref

def process_file(args):
    log, f, output_dir, suffix= args  # Unpack the tuple
    f_name = os.path.basename(f)
    oid, oid2, oid3, chip, exten, old_suffix = f_name.split('_')
    imag = fnoise_reduction(f, output_dir)
    data = fits.open(f)
    data[1].data = imag
    output_filename = os.path.join(output_dir, f"{oid}_{oid2}_{oid3}_{chip}_{exten}{suffix}.fits")
    data.writeto(output_filename, overwrite = True)
    data.close()

def process_files(log, files, nproc, output_dir, suffix):
    effective_nproc = nproc #min(nproc, ...)
    task_args = [(log, img, output_dir, suffix) for img in files]
    if effective_nproc != 0:
        with Pool(processes=effective_nproc) as pool:
            with tqdm(total=len(files), file=sys.stdout) as pbar:
                for _ in pool.imap_unordered(process_file, task_args):
                    pbar.update(1)

def parse_args():

    parser = argparse.ArgumentParser(description='Measure and remove horizontal and vertical striping pattern (1/f noise) from rate file')
    parser.add_argument('--files', dest='files', action='store', nargs='+', type=str, required=False, default='./*_cal.fits')
    parser.add_argument('--nproc', dest='nproc', action='store', type=int, required=False, default=6)
    parser.add_argument('--output_dir', dest='output_dir', action='store', type=str, required=False, default='./')
    parser.add_argument('--suffix', dest='suffix', action='store', type=str, required=False, default='_wisp')
    args = parser.parse_args()

    return args

if __name__=='__main__':
    # Get the command line arguments
    args = parse_args()
    log, log_file_path = setup_logger(args.output_dir)
    # Process the input files
    results = process_files(log, **vars(args))
    files_to_remove = glob(os.path.join(args.output_dir, "**", "*_wisp.fits"), recursive=True)
    for file in files_to_remove:
        os.remove(file)
    log.info('removed _wisp files.')
    log.info('fnoise_reduction.py complete.')