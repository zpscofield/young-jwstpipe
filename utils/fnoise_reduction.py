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
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
from photutils.segmentation import detect_threshold
from photutils.segmentation import detect_sources as phot_detect_sources
from scipy.ndimage import binary_dilation

import yaml

from log_utils import archive_existing_log

# Tunable settings come from config.yaml (set via the Streamlit UI). Each
# read below falls back to the original hardcoded value so behaviour is
# unchanged unless the user overrides it.
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file) or {}


def setup_logger(output_dir):
    """
    Setup a logger for the pipeline using the provided output directory.
    """
    parent_dir = os.path.dirname(output_dir)
    log_file_path = os.path.join(parent_dir, "logs/pipeline_cfnoise.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    archive_existing_log(log_file_path)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    with open(log_file_path, 'a') as log_file:
        log_file.write("\n----------------\n")
        log_file.write("CAL fnoise reduction\n")
        log_file.write("----------------\n\n")

    return log, log_file_path

def detect_sources(image, threshold):
    return image > threshold

def compute_x_derivatives(image):
    dx = np.diff(image, axis=1)
    dx = np.pad(dx, ((0, 0), (1, 0)), mode='constant', constant_values=0)
    return dx

def compute_y_derivatives(image):
    dy = np.diff(image, axis=0)
    dy = np.pad(dy, ((1, 0), (0, 0)), mode='constant', constant_values=0)
    return dy

def reconstruct_image_from_dx(dx, initial_values):
    reconstructed_image = np.cumsum(dx, axis=1)
    reconstructed_image += initial_values[:, np.newaxis]
    return reconstructed_image

def reconstruct_image_from_dy(dy, initial_values):
    reconstructed_image = np.cumsum(dy, axis=0)
    reconstructed_image += initial_values[np.newaxis,:]
    return reconstructed_image

def fnoise_reduction(ori_imag, output_dir, threshold1=1, threshold2 = 98):
    ### Step 1. dy estimation
    #### step 1-1. Load image
    ori_imag_cache = cp.deepcopy(ori_imag)
    ori_imag[np.isnan(ori_imag)] = 0


    #### step 1-2. Set two mask
    # Broad mask (To conserve diffuse broad signals)

    # keep a NaN mask for mask_map1 only
    nanmask = ~np.isfinite(ori_imag_cache)

    # build mask_map1 from photutils
    data = ori_imag_cache.copy()
    data_filled = data.copy()

    thr1 = detect_threshold(data_filled, nsigma=threshold1, mask=nanmask)
    segm1 = phot_detect_sources(data_filled, thr1, npixels=config.get('cfnoise_npixels', 200), mask=nanmask)

    if segm1 is None:
        mask_map1 = np.zeros_like(data_filled, dtype=bool)
    else:
        mask_map1 = segm1.make_source_mask(size=config.get('cfnoise_mask_size', 11))

    # mask NaN region in mask_map1
    mask_map1[nanmask] = False

    # Narrow mask (To mask the dx image)
    pix_threshold2 = np.nanpercentile(ori_imag, threshold2)
    mask_map2 = detect_sources(ori_imag, pix_threshold2)

    #### step 1-2-1. Determine y_noise correction should be splitted or not
    left0  = np.sum(mask_map1[:, 0:512], axis=1)
    right0 = np.sum(np.isnan(ori_imag_cache[:, 0:512]) == False, axis=1)
    valid0 = ~((left0 == 0) & (right0 == 0))
    frac0  = np.zeros_like(left0, dtype=float)
    frac0[valid0] = left0[valid0] / right0[valid0]
    test0  = np.sum((frac0 >= 0.75) & valid0)

    left1  = np.sum(mask_map1[:, 512:1024], axis=1)
    right1 = np.sum(np.isnan(ori_imag_cache[:, 512:1024]) == False, axis=1)
    valid1 = ~((left1 == 0) & (right1 == 0))
    frac1  = np.zeros_like(left1, dtype=float)
    frac1[valid1] = left1[valid1] / right1[valid1]
    test1  = np.sum((frac1 >= 0.75) & valid1)

    left2  = np.sum(mask_map1[:, 1024:1536], axis=1)
    right2 = np.sum(np.isnan(ori_imag_cache[:, 1024:1536]) == False, axis=1)
    valid2 = ~((left2 == 0) & (right2 == 0))
    frac2  = np.zeros_like(left2, dtype=float)
    frac2[valid2] = left2[valid2] / right2[valid2]
    test2  = np.sum((frac2 >= 0.75) & valid2)

    left3  = np.sum(mask_map1[:, 1536:2048], axis=1)
    right3 = np.sum(np.isnan(ori_imag_cache[:, 1536:2048]) == False, axis=1)
    valid3 = ~((left3 == 0) & (right3 == 0))
    frac3  = np.zeros_like(left3, dtype=float)
    frac3[valid3] = left3[valid3] / right3[valid3]
    test3  = np.sum((frac3 >= 0.75) & valid3)


    # Optional add-on: force whole-image (non-split) correction regardless of
    # the per-channel auto-detection above. When disabled (the default), the
    # original auto-detection logic is preserved exactly.
    if config.get('cfnoise_whole_image', False):
        split = False
    elif test0 + test1 + test2 + test3 == 0:
        split = True
    else:
        split = False
    #### step 1-3. calculate dy_ref, dy_med
    dy = compute_y_derivatives(ori_imag)
    dy_ref = cp.deepcopy(dy)
    dy[mask_map2] = np.nan

    if split == True:
        dy_0 = dy[:,0:512]
        dy_1 = dy[:,512:1024]
        dy_2 = dy[:,1024:1536]
        dy_3 = dy[:,1536:2048]

        dy_0ref = cp.deepcopy(dy_ref[:,0:512])
        dy_0med = np.nanmedian(dy_0, axis = 1)
        dy_0med = np.repeat(dy_0med, 512)
        dy_0med = np.reshape(dy_0med, (2048, 512))

        dy_1ref = cp.deepcopy(dy_ref[:,512:1024])
        dy_1med = np.nanmedian(dy_1, axis = 1)
        dy_1med = np.repeat(dy_1med, 512)
        dy_1med = np.reshape(dy_1med, (2048, 512))

        dy_2ref = cp.deepcopy(dy_ref[:,1024:1536])
        dy_2med = np.nanmedian(dy_2, axis = 1)
        dy_2med = np.repeat(dy_2med, 512)
        dy_2med = np.reshape(dy_2med, (2048, 512))

        dy_3ref = cp.deepcopy(dy_ref[:,1536:2048])
        dy_3med = np.nanmedian(dy_3, axis = 1)
        dy_3med = np.repeat(dy_3med, 512)
        dy_3med = np.reshape(dy_3med, (2048, 512))

        dy_med = np.zeros((2048,2048))
        dy_med[:,0:512] += dy_0med
        dy_med[:,512:1024] += dy_1med
        dy_med[:,1024:1536] += dy_2med
        dy_med[:,1536:2048] += dy_3med
    else:
        dy_med = np.nanmedian(dy, axis = 1)
        dy_med = np.repeat(dy_med, 2048)
        dy_med = np.reshape(dy_med, (2048,2048))


    ### Step 2. Reconstruct image without y-noise
    #### step 2-1. reduce dy_ref image
    if split == True:
        dy_0ref -= dy_0med
        dy_0ref_mask = cp.deepcopy(dy_0ref)
        dy_1ref -= dy_1med
        dy_1ref_mask = cp.deepcopy(dy_1ref)
        dy_2ref -= dy_2med
        dy_2ref_mask = cp.deepcopy(dy_2ref)
        dy_3ref -= dy_3med
        dy_3ref_mask = cp.deepcopy(dy_3ref)
        dy_ref = np.zeros((2048,2048))
        dy_ref[:,0:512] += dy_0ref
        dy_ref[:,512:1024] += dy_1ref
        dy_ref[:,1024:1536] += dy_2ref
        dy_ref[:,1536:2048] += dy_3ref
        dy_ref_mask = np.zeros((2048,2048))
        dy_ref_mask[:,0:512] += dy_0ref_mask
        dy_ref_mask[:,512:1024] += dy_1ref_mask
        dy_ref_mask[:,1024:1536] += dy_2ref_mask
        dy_ref_mask[:,1536:2048] += dy_3ref_mask
        dy_ref_mask[np.isnan(dy_ref)] = np.nan
    else:
        dy_ref -= dy_med
        dy_ref_mask = cp.deepcopy(dy_ref)
        dy_ref_mask[np.isnan(dy_ref)] = np.nan


    dy_ref_med = np.nanmedian(dy_ref_mask, axis = 1)
    dy_ref_med = np.repeat(dy_ref_med, 2048)
    dy_ref_med = np.reshape(dy_ref_med, (2048,2048))
    dy_ref[np.isnan(dy_ref)] = dy_ref_med[np.isnan(dy_ref)]
    dy_ref -= np.nanmedian(dy_ref[mask_map2==False])

    #### step 2-2. reconstruct image from dy_ref
    if split == True:
        dy_0ref = dy_ref[:,0:512]
        dy_1ref = dy_ref[:,512:1024]
        dy_2ref = dy_ref[:,1024:1536]
        dy_3ref = dy_ref[:,1536:2048]
        y_0ref = reconstruct_image_from_dy(dy_0ref, initial_values=ori_imag[0,0:512])
        y_1ref = reconstruct_image_from_dy(dy_1ref, initial_values=ori_imag[0,512:1024])
        y_2ref = reconstruct_image_from_dy(dy_2ref, initial_values=ori_imag[0,1024:1536])
        y_3ref = reconstruct_image_from_dy(dy_3ref, initial_values=ori_imag[0,1536:2048])
        y_ref = np.zeros((2048,2048))
        y_ref[:,0:512] += y_0ref
        y_ref[:,512:1024] += y_1ref
        y_ref[:,1024:1536] += y_2ref
        y_ref[:,1536:2048] += y_3ref
    else:
        y_ref = reconstruct_image_from_dy(dy_ref, initial_values=ori_imag[0,:])

    step = config.get('cfnoise_interp_step', 4)
    if split == True:
        bg0 = (mask_map1[:, 0:512] == False)
        bg1 = (mask_map1[:, 512:1024] == False)
        bg2 = (mask_map1[:, 1024:1536] == False)
        bg3 = (mask_map1[:, 1536:2048] == False)
        ny = y_0ref.shape[0]
        anchors = np.arange(0, ny, step)
        if anchors[-1] != ny - 1:
            anchors = np.append(anchors, ny - 1)
        vals0 = []
        vals1 = []
        vals2 = []
        vals3 = []
        for r in anchors:
            v0 = y_0ref[r, :].copy()
            v0[~bg0[r, :]] = np.nan
            vals0.append(np.nanmedian(v0))
            v1 = y_1ref[r, :].copy()
            v1[~bg1[r, :]] = np.nan
            vals1.append(np.nanmedian(v1))
            v2 = y_2ref[r, :].copy()
            v2[~bg2[r, :]] = np.nan
            vals2.append(np.nanmedian(v2))
            v3 = y_3ref[r, :].copy()
            v3[~bg3[r, :]] = np.nan
            vals3.append(np.nanmedian(v3))
        vals0 = np.array(vals0)
        vals1 = np.array(vals1)
        vals2 = np.array(vals2)
        vals3 = np.array(vals3)
        baseline0 = np.interp(np.arange(ny), anchors, vals0)[:, None]
        baseline1 = np.interp(np.arange(ny), anchors, vals1)[:, None]
        baseline2 = np.interp(np.arange(ny), anchors, vals2)[:, None]
        baseline3 = np.interp(np.arange(ny), anchors, vals3)[:, None]
        y_0ref = y_0ref - baseline0
        y_1ref = y_1ref - baseline1
        y_2ref = y_2ref - baseline2
        y_3ref = y_3ref - baseline3


        y_ref = np.zeros((2048,2048))
        y_ref[:,0:512] += y_0ref
        y_ref[:,512:1024] += y_1ref
        y_ref[:,1024:1536] += y_2ref
        y_ref[:,1536:2048] += y_3ref
        lin_y_ref = np.zeros((2048,2048))
        lin_y_ref[:,0:512] += baseline0
        lin_y_ref[:,512:1024] += baseline1
        lin_y_ref[:,1024:1536] += baseline2
        lin_y_ref[:,1536:2048] += baseline3

    else:
        bg = (mask_map1 == False)
        ny = y_ref.shape[0]
        anchors = np.arange(0, ny, step)
        if anchors[-1] != ny - 1:
            anchors = np.append(anchors, ny - 1)
        vals = []
        for r in anchors:
            v = y_ref[r, :].copy()
            v[~bg[r, :]] = np.nan
            vals.append(np.nanmedian(v))
        vals = np.array(vals)
        baseline = np.interp(np.arange(ny), anchors, vals)[:, None]
        y_ref = y_ref - baseline
        lin_y_ref = np.zeros((2048,2048))
        lin_y_ref += baseline

    y_ref[ori_imag_cache==0]=0

    ynoise = ori_imag - y_ref

    y_ref[np.isnan(ori_imag_cache)]=0

    ### Step 3. dx estimation
    #### step 3-1. calculate dx_ref, dx_med
    dx = compute_x_derivatives(y_ref)

    dx_ref = cp.deepcopy(dx)
    dx[mask_map2] = np.nan
    dx_med = np.nanmedian(dx, axis = 0)
    dx_med = np.repeat(dx_med, 2048)
    dx_med = np.reshape(dx_med, (2048,2048))
    dx_med = dx_med.T

    ### Step 4. Reconstruct image without x_noise
    #### step 4-1. reduce dx_ref image
    dx_ref -= dx_med
    dx_ref_mask = cp.deepcopy(dx_ref)

    dx_ref_mask[np.isnan(ori_imag_cache)] = np.nan
    dx_ref_med = np.nanmedian(dx_ref_mask, axis = 0)
    dx_ref_med = np.repeat(dx_ref_med, 2048)
    dx_ref_med = np.reshape(dx_ref_med, (2048,2048))
    dx_ref_med = dx_ref_med.T
    dx_ref[np.isnan(dx_ref)] = dx_ref_med[np.isnan(dx_ref)]
    dx_ref -= np.nanmedian(dx_ref[mask_map2==False])

    #### step 4-2. reconstruct image from dy_ref
    x_ref = reconstruct_image_from_dx(dx_ref, initial_values=np.zeros(2048))
    nx = x_ref.shape[1]
    step = config.get('cfnoise_interp_step', 4)
    bg = (mask_map1 == False)
    anchors = np.arange(0, nx, step)
    if anchors[-1] != nx - 1:
        anchors = np.append(anchors, nx - 1)
    vals = []
    for c in anchors:
        v = x_ref[:, c].copy()
        v[~bg[:, c]] = np.nan
        vals.append(np.nanmedian(v))
    vals = np.array(vals)
    baseline = np.interp(np.arange(nx), anchors, vals)[None, :]
    x_ref = x_ref - baseline

    x_ref_mask = cp.deepcopy(x_ref)
    x_ref_mask[mask_map1] = np.nan
    lin_x_ref = np.zeros((2048,2048))
    lin_x_ref += baseline

    x_ref[ori_imag_cache==0]=0

    xnoise = y_ref - x_ref


    ### Step 5. Noise and Denoised Image
    all_noise = xnoise + ynoise
    all_noise -= np.nanmean(all_noise)

    # Guard: the intermediate baseline/median steps can produce NaN whenever a
    # full row or column of an amplifier region is source-masked (e.g. near
    # cluster cores). We don't want those algorithmic NaNs to leak into the
    # denoised output -- only positions that were NaN in the input should be
    # NaN in the output. Zero out the noise at any position that was finite
    # in the input but ended up NaN here.
    finite_input = ~np.isnan(ori_imag_cache)
    unwanted_nan = np.isnan(all_noise) & finite_input
    if unwanted_nan.any():
        all_noise[unwanted_nan] = 0.0

    denoise = ori_imag - all_noise
    denoise[np.isnan(ori_imag_cache)] = np.nan


    return denoise, all_noise


def _cfnoise_output_names(input_path: str) -> tuple[str, str]:
    """Given a _cal.fits or _cal_wisp.fits input path, return the (denoised, model) output paths."""
    if "_cal_wisp.fits" in input_path:
        denoised = input_path.replace("_cal_wisp.fits", "_cal_cfnoise.fits")
        model = input_path.replace("_cal_wisp.fits", "_cal_cfnoise_model.fits")
    elif "_cal.fits" in input_path:
        denoised = input_path.replace("_cal.fits", "_cal_cfnoise.fits")
        model = input_path.replace("_cal.fits", "_cal_cfnoise_model.fits")
    else:
        denoised = input_path.replace(".fits", "_cfnoise.fits")
        model = input_path.replace(".fits", "_cfnoise_model.fits")
    return denoised, model


def process_file(args):
    log, f, output_dir = args  # Unpack the tuple
    log.info(f'File name: {f}')
    data = fits.open(f)
    ori_imag = data[1].data
    ori_imag_cache = cp.deepcopy(ori_imag)
    isnan = np.isnan(ori_imag_cache)
    x_all = np.zeros((2048,2048))
    y_all = np.zeros((2048,2048))

    denoise, noise = fnoise_reduction(
        ori_imag, output_dir,
        threshold1=config.get('cfnoise_threshold1', 1),
        threshold2=config.get('cfnoise_threshold2', 98),
    )
    ori_imag -= noise

    save_denoise = cp.deepcopy(ori_imag)
    save_denoise[isnan] = np.nan
    data[1].data = save_denoise

    denoised_path, model_path = _cfnoise_output_names(f)
    data.writeto(denoised_path, overwrite=True)
    log.info(f'Result saved: {denoised_path}')

    noise_model = ori_imag_cache - ori_imag
    data[1].data = noise_model
    data.writeto(model_path, overwrite=True)
    log.info(f'Result saved: {model_path}')
    data.close()


def process_files(log, files, nproc, output_dir):
    effective_nproc = nproc
    task_args = [(log, img, output_dir) for img in files]
    if effective_nproc != 0:
        with Pool(processes=effective_nproc) as pool:
            with tqdm(total=len(files), file=sys.stdout) as pbar:
                for _ in pool.imap_unordered(process_file, task_args):
                    pbar.update(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure and remove horizontal and vertical striping pattern (1/f noise) from cal files.'
    )
    parser.add_argument('--files', dest='files', action='store', nargs='+', type=str, required=False,
                        default='./*_cal.fits',
                        help='Input files. Both *_cal.fits and *_cal_wisp.fits are accepted; the output filename is derived per-file.')
    parser.add_argument('--nproc', dest='nproc', action='store', type=int, required=False, default=6)
    parser.add_argument('--output_dir', dest='output_dir', action='store', type=str, required=False, default='./')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()
    log, log_file_path = setup_logger(args.output_dir)
    process_files(log, **vars(args))
    # Remove the original inputs (both cal and cal_wisp variants) now that
    # they have been superseded by *_cal_cfnoise.fits.
    patterns = ["*_cal.fits", "*_cal_wisp.fits"]
    files_to_remove: list[str] = []
    for pat in patterns:
        files_to_remove.extend(glob(os.path.join(args.output_dir, "**", pat), recursive=True))
    for file in files_to_remove:
        try:
            os.remove(file)
            log.info(f'removed {file}')
        except OSError as exc:
            log.warning(f'could not remove {file}: {exc}')
    log.info('fnoise_reduction.py complete.')
