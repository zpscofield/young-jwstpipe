from astropy.io import fits
from astropy.stats import sigma_clipped_stats, sigma_clip, biweight_location
from scipy.optimize import curve_fit
import numpy as np
import os
import copy as cp
import logging
import sys
# import json
import yaml
import background_subtraction
import compute_cal_sky_variance
import pprint
from jwst.datamodels import ImageModel
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
import argparse

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

def setup_logger(output_dir):
    """
    Setup a logger for the pipeline using the provided output directory.
    """
    parent_dir = os.path.dirname(output_dir)
    log_file_path = os.path.join(parent_dir, "logs/pipeline_bkg.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Ensure log directory exists

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    stpipe_log = logging.getLogger("stpipe")
    stpipe_log.setLevel(logging.INFO)
    stpipe_handler = logging.FileHandler(log_file_path, mode='a')
    stpipe_handler.setFormatter(formatter)
    stpipe_log.addHandler(stpipe_handler)
    for handler in stpipe_log.handlers: 
        if isinstance(handler, logging.StreamHandler):
            stpipe_log.removeHandler(handler)

    with open(log_file_path, 'a') as log_file:
        log_file.write("\n----------------------\n")
        log_file.write("Background Subtraction\n")
        log_file.write("----------------------\n\n")

    return log, log_file_path

def log_error(log, message, exc=None):
    log.error(message)
    if exc is not None:
        log.exception(exc)  # This logs the exception traceback

def gaussian(x, a, mu, sig):
    return a * np.exp(-(x-mu)**2/(2*sig**2))

def fit_sky(data, plot_sky=False, ax=None, color='C1', label=None, **kwargs):
    """Fit distribution of sky fluxes with a Gaussian"""
    bins = np.arange(-1.5, 2.0, 0.001)
    h, b = np.histogram(data, bins=bins)
    bc = 0.5 * (b[1:] + b[:-1])
    binsize = b[1] - b[0]

    p0 = [10, bc[np.argmax(h)], 0.01]
    popt, pcov = curve_fit(gaussian, bc, h, p0=p0)

    if plot_sky:
        ax.plot(bc, gaussian(bc, *popt), color, label=label, **kwargs)

    return popt[1]

def bkgsub(directory, img, log, output_dir, suffix, plot_sky=False):
    img_path = img#os.path.join(directory, img)
    
    # Check if the file exists
    if not os.path.exists(img_path):
        log.info(f"File {img_path} does not exist or is not accessible.")
        return
    
    bkg_suffix = 'bkgsub1'
    file_suffix = suffix
    bs = background_subtraction.SubtractBackground(log=log)
    bs.suffix = bkg_suffix
    bs.replace_sci = True
    bs.do_background_subtraction(directory, img)
    bkgsub_file = os.path.join(img_path.replace(f'{suffix}.fits', '_%s.fits' % bkg_suffix))
    with fits.open(bkgsub_file) as hdul:
        mask = hdul[9].data

    model = ImageModel(img_path)
    dq = model.dq
    sci = model.data
    original_data = cp.deepcopy(model.data)

    w = np.where((dq == 0) & (mask == 0))
    data = sci[w]
    data = data.flatten()

    if plot_sky:
        bins = np.arange(-1.5, 2.0, 0.001)
        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(15, 8))
        ax.hist(data, bins=bins, color='k', alpha=0.3, label='Original data (masked)')
    else:
        ax = None

    try:
        sky = fit_sky(data, plot_sky=plot_sky, ax=ax, color='C1', label='Original data - fit',
                      alpha=0.5, lw=2)
    except RuntimeError as e:
        log.info('!!! Error %s !!!' % img)
        err = open('errors.list', 'a')
        err.write('{}\t{}\n'.format(img, e))
        err.close()
        sky = 0

    meddata = np.median(data)

    # iterate on sigma clipping
    clipped = sigma_clip(data, sigma=5, sigma_upper=0, sigma_lower=10,
                         maxiters=5, masked=False)
    medclip = np.median(clipped)
    biweight = biweight_location(clipped)

    if plot_sky:
        ax.hist(clipped, bins=bins, color='C0', alpha=0.3, label='Clipped data')
        ax.axvline(meddata, color='C1', lw=1, label=f'Orig median: {meddata:.4f}')
        ax.axvline(medclip, color='C2', lw=1, label=f'Clipped median {medclip:.4f}')
        ax.axvline(biweight, color='C3', lw=1, label=f'Biweight location: {biweight:.4f}')

    try:
        sky = fit_sky(clipped, plot_sky=plot_sky, ax=ax, color='C2', label='Clipped data - fit',
                      alpha=0.6, lw=1)
    except RuntimeError as e:
        log.info('!!! Error %s !!!' % img)
        err = open('errors.list', 'a')
        err.write('{}\t{}\n'.format(img, e))
        err.close()
        sky = 0

    if plot_sky:
        skysub_data = data - sky
        skysub_med = np.median(skysub_data)
        ax.hist(skysub_data, bins=bins, color='C5', alpha=0.3, label='Sky subtracted data (masked)')
        ax.axvline(skysub_med, color='C6', lw=1, label=f'Sky sub median: {skysub_med:.4f}')
        plt.legend()
        plt.savefig(img.replace('.fits', '_sky.png'))

    log.info('%s' % img)
    log.info('  clipped median: %f' % medclip)
    log.info('  biweight background: %f' % biweight_location(clipped))
    log.info('  gaussian-fit background: %f' % sky)
    log.info('%s subtracting sky: %f' % (img, sky))
    # subtract off sky
    processed_data = original_data - sky

    model.meta.background.level = sky
    model.meta.background.subtracted = True
    model.meta.background.method = 'local'

    ### rescale variance maps
    log.info('%s rescaling readnoise variance' % img)
    # Instantiate the SubtractBackground object. Set the output suffix
    sv = compute_cal_sky_variance.ScaledVariance(log=log)

    # Print out the parameters being used
    log.info("ScaledVariance parameters:\n%s", pprint.pformat(sv.__dict__))

    # use the 2D background subtracted image
    fitsfile = img.replace('%s.fits' % file_suffix, '_%s.fits' % bkg_suffix)

    sv.read_file(directory, fitsfile)
    # directly pull corrected readnoise, rather than writing to file
    sv.correct_the_variance()
    varcorr = sv.predicted_skyvar
    model.var_rnoise = varcorr

    ### fix holes in variance maps
    log.info('%s fixing variance map holes' % img)
    rnoise = model.var_rnoise
    poisson = model.var_poisson
    flat = model.var_flat

    w = np.where(rnoise == 0)
    rnoise[w] = np.inf

    w = np.where(poisson == 0)
    poisson[w] = np.inf

    w = np.where(flat == 0)
    flat[w] = np.inf

    model.var_rnoise = rnoise
    model.var_poisson = poisson
    model.flat = flat
    log.info('success %s' % img)

    renamed_cal = img_path.replace(f'cal{suffix}.fits', f'cal{suffix}_pre_bkg.fits')
    os.rename(img_path, renamed_cal)
    # save output
    model.data = processed_data
    model.save(os.path.join(output_dir, img))

    log.info('finished: %s' % img)

def cleanup_intermediate_files(output_dir, image_filename, log):
    base_filename = os.path.basename(image_filename).replace(f'_cal{suffix}.fits', '')
    intermediate_files = [
        os.path.join(output_dir, base_filename + '_cal_bkgsub1.fits'),
        os.path.join(output_dir, base_filename + f'_cal{suffix}_pre_bkg.fits')
    ]
    for file in intermediate_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                log.info(f"Deleted intermediate file: {file}")
            except Exception as e:
                log.error(f"Error deleting file: {file}, {e}")

def process_file(args):
    directory, output_dir, img, plot_sky, suffix, log, log_file_path = args
    try:
        bkgsub(directory, img, log, output_dir, suffix, plot_sky)
        cleanup_intermediate_files(directory, img, log)
    except Exception as e:
        log_error(log, f"Error in process_file for {img}", exc=e)
        with open(log_file_path, "a") as f:
            f.write(f"{img}: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Background subtraction.')
    parser.add_argument('--files', dest='files', action='store', nargs='+', type=str, required=False, default='./*_cal.fits')
    parser.add_argument('--output_dir', type=str, help='Directory where output will be written')
    parser.add_argument('--input_dir', type=str, help='Directory where input is located')
    parser.add_argument('--nproc', type=int, default=cpu_count() // 2,
                        help='Number of parallel processes to use (default: half of available cores)')
    parser.add_argument('--suffix', dest='suffix', action='store', type=str, required=False, default='_final')
    args = parser.parse_args()

    log, log_file_path = setup_logger(args.output_dir)
    path = args.input_dir
    output_dir = args.output_dir
    files = args.files
    suffix = args.suffix

    pool_args = [(path, output_dir, img, config['plot_sky'], suffix, log, log_file_path) for img in files]

    log.info("Starting multiprocessing for background subtraction...")
    effective_nproc = min(args.nproc, len(pool_args))
    with Pool(processes=effective_nproc) as pool:
        with tqdm(total=len(pool_args), file=sys.stdout) as pbar:
            for result in pool.imap_unordered(process_file, pool_args):
                pbar.update(1) 

    log.info("Completed processing all files.")
