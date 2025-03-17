__author__ = "Micaela B. Bagley, UT Austin"
__version__ = "0.3.0"
__license__ = "BSD3"

# Modified for cluster fields by Zachary Scofield - Yonsei University

import os
import sys
# import json
import yaml
import shutil
import logging
from datetime import datetime
import argparse
import numpy as np
from astropy.io import fits
import astropy.stats as astrostats
from astropy.convolution import Ring2DKernel, Gaussian2DKernel, convolve_fft
from scipy.optimize import curve_fit
from scipy.ndimage import binary_dilation
from scipy.ndimage import median_filter
from photutils.segmentation import SegmentationImage, detect_sources
from glob import glob
from jwst.datamodels import ImageModel, FlatModel, dqflags
from jwst.flatfield.flat_field import do_correction
from stdatamodels import util
import crds
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

os.environ['CRDS_PATH'] = config['crds_path']
os.environ['CRDS_SERVER_URL'] = config['crds_server_url']

def setup_logger(output_dir):
    """
    Setup a logger for the pipeline using the provided output directory.
    """
    parent_dir = os.path.dirname(output_dir)
    log_file_path = os.path.join(parent_dir, "logs/pipeline_fnoise.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Ensure log directory exists

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    with open(log_file_path, 'a') as log_file:
        log_file.write("\n-----------------\n")
        log_file.write("1/f Noise Removal\n")
        log_file.write("-----------------\n\n")

    crds_log = logging.getLogger("CRDS")
    crds_log.setLevel(logging.INFO)
    crds_handler = logging.FileHandler(log_file_path, mode='a')
    crds_handler.setFormatter(formatter)
    crds_log.addHandler(crds_handler)
    for handler in crds_log.handlers: 
        if isinstance(handler, logging.StreamHandler):
            crds_log.removeHandler(handler)

    stpipe_log = logging.getLogger("stpipe")
    stpipe_log.setLevel(logging.INFO) 
    stpipe_handler = logging.FileHandler(log_file_path, mode='a')
    stpipe_handler.setFormatter(formatter)
    stpipe_log.addHandler(stpipe_handler)
    for handler in stpipe_log.handlers:
        if isinstance(handler, logging.StreamHandler):
            stpipe_log.removeHandler(handler)

    return log, log_file_path
        
MASKTHRESH = 0.8

NIR_reference_sections = {'A': {'top': (2044, 2048, 0, 512),
                                'bottom': (0, 4, 0, 512),
                                'side': (0, 2048, 0, 4),
                                'data': (0, 2048, 0, 512)},
                          'B': {'top': (2044, 2048, 512, 1024),
                                'bottom': (0, 4, 512, 1024),
                                'data': (0, 2048, 512, 1024)},
                          'C': {'top': (2044, 2048, 1024, 1536),
                                'bottom': (0, 4, 1024, 1536),
                                'data': (0, 2048, 1024, 1536)},
                          'D': {'top': (2044, 2048, 1536, 2048),
                                'bottom': (0, 4, 1536, 2048),
                                'side': (0, 2048, 2044, 2048),
                                'data': (0, 2048, 1536, 2048)}
                         }

NIR_amps = {'A': {'data': (4, 2044, 4, 512)},
            'B': {'data': (4, 2044, 512, 1024)},
            'C': {'data': (4, 2044, 1024, 1536)},
            'D': {'data': (4, 2044, 1536, 2044)}
            }

def gaussian(x, a, mu, sig):
    return a * np.exp(-(x-mu)**2/(2*sig**2))

def fit_sky(data):
    """Fit distribution of sky fluxes with a Gaussian"""
    bins = np.arange(-1, 1.5, 0.001)
    h, b = np.histogram(data, bins=bins)
    bc = 0.5 * (b[1:] + b[:-1])
    binsize = b[1] - b[0]

    p0 = [10, bc[np.argmax(h)], 0.01]
    popt, pcov = curve_fit(gaussian, bc, h, p0=p0)

    return popt[1]

def collapse_image(im, mask, dimension='y', sig=2.):
    """collapse an image along one dimension to check for striping."""
    if dimension == 'y':
        res = astrostats.sigma_clipped_stats(im, mask=mask, sigma=sig, 
                                             cenfunc='median',
                                             stdfunc='std', axis=1)
    elif dimension == 'x':
        res = astrostats.sigma_clipped_stats(im, mask=mask, sigma=sig, 
                                             cenfunc='median',
                                             stdfunc='std', axis=0)
    return res[1]

def masksources(log, image, output_dir):
    """Detect sources in an image using a tiered approach for different source sizes."""
    model = ImageModel(image)
    sci = model.data
    err = model.err
    wht = model.wht
    dq = model.dq

    bpflag = dqflags.pixel['DO_NOT_USE']
    bp = np.bitwise_and(dq, bpflag)
    bpmask = np.logical_not(bp == 0)

    log.info('masking, estimating background')
    sci_nan = np.choose(np.isnan(err), (sci, err))
    robust_mean_background = astrostats.biweight_location(sci_nan, c=6., ignore_nan=True)
    sci_filled = np.copy(sci)
    sci_filled[np.isnan(sci)] = robust_mean_background
    
    log.info('masking, initial source mask')
    ring = Ring2DKernel(40, 3)
    filtered = median_filter(sci_filled, footprint=ring.array)

    log.info('masking, mask tier 1')
    convolved_difference = convolve_fft(sci-filtered, Gaussian2DKernel(25))
    threshold = 3 * astrostats.mad_std(convolved_difference)
    segm_array = detect_sources(convolved_difference, threshold, npixels=15)
    segm = SegmentationImage(segm_array.data.astype(int))
    mask1 = segm.make_source_mask()

    temp = np.zeros(sci.shape)
    temp[mask1] = 1
    sources = np.logical_not(temp == 0)
    dilation_sigma = 10
    dilation_window = 11
    dilation_kernel = Gaussian2DKernel(dilation_sigma, x_size=dilation_window, y_size=dilation_window)
    source_wings = binary_dilation(sources, dilation_kernel)
    temp[source_wings] = 1
    mask1 = np.logical_not(temp == 0)

    log.info('masksources: mask tier 2')
    convolved_difference = convolve_fft(sci-filtered, Gaussian2DKernel(15))
    threshold = 3 * astrostats.mad_std(convolved_difference)
    segm_array = detect_sources(convolved_difference, threshold, npixels=15)
    segm = SegmentationImage(segm_array.data.astype(int))
    mask2 = segm.make_source_mask() | mask1

    log.info('masksources: mask tier 3')
    convolved_difference = convolve_fft(sci-filtered, Gaussian2DKernel(5))
    threshold = 3 * astrostats.mad_std(convolved_difference)
    segm_array = detect_sources(convolved_difference, threshold, npixels=5)
    segm = SegmentationImage(segm_array.data.astype(int))
    mask3 = segm.make_source_mask() | mask2

    log.info('masksources: mask tier 4')
    convolved_difference = convolve_fft(sci-filtered, Gaussian2DKernel(2))
    threshold = 3 * astrostats.mad_std(convolved_difference)
    segm_array = detect_sources(convolved_difference, threshold, npixels=3)
    segm = SegmentationImage(segm_array.data.astype(int))
    mask4 = segm.make_source_mask() | mask3

    finalmask = mask4

    outputbase = os.path.join(output_dir, os.path.basename(image))
    maskname = outputbase.replace('.fits', '_1fmask_new.fits')
    log.info('masksources: saving mask to %s' % maskname)
    outmask = np.zeros(finalmask.shape, dtype=int)
    outmask[finalmask] = 1
    fits.writeto(maskname, outmask, overwrite=True)
    return outmask

def measure_fullimage_striping(fitdata, mask):
    """Measures striping in countrate images using the full rows."""
    horizontal_striping = collapse_image(fitdata, mask, dimension='y')
    temp_image = fitdata.T - horizontal_striping
    temp_image2 = temp_image.T
    vertical_striping = collapse_image(temp_image2, mask, dimension='x')
    return horizontal_striping, vertical_striping

def measure_striping(log, image, origfilename, output_dir, thresh=None, apply_flat=True, mask_sources=True, save_patterns=False, flat_file=None):
    """Removes striping in rate.fits files before flat fielding."""

    if thresh is None:
        thresh = MASKTHRESH

    outputbase = os.path.join(output_dir, os.path.basename(image))

    model = ImageModel(image)
    log.info('Measuring image striping')
    log.info('Working on %s' % os.path.basename(image))

    if apply_flat:
        try:
            flatfile = flat_file
        except KeyError:
            log.error('Flat was not found')
            exit()
        log.info('Using flat: %s' % (os.path.basename(flatfile)))
        with FlatModel(flatfile) as flat:
            model, applied_flat = do_correction(model, flat)
            model.save(outputbase.replace('.fits', '_flat-fielded.fits'))

    mask = np.zeros(model.data.shape, dtype=bool)
    mask[model.dq > 0] = True

    if mask_sources:
        srcmask = outputbase.replace('.fits', '_1fmask_new.fits')
        if os.path.exists(srcmask):
            log.info('Using existing source mask %s' % srcmask)
            seg = fits.getdata(srcmask)
        else:
            log.info('Detecting sources to mask out source flux')
            seg = masksources(log, image, output_dir)
        wobj = np.where(seg > 0)
        mask[wobj] = True

    log.info('Measuring the pedestal in the image')
    pedestal_data = model.data[~mask]
    pedestal_data = pedestal_data.flatten()
    median_image = np.median(pedestal_data)
    log.info('Image median (unmasked and DQ==0): %f' % (median_image))
    try:
        pedestal = fit_sky(pedestal_data)
    except RuntimeError as e:
        log.error("Can't fit sky, using median value instead")
        pedestal = median_image
    else:
        log.info('Fit pedestal: %f' % pedestal)

    model.data -= pedestal
    full_horizontal, vertical_striping = measure_fullimage_striping(model.data, mask)

    horizontal_striping = np.zeros(model.data.shape)
    vertical_striping = np.zeros(model.data.shape)

    ampcounts = []
    for amp in ['A', 'B', 'C', 'D']:
        ampcount = 0
        rowstart, rowstop, colstart, colstop = NIR_amps[amp]['data']
        ampdata = model.data[:, colstart:colstop]
        ampmask = mask[:, colstart:colstop]
        hstriping_amp = collapse_image(ampdata, ampmask, dimension='y')
        nmask = np.sum(ampmask, axis=1)
        for i, row in enumerate(ampmask):
            if nmask[i] > (ampmask.shape[1] * thresh):
                horizontal_striping[i, colstart:colstop] = full_horizontal[i]
                ampcount += 1
            elif (hstriping_amp[i] > 2 * (np.nanstd(full_horizontal))):
                horizontal_striping[i, colstart:colstop] = full_horizontal[i]
            else:
                horizontal_striping[i, colstart:colstop] = hstriping_amp[i]
        ampcounts.append('%s-%i' % (amp, ampcount))

    ampinfo = ', '.join(ampcounts)
    log.info('%s, full row medians used: %s /%i' % (os.path.basename(image), ampinfo, rowstop-rowstart))

    log.info('%s, checking for any problematic amplifier medians...')
    # for i in range(horizontal_striping.shape[0]):  # Loop through rows
    #     amp_values = [] 
    #     amp_columns = []

    #     for amp in ['A', 'B', 'C', 'D']:
    #         _, _, colstart, colstop = NIR_amps[amp]['data']
    #         amp_median = np.nanmedian(horizontal_striping[i, colstart:colstop])
    #         amp_values.append(amp_median)
    #         amp_columns.append((colstart, colstop))

    #     # Identify overestimated amplifier values
    #     for j, amp_value in enumerate(amp_values):
    #         if amp_value > (2 * np.nanstd(full_horizontal)): 
    #             other_values = np.delete(amp_values, j) 
    #             replacement_value = np.nanmin(other_values)
    #             colstart, colstop = amp_columns[j]
    #             horizontal_striping[i, colstart:colstop] = replacement_value

    temp_sub = model.data - horizontal_striping
    vstriping = collapse_image(temp_sub, mask, dimension='x')
    vertical_striping[:, :] = vstriping

    if save_patterns:
        fits.writeto(outputbase.replace('.fits', '_horiz.fits'), horizontal_striping, overwrite=True)
        fits.writeto(outputbase.replace('.fits', '_vert.fits'), vertical_striping, overwrite=True)
        fits.writeto(outputbase.replace('.fits', '_full_horizontal.fits'), full_horizontal, overwrite=True)

    model.close()

    with ImageModel(image) as immodel:
        sci = immodel.data
        wzero = np.where(sci == 0)
        temp_sci = sci - horizontal_striping
        outsci = temp_sci - vertical_striping
        outsci[wzero] = 0
        wnan = np.isnan(outsci)
        bpflag = dqflags.pixel['DO_NOT_USE']
        outsci[wnan] = 0
        immodel.dq[wnan] = np.bitwise_or(immodel.dq[wnan], bpflag)
        immodel.data = outsci
        time = datetime.now()
        stepdescription = 'Removed horizontal,vertical striping; remstriping.py %s' % time.strftime('%Y-%m-%d %H:%M:%S')
        software_dict = {'name': 'remstriping.py', 'author': 'Micaela Bagley', 'version': '1.0', 'homepage': 'ceers.github.io'}
        substr = util.create_history_entry(stepdescription, software=software_dict)
        immodel.history.append(substr)
        log.info('Saving cleaned image to %s' % outputbase)
        original_file = image  # This is the 'rate.fits' file
        backup_file = origfilename #image.replace('rate.fits', 'rate_pre_fnoise.fits')  # This will be 'rate_pre_fnoise.fits'
        temporary_file = image.replace('rate.fits', 'rate_tmp.fits')
        try:
            immodel.save(temporary_file)
            log.info(f"Saved temporary file: {temporary_file}")
        except Exception as e:
            log.error(f"Error saving file: {e}")
        else:
            if os.path.exists(temporary_file):
                if os.path.exists(original_file):
                    try:
                        os.rename(original_file, backup_file)
                        log.info(f"Renamed original file to backup: {backup_file}")
                    except Exception as e:
                        log.error(f"Error renaming original file to backup: {e}")
                        raise
                else:
                    log.warning(f"Original file not found: {original_file}")
                try:
                    os.rename(temporary_file, original_file)
                    log.info(f"Renamed temporary file to original filename: {original_file}")
                except Exception as e:
                    log.error(f"Error renaming temporary file to original: {e}")
                    raise
            else:
                log.error("Temporary file not found after save operation")

def cleanup_intermediate_files(log, output_dir, image_filename):
    log, log_file_path = setup_logger(output_dir)
    base_filename = os.path.basename(image_filename).replace('rate.fits', '')
    intermediate_files = [
        os.path.join(output_dir, base_filename + 'rate_pre1f.fits'),
        os.path.join(output_dir, base_filename + 'rate_1fmask_new.fits'),
        os.path.join(output_dir, base_filename + 'rate_flat-fielded.fits')
    ]
    for file in intermediate_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                log.info(f"Deleted intermediate file: {file}")
            except Exception as e:
                log.error(f"Error deleting file: {file}, {e}")

def process_file(args):
    image, pre1f, output_dir, thresh, apply_flat, mask_sources, save_patterns, flat_file, log = args
    measure_striping(log, image, pre1f, output_dir, thresh=thresh, apply_flat=apply_flat, mask_sources=mask_sources, save_patterns=save_patterns, flat_file=flat_file)
    cleanup_intermediate_files(log, output_dir, image)

def main():
    parser = argparse.ArgumentParser(description='Measure and remove horizontal and vertical striping pattern (1/f noise) from rate file')

    parser.add_argument('--output_dir', type=str, default='./', help='Directory where rate images are stored and output will be written')
    parser.add_argument('--thresh', type=float, help='The threshold (fraction of masked pixels in an amp-row) above which to switch to a full-row median')
    parser.add_argument('--save_patterns', action='store_true', help='Save the horizontal and vertical striping patterns as FITS files')
    parser.add_argument('--nproc', type=int, default=cpu_count() // 2,
                        help='Number of parallel processes to use (default: half of available cores)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--runone', type=str, help='Filename of single file to clean. Overrides the runall argument')
    group.add_argument('--runall', action='store_true', help='Set to run all *rate.fits images in the output_dir directory.')
    parser.add_argument('image', nargs='?', type=str, help='Filename of rate image for pattern subtraction (required if --runone is not used)')
    parser.add_argument('--apply_flat', dest='apply_flat', action=argparse.BooleanOptionalAction, required=False, default=True)
    parser.add_argument('--mask_sources', dest='mask_sources', action=argparse.BooleanOptionalAction, required=False, default=True)
    args = parser.parse_args()

    log, log_file_path = setup_logger(args.output_dir)

    if args.runone:
        images = [os.path.join(args.output_dir, args.runone)]
    elif args.runall:
        images = glob(os.path.join(args.output_dir, '*rate.fits'))
        images.sort()
        
    crds_context = os.environ.get('CRDS_CONTEXT', None)
    if not crds_context:
        crds_context = crds.get_default_context()
    # Pre-fetch flats for all images
    flats_dict = {}
    for image in images:
        model = ImageModel(image)
        crds_dict = {
            'INSTRUME': 'NIRCAM',
            'DETECTOR': model.meta.instrument.detector,
            'FILTER': model.meta.instrument.filter,
            'PUPIL': model.meta.instrument.pupil,
            'DATE-OBS': model.meta.observation.date,
            'TIME-OBS': model.meta.observation.time
        }
        flats = crds.getreferences(crds_dict, reftypes=['flat'])
        if 'flat' in flats:
            flats_dict[image] = flats['flat']
            log.info(f'Flat file {flats["flat"]} for {os.path.basename(image)} is available.')

    if args.runone:
        pre1f = images[0].replace('rate.fits', 'rate_pre1f.fits')
        measure_striping(log, images[0], pre1f, args.output_dir, thresh=args.thresh, apply_flat=args.apply_flat, mask_sources=args.mask_sources, save_patterns=args.save_patterns, flat_file=flats_dict[images[0]])
        cleanup_intermediate_files(log, args.output_dir, args.runone)
    elif args.runall:
        pool_args = [(rate, rate.replace('rate.fits', 'rate_pre1f.fits'), args.output_dir, args.thresh, args.apply_flat, args.mask_sources, args.save_patterns, flats_dict[rate], log) for rate in images]
        effective_nproc = min(args.nproc, len(pool_args))
        with Pool(processes=effective_nproc) as pool:
            with tqdm(total=len(pool_args), file=sys.stdout) as pbar:
                for _ in pool.imap_unordered(process_file, pool_args):
                    pbar.update(1)

if __name__ == '__main__':
    main()
