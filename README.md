# Yonsei Observable UNiverse Group (YOUNG) JWST Calibration Pipeline

A streamlined implementation of the James Webb Space Telescope (JWST) calibration pipeline with additions to improve both the final mosaic image quality and to
simplify the execution of the various stages of the pipeline.

## Authors & Contributors

**[Zachary P. Scofield](https://github.com/zpscofield)**

**[Hyungjin Joo](https://github.com/Hyungjin-Joo)**

## What this JWST calibration pipeline implementation does

- Uses a directory containing uncalibrated JWST data files (`_uncal.fits`) as input.
    - These files can be from the same observation or overlapping observations.  
      A new association is created during the final processing stage, so it is unnecessary to use the default associations from the JWST pipeline.
- Executes stages 1, 2, and 3 of the default JWST calibration pipeline, with added calibration steps:
    - **1/f noise correction** (Modified version of Micaela Bagley's *remstriping* algorithm).  
      An alternative gradient-based method is also available for improved performance in fields with extended light.
    - **Wisp correction** (Modified version of Ben Sunnquist's wisp correction algorithm, JWST documentation version 3).
    - **Background subtraction** (Modified version of Henry C. Ferguson's background subtraction).
        - The modifications mostly involve efficiency improvements or changes to improve integration into the YOUNG JWST Pipeline.
        - The 1/f noise correction is significantly modified to handle bright, extended sources and perform better in cluster fields.  
          It uses full-row estimates when amplifier estimates are unreliable and incorporates multiprocessing.
        - The original 1/f noise correction and background subtraction codes are courtesy of the [*CEERS team*](https://github.com/ceers/ceers-nircam).
- Speeds up pipeline execution by implementing MPI for Stage 1 and 2 processing.
- Organizes calibrated exposures by filter and uses the source catalog from the longest-wavelength filter for astrometric alignment.
    - The longest-wavelength filter can also be matched to an external catalog provided by the user.  
      If no external catalog is provided, the longest-wavelength filter will not be matched to any catalog.
    - Resampling parameters can be set in the `config.yaml` file.  
      The pipeline ensures that all necessary parameters (pixel scale, pixfrac, resampling kernel, center pixel, center RA and DEC, output shape, rotation) are consistent between filters.
- Produces aligned mosaic images, segmentation maps, and source catalogs for each filter in an observation or set of overlapping observations.

## Installation and Requirements

Installation instructions can be found on the [JWST Calibration Pipeline site](https://jwst-pipeline.readthedocs.io/en/latest/), along with other information about the default pipeline.

It is also necessary to have the command-line YAML processor [`yq`](https://pypi.org/project/yq/) installed, as well as the packages [`psutil`](https://anaconda.org/conda-forge/psutil/) and [`tqdm`](https://anaconda.org/conda-forge/tqdm/).

The wisp templates necessary for the wisp correction step can be found [here](https://stsci.app.box.com/s/1bymvf1lkrqbdn9rnkluzqk30e8o2bne/folder/275049066832?page=2).  
These are the version 3 templates provided in the [JWST documentation](https://jwst-docs.stsci.edu/known-issues-with-jwst-data/nircam-known-issues/nircam-scattered-light-artifacts#gsc.tab=0).  
These templates (`FITS` files) should be placed in a location specified by the user in the `config.yaml` file.

> **Tip:**  
> It is strongly recommended to have a large directory available when running this pipeline, as intermediate files are saved.

## Usage

To use the YOUNG JWST calibration pipeline, modify the `config.yaml` file as necessary.  
Ensure you set the **CRDS environment variables** and specify the **input directory containing the uncalibrated (`uncal.fits`) files**.

Different steps of the pipeline can be skipped by listing them in the `skip_steps` section of the `config.yaml` file.  
This is useful if you have already finished certain steps and want to avoid repeating them.

### Skipping calibration steps
Optional calibration steps, such as **1/f noise correction**, **wisp subtraction**, and **background subtraction**, can be safely skipped.  
These steps are designed to detect existing input files and adjust automatically, so skipping them does not interfere with the pipeline's ability to proceed.  
Skipping these steps **will not cause errors**, but may affect the quality of the final mosaics.

### Critical pipeline dependencies
The **main pipeline stages have strict dependencies and cannot be skipped unless their outputs already exist and are valid**:
- **Stage 2 requires the output from Stage 1.**
- **Stage 3 requires the output from Stage 2.**

Attempting to start the pipeline at Stage 2 or Stage 3 **without previously running the required stages will result in errors**.

### Note on file overwriting behavior
Calibration steps **modify the pipeline outputs in place**.  
For example:
- Running `fnoise_correction` will permanently alter the Stage 1 output unless you re-run Stage 1.
- Similarly, running `wisp_subtraction`, `cal_fnoise_reduction`, or `background_subtraction` will overwrite the previous files in Stage 2 output.

If you want to preserve the **default JWST stage outputs**, you should:
- Run only the main stages (`stage1`, `stage2`, `stage3`) **without any additional calibration steps**.
- Or manually back up your outputs before running additional steps.

> **Notes:**  
> It is highly recommended **not to skip the `download_uncal_references`, `download_rate_references`, and `download_cal_references` steps during the first pipeline execution**.  
> These steps ensure that the required CRDS reference files are downloaded and cached locally before the main pipeline stages run.  
>  
> Skipping these steps without a fully populated CRDS cache may cause **multiprocessing stages to simultaneously request references from the CRDS server**, potentially overloading the
> CRDS API and causing the pipeline to crash.
>
> **IMPORTANT:**
> Ensure that your **CRDS cache is located on a local drive** within your system.  
> Using a network-attached storage (NAS) or remote drive can cause **severe slowdowns during Stage 1, Stage 2, and Stage 3**, due to heavy file I/O demands.

---

To run the pipeline:
```bash
$ ./young_pipeline.sh
```

Once the pipeline execution begins, a new directory will be created (output_directory/program) which will contain the output from all stages of the pipeline. Additionally, log files (.log) will be created to keep track of the detailed output from the pipeline. For stage 3 processing, log files will be located in the final filter directories.

## Acknowledgements

This project includes code or functionality derived from the [*jwst* project](https://github.com/spacetelescope/jwst), developed by the Space Telescope Science Institute (STScI) and the Association of Universities for Research in Astronomy (AURA). This project also incorporates algorithms from the [*ceers-nircam* project](https://github.com/ceers/ceers-nircam).
