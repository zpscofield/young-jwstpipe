# Yonsei Observable UNiverse Group (YOUNG) JWST Calibration Pipeline

A streamlined implementation of the James Webb Space Telescope (JWST) calibration pipeline with additions to improve both the final mosaic image quality and to
simplify the execution of the various stages of the pipeline.

## Authors & Contributors

**[Zachary P. Scofield](https://github.com/zpscofield)**, 
**[Hyungjin Joo](https://github.com/Hyungjin-Joo)**

## What this JWST calibration pipeline implementation does

- Uses a directory containing uncalibrated JWST data files (_uncal.fits) as input
    - These uncalibrated data files can be from the same observation or different overlapping observations. A new association is made in the final processing stage, so it is 
    unnecessary to use the default associations from the pipeline.
- Executes stages 1, 2, and 3 of the default JWST calibration pipeline, with added calibration steps throughout. These calibration steps include:
    - 1/f noise correction (Modified version of Micaela Bagley's "remstriping" algorithm)
    - Wisp correction (Modified version of Ben Sunnquist's wisp correction algorithm provided in the JWST documentation (version 3))
    - Background subtraction (Modified version of Henry C. Ferguson's background subtraction)
        - The modifications of these calibration steps mostly involve efficiency improvements or changes to help with integration into the YOUNG JWST Pipeline.
        - The modified 1/f noise correction algorithm is more significantly modified to handle bright, extended sources and perform better in cluster fields. In general, 
        it does this by using full-row 1/f noise estimates when the amplifier estimate is unreasonably high. Additionally, this algorithm was updated to incorporate multiprocessing.
        - The original 1/f noise correction and background subtraction codes are courtesy of the [*CEERS team*](https://github.com/ceers/ceers-nircam).
- Speeds up pipeline execution by implementing MPI for stage 1 and 2 processing.
- Organizes calibrated exposures based on filter and uses the source catalog from the longest wavelength filter as the astrometric calibration catalog for other filters to ensure
proper alignment.
    - The longest wavelength filter can also be matched to an external catalog provided by the user, and if no external catalog is provided then the initial absolute astrometric
    catalog will be the GAIADR3 catalog.
    - Resampling parameters can be set in the config.yaml file. The pipeline insures that all necessary parameters (pixel scale, pixfrac, resampling kernel, center pixel, center RA 
    and DEC, output shape, rotation) are matched between filters.
- Produces aligned mosaic images, segmentation maps, and source catalogs for each filter in an observation/set of overlapping observations.

## Installation and Requirements

Installation instructions can be found on the [*JWST calibration pipeline*](https://jwst-pipeline.readthedocs.io/en/latest/) site, along with other information regarding the default 
pipeline. It is also necessary to have the command-line YAML processor [*yq*](https://pypi.org/project/yq/) installed. 

The wisp templates necessary for the wisp correction step can be found [*here*](https://stsci.app.box.com/s/1bymvf1lkrqbdn9rnkluzqk30e8o2bne/folder/275049066832?page=2), and are the version 3 templates provided in the [*JWST documentation*](https://jwst-docs.stsci.edu/known-issues-with-jwst-data/nircam-known-issues/nircam-scattered-light-artifacts#gsc.tab=0). These templates (FITS files) should be placed in the provided wisp-template folder within the ./utils directory.

It is strongly encouraged to have a large directory available when running this JWST pipeline implementation, given that the intermediate files are saved.

## Usage

To use the YOUNG JWST calibration pipeline, simply modify the config.yaml file as necessary. It is important to properly set the CRDS environment variables as well as to specify the uncalibrated file directory. Different steps of the pipeline can be skipped by uncommenting the steps within the config.yaml file, which can be helpful if you have already finished stage 1 processing and do not want to repeat it. Added calibration steps can also be skipped, such as the 1/f noise correction, wisp removal, and background subtraction. These additional calibration steps introduce new files and can also rename the default pipeline output files, so we suggest always starting the pipeline at a certain stage (stage 1, 2, or 3) rather than at an one of the added calibration steps. This prevents any complications due to changes in file names. If it is necessary to start the pipeline at a calibration step rather than a main stage of the pipeline, do so only if you are certain that this calibration step has not been run previously and that you have the necessary input files for the step. Take the following examples:
    - You have previously executed the pipeline up to and including stage 1. It is then safe to start the pipeline at the fnoise_correction step since this step has not been executed previously.
    - You have previously executed the pipeline up to and including stage 2. It is then safe to start the pipeline at the wisp_correction step since this step has not been executed previously.

Finally, to run the pipeline, simply use the following command:
- $ ./young_pipeline.sh

Once the pipeline execution begins, a new directory will be created (.[target]/output) which will contain the output from all stages of the pipeline. Additionally, a log file (pipeline.log) will be created to keep track of the detailed output from the pipeline.
