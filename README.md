# Yonsei Observable UNiverse Group (YOUNG) JWST Calibration Pipeline

A streamlined implementation of the James Webb Space Telescope (JWST) calibration pipeline with additions to improve both the final mosaic image quality and to simplify the execution of the various stages of the pipeline. The pipeline is driven from a web-based (Streamlit) interface that handles data discovery, configuration, execution, and color-image creation — all from one page.

## Authors & Contributors

**[Zachary P. Scofield](https://github.com/zpscofield)**

**[Hyungjin Joo](https://github.com/Hyungjin-Joo)**

## What this JWST calibration pipeline implementation does

- Searches MAST and downloads uncalibrated JWST NIRCam data (`_uncal.fits`) directly from the interface — by target name, RA/Dec, or program ID — or uses a directory of `_uncal.fits` files you already have.
    - Files can be from the same observation or overlapping observations. A new association is created during the final processing stage, so it is unnecessary to use the default associations from the JWST pipeline.
- Executes Stages 1, 2, and 3 of the default JWST calibration pipeline, with added calibration steps:
    - **1/f noise correction** — a gradient-based technique that estimates the correlated 1/f striping from the image derivatives and removes it from the cal-level images. Source masking protects bright, extended emission, and estimates can be made per amplifier (512-pixel sections) or across the whole image, which improves performance in cluster fields.
    - **Wisp correction** (modified version of Ben Sunnquist's wisp correction algorithm, JWST documentation version 4).
    - **Background subtraction** (modified version of Henry C. Ferguson's tiered-source-masking background subtraction). The background subtraction code is courtesy of the [*CEERS team*](https://github.com/ceers/ceers-nircam).
- Speeds up execution with parallel processing (Python multiprocessing) across exposures and filters; worker counts are configurable per stage.
- Organizes calibrated exposures by filter and uses the source catalog from the longest-wavelength filter for astrometric alignment.
    - The longest-wavelength filter can also be matched to an external catalog (e.g. `GAIADR3`). If no external catalog is provided, it is not matched to any catalog.
    - Resampling parameters are set in the interface. The pipeline keeps all necessary parameters (pixel scale, pixfrac, resampling kernel, center pixel, center RA/Dec, output shape, rotation) consistent between filters.
- Exposes every JWST step parameter for Stages 1, 2, and 3 as guided overrides, introspected from the installed `jwst` version, so any sub-step parameter can be tuned without editing code.
- Produces aligned mosaic images, segmentation maps, and source catalogs for each filter, and can build a stretched RGB **color image** from the resulting mosaics.

## Installation and Requirements

### 1. Python environment

Create the environment from the provided files. With conda:

```bash
conda env create -f environment.yml
conda activate young-jwstpipe
```

Or with pip (into a Python 3.11 environment):

```bash
pip install -r requirements.txt
```

General JWST pipeline installation notes are on the [JWST Calibration Pipeline site](https://jwst-pipeline.readthedocs.io/en/latest/).

### 2. Command-line YAML processor (`yq` + `jq`)

The pipeline shell script reads `config.yaml` with [`yq`](https://pypi.org/project/yq/) using `jq`-style filters. This `yq` is a thin wrapper around [`jq`](https://stedolan.github.io/jq/), so **both** must be installed.

`jq` is a system binary and is installed differently from Python packages:

```bash
# Linux (Debian/Ubuntu)
sudo apt install jq

# macOS (Homebrew)
brew install jq
```

Then install `yq` (the Python wrapper) into your environment:

```bash
pip install yq
```

> Verify with `jq --version` and `yq --version`. If `yq` errors about `jq` not being found, `jq` is not installed or not on your `PATH`.

### 3. Wisp templates

The wisp templates for the wisp-correction step are available at [stsci.app.box.com](https://stsci.app.box.com/s/1bymvf1lkrqbdn9rnkluzqk30e8o2bne). Use the version 4 templates. Place the `FITS` files in a folder and point the **WISP templates directory** field at it.

> **Tip:** Have a large directory available — intermediate files for every stage are saved.

## Usage

### Launch the interface

```bash
./start.sh
```

On macOS you can also double-click `start.command`. Streamlit opens your browser automatically at `http://localhost:8501`.

**Running on a remote server over SSH:** `start.sh` detects the SSH session, starts Streamlit headless, and prints the exact `ssh -L 8501:localhost:8501 ...` port-forward command to run on your laptop. (VSCode/Cursor Remote-SSH usually auto-forwards the port and offers an "Open in Browser" popup, in which case you can skip that step.)

### Configure and run

The page is organized top to bottom:

1. **Data source** — search MAST (by target, RA/Dec, or program ID) and download, or point at an existing directory of `_uncal.fits` files.
2. **Output & grouping** — output directory, a custom run name, and how observations are grouped (by program ID, by subdirectory, or combined).
3. **Calibration steps** — toggle which stages and optional calibration steps run, and set the WISP templates directory.
4. **Performance** — parallel worker counts per stage, plus stage-3 multiprocessing options.
5. **CRDS** — CRDS cache path and server URL.
6. **Advanced settings** — guided per-step parameter overrides for Stages 1/2/3 (with a reference of every parameter for your installed `jwst` version), the curated Stage 3 settings (resample, outlier detection, tweakreg, skymatch), and the background / WISP / 1/f-noise step settings.
7. **Color image** — per-filter hues and stretch settings; generate an RGB color image from the stage-3 mosaics.

Click **Save & Run pipeline** to write `config.yaml` and start the run; the log streams live in the page (full per-stage detail is also written to `<output>/<obs>/logs/`).

### Important notes

**Skipping calibration steps.** The optional steps (**1/f noise correction**, **wisp subtraction**, **background subtraction**) can be safely skipped — they detect existing files and adjust automatically. Skipping them will not cause errors but may affect final mosaic quality.

**Critical stage dependencies.** The main stages cannot be skipped unless their outputs already exist and are valid:
- Stage 2 requires the output from Stage 1.
- Stage 3 requires the output from Stage 2.

Starting at Stage 2 or Stage 3 without the required prior outputs will result in errors.

**File overwriting.** Calibration steps modify the pipeline outputs in place. For example, running the 1/f noise correction permanently alters the Stage 1 output unless you re-run Stage 1; `wisp_subtraction`, `cal_fnoise_reduction`, and `background_subtraction` overwrite the previous Stage 2 files. To preserve the default JWST stage outputs, run only the main stages without additional calibration steps, or back up your outputs first.

**CRDS reference downloads.** During the first run, do **not** skip the `download_uncal_references`, `download_rate_references`, and `download_cal_references` steps. They populate the CRDS cache before the main stages so the multiprocessing stages don't all hit the CRDS server at once (which can overload the API and crash the run).

**CRDS cache location.** Keep the CRDS cache on a **local drive**. A network-attached (NAS) or remote drive causes severe slowdowns in Stages 1–3 due to heavy file I/O.

### Running without the interface

The interface writes `config.yaml` and then runs `young_pipeline.sh`, which is the actual execution engine. You can also edit `config.yaml` by hand and run the engine directly:

```bash
./young_pipeline.sh
```

A new directory is created at `output_directory/<run name>` containing the output from every stage, along with `.log` files. Stage 3 log files are located in the per-filter output directories.

## Acknowledgements

This project includes code or functionality derived from the [*jwst* project](https://github.com/spacetelescope/jwst), developed by the Space Telescope Science Institute (STScI) and the Association of Universities for Research in Astronomy (AURA). This project also incorporates algorithms from the [*ceers-nircam* project](https://github.com/ceers/ceers-nircam).
