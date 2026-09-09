# Yonsei Observable UNiverse Group (YOUNG) JWST Calibration Pipeline

A streamlined implementation of the James Webb Space Telescope (JWST) calibration pipeline with additions to improve both the final mosaic image quality and to simplify the execution of the various stages of the pipeline. The pipeline is driven from a web-based (Streamlit) interface that handles data discovery, configuration, execution, and color-image creation — all from one page.

## Authors & Contributors

**[Zachary P. Scofield](https://github.com/zpscofield)**

**[Hyungjin Joo](https://github.com/Hyungjin-Joo)**

**[Kyle Finner](https://github.com/kfinner)**

## What this JWST calibration pipeline implementation does

- Searches MAST and downloads uncalibrated JWST NIRCam data (`_uncal.fits`) directly from the interface — by target name, RA/Dec, or program ID — or uses a directory of `_uncal.fits` files you already have.
    - Files can be from the same observation or overlapping observations. A new association is created during the final processing stage, so it is unnecessary to use the default associations from the JWST pipeline.
- Executes Stages 1, 2, and 3 of the default JWST calibration pipeline, with added calibration steps:
    - **1/f noise correction** — a gradient-based technique that estimates the correlated 1/f striping from the image derivatives and removes it from the cal-level images. Source masking protects bright, extended emission, and estimates can be made per amplifier (512-pixel sections) or across the whole image, which improves performance in cluster fields.
    - **Wisp correction** (modified version of Ben Sunnquist's wisp correction algorithm, JWST documentation version 4).
    - **Background subtraction** (modified version of Henry C. Ferguson's tiered-source-masking background subtraction). The background subtraction code is courtesy of the [*CEERS team*](https://github.com/ceers/ceers-nircam).
- Speeds up execution with parallel processing (Python multiprocessing) across exposures and filters; worker counts are configurable per stage.
- Organizes calibrated exposures by filter and aligns every filter to the source catalog of a **reference filter**. By default the reference is the filter covering the largest sky area, so every other filter has reference sources across as much of its footprint as possible; the longest wavelength breaks exact ties. It can also be chosen explicitly.
    - The reference filter can also be matched to an external catalog (e.g. `GAIADR3`). If no external catalog is provided, it is not matched to any catalog.
    - All filters are resampled onto one shared pixel grid. By default that grid covers the combined footprint of every filter, so combining programs with different coverage never crops a filter; optionally it can be limited to the reference filter's footprint.
    - Resampling parameters are set in the interface. The pipeline keeps all necessary parameters (pixel scale, pixfrac, resampling kernel, center pixel, center RA/Dec, output shape, rotation) consistent between filters.
- Exposes every JWST step parameter for Stages 1, 2, and 3 as guided overrides, introspected from the installed `jwst` version, so any sub-step parameter can be tuned without editing code.
- Produces aligned mosaic images, segmentation maps, and source catalogs for each filter, and can build a stretched RGB **color image** from the resulting mosaics.

## Installation and Requirements

The pipeline runs on Linux and macOS (on Windows, use WSL2). Everything it needs comes from one Python environment; there are no other system tools to install.

### 1. Before you start

- **git** and either **conda** ([Miniforge](https://github.com/conda-forge/miniforge) is recommended) or **Python 3.11**.
- **Internet access** from the machine that runs the pipeline. It downloads data from MAST and calibration reference files from CRDS.
- **Disk space.** Intermediate files from every stage are kept, so a NIRCam program can take hundreds of GB. The CRDS reference cache grows to tens of GB and must be on a **local** drive; a network drive makes Stages 1–3 very slow.
- **Memory.** Stage 3 builds each mosaic in memory. Large fields at 0.02"/pixel can need tens of GB of RAM.

### 2. Get the code

```bash
git clone https://github.com/zpscofield/young-jwstpipe.git
cd young-jwstpipe
```

### 3. Create the Python environment

With conda (recommended):

```bash
conda env create -f environment.yml
conda activate young-jwstpipe
```

Or with pip, in a fresh Python 3.11 virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs the STScI `jwst` pipeline, `crds`, Streamlit, and everything else the pipeline uses. General notes on the JWST pipeline itself are on the [JWST Calibration Pipeline site](https://jwst-pipeline.readthedocs.io/en/latest/).

### 4. Check the installation

With the environment activated:

```bash
python -c "import jwst; print(jwst.__version__)"
streamlit --version
```

Both should print a version number.

### 5. CRDS reference files

Calibration reference files are fetched from CRDS automatically during the first run and cached in a directory you choose in the **CRDS** section of the interface. No account is needed. Choose a directory on a local drive and keep using the same one; it is shared by every run.

### 6. Wisp templates

The wisp templates for the wisp-correction step are available at [stsci.app.box.com](https://stsci.app.box.com/s/1bymvf1lkrqbdn9rnkluzqk30e8o2bne). Use the version 4 templates. Place the `FITS` files in a folder and point the **WISP templates directory** field at it. If you skip wisp subtraction, you do not need them.

## Usage

### Launch the interface

```bash
./run.sh
```

On macOS you can also double-click `run.command`. Streamlit opens your browser automatically at `http://localhost:8501`.

**Running on a remote server over SSH:** `run.sh` detects the SSH session, starts Streamlit headless, and prints the exact `ssh -L 8501:localhost:8501 ...` port-forward command to run on your laptop. (VSCode/Cursor Remote-SSH usually auto-forwards the port and offers an "Open in Browser" popup, in which case you can skip that step.)

### Configure and run

The page is organized top to bottom:

1. **Data source** — search MAST (by target, RA/Dec, or program ID) and download, or point at an existing directory of `_uncal.fits` files.
2. **Output & grouping** — output directory, a custom run name, and how observations are grouped (by program ID, by subdirectory, or combined).
3. **Calibration steps** — toggle which stages and optional calibration steps run, and set the WISP templates directory.
4. **Performance** — parallel worker counts per stage, plus stage-3 multiprocessing options.
5. **CRDS** — CRDS cache path and server URL.
6. **Advanced settings** — guided per-step parameter overrides for Stages 1/2/3 (with a reference of every parameter for your installed `jwst` version), and the background / WISP / 1/f-noise step settings.
7. **Required mosaic creation settings** — the stage-3 settings you should review for your data: reference filter and mosaic footprint, resample (pixel scale, pixfrac, rotation, kernel), outlier detection, tweakreg (reference catalog and alignment), and skymatch.
8. **Color image** — per-filter hues and stretch settings; generate an RGB color image from the stage-3 mosaics.

Settings are loaded from `config.default.yaml` the first time and saved to `config.yaml` next to it, so the shipped defaults are never overwritten; **Reset to defaults** deletes your `config.yaml` and reloads the template.

**Check the setup first.** **Check setup** takes a few seconds and processes nothing. It checks the Python environment, reads every uncal header (instrument, exposure type, filters, detectors, truncated files), shows how the observations will be grouped, confirms the output directory is writable with enough space, checks that a wisp template exists for every detector and filter present, asks CRDS which reference files the run needs and reports which are not cached yet (they download at the start of the run) or are truncated from an interrupted download, estimates the stage 3 grid size and memory, and compares the worker counts with the CPU count. From the command line: `./young_pipeline.sh --check`.

**Save & Run pipeline** writes `config.yaml`, runs the same check, and starts the run only if it finds no problems; warnings show the report and ask you to confirm with **Run anyway**. The log streams live in the page (full per-stage detail is also written to `<output>/<obs>/logs/`).

**Small real test run (command line).** `./young_pipeline.sh --test` runs every enabled step on two exposures per filter (one wisp-affected short-wavelength detector and its long-wavelength partner) and writes to `<output>/<observation>_test`. It needs the same reference files as a full run, so the first time it can spend a while downloading them; after that it takes a few minutes.

**Runs survive disconnects.** The pipeline is started as a detached process on the machine running the interface, with its log written to `.pipeline_run/pipeline.log` in the pipeline directory. If you are working on a server, you can close the browser tab, drop the SSH tunnel, or shut your laptop; the reduction keeps going on the server. Reopen the page (running `./run.sh` again if needed) and it reattaches to the run, live or finished. A **Stop pipeline** button ends a run early. If you come back and the page shows **Connecting…**, the run is not lost: the SSH port forward died with your session. Re-run the `ssh -L` command that `run.sh` printed (or let VSCode reconnect and re-forward the port), or open the **Network URL** Streamlit printed if your computer is on the same network as the server.

To keep a dead tunnel from lingering and holding port 8501 on your laptop after it sleeps, add keepalives to `~/.ssh/config` on the laptop; VSCode Remote-SSH uses the same file:

```
Host *
    ServerAliveInterval 15
    ServerAliveCountMax 3
```

If `localhost:8501` still hangs while the Network URL works, a stale `ssh` process on the laptop is holding the port. Find it with `lsof -nP -iTCP:8501 -sTCP:LISTEN` and kill it; VSCode will reconnect and forward the port again. When working in VSCode, don't also run a manual `ssh -L` tunnel for the same port. If the pipeline is running on your own computer, putting it to sleep or shutting it down stops the reduction like any other process.

### Important notes

**Skipping calibration steps.** The optional steps (**1/f noise correction**, **wisp subtraction**, **background subtraction**) can be safely skipped — they detect existing files and adjust automatically. Skipping them will not cause errors but may affect final mosaic quality.

**Critical stage dependencies.** The main stages cannot be skipped unless their outputs already exist and are valid:
- Stage 2 requires the output from Stage 1.
- Stage 3 requires the output from Stage 2.

Starting at Stage 2 or Stage 3 without the required prior outputs will result in errors.

**File overwriting.** Calibration steps modify the pipeline outputs in place. For example, running the 1/f noise correction permanently alters the Stage 1 output unless you re-run Stage 1; `wisp_subtraction`, `cal_fnoise_reduction`, and `background_subtraction` overwrite the previous Stage 2 files. To preserve the default JWST stage outputs, run only the main stages without additional calibration steps, or back up your outputs first.

**CRDS reference downloads.** During the first run, do **not** skip the `download_uncal_references`, `download_rate_references`, and `download_cal_references` steps. They populate the CRDS cache before the main stages so the multiprocessing stages don't all hit the CRDS server at once (which can overload the API and crash the run).

**Failures stop the run.** If any exposure fails in Stage 1 or 2, or any filter fails in Stage 3, the stage reports how many failed, the run stops with a non-zero exit status, and the interface shows it as failed. Earlier versions kept going and quietly produced mosaics missing the failed exposures. The per-file reason is in `<output>/<obs>/logs/`.

**Truncated reference files.** Stopping a run (or losing the connection) during a CRDS reference download can leave a partially written reference file in the cache. CRDS then treats it as present, and every exposure that needs it fails in Stage 1 with a message such as `cannot reshape array` or `buffer is too small`. Delete that file from the cache and rerun; it will be downloaded again.

**CRDS cache location.** Keep the CRDS cache on a **local drive**. A network-attached (NAS) or remote drive causes severe slowdowns in Stages 1–3 due to heavy file I/O.

### Running without the interface

The interface writes `config.yaml` and then runs `young_pipeline.sh`, which is the actual execution engine. You can also create `config.yaml` by hand and run the engine directly:

```bash
cp config.default.yaml config.yaml   # then edit the paths
./young_pipeline.sh
```

A new directory is created at `output_directory/<run name>` containing the output from every stage, along with `.log` files. Stage 3 log files are located in the per-filter output directories.

## Acknowledgements

This project includes code or functionality derived from the [*jwst* project](https://github.com/spacetelescope/jwst), developed by the Space Telescope Science Institute (STScI) and the Association of Universities for Research in Astronomy (AURA). This project also incorporates algorithms from the [*ceers-nircam* project](https://github.com/ceers/ceers-nircam).
