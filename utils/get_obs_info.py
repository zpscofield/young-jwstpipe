#!/usr/bin/env python3
import os
import glob
import sys
from astropy.io import fits
import yaml


def safe_str(x, default=""):
    if x is None:
        return default
    s = str(x).strip()
    return s if s else default


def read_program_id(path: str) -> str:
    """Read PROGRAM from primary header (fallback '00000')."""
    try:
        with fits.open(path) as hdul:
            return safe_str(hdul[0].header.get("PROGRAM", "00000"), "00000")
    except Exception:
        return "00000"


def find_uncal_files(data_dir: str):
    """Recursively find all *_uncal.fits under data_dir."""
    pattern = os.path.join(os.path.abspath(data_dir), "**", "*_uncal.fits")
    return sorted(glob.glob(pattern, recursive=True))


# Detectors to prefer for a test run. nrca3 and nrcb4 are the wisp-affected
# short-wavelength detectors, so a test run exercises the wisp code; each
# short-wavelength pick is paired with its module's long-wavelength detector
# because wisp subtraction builds its source mask from the LW image.
TEST_SW_PREFERENCE = ["nrca3", "nrcb4", "nrca1", "nrca2", "nrca4", "nrcb1", "nrcb2", "nrcb3"]


def _read_filter_and_detector(path: str):
    try:
        with fits.open(path) as hdul:
            h = hdul[0].header
            return safe_str(h.get("FILTER"), "UNKNOWN"), safe_str(h.get("DETECTOR"), "").lower()
    except Exception:
        return "UNKNOWN", ""


def select_test_subset(files, exposures_per_filter: int = 2):
    """Pick a small set of uncal files that still exercises every step.

    Groups files by exposure (filename without the detector), keeps the
    first ``exposures_per_filter`` exposures of every filter, and from each
    kept exposure takes one preferred SW detector plus its module's LW
    detector. Two exposures per filter is the minimum for outlier detection
    and sky matching to do real work in stage 3.
    """
    exposures = {}
    for f in files:
        base = os.path.basename(f)
        parts = base.split("_")
        detector = parts[-2].lower() if len(parts) >= 3 else ""
        exp_key = "_".join(parts[:-2])
        filt, det_hdr = _read_filter_and_detector(f)
        detector = det_hdr or detector
        exposures.setdefault(exp_key, {})[detector] = (f, filt)

    covered = {}
    chosen = []
    for exp_key in sorted(exposures):
        filters = {filt for _, filt in exposures[exp_key].values()}
        if any(covered.get(flt, 0) < exposures_per_filter for flt in filters):
            chosen.append(exp_key)
            for flt in filters:
                covered[flt] = covered.get(flt, 0) + 1

    subset = []
    for exp_key in chosen:
        dets = exposures[exp_key]
        sw = next((d for d in TEST_SW_PREFERENCE if d in dets), None)
        if sw is not None:
            subset.append(dets[sw][0])
            lw = f"nrc{sw[3]}long"
            if lw in dets:
                subset.append(dets[lw][0])
        else:
            lw_dets = sorted(d for d in dets if d.endswith("long"))
            if lw_dets:
                subset.append(dets[lw_dets[0]][0])
    return sorted(set(subset))


def get_observation_info(data_dir: str, combine: bool, group_by_directory: bool, name: str):
    """
    Implements user rules:

    1) combine=False, group_by_directory=False:
       - group by PROGRAM id (proposal id), one job per program
       - payload is comma-separated uncal file list for that program

    2) combine=True:
       - one job total containing ALL uncal files under data_dir
       - payload is comma-separated uncal file list

    3) group_by_directory=True (overrides combine):
       - data_dir should contain subdirectories
       - one job per IMMEDIATE subdirectory that contains uncal files (recursive within that subdir)
       - name: Output_<subdir>
       - payload is comma-separated file list for that subdir
    """
    data_dir = os.path.abspath(data_dir)
    uncal_files = find_uncal_files(data_dir)

    obs_info = []
    program_ids = set()

    if not uncal_files:
        return obs_info, []

    # Rule C: group_by_directory overrides combine
    if group_by_directory:
        # immediate subdirs only
        subdirs = sorted(
            d for d in glob.glob(os.path.join(data_dir, "*"))
            if os.path.isdir(d)
        )

        for subdir in subdirs:
            files = sorted(glob.glob(os.path.join(subdir, "**", "*_uncal.fits"), recursive=True))
            if not files:
                continue

            subname = os.path.basename(subdir)
            obs_name = f"Output_{subname}".replace(" ", "_").replace(".", "_")

            for f in files:
                program_ids.add(read_program_id(f))

            obs_info.append((obs_name, ",".join(files)))

        return obs_info, sorted(program_ids)

    # Rule B: combine everything
    if combine:
        for f in uncal_files:
            program_ids.add(read_program_id(f))
        obs_info.append((name, ",".join(uncal_files)))
        return obs_info, sorted(program_ids)

    # Rule A: default => group by PROGRAM id
    files_by_program = {}
    for f in uncal_files:
        pid = read_program_id(f)
        program_ids.add(pid)
        files_by_program.setdefault(pid, []).append(f)

    for pid in sorted(files_by_program.keys()):
        files = sorted(files_by_program[pid])
        obs_info.append((pid, ",".join(files)))

    return obs_info, sorted(program_ids)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    test_subset = "--test-subset" in sys.argv[1:]
    pipeline_dir = args[0] if args else "."
    config_file = os.path.join(pipeline_dir, "config.yaml")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f) or {}

    data_dir = config.get("data_directory") or pipeline_dir
    combine_observations = bool(config.get("combine_observations", False))
    group_directory = bool(config.get("group_by_directory", False))
    custom_name = safe_str(config.get("custom_name", "Combined_Observation"), "Combined_Observation")

    obs_info, program_names = get_observation_info(
        data_dir=data_dir,
        combine=combine_observations,
        group_by_directory=group_directory,
        name=custom_name,
    )

    for obs_name, payload in obs_info:
        if test_subset:
            # Test runs use a handful of files and keep their outputs apart
            # from real reductions by adding a _test suffix to the name.
            subset = select_test_subset(payload.split(","))
            print(f"OBS:{obs_name}_test:{','.join(subset)}")
        else:
            print(f"OBS:{obs_name}:{payload}")

    print(f"TARGET_NAMES:{','.join(program_names)}")


if __name__ == "__main__":
    main()
