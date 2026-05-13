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
    pipeline_dir = sys.argv[1] if len(sys.argv) > 1 else "."
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
        print(f"OBS:{obs_name}:{payload}")

    print(f"TARGET_NAMES:{','.join(program_names)}")


if __name__ == "__main__":
    main()
