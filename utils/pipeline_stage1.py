#!/usr/bin/env python3
import os
import glob
import yaml
import numpy as np
from jwst.pipeline import Detector1Pipeline
from tqdm.auto import tqdm
import logging
import sys
import argparse
from multiprocessing import Pool, cpu_count
from contextlib import contextmanager

from log_utils import archive_existing_log


@contextmanager
def redirect_output_to_file(log_file):
    """Temporarily redirect stdout and stderr to a log file."""
    log_fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(log_fd)


# Load configuration (same behavior as your original script)
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file) or {}

# CRDS env
if "crds_path" in config:
    os.environ["CRDS_PATH"] = os.path.expanduser(str(config["crds_path"]))
if "crds_server_url" in config:
    os.environ["CRDS_SERVER_URL"] = str(config["crds_server_url"])


def setup_logger(output_dir):
    """
    Setup a logger for the pipeline using the provided output directory.
    Writes to: <parent_of_output_dir>/logs/pipeline_stage1.log
    """
    parent_dir = os.path.dirname(output_dir.rstrip("/"))
    log_file_path = os.path.join(parent_dir, "logs", "pipeline_stage1.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    archive_existing_log(log_file_path)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if script is called multiple times
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file_path for h in log.handlers):
        file_handler = logging.FileHandler(log_file_path, mode="a")
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)

    with open(log_file_path, "a") as log_file:
        log_file.write("------------------\n")
        log_file.write("Stage 1 Processing\n")
        log_file.write("------------------\n\n")

    return log, log_file_path


def _write_error(log_file, msg):
    try:
        with open(log_file, "a") as f:
            f.write(msg.rstrip() + "\n")
    except Exception:
        # last resort: print
        print(msg, file=sys.stderr)


def process_file(args):
    """
    Process a single file.
    NOTE: we do NOT pass a logger object into multiprocessing workers (pickle issues on macOS/spawn).
    """
    img, output_dir, log_file = args
    try:
        # maximum_cores must default to "1": the step's own multiprocessing
        # would otherwise contend with the outer per-file Pool wrapper.
        steps = {
            "ramp_fit": {"maximum_cores": config.get("ramp_fit_cores") or "1"},
            "jump": {"maximum_cores": config.get("jump_cores") or "1"},
        }
        # Deep-merge any guided per-step overrides from the UI on top of the
        # defaults above (so e.g. jump.rejection_threshold merges with the
        # jump.maximum_cores set here). Absent => unchanged behaviour.
        for _step, _params in (config.get("stage1_step_overrides") or {}).items():
            steps.setdefault(_step, {}).update(_params or {})
        with redirect_output_to_file(log_file):
            Detector1Pipeline.call(
                img,
                steps=steps,
                output_dir=output_dir,
                save_results=True,
            )
    except Exception as e:
        _write_error(log_file, f"Failed to process {img}: {e}")


def build_uncal_list(combined_mode: bool, input_dir_or_list: str):
    """
    Returns a sorted numpy array of *_uncal.fits files.

    - combined_mode=True:
        * If input is a directory: recurse and find all *_uncal.fits under it.
        * Else: treat input as comma-separated list of files.
    - combined_mode=False:
        * input must be a directory; search only that directory for *_uncal.fits (non-recursive).
    """
    s = input_dir_or_list.strip()

    if combined_mode:
        if os.path.isdir(s):
            files = glob.glob(os.path.join(s, "**", "*_uncal.fits"), recursive=True)
            return np.sort(np.array(files, dtype=str))
        else:
            items = [x.strip() for x in s.split(",") if x.strip()]
            return np.sort(np.array(items, dtype=str))

    # non-combined: expect a directory
    if not os.path.isdir(s):
        # allow passing a single file by accident
        if os.path.isfile(s) and s.endswith("_uncal.fits"):
            return np.sort(np.array([s], dtype=str))
        return np.sort(np.array([], dtype=str))

    files = glob.glob(os.path.join(s, "*_uncal.fits"))
    return np.sort(np.array(files, dtype=str))


def main(combined_mode, input_dir, output_dir, nproc, log, log_file_path):
    uncal_list = build_uncal_list(combined_mode, input_dir)

    log.info(f"combined_mode={combined_mode}")
    log.info(f"input_dir={input_dir}")
    log.info(f"output_dir={output_dir}")
    log.info(f"Total files to process: {len(uncal_list)}")

    if len(uncal_list) == 0:
        log.error("No *_uncal.fits files found. Check --input_dir and file naming.")
        return 1

    os.makedirs(output_dir, exist_ok=True)

    task_args = [(img, output_dir, log_file_path) for img in uncal_list]

    # --- NEW: report CPU + process usage ---
    host_cores = cpu_count()
    requested = int(nproc)
    log.info(f"Host CPU cores detected: {host_cores}")
    log.info(f"Requested nproc: {requested}")

    if requested <= 1:
        log.info("Running stage1 sequentially (nproc=1).")
        with tqdm(total=len(task_args), file=sys.stderr) as pbar:
            for a in task_args:
                process_file(a)
                pbar.update(1)
    else:
        effective_nproc = min(requested, len(task_args))
        log.info(f"Running stage1 with multiprocessing: {effective_nproc} worker process(es) "
                 f"(min(requested={requested}, nfiles={len(task_args)})).")
        print(f"[Stage1] Using {effective_nproc} process(es) (requested={requested}, nfiles={len(task_args)}, cores={host_cores})", flush=True)

        with Pool(processes=effective_nproc) as pool:
            with tqdm(total=len(task_args), file=sys.stdout) as pbar:
                for _ in pool.imap_unordered(process_file, task_args):
                    pbar.update(1)

    return 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1 of the JWST data reduction pipeline."
    )
    parser.add_argument(
        "--combined_mode",
        action="store_true",
        help="If set: input_dir may be a comma-separated list of files, or a directory (recursive search).",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing *_uncal.fits (non-combined mode), or comma-separated file list (combined mode).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where output will be written",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=max(1, cpu_count() // 2),
        help="Number of parallel processes to use (default: half of available cores)",
    )
    args = parser.parse_args()

    log, log_file_path = setup_logger(args.output_dir)
    sys.exit(main(args.combined_mode, args.input_dir, args.output_dir, args.nproc, log, log_file_path))
