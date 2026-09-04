"""Fast setup checks for a pipeline run: seconds, no processing, no downloads.

Checks the Python environment, the configuration, the input data, the
observation grouping, the output location, wisp templates, the CRDS cache
and server (which reference files are needed, which are missing, which are
truncated), the stage 3 grid, and the parallel settings.

    python utils/preflight.py [pipeline_dir]      # from the command line
    ./young_pipeline.sh --check                    # same thing

Each result is a Check(status, name, detail) with status "ok", "warn" or
"fail". A "fail" means the run cannot succeed as configured.
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from get_obs_info import find_uncal_files, get_observation_info, select_test_subset  # noqa: E402
from mosaic_footprint import _tangent_plane, choose_reference_filter, footprint_areas, parse_s_region  # noqa: E402
from nircam_filters import FILTER_PIVOT_WAVELENGTHS_UM  # noqa: E402

WISP_DETECTORS = ("NRCA3", "NRCA4", "NRCB3", "NRCB4")
REQUIRED_MODULES = ["jwst", "stcal", "crds", "astropy", "photutils", "numpy", "scipy",
                    "matplotlib", "yaml", "tqdm", "astroquery", "streamlit"]
NETWORK_FS = {"cifs", "nfs", "nfs4", "smb3", "smbfs", "fuse.sshfs", "afpfs"}


def crds_command() -> str | None:
    """Path to the crds CLI: on PATH, or next to the running Python."""
    found = shutil.which("crds")
    if found:
        return found
    sibling = Path(sys.executable).parent / "crds"
    return str(sibling) if sibling.exists() else None


@dataclass
class Check:
    status: str  # ok | warn | fail
    name: str
    detail: str = ""


def _versions() -> list[Check]:
    out = []
    missing, found = [], []
    for mod in REQUIRED_MODULES:
        try:
            m = importlib.import_module(mod)
            found.append(f"{mod} {getattr(m, '__version__', '')}".strip())
        except Exception:
            missing.append(mod)
    py = ".".join(str(v) for v in sys.version_info[:3])
    if missing:
        out.append(Check("fail", "Python environment",
                         f"Missing packages: {', '.join(missing)}. Activate the pipeline environment "
                         "or reinstall from environment.yml / requirements.txt."))
    else:
        out.append(Check("ok", "Python environment", f"Python {py}; " + ", ".join(found)))
    if crds_command() is None:
        out.append(Check("fail", "crds command", "'crds' was not found on PATH or next to this Python; it is "
                                                 "installed with the jwst package. Activate the pipeline environment."))
    return out


def _config_keys(config: dict, repo_root: Path) -> list[Check]:
    default_path = repo_root / "config.default.yaml"
    if not default_path.exists():
        return []
    with open(default_path) as f:
        defaults = yaml.safe_load(f) or {}
    missing = [k for k in defaults if k not in config]
    if missing:
        return [Check("warn", "Configuration keys",
                      f"Not set (pipeline defaults will be used): {', '.join(missing)}")]
    return [Check("ok", "Configuration keys", "All settings present.")]


def _read_uncal_headers(files: list[str]):
    rows, problems = [], []
    for f in files:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                with fits.open(f) as hdul:
                    h0 = hdul[0].header
                    sci = hdul["SCI"].header if "SCI" in hdul else {}
                    rows.append({
                        "path": f,
                        "instrume": str(h0.get("INSTRUME", "")).upper(),
                        "exp_type": str(h0.get("EXP_TYPE", "")).upper(),
                        "filter": str(h0.get("FILTER", "")).upper(),
                        "pupil": str(h0.get("PUPIL", "")).upper(),
                        "detector": str(h0.get("DETECTOR", "")).upper(),
                        "program": str(h0.get("PROGRAM", "")),
                        "readpatt": str(h0.get("READPATT", "")),
                        "subarray": str(h0.get("SUBARRAY", "")),
                        "s_region": str(sci.get("S_REGION", "")),
                        "size": os.path.getsize(f),
                    })
            except Exception as e:
                problems.append(f"{os.path.basename(f)}: {str(e)[:80]}")
                continue
            for x in w:
                if "truncated" in str(x.message):
                    problems.append(f"{os.path.basename(f)}: file is truncated")
    return rows, problems


def _data(config: dict) -> tuple[list[Check], list[dict]]:
    out = []
    raw = str(config.get("data_directory") or "").strip()
    if not raw:
        return [Check("fail", "Data directory", "No data directory set.")], []
    data_dir = Path(raw).expanduser()
    if not data_dir.is_dir():
        return [Check("fail", "Data directory", f"Does not exist: {data_dir}")], []
    files = find_uncal_files(str(data_dir))
    if not files:
        return [Check("fail", "Input data", f"No *_uncal.fits files under {data_dir} (search is recursive).")], []

    rows, problems = _read_uncal_headers(files)
    if problems:
        shown = "; ".join(problems[:5]) + (" …" if len(problems) > 5 else "")
        out.append(Check("fail", "Unreadable input files", f"{len(problems)} of {len(files)}: {shown}"))

    bad_inst = sorted({r["instrume"] for r in rows if r["instrume"] != "NIRCAM"})
    if bad_inst:
        out.append(Check("fail", "Instrument", f"Only NIRCam imaging is supported; found {', '.join(bad_inst)}."))
    bad_exp = sorted({r["exp_type"] for r in rows if r["exp_type"] != "NRC_IMAGE"})
    if bad_exp:
        out.append(Check("fail", "Exposure type", f"Only NRC_IMAGE is supported; found {', '.join(bad_exp)}."))
    unknown = sorted({r["filter"] for r in rows if r["filter"] not in FILTER_PIVOT_WAVELENGTHS_UM})
    if unknown:
        out.append(Check("fail", "Filters", f"Not in the pipeline's filter table (stage 3 orders filters by "
                                            f"wavelength): {', '.join(unknown)}."))

    filters = {}
    for r in rows:
        filters[r["filter"]] = filters.get(r["filter"], 0) + 1
    programs = sorted({r["program"] for r in rows})
    detectors = sorted({r["detector"] for r in rows})
    total_gb = sum(r["size"] for r in rows) / 1e9
    out.append(Check("ok", "Input data",
                     f"{len(rows)} uncal files ({total_gb:.1f} GB); program(s) {', '.join(programs)}; filters "
                     + ", ".join(f"{k} ({v})" for k, v in sorted(filters.items())) + f"; detectors {', '.join(detectors)}."))
    return out, rows


def _grouping(config: dict, rows: list[dict]) -> list[Check]:
    if not rows:
        return []
    data_dir = str(Path(str(config.get("data_directory"))).expanduser())
    obs_info, _ = get_observation_info(
        data_dir=data_dir,
        combine=bool(config.get("combine_observations", False)),
        group_by_directory=bool(config.get("group_by_directory", False)),
        name=str(config.get("custom_name") or "Combined_Observation"),
    )
    if not obs_info:
        return [Check("fail", "Observation grouping", "No observations would be processed.")]
    parts = []
    for name, payload in obs_info:
        n = len(payload.split(","))
        n_test = len(select_test_subset(payload.split(",")))
        parts.append(f"{name} ({n} files; {n_test} in a --test run)")
    out = [Check("ok", "Observation grouping", "; ".join(parts))]

    output_dir = Path(str(config.get("output_directory") or ".")).expanduser()
    skip = set(config.get("skip_steps") or [])
    existing = []
    for name, _ in obs_info:
        for stage in ("stage1", "stage2", "stage3"):
            if stage not in skip and (output_dir / name / f"{stage}_output").exists():
                existing.append(f"{name}/{stage}_output")
    if existing:
        out.append(Check("warn", "Existing outputs",
                         "These will be deleted and rebuilt: " + ", ".join(existing)))
    return out


def _fs_type(path: Path) -> str:
    """Filesystem type of the mount holding path (Linux only; '' elsewhere)."""
    try:
        best, fstype = "", ""
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and str(path).startswith(parts[1]) and len(parts[1]) > len(best):
                    best, fstype = parts[1], parts[2]
        return fstype
    except OSError:
        return ""


def _writable(path: Path) -> str | None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".preflight_", delete=True):
            pass
        return None
    except Exception as e:
        return str(e)


def _output(config: dict, rows: list[dict]) -> list[Check]:
    out = []
    output_dir = Path(str(config.get("output_directory") or ".")).expanduser().resolve()
    err = _writable(output_dir)
    if err:
        return [Check("fail", "Output directory", f"Cannot write to {output_dir}: {err}")]
    free_gb = shutil.disk_usage(output_dir).free / 1e9
    need_gb = sum(r["size"] for r in rows) * 6 / 1e9 if rows else 0.0
    if need_gb and free_gb < need_gb:
        out.append(Check("warn", "Output directory",
                         f"{output_dir}: {free_gb:.0f} GB free, roughly {need_gb:.0f} GB expected for all "
                         "intermediate products."))
    else:
        out.append(Check("ok", "Output directory", f"{output_dir}: writable, {free_gb:.0f} GB free"
                                                   + (f" (roughly {need_gb:.0f} GB expected)." if need_gb else ".")))
    fstype = _fs_type(output_dir)
    if fstype in NETWORK_FS:
        out.append(Check("warn", "Output on a network filesystem",
                         f"{output_dir} is on {fstype}; every stage does heavy file I/O and will be much slower "
                         "than on a local disk."))
    return out


def _wisp(config: dict, rows: list[dict]) -> list[Check]:
    if "wisp_subtraction" in set(config.get("skip_steps") or []):
        return [Check("ok", "Wisp templates", "Wisp subtraction is skipped.")]
    raw = str(config.get("wisp_directory") or "").strip()
    if not raw:
        return [Check("fail", "Wisp templates", "Wisp subtraction is enabled but no templates directory is set.")]
    wisp_dir = Path(raw).expanduser()
    if not wisp_dir.is_dir():
        return [Check("fail", "Wisp templates", f"Directory does not exist: {wisp_dir}")]
    needed = sorted({(r["detector"], r["filter"], r["pupil"]) for r in rows if r["detector"] in WISP_DETECTORS})
    missing = [f"WISP_{d}_{f}_{p}.fits" for d, f, p in needed if not (wisp_dir / f"WISP_{d}_{f}_{p}.fits").exists()]
    if missing:
        return [Check("fail", "Wisp templates", f"Missing in {wisp_dir}: {', '.join(missing)}. Wisp subtraction "
                                                "fails for exposures without a template.")]
    if not needed:
        return [Check("ok", "Wisp templates", "No wisp-affected detectors (NRCA3/A4/B3/B4) in the data.")]
    return [Check("ok", "Wisp templates", f"{len(needed)} template(s) found for the detectors and filters present.")]


def _crds(config: dict, rows: list[dict]) -> list[Check]:
    out = []
    raw = str(config.get("crds_path") or "").strip()
    if not raw:
        return [Check("fail", "CRDS cache", "No CRDS cache path set.")]
    crds_path = Path(raw).expanduser()
    err = _writable(crds_path)
    if err:
        return [Check("fail", "CRDS cache", f"Cannot write to {crds_path}: {err}")]
    fstype = _fs_type(crds_path.resolve())
    if fstype in NETWORK_FS:
        out.append(Check("warn", "CRDS cache location", f"{crds_path} is on {fstype}; keep the cache on a local "
                                                        "disk, stages 1-3 read reference files constantly."))

    server = str(config.get("crds_server_url") or "").strip()
    reachable = False
    if not server:
        out.append(Check("fail", "CRDS server", "No CRDS server URL set."))
    else:
        try:
            urllib.request.urlopen(server, timeout=10).read(1)
            reachable = True
            out.append(Check("ok", "CRDS server", f"{server} reachable."))
        except Exception as e:
            out.append(Check("warn", "CRDS server", f"{server} not reachable ({str(e)[:60]}). Only already-cached "
                                                    "reference files can be used."))

    crds_cmd = crds_command()
    if not rows or crds_cmd is None:
        return out

    # One representative file per instrument configuration is enough to learn
    # which reference files a run needs; bestrefs without syncing downloads nothing.
    reps = {}
    for r in rows:
        reps.setdefault((r["detector"], r["filter"], r["pupil"], r["readpatt"], r["subarray"]), r["path"])
    env = dict(os.environ, CRDS_PATH=str(crds_path), CRDS_SERVER_URL=server)
    try:
        proc = subprocess.run([crds_cmd, "bestrefs", "--files", *reps.values(), "--print-new-references"],
                              capture_output=True, text=True, timeout=300, env=env)
    except Exception as e:
        out.append(Check("warn", "CRDS reference files", f"Could not query best references: {str(e)[:80]}"))
        return out
    needed = set()
    for line in proc.stdout.splitlines() + proc.stderr.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[1] == "nircam" and parts[4].lower() not in ("n/a", "none"):
            needed.add(parts[4])
    if proc.returncode != 0 and not needed:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
        out.append(Check("warn" if reachable else "fail", "CRDS reference files",
                         f"crds bestrefs failed: {tail[0][:120]}"))
        return out

    ref_dir = crds_path / "references" / "jwst" / "nircam"
    missing, truncated = [], []
    for name in sorted(needed):
        p = ref_dir / name
        if not p.exists():
            missing.append(name)
            continue
        if p.suffix == ".fits":
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    with fits.open(p, lazy_load_hdus=False):
                        pass
                except Exception:
                    truncated.append(name)
                    continue
                if any("truncated" in str(x.message) for x in w):
                    truncated.append(name)
        elif p.stat().st_size == 0:
            truncated.append(name)
    if truncated:
        out.append(Check("fail", "Truncated CRDS reference files",
                         f"Incomplete downloads in {ref_dir}: {', '.join(truncated)}. Delete them; they will be "
                         "downloaded again on the next run."))
    if missing:
        darks = sum(1 for m in missing if "_dark_" in m)
        out.append(Check("warn", "CRDS reference files",
                         f"{len(needed)} needed, {len(missing)} not cached yet and will be downloaded at the start "
                         f"of the run ({darks} dark file(s) of ~2.5 GB each): {', '.join(missing)}"
                         + ("" if reachable else ". The CRDS server is not reachable, so the run will fail.")))
    elif not truncated:
        out.append(Check("ok", "CRDS reference files", f"All {len(needed)} needed reference files are cached and intact."))
    return out


def _stage3(config: dict, rows: list[dict]) -> list[Check]:
    if not rows or "stage3" in set(config.get("skip_steps") or []):
        return []
    regions = {}
    for r in rows:
        if r["s_region"]:
            regions.setdefault(r["filter"], []).append(r["s_region"])
    if not regions:
        return [Check("warn", "Stage 3 grid", "No S_REGION footprints in the uncal headers; cannot estimate.")]
    areas = footprint_areas(regions)
    ref, reason = choose_reference_filter(str(config.get("reference_filter", "auto") or "auto"), areas,
                                          FILTER_PIVOT_WAVELENGTHS_UM)
    out = [Check("ok", "Reference filter", f"{ref} ({reason}). Footprints: "
                 + ", ".join(f"{k} {v:.1f}'²" for k, v in sorted(areas.items(), key=lambda kv: -kv[1])) + ".")]

    # Grid size estimate on a tangent plane at the configured rotation.
    footprint_mode = str(config.get("mosaic_footprint", "all_filters") or "all_filters")
    polys = [parse_s_region(s) for f, ss in regions.items() if footprint_mode == "all_filters" or f == ref for s in ss]
    verts = np.vstack(polys)
    plane = _tangent_plane(verts, 1.0)
    x, y = plane.all_world2pix(verts[:, 0], verts[:, 1], 0)
    rot = np.deg2rad(float(config.get("rotation", 0.0) or 0.0))
    xr = x * np.cos(rot) - y * np.sin(rot)
    yr = x * np.sin(rot) + y * np.cos(rot)
    scale = float(config.get("pixel_scale", 0.02) or 0.02)
    nx = int(np.ceil((xr.max() - xr.min()) / scale))
    ny = int(np.ceil((yr.max() - yr.min()) / scale))
    per_mosaic_gb = nx * ny * 4 * 8 / 1e9  # ~8 float32 planes in flight during resample
    try:
        ram_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (ValueError, OSError, AttributeError):
        ram_gb = 0.0
    n_parallel = int(config.get("min_processes", 1) or 1) if config.get("stage3_use_multiprocessing") else 1
    detail = (f"about {nx} x {ny} pixels at {scale} arcsec/pixel "
              f"({'all filters' if footprint_mode == 'all_filters' else 'reference filter only'}); "
              f"roughly {per_mosaic_gb:.1f} GB of memory per filter being resampled, {n_parallel} in parallel")
    if ram_gb and per_mosaic_gb * n_parallel > 0.5 * ram_gb:
        out.append(Check("warn", "Stage 3 grid", detail + f", against {ram_gb:.0f} GB of RAM. Consider fewer parallel "
                                                          "filters, a coarser pixel scale, or turning off in-memory resampling."))
    else:
        out.append(Check("ok", "Stage 3 grid", detail + (f", {ram_gb:.0f} GB of RAM available." if ram_gb else ".")))
    return out


def _performance(config: dict) -> list[Check]:
    cores = os.cpu_count() or 1
    over = [k for k in ("stage1_nproc", "stage2_nproc", "wisp_nproc", "cfnoise_nproc", "bkg_nproc")
            if int(config.get(k, 1) or 1) > cores]
    if over:
        return [Check("warn", "Parallel workers", f"{', '.join(over)} exceed the {cores} CPU cores on this machine.")]
    return [Check("ok", "Parallel workers", f"Within the {cores} CPU cores on this machine.")]


def run_preflight(config: dict, repo_root: Path) -> list[Check]:
    checks: list[Check] = []
    checks += _versions()
    checks += _config_keys(config, repo_root)
    data_checks, rows = _data(config)
    checks += data_checks
    checks += _grouping(config, rows)
    checks += _output(config, rows)
    checks += _wisp(config, rows)
    checks += _crds(config, rows)
    try:
        checks += _stage3(config, rows)
    except Exception as e:  # an estimate must never block the report
        checks.append(Check("warn", "Stage 3 grid", f"Could not estimate: {str(e)[:80]}"))
    checks += _performance(config)
    return checks


def main() -> int:
    repo_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    config_path = repo_root / "config.yaml"
    if not config_path.exists():
        print(f"[FAIL] config.yaml not found in {repo_root}. Copy config.default.yaml to config.yaml and edit it.")
        return 1
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    checks = run_preflight(config, repo_root)
    tag = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}
    for c in checks:
        print(f"{tag[c.status]} {c.name}: {c.detail}")
    n_fail = sum(c.status == "fail" for c in checks)
    n_warn = sum(c.status == "warn" for c in checks)
    print(f"\n{len(checks) - n_fail - n_warn} passed, {n_warn} warning(s), {n_fail} problem(s).")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
