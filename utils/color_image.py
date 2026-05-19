"""Build a color image from per-filter stage-3 i2d.fits mosaics.

For each observation, this module:
  - Reads each filter's SCI extension (the i2d mosaic).
  - Applies an asinh stretch with tunable parameters.
  - Saves each stretched mosaic as a grayscale 8-bit TIFF (one per filter)
    so the user can take individual filter images into Photoshop / GIMP.
  - Combines the stretched mosaics into a single RGB color image using
    user-specified hues per filter (when there are 3 or more filters) or
    a luminance-preserving blue/red/green-mean recipe (when there are 2).
  - Saves the combined image as a TIFF plus a downscaled PNG preview for
    the Streamlit UI.

i2d files are expected at:
  <obs_dir>/stage3_output/<filter>/output_files/<target>_nircam_clear-<filter>_i2d.fits

All stage-3 i2d arrays share the same WCS grid (stage 3 aligns them to
the longest-wavelength filter), so they can be combined element-wise
without any reprojection step.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import logging
import os
import sys
from glob import glob
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from PIL import Image

from log_utils import archive_existing_log
from nircam_filters import FILTER_PIVOT_WAVELENGTHS_UM


# ---------------------------------------------------------------------------
# Sky normalization
# ---------------------------------------------------------------------------

def estimate_sky_level(
    data: np.ndarray, sigma: float = 3.0, maxiters: int = 5
) -> float:
    """Return a robust per-filter sky level via sigma-clipped median.

    Bright sources (and any leftover hot pixels) clip out after a few
    iterations, so the median converges on the residual sky pedestal
    left over after stage-2/3 background subtraction. This is what we
    subtract to make sure every filter's true sky maps to the same
    'black' before the asinh stretch.
    """
    _mean, median, _std = sigma_clipped_stats(data, sigma=sigma, maxiters=maxiters)
    return float(median)


def subtract_sky(
    data: np.ndarray, sigma: float = 3.0, maxiters: int = 5
) -> tuple[np.ndarray, float]:
    """Return (sky-subtracted data, sky level)."""
    sky = estimate_sky_level(data, sigma=sigma, maxiters=maxiters)
    return data - sky, sky


# ---------------------------------------------------------------------------
# Stretch + color helpers
# ---------------------------------------------------------------------------

def asinh_stretch(
    data: np.ndarray,
    min_level: float = 0.001,
    max_quantile: float = 0.99999,
    gamma: float = 2.2,
) -> np.ndarray:
    """Apply an asinh stretch to a 2-D image, returning values in [0, 1].

    Mirrors the recipe the user supplied: clip negatives, set vmin/vmax,
    do an asinh remap with softening parameter (vmax - vmin) / 10, clip
    to [0, 1], then apply a gamma correction.
    """
    data = np.maximum(data, 0.0)

    vmin = min_level
    vmax = float(np.nanquantile(data, max_quantile))
    # Guard against degenerate images where the high quantile collapses to vmin.
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-9

    a = (vmax - vmin) / 10.0
    stretched = np.arcsinh((data - vmin) / a) / np.arcsinh((vmax - vmin) / a)
    stretched = np.clip(stretched, 0.0, 1.0)
    # Replace any NaNs from the input with zero in the stretched result.
    stretched = np.nan_to_num(stretched, nan=0.0, posinf=1.0, neginf=0.0)

    return stretched ** (1.0 / gamma)


def hue_to_rgb(hue_deg: float) -> tuple[float, float, float]:
    """Convert a hue in degrees to an (R, G, B) triple with S=1, V=1."""
    hue = float(hue_deg) % 360.0
    return colorsys.hsv_to_rgb(hue / 360.0, 1.0, 1.0)


def default_hues_for_filters(filters_sorted: list[str]) -> dict[str, float]:
    """Pre-populate per-filter hue defaults from the wavelength→hue ramp.

    The shortest filter in the input list lands at 240° (pure blue) and the
    longest at 0° (pure red), with intermediate filters linearly interpolated
    by their pivot wavelength. Anchoring the endpoints to the actual filter
    set means F090W is exactly blue when it's the bluest filter present,
    regardless of whether F070W or F480M happens to exist in the wavelength
    table. The user can override any value from the Streamlit UI.
    """
    known = [
        (name, FILTER_PIVOT_WAVELENGTHS_UM[name])
        for name in filters_sorted
        if name in FILTER_PIVOT_WAVELENGTHS_UM
    ]
    if not known:
        return {name: 120.0 for name in filters_sorted}

    wl_min = min(wl for _, wl in known)
    wl_max = max(wl for _, wl in known)
    span = wl_max - wl_min

    hues: dict[str, float] = {}
    for name in filters_sorted:
        wl = FILTER_PIVOT_WAVELENGTHS_UM.get(name)
        if wl is None:
            hues[name] = 120.0  # green fallback for filters not in the table
            continue
        if span <= 0:
            hues[name] = 240.0  # single known wavelength → pure blue
            continue
        hue = 240.0 * (1.0 - (wl - wl_min) / span)
        hues[name] = float(np.clip(hue, 0.0, 240.0))
    return hues


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_i2d_files(obs_dir: str | os.PathLike, target: str) -> dict[str, Path]:
    """Return {filter: i2d_path} for every filter with a stage-3 mosaic."""
    obs_dir = Path(obs_dir)
    stage3 = obs_dir / "stage3_output"
    found: dict[str, Path] = {}
    if not stage3.is_dir():
        return found

    pattern = f"{target}_nircam_clear-*_i2d.fits"
    for path in stage3.glob(f"*/output_files/{pattern}"):
        # Filter name is the directory two levels up from the file.
        filter_name = path.parent.parent.name
        found[filter_name] = path

    return found


def sort_filters_by_wavelength(filters: Iterable[str]) -> list[str]:
    """Sort filters by pivot wavelength, shortest first."""
    return sorted(
        filters,
        key=lambda name: FILTER_PIVOT_WAVELENGTHS_UM.get(name, float("inf")),
    )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)


def _save_grayscale_tiff(arr_01: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(arr_01), mode="L").save(str(path), format="TIFF")


def _save_rgb_tiff(rgb_01: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(rgb_01), mode="RGB").save(str(path), format="TIFF")


def _save_preview_png(rgb_01: np.ndarray, path: Path, max_size: int = 1200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(_to_uint8(rgb_01), mode="RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    img.save(str(path), format="PNG", optimize=True)


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------

def _normalize_channel_to_unit(arr: np.ndarray) -> np.ndarray:
    """Scale a 2-D array so its 99.99th percentile lands at 1.0, clipped."""
    if not np.any(np.isfinite(arr)):
        return np.zeros_like(arr)
    top = float(np.nanquantile(arr, 0.9999))
    if top <= 0:
        return np.clip(arr, 0.0, 1.0)
    return np.clip(arr / top, 0.0, 1.0)


def _combine_two_filters(
    stretched: dict[str, np.ndarray], filters_sorted: list[str]
) -> np.ndarray:
    """Blue/red/green-mean recipe for N=2 filters."""
    blue = stretched[filters_sorted[0]]
    red = stretched[filters_sorted[1]]
    green = 0.5 * (blue + red)
    return np.stack(
        [
            _normalize_channel_to_unit(red),
            _normalize_channel_to_unit(green),
            _normalize_channel_to_unit(blue),
        ],
        axis=-1,
    )


def _combine_with_hues(
    stretched: dict[str, np.ndarray],
    filters_sorted: list[str],
    filter_hues: dict[str, float],
) -> np.ndarray:
    """Weighted RGB sum using user-supplied hues for N>=3."""
    shape = next(iter(stretched.values())).shape
    r = np.zeros(shape, dtype=np.float32)
    g = np.zeros(shape, dtype=np.float32)
    b = np.zeros(shape, dtype=np.float32)
    for name in filters_sorted:
        weight_r, weight_g, weight_b = hue_to_rgb(filter_hues[name])
        s = stretched[name]
        r += weight_r * s
        g += weight_g * s
        b += weight_b * s
    return np.stack(
        [
            _normalize_channel_to_unit(r),
            _normalize_channel_to_unit(g),
            _normalize_channel_to_unit(b),
        ],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def make_color_image(
    obs_dir: str | os.PathLike,
    target: str,
    min_level: float = 0.001,
    max_quantile: float = 0.99999,
    gamma: float = 2.2,
    filter_hues: dict[str, float] | None = None,
    subtract_sky_per_filter: bool = True,
    log: logging.Logger | None = None,
) -> dict:
    """Top-level entry. See module docstring for output layout."""
    log = log or logging.getLogger(__name__)
    obs_dir = Path(obs_dir)
    color_dir = obs_dir / "color"
    color_dir.mkdir(parents=True, exist_ok=True)

    filter_paths = find_i2d_files(obs_dir, target)
    if not filter_paths:
        log.warning(f"No stage-3 i2d files found under {obs_dir}; nothing to do.")
        return {"tiff": None, "preview": None, "per_filter_tiffs": {}, "n_filters": 0}

    filters_sorted = sort_filters_by_wavelength(filter_paths.keys())
    log.info(f"Found {len(filters_sorted)} filters: {filters_sorted}")

    # Stretch each filter and save its grayscale TIFF.
    stretched: dict[str, np.ndarray] = {}
    per_filter_tiffs: dict[str, Path] = {}
    sky_levels: dict[str, float] = {}
    for name in filters_sorted:
        i2d_path = filter_paths[name]
        log.info(f"Stretching {name} from {i2d_path}")
        with fits.open(i2d_path) as hdul:
            data = np.flipud(hdul[1].data.astype(np.float64))

        if subtract_sky_per_filter:
            data, sky_level = subtract_sky(data)
            sky_levels[name] = sky_level
            log.info(f"{name}: subtracted sky level {sky_level:.6f}")

        s = asinh_stretch(data, min_level=min_level, max_quantile=max_quantile, gamma=gamma)
        stretched[name] = s.astype(np.float32)

        tiff_path = color_dir / f"{target}_{name}_stretched.tiff"
        _save_grayscale_tiff(s, tiff_path)
        per_filter_tiffs[name] = tiff_path
        log.info(f"Wrote {tiff_path}")

    n = len(filters_sorted)
    result = {
        "tiff": None,
        "preview": None,
        "per_filter_tiffs": per_filter_tiffs,
        "sky_levels": sky_levels,
        "n_filters": n,
    }

    if n < 2:
        log.info("Only one filter present; skipping combined color image.")
        return result

    if n == 2:
        log.info("Combining two filters as blue / green-mean / red.")
        rgb = _combine_two_filters(stretched, filters_sorted)
    else:
        # Fill in any missing hues with defaults so the script remains usable
        # from the CLI even if filter_hues is partial.
        defaults = default_hues_for_filters(filters_sorted)
        hues: dict[str, float] = dict(defaults)
        if filter_hues:
            for name in filters_sorted:
                if name in filter_hues:
                    hues[name] = float(filter_hues[name])
        log.info(f"Combining {n} filters with hues: {hues}")
        rgb = _combine_with_hues(stretched, filters_sorted, hues)

    tiff_path = color_dir / f"{target}_color.tiff"
    preview_path = color_dir / f"{target}_color_preview.png"
    _save_rgb_tiff(rgb, tiff_path)
    _save_preview_png(rgb, preview_path)
    log.info(f"Wrote {tiff_path}")
    log.info(f"Wrote {preview_path}")

    result["tiff"] = tiff_path
    result["preview"] = preview_path
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _setup_logger(obs_dir: Path) -> logging.Logger:
    log_dir = obs_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline_color_image.log"
    archive_existing_log(str(log_path))

    log = logging.getLogger("color_image")
    log.setLevel(logging.DEBUG)
    # Avoid duplicate handlers when re-invoked in the same process.
    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_path)
        for h in log.handlers
    ):
        handler = logging.FileHandler(log_path, mode="a")
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        log.addHandler(handler)

    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.INFO)
    stream.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(stream)
    return log


def _parse_filter_hues(raw: str | None) -> dict[str, float]:
    if not raw or not raw.strip() or raw.strip() in {"{}", "null", "None"}:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fall back to YAML in case the shell passed block YAML through.
        import yaml
        parsed = yaml.safe_load(raw) or {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): float(v) for k, v in parsed.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a color image from per-filter stage-3 i2d mosaics."
    )
    parser.add_argument("--obs-dir", required=True, help="Per-observation output directory (contains stage3_output/).")
    parser.add_argument("--target", required=True, help="Target name (the program ID or custom name used by the pipeline).")
    parser.add_argument("--min-level", type=float, default=0.001)
    parser.add_argument("--max-quantile", type=float, default=0.99999)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument(
        "--filter-hues",
        type=str,
        default="{}",
        help='JSON dict of {filter_name: hue_degrees}, e.g. \'{"F090W":240,"F277W":120}\'. Missing entries default to a wavelength ramp.',
    )
    parser.add_argument(
        "--subtract-sky",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Estimate and subtract each filter's residual sky pedestal before stretching (default: on).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    obs_dir = Path(args.obs_dir)
    log = _setup_logger(obs_dir)
    log.info("------------------")
    log.info("Color image step")
    log.info("------------------")
    filter_hues = _parse_filter_hues(args.filter_hues)
    make_color_image(
        obs_dir=obs_dir,
        target=args.target,
        min_level=args.min_level,
        max_quantile=args.max_quantile,
        gamma=args.gamma,
        filter_hues=filter_hues,
        subtract_sky_per_filter=args.subtract_sky,
        log=log,
    )
    log.info("color_image.py complete.")


if __name__ == "__main__":
    main()
