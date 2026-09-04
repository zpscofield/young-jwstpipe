"""Reference-filter selection and shared output grid for stage 3.

Two decisions used to be made implicitly by "process the longest-wavelength
filter first": which filter's source catalog every other filter is aligned
to, and which filter's mosaic footprint every other filter is cropped to.
This module makes both explicit.

- ``footprint_areas`` measures each filter's sky coverage from the S_REGION
  polygons of its calibrated exposures.
- ``choose_reference_filter`` picks the reference: an explicit filter from
  the config, or automatically the largest footprint with the longest
  wavelength breaking near-ties.
- ``write_union_output_wcs`` builds one output WCS covering every exposure
  of every filter and saves it as ASDF for the resample step's
  ``output_wcs`` parameter, so no filter is cropped to another's footprint.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.path import Path as MplPath

# Footprints this close to the largest count as equal; the longest wavelength
# then wins. Half a percent only absorbs rasterisation noise, so a filter
# that genuinely covers more sky, even by a few percent, is the reference.
AREA_TIE_TOLERANCE = 0.005


def parse_s_region(s_region: str) -> np.ndarray:
    """Return an (N, 2) array of RA/Dec vertices from an S_REGION string."""
    tokens = s_region.split()
    values = [float(t) for t in tokens if _is_float(t)]
    return np.asarray(values, dtype=float).reshape(-1, 2)


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def read_sregions(filter_to_paths: dict[str, list[str]]) -> dict[str, list[str]]:
    """Read the S_REGION keyword of every exposure, grouped by filter."""
    out: dict[str, list[str]] = {}
    for filt, paths in filter_to_paths.items():
        regions = []
        for path in paths:
            header = fits.getheader(path, "SCI")
            region = header.get("S_REGION")
            if region:
                regions.append(str(region))
        out[filt] = regions
    return out


def _tangent_plane(all_vertices: np.ndarray, resolution_arcsec: float) -> WCS:
    ra = np.deg2rad(all_vertices[:, 0])
    dec = np.deg2rad(all_vertices[:, 1])
    x = np.mean(np.cos(dec) * np.cos(ra))
    y = np.mean(np.cos(dec) * np.sin(ra))
    z = np.mean(np.sin(dec))
    ra0 = np.rad2deg(np.arctan2(y, x)) % 360.0
    dec0 = np.rad2deg(np.arctan2(z, np.hypot(x, y)))

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = [0.0, 0.0]
    step = resolution_arcsec / 3600.0
    w.wcs.cdelt = [-step, step]
    return w


def footprint_areas(
    sregions_by_filter: dict[str, Iterable[str]], resolution_arcsec: float = 1.0
) -> dict[str, float]:
    """Sky area covered by each filter, in square arcminutes.

    Each filter's exposure polygons are rasterised on a common tangent plane
    and unioned, so overlapping exposures are not double counted.
    """
    polygons = {
        filt: [parse_s_region(s) for s in regions]
        for filt, regions in sregions_by_filter.items()
    }
    all_vertices = [p for polys in polygons.values() for p in polys]
    if not all_vertices:
        return {filt: 0.0 for filt in polygons}

    plane = _tangent_plane(np.vstack(all_vertices), resolution_arcsec)
    projected = {
        filt: [np.column_stack(plane.all_world2pix(p[:, 0], p[:, 1], 0)) for p in polys]
        for filt, polys in polygons.items()
    }
    stacked = np.vstack([p for polys in projected.values() for p in polys])
    x0, y0 = np.floor(stacked.min(axis=0)).astype(int) - 1
    x1, y1 = np.ceil(stacked.max(axis=0)).astype(int) + 1
    ny, nx = y1 - y0 + 1, x1 - x0 + 1

    areas: dict[str, float] = {}
    for filt, polys in projected.items():
        mask = np.zeros((ny, nx), dtype=bool)
        for poly in polys:
            px0, py0 = np.floor(poly.min(axis=0)).astype(int)
            px1, py1 = np.ceil(poly.max(axis=0)).astype(int)
            xs = np.arange(px0, px1 + 1)
            ys = np.arange(py0, py1 + 1)
            gx, gy = np.meshgrid(xs, ys)
            inside = MplPath(poly).contains_points(np.column_stack([gx.ravel(), gy.ravel()]))
            sub = inside.reshape(gy.shape)
            mask[py0 - y0 : py1 - y0 + 1, px0 - x0 : px1 - x0 + 1] |= sub
        areas[filt] = float(mask.sum()) * resolution_arcsec**2 / 3600.0
    return areas


def choose_reference_filter(
    requested: str | None,
    areas: dict[str, float],
    wavelengths: dict[str, float],
) -> tuple[str, str]:
    """Return (filter, reason) for the astrometric/grid reference filter.

    ``requested`` is a filter name or "auto". Automatic choice: the largest
    footprint, with the longest wavelength breaking (near-exact) ties. An
    explicit filter that is not present in the data falls back to automatic.
    """
    filters = list(areas)
    if not filters:
        raise ValueError("No filters to choose a reference from.")

    requested = (requested or "auto").strip()
    if requested.lower() != "auto":
        if requested.upper() in filters:
            return requested.upper(), "chosen in the configuration"
        fallback_note = f"requested filter {requested} not found in the data; "
    else:
        fallback_note = ""

    largest = max(areas.values())
    candidates = [f for f in filters if areas[f] >= (1.0 - AREA_TIE_TOLERANCE) * largest]
    chosen = max(candidates, key=lambda f: wavelengths.get(f, 0.0))
    if len(candidates) == 1:
        reason = "largest footprint"
    else:
        reason = f"longest wavelength among the filters with equal footprints ({', '.join(sorted(candidates))})"
    return chosen, fallback_note + reason


def write_union_output_wcs(
    sregions: Iterable[str],
    reference_exposure: str,
    pixel_scale_arcsec: float,
    rotation_deg: float,
    out_path: str,
) -> tuple[str, tuple[int, int]]:
    """Build one output WCS covering every S_REGION and save it as ASDF.

    Uses the same stcal routine the jwst resample step uses internally, so
    the grid is identical to what resample would build for that set of
    exposures. Returns (path, (ny, nx)).
    """
    import asdf
    from stcal.alignment.util import wcs_from_sregions
    from stdatamodels.jwst import datamodels

    with datamodels.open(reference_exposure) as model:
        ref_wcs = model.meta.wcs
        ref_wcsinfo = model.meta.wcsinfo.instance

    wcs = wcs_from_sregions(
        list(sregions),
        ref_wcs,
        ref_wcsinfo,
        pscale=pixel_scale_arcsec / 3600.0,
        rotation=rotation_deg,
    )
    shape = tuple(int(v) for v in wcs.array_shape)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    asdf.AsdfFile({"wcs": wcs, "pixel_scale": float(pixel_scale_arcsec)}).write_to(out_path)
    return out_path, shape
