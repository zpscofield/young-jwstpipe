"""MAST search helpers for JWST NIRCam imaging observations.

Two ways to find data:
- search_by_target("Abell 2744")  — name lookup plus cone search
- search_by_proposal(2756)        — all observations in a program

Both return an astropy Table of observations, already filtered to NIRCam
imaging and deduplicated. Pass that table to summarize() for a
human-readable list, or to mast_download.download_uncal() to fetch the
uncalibrated files.
"""

from __future__ import annotations

from collections import defaultdict


DEFAULT_CONE_RADIUS_ARCSEC = 60.0


def _import_astroquery():
    from astroquery.mast import Observations
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    return Observations, SkyCoord, u


def _row_value(row, *names):
    colnames = row.colnames
    for name in names:
        if name in colnames:
            return row[name]
    return None


def _norm(value) -> str:
    if value is None:
        return ""
    try:
        import numpy as np
        if np.ma.is_masked(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _is_nircam_imaging(row) -> bool:
    if _norm(_row_value(row, "obs_collection")).upper() != "JWST":
        return False
    dataproduct = _norm(_row_value(row, "dataproduct_type")).upper()
    if dataproduct and dataproduct != "IMAGE":
        return False
    if "NIRCAM" not in _norm(_row_value(row, "instrument_name")).upper():
        return False
    intent = _norm(_row_value(row, "intentType", "intent_type")).upper()
    if intent and intent != "SCIENCE":
        return False
    return True


def _filter_nircam_imaging(observations):
    if len(observations) == 0:
        return observations
    keep = [i for i, row in enumerate(observations) if _is_nircam_imaging(row)]
    return observations[keep]


def _dedupe(observations):
    if len(observations) == 0:
        return observations
    seen = set()
    keep = []
    for i, row in enumerate(observations):
        key = _row_value(row, "obsid", "obs_id", "obsID", "observationid")
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return observations[keep]


def search_by_target(target_name: str, radius_arcsec: float = DEFAULT_CONE_RADIUS_ARCSEC):
    """Return NIRCam imaging observations matching a target name.

    Tries an exact target_name query first, then a cone search around the
    name-resolved coordinates, and combines/deduplicates the results.
    """
    Observations, SkyCoord, u = _import_astroquery()

    # Narrow query: exact target_name match.
    try:
        narrow_matches = Observations.query_criteria(
            obs_collection="JWST",
            target_name=target_name,
            dataproduct_type="image",
            instrument_name="NIRCam/IMAGE",
        )
    except Exception:
        narrow_matches = None

    # Broad fallback if the narrow query was empty/failed.
    if narrow_matches is None or len(narrow_matches) == 0:
        try:
            narrow_matches = Observations.query_criteria(
                obs_collection="JWST",
                target_name=target_name,
            )
        except Exception:
            narrow_matches = None

    name_filtered = _filter_nircam_imaging(narrow_matches) if narrow_matches is not None else None

    # Cone search around name-resolved coordinates.
    cone_filtered = None
    try:
        coord = SkyCoord.from_name(target_name)
        cone_matches = Observations.query_region(coord, radius=float(radius_arcsec) * u.arcsec)
        cone_filtered = _filter_nircam_imaging(cone_matches)
    except Exception:
        cone_filtered = None

    if name_filtered is None and cone_filtered is None:
        raise RuntimeError(
            f"Could not resolve '{target_name}' to coordinates and no target-name match was found."
        )

    if name_filtered is None or len(name_filtered) == 0:
        return _dedupe(cone_filtered)
    if cone_filtered is None or len(cone_filtered) == 0:
        return _dedupe(name_filtered)

    from astropy.table import vstack
    combined = vstack([name_filtered, cone_filtered], metadata_conflicts="silent")
    return _dedupe(combined)


def search_by_proposal(proposal_id):
    """Return NIRCam imaging observations from a single JWST proposal."""
    Observations, _SkyCoord, _u = _import_astroquery()
    observations = Observations.query_criteria(
        obs_collection="JWST",
        proposal_id=str(proposal_id),
    )
    return _dedupe(_filter_nircam_imaging(observations))


def summarize(observations) -> list[dict]:
    """Group rows by program and filter for human-readable display.

    Returns a list of dicts:
        {program, target, filters, n_frames}
    sorted by program then target.
    """
    counts = defaultdict(lambda: defaultdict(int))
    targets = defaultdict(set)

    for row in observations:
        program = _norm(_row_value(row, "proposal_id", "proposalid", "proposal")) or "unknown"
        target = _norm(_row_value(row, "target_name", "target")) or "unknown"
        filter_value = _norm(_row_value(row, "filters", "filter")) or "unknown filter"
        counts[program][filter_value] += 1
        targets[program].add(target)

    rows = []
    for program in sorted(counts):
        for filter_value in sorted(counts[program]):
            rows.append(
                {
                    "program": program,
                    "target": ", ".join(sorted(targets[program])),
                    "filters": filter_value,
                    "n_frames": counts[program][filter_value],
                }
            )
    return rows
