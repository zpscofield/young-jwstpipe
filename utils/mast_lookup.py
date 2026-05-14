"""MAST search helpers for JWST NIRCam imaging observations.

Three ways to find data:
- search_by_target("Abell 2744")        — name lookup plus cone search
- search_by_coordinates(ra_deg, dec_deg) — cone search at coordinates
- search_by_proposal(2756)               — all observations in a program

All return an astropy Table of observations, already filtered to NIRCam
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


def search_by_coordinates(ra_deg: float, dec_deg: float, radius_arcsec: float = DEFAULT_CONE_RADIUS_ARCSEC):
    """Cone search for NIRCam imaging observations at the given coordinates."""
    Observations, SkyCoord, u = _import_astroquery()
    coord = SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg)
    matches = Observations.query_region(coord, radius=float(radius_arcsec) * u.arcsec)
    return _dedupe(_filter_nircam_imaging(matches))


def search_by_proposal(proposal_id):
    """Return NIRCam imaging observations from one or more JWST proposals.

    proposal_id may be:
      - a single ID (int or str): "2756"
      - a list/tuple of IDs: ["2756", "1837"]
      - a comma- or space-separated string: "2756, 1837"
    """
    Observations, _SkyCoord, _u = _import_astroquery()
    ids = _parse_proposal_ids(proposal_id)
    if not ids:
        raise ValueError("No proposal IDs provided.")

    per_proposal = []
    for pid in ids:
        observations = Observations.query_criteria(
            obs_collection="JWST",
            proposal_id=pid,
        )
        per_proposal.append(_filter_nircam_imaging(observations))

    non_empty = [t for t in per_proposal if len(t) > 0]
    if not non_empty:
        return per_proposal[0]  # an empty table of the right type
    if len(non_empty) == 1:
        return _dedupe(non_empty[0])

    from astropy.table import vstack
    combined = vstack(non_empty, metadata_conflicts="silent")
    return _dedupe(combined)


def _parse_proposal_ids(value) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    # Split on commas or whitespace.
    parts = [p.strip() for chunk in text.split(",") for p in chunk.split()]
    return [p for p in parts if p]


def filter_by_summary_rows(observations, selected_summary_rows):
    """Filter an observations table down to entries matching the given summary rows.

    selected_summary_rows is a subset of what summarize() returned (each row
    has 'program' and 'filters' keys). Returns the matching subset of
    observations.
    """
    if not selected_summary_rows:
        return observations
    keys = {(row["program"], row["filters"]) for row in selected_summary_rows}
    keep = []
    for i, obs_row in enumerate(observations):
        program = _norm(_row_value(obs_row, "proposal_id", "proposalid", "proposal")) or "unknown"
        filters = _norm(_row_value(obs_row, "filters", "filter")) or "unknown filter"
        if (program, filters) in keys:
            keep.append(i)
    return observations[keep]


def filter_products_by_summary_rows(observations, products, selected_summary_rows):
    """Filter a products table to only those whose parent observation matches.

    Joins products back to observations via parent_obsid/obsID. Returns the
    subset of products belonging to (program, filters) pairs in the selected
    summary rows. If no rows are selected, returns the full products table.
    """
    if not selected_summary_rows:
        return products

    # Build obs_id -> (program, filters) from the observations table.
    obs_meta = {}
    for obs_row in observations:
        obs_id = _norm(_row_value(obs_row, "obsid", "obs_id", "obsID", "observationid"))
        program = _norm(_row_value(obs_row, "proposal_id", "proposalid", "proposal")) or "unknown"
        filters = _norm(_row_value(obs_row, "filters", "filter")) or "unknown filter"
        if obs_id:
            obs_meta[obs_id] = (program, filters)

    keys = {(row["program"], row["filters"]) for row in selected_summary_rows}
    keep = []
    for i, prod_row in enumerate(products):
        obs_id = _norm(_row_value(prod_row, "parent_obsid", "obsID", "obsid", "obs_id"))
        meta = obs_meta.get(obs_id)
        if meta is not None and meta in keys:
            keep.append(i)
    return products[keep]


def resolve_target(target_name: str) -> tuple[float, float] | None:
    """Resolve a target name to (RA, Dec) in degrees. Returns None on failure."""
    try:
        _Observations, SkyCoord, _u = _import_astroquery()
        coord = SkyCoord.from_name(target_name)
        return float(coord.ra.deg), float(coord.dec.deg)
    except Exception:
        return None


def summarize(observations, uncal_products=None) -> list[dict]:
    """Group rows by program and filter for human-readable display.

    Returns a list of dicts:
        {program, target, filters, n_frames}
    sorted by program then target.

    If uncal_products is provided (from mast_download.get_uncal_products),
    n_frames counts the actual UNCAL files that would be downloaded for
    each (program, filters) pair. Without it, n_frames counts MAST
    observations, which usually understates the file count because each
    observation contains many exposures.
    """
    counts = defaultdict(lambda: defaultdict(int))
    targets = defaultdict(set)

    if uncal_products is not None:
        # Build obs_id -> (program, filters, target) from the observations.
        obs_meta = {}
        for row in observations:
            obs_id = _norm(_row_value(row, "obsid", "obs_id", "obsID", "observationid"))
            program = _norm(_row_value(row, "proposal_id", "proposalid", "proposal")) or "unknown"
            target = _norm(_row_value(row, "target_name", "target")) or "unknown"
            filter_value = _norm(_row_value(row, "filters", "filter")) or "unknown filter"
            if obs_id:
                obs_meta[obs_id] = (program, filter_value, target)

        # Count uncal products, joining each back to its observation.
        for prod_row in uncal_products:
            obs_id = _norm(_row_value(prod_row, "parent_obsid", "obsID", "obsid", "obs_id"))
            meta = obs_meta.get(obs_id)
            if meta is None:
                continue
            program, filter_value, target = meta
            counts[program][filter_value] += 1
            targets[program].add(target)
    else:
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
