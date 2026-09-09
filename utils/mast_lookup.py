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

import concurrent.futures
from collections import defaultdict


DEFAULT_CONE_RADIUS_ARCSEC = 60.0

# Hard ceiling for any single MAST/name-resolver network call. The MAST
# servers (and the Sesame name resolver in particular) occasionally stop
# responding without ever closing the connection, which would otherwise hang
# the UI indefinitely. See _run_with_timeout.
DEFAULT_QUERY_TIMEOUT_SEC = 30.0


class QueryTimeout(RuntimeError):
    """Raised when a MAST/name-resolver call exceeds its time budget."""


def _run_with_timeout(func, *args, timeout: float = DEFAULT_QUERY_TIMEOUT_SEC, **kwargs):
    """Run func(*args, **kwargs), raising QueryTimeout if it takes too long.

    Uses a worker thread so a hung network call can't block the caller. If the
    call times out we don't wait for the worker to finish (it may be stuck in a
    socket read) — we abandon it and let it die on its own.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        raise QueryTimeout(
            f"MAST query exceeded {timeout:.0f}s and was abandoned. "
            "The archive may be slow or unreachable — try again."
        )
    finally:
        # Don't block on a possibly-hung worker thread.
        executor.shutdown(wait=False)


def _import_astroquery():
    from astroquery.mast import Observations
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    # Make the underlying HTTP requests give up instead of hanging forever, so
    # the abandoned worker thread in _run_with_timeout can actually terminate.
    try:
        Observations.TIMEOUT = DEFAULT_QUERY_TIMEOUT_SEC
    except Exception:
        pass
    try:
        from astropy.utils.data import conf as _data_conf
        if _data_conf.remote_timeout < DEFAULT_QUERY_TIMEOUT_SEC:
            _data_conf.remote_timeout = DEFAULT_QUERY_TIMEOUT_SEC
    except Exception:
        pass

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


def _column_values(table, *names) -> list[str]:
    """Return one normalized string per row for the first matching column.

    Resolves the column name once and reads the whole column in one shot
    (``Column.tolist()``), which is far faster than iterating Table rows and
    re-resolving names per row. Masked entries come back as "" via _norm.
    """
    colnames = table.colnames
    for name in names:
        if name in colnames:
            return [_norm(v) for v in table[name].tolist()]
    return [""] * len(table)


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
        narrow_matches = _run_with_timeout(
            Observations.query_criteria,
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
            narrow_matches = _run_with_timeout(
                Observations.query_criteria,
                obs_collection="JWST",
                target_name=target_name,
            )
        except Exception:
            narrow_matches = None

    name_filtered = _filter_nircam_imaging(narrow_matches) if narrow_matches is not None else None

    # Cone search around name-resolved coordinates.
    cone_filtered = None
    try:
        coord = _run_with_timeout(SkyCoord.from_name, target_name)
        cone_matches = _run_with_timeout(
            Observations.query_region, coord, radius=float(radius_arcsec) * u.arcsec
        )
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
    matches = _run_with_timeout(
        Observations.query_region, coord, radius=float(radius_arcsec) * u.arcsec
    )
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
        observations = _run_with_timeout(
            Observations.query_criteria,
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
    has 'program', 'target' and 'filters' keys). Returns the matching subset of
    observations.
    """
    if not selected_summary_rows:
        return observations
    keys = {(row["program"], row["target"], row["filters"]) for row in selected_summary_rows}
    keep = [i for i, key in enumerate(_summary_keys(observations)) if key in keys]
    return observations[keep]


def filter_products_by_summary_rows(observations, products, selected_summary_rows):
    """Filter a products table to only those whose parent observation matches.

    Joins products back to observations via parent_obsid/obsID. Returns the
    subset of products belonging to (program, target, filters) groups in the
    selected summary rows. If no rows are selected, returns the full products
    table.
    """
    if not selected_summary_rows:
        return products

    # Build obs_id -> (program, target, filters) from the observations table.
    obs_ids = _column_values(observations, "obsid", "obs_id", "obsID", "observationid")
    obs_meta = {oid: key for oid, key in zip(obs_ids, _summary_keys(observations)) if oid}

    keys = {(row["program"], row["target"], row["filters"]) for row in selected_summary_rows}
    parents = _column_values(products, "parent_obsid", "obsID", "obsid", "obs_id")
    keep = [
        i for i, parent in enumerate(parents)
        if obs_meta.get(parent) in keys
    ]
    return products[keep]


def resolve_target(target_name: str) -> tuple[float, float] | None:
    """Resolve a target name to (RA, Dec) in degrees. Returns None on failure."""
    try:
        _Observations, SkyCoord, _u = _import_astroquery()
        coord = _run_with_timeout(SkyCoord.from_name, target_name)
        return float(coord.ra.deg), float(coord.dec.deg)
    except Exception:
        return None


def summarize(observations, uncal_products=None) -> list[dict]:
    """Group rows by program, target and filter for human-readable display.

    Returns a list of dicts:
        {program, target, filters, n_frames}
    sorted by program, then target, then filter. Each individual target in a
    multi-target proposal gets its own rows so they can be selected and
    downloaded independently.

    If uncal_products is provided (from mast_download.get_uncal_products),
    n_frames counts the actual UNCAL files that would be downloaded for
    each (program, target, filters) group. Without it, n_frames counts MAST
    observations, which usually understates the file count because each
    observation contains many exposures.
    """
    counts = defaultdict(int)  # keyed by (program, target, filter_value)
    obs_keys = _summary_keys(observations)

    if uncal_products is not None:
        # Build obs_id -> (program, target, filters) from the observations,
        # then count products by joining each back to its observation.
        obs_ids = _column_values(observations, "obsid", "obs_id", "obsID", "observationid")
        obs_meta = {oid: key for oid, key in zip(obs_ids, obs_keys) if oid}

        parents = _column_values(uncal_products, "parent_obsid", "obsID", "obsid", "obs_id")
        for parent in parents:
            key = obs_meta.get(parent)
            if key is not None:
                counts[key] += 1
    else:
        for key in obs_keys:
            counts[key] += 1

    rows = []
    for (program, target, filter_value) in sorted(counts):
        rows.append(
            {
                "program": program,
                "target": target,
                "filters": filter_value,
                "n_frames": counts[(program, target, filter_value)],
            }
        )
    return rows


def _summary_keys(table) -> list[tuple[str, str, str]]:
    """Return the (program, target, filters) grouping key for every row."""
    programs = _column_values(table, "proposal_id", "proposalid", "proposal")
    targets = _column_values(table, "target_name", "target")
    filters = _column_values(table, "filters", "filter")
    return [
        (p or "unknown", t or "unknown", f or "unknown filter")
        for p, t, f in zip(programs, targets, filters)
    ]
