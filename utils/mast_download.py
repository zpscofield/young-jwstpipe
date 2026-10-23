"""Download JWST NIRCam _uncal.fits files from MAST.

Given an observations table (typically from utils.mast_lookup), fetches
the uncalibrated science products and writes them to a single directory.
That directory is the value the pipeline will use as data_directory.

Direct HTTPS downloads via the MAST API are used (rather than
astroquery's download_products) so the UI can show per-file progress and
recover from transient connection failures with a simple retry.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import requests


MAST_DOWNLOAD_BASE = "https://mast.stsci.edu/api/v0.1/Download/file"
CONNECT_TIMEOUT_SECONDS = 30
READ_TIMEOUT_SECONDS = 600
CHUNK_BYTES = 1024 * 1024


def _import_observations():
    from astroquery.mast import Observations
    return Observations


def _auth_headers() -> dict:
    token = os.environ.get("MAST_API_TOKEN") or os.environ.get("MAST_TOKEN")
    return {"Authorization": f"token {token}"} if token else {}


def get_uncal_products(observations):
    """Return the _uncal.fits SCIENCE products for the given observations.

    Use this once per search; the products table can be reused for both
    counting (via mast_lookup.summarize) and downloading (via
    download_uncal_products) without re-hitting MAST.
    """
    Observations = _import_observations()
    products = Observations.get_product_list(observations)
    filtered = Observations.filter_products(
        products,
        productType="SCIENCE",
        productSubGroupDescription="UNCAL",
        extension="fits",
    )
    keep = [
        i for i, row in enumerate(filtered)
        if str(row["productFilename"] or "").endswith("_uncal.fits")
    ]
    return filtered[keep]


# Backwards-compatible alias.
_uncal_products = get_uncal_products


def _download_one(data_uri: str, dest: Path, retries: int = 2) -> bool:
    url = f"{MAST_DOWNLOAD_BASE}?uri={data_uri}"
    headers = _auth_headers()
    temp_path = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(retries):
        if temp_path.exists():
            temp_path.unlink()
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            ) as response:
                if response.status_code >= 400:
                    if attempt + 1 < retries:
                        continue
                    return False
                with open(temp_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                        if chunk:
                            handle.write(chunk)
        except (requests.RequestException, OSError):
            if attempt + 1 < retries:
                continue
            return False

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            if attempt + 1 < retries:
                continue
            return False

        temp_path.replace(dest)
        return True

    return False


def download_uncal_products(
    products,
    dest_dir,
    progress: Callable[[int, int, str, str], None] | None = None,
) -> dict:
    """Download every file in the products table to dest_dir.

    progress(i, total, filename, status) is called once per file with
    status in {"downloading", "cached", "failed"}.

    Returns: {"path": Path(dest_dir), "downloaded": [Path, ...], "failed": [filename, ...]}.
    Already-present files are reused (the function is safe to re-run).
    """
    dest = Path(dest_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    total = len(products)
    downloaded: list[Path] = []
    failed: list[str] = []

    for i, row in enumerate(products, start=1):
        filename = str(row["productFilename"])
        data_uri = row["dataURI"]
        local_path = dest / filename

        if local_path.exists() and local_path.stat().st_size > 0:
            downloaded.append(local_path)
            if progress is not None:
                progress(i, total, filename, "cached")
            continue

        if progress is not None:
            progress(i, total, filename, "downloading")

        if _download_one(data_uri, local_path):
            downloaded.append(local_path)
        else:
            failed.append(filename)
            if progress is not None:
                progress(i, total, filename, "failed")

    return {"path": dest, "downloaded": downloaded, "failed": failed}


def download_uncal(
    observations,
    dest_dir,
    progress: Callable[[int, int, str, str], None] | None = None,
) -> dict:
    """Convenience wrapper: query products from observations, then download.

    For new code, prefer get_uncal_products() + download_uncal_products()
    so the products table can be shared with the summary step.
    """
    products = get_uncal_products(observations)
    return download_uncal_products(products, dest_dir, progress=progress)
