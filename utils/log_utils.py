"""Small shared logging helpers for the pipeline."""

import os
from datetime import datetime


def archive_existing_log(log_path: str) -> str | None:
    """If a log file exists at log_path, rename it with a timestamp suffix.

    Returns the archive path, or None if there was nothing to archive.
    Use this at the top of a stage's setup_logger so re-running the stage
    keeps prior runs' logs instead of appending to them.
    """
    if not os.path.exists(log_path):
        return None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_path = f"{log_path}.{timestamp}"

    # Extremely unlikely, but handle the case where two archives collide.
    counter = 1
    while os.path.exists(archive_path):
        archive_path = f"{log_path}.{timestamp}-{counter}"
        counter += 1

    os.rename(log_path, archive_path)
    return archive_path
