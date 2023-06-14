"""Utils to cache the results of a function call."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
from typing import Any


def get_cache_path(
    cache_root: str,
    params: dict[str, Any],
    extension: str | None = None,
) -> str:
    """Get a path to a cache.

    Args:
        task: The task to cache for.
        params: The parameters that the cache should represent.
        extension: The extension to use for the cache file, or None for no
            extension.

    Returns:
        The path to the cache file or directory.
    """
    if extension == "zbp":
        raise ValueError(
            'Cannot use extension "zbp", as it is reserved for the parameters.'
        )
    pathlib.Path(cache_root).mkdir(parents=True, exist_ok=True)

    # Find a name with no hash collisions
    dumped_params = json.dumps(params, sort_keys=True)
    m = hashlib.sha256()
    while True:
        m.update(dumped_params.encode("utf-8"))
        base_name = m.hexdigest()
        param_file = os.path.join(cache_root, f"{base_name}.zbp")
        if not os.path.exists(param_file):
            break
        with open(param_file, "r") as f:
            if f.read() == dumped_params:
                break
    with open(param_file, "w") as f:
        f.write(dumped_params)
    return os.path.join(
        cache_root, base_name if extension is None else f"{base_name}.{extension}"
    )


def fail_cache(cache_file: str, cache_message: str | None) -> None:
    """Mark a cache as failed."""
    with open(cache_file + ".zbfail", "w") as f:
        if cache_message is not None:
            print(cache_message, file=f)


class CacheLock:
    """A lock for a cache file."""

    def __init__(self, filename: str):
        """Initialize the lock."""
        self.lock_path = f"{filename}.zblock"
        self.skipped = False

    def __enter__(self) -> bool:
        """Enter the cache lock.

        If the lock file does not exist, create it and return True.
        Otherwise, return False.
        """
        # Skip if the lock file exists
        if os.path.exists(self.lock_path):
            logging.getLogger(__name__).info(f"Skipping {self.lock_path}")
            self.skipped = True
            return False

        # Create the lock file
        with open(self.lock_path, "w"):
            logging.getLogger(__name__).info(f"Created lock file {self.lock_path}")
            pass

        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove the lock file."""
        logging.getLogger(__name__).info(f"Exiting {self.lock_path}")
        if not self.skipped:
            os.remove(self.lock_path)
