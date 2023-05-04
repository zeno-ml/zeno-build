"""Utils to cache the results of a function call."""

from __future__ import annotations

import hashlib
import json
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
