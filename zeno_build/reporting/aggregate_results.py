"""A program to aggregate results from a cache directory into a single json."""

from __future__ import annotations

import argparse
import json
import os


def aggregate_results(
    cache_dir: str | None,
    cache_files: list[str] | None,
    output_file: str,
):
    """Aggregate results from a cache directory into a single json.

    Args:
        cache_dir: The directory containing the cached results.
        cache_files: The files containing the cached results. Must all be .json files
            with a corresponding .zbp in the same directory.
        output_file: The file to write the aggregated results to.

    Returns:
        None
    """
    # Get files
    if cache_dir is not None:
        cache_files = [
            os.path.join(cache_dir, x)
            for x in os.listdir(cache_dir)
            if x.endswith(".json")
        ]
    elif cache_files is not None:
        cache_files = [os.path.abspath(x) for x in cache_files]
    else:
        raise ValueError("Must provide either cache_dir or cache_files.")

    # Validation
    if len(cache_files) == 0:
        raise FileNotFoundError("No cache files found.")
    for cache_file in cache_files:
        if not cache_file.endswith(".json"):
            raise ValueError(f"Cache file {cache_file} must be a .json file.")
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file {cache_file} does not exist.")
        if not os.path.exists(cache_file[:-5] + ".zbp"):
            raise FileNotFoundError(
                f"Cache file {cache_file} does not have a corresponding .zbp file."
            )

    # Load the results
    all_results = []
    for cache_file in cache_files:
        with open(cache_file, "r") as f:
            predictions = json.load(f)
        with open(cache_file[:-5] + ".zbp", "r") as f:
            parameters = json.load(f)
        all_results.append(
            {
                "parameters": parameters,
                "predictions": predictions,
                "eval_result": 0.0,
            }
        )

    # Write the results
    with open(output_file, "w") as f:
        json.dump(all_results, f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    cache_group = p.add_mutually_exclusive_group(required=True)
    cache_group.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory containing the cached results.",
    )
    cache_group.add_argument(
        "--cache_files",
        type=str,
        nargs="+",
        default=None,
        help="The files containing the cached results. "
        "Must all be .json files with a corresponding .zbp in the same directory.",
    )
    p.add_argument(
        "--output_file",
        type=str,
        help="The file to write the aggregated results to.",
        required=True,
    )
    args = p.parse_args()
    aggregate_results(**vars(args))
