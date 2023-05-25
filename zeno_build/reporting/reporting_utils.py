"""Utility functions to help with reporting."""

from typing import Any

from zeno_build.experiments import search_space
from zeno_build.experiments.search_space import CombinatorialSearchSpace


def parameters_to_name(
    parameters: dict[str, Any], space: CombinatorialSearchSpace
) -> str:
    """Convert parameters into a readable model name.

    Args:
        parameters: The parameters to convert.
        space: The search space to use.

    Returns:
        A string representation of the model.
    """
    return " ".join(
        [
            parameters[k] if isinstance(parameters[k], str) else f"{k}={parameters[k]}"
            for k, v in space.dimensions.items()
            if not isinstance(v, search_space.Constant)
        ]
    )
