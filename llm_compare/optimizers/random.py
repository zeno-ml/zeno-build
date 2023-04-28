"""An optimizer using random search."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any, TypeVar

from pandas import DataFrame
from zeno import DistillReturn, MetricReturn, ZenoOptions

from llm_compare import search_space
from llm_compare.optimizers.base import Optimizer

T = TypeVar("T")


class RandomOptimizer(Optimizer):
    """An optimizer using random search."""

    def __init__(
        self,
        space: dict[str, search_space.SearchDimension],
        constants: dict[str, Any],
        distill_functions: list[Callable[[DataFrame, ZenoOptions], DistillReturn]],
        metric: Callable[[DataFrame, ZenoOptions], MetricReturn],
        seed: int | None = None,
    ):
        """Initialize a random optimizer.

        Args:
            space: The space to search over.
            constants: The constants to use.
            distill_functions: The distill functions to use.
            metric: The metric to use.
            seed: The random seed to use.
        """
        super().__init__(space, constants, distill_functions, metric)
        if seed is not None:
            random.seed(seed)

    def get_parameters(self) -> dict[str, Any]:
        """Randomize the parameters in a space.

        Args:
            space: The space to randomize.

        Returns:
            A dictionary of randomized parameters.
        """
        params = {}
        for name, dimension in self.space.items():
            if isinstance(dimension, search_space.Categorical) or isinstance(
                dimension, search_space.Discrete
            ):
                params[name] = random.choice(dimension.choices)
            elif isinstance(dimension, search_space.Float):
                params[name] = random.uniform(dimension.lower, dimension.upper)
            elif isinstance(dimension, search_space.Int):
                params[name] = random.randint(dimension.lower, dimension.upper)
            else:
                raise ValueError(f"Unknown search dimension: {dimension}")
        return params
