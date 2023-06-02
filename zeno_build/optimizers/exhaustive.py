"""An optimizer using random search."""
from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from pandas import DataFrame
from zeno import DistillReturn, MetricReturn, ZenoOptions

from zeno_build.experiments import search_space
from zeno_build.optimizers.base import Optimizer

T = TypeVar("T")


class ExhaustiveOptimizer(Optimizer):
    """An optimizer that enumerates all possibilities."""

    def __init__(
        self,
        space: search_space.SearchSpace,
        distill_functions: list[Callable[[DataFrame, ZenoOptions], DistillReturn]],
        metric: Callable[[DataFrame, ZenoOptions], MetricReturn],
        seed: int | None = None,
        num_trials: int | None = None,
    ):
        """Initialize an exhaustive optimizer.

        Args:
            space: The space to search over.
            distill_functions: The distill functions to use.
            metric: The metric to use.
            seed: The random seed to use.
            num_trials: The maximum number of trials to run.

        Raises:
            ValueError: If the space contains floats.
        """
        super().__init__(space, distill_functions, metric, num_trials)
        if isinstance(self.space, search_space.CombinatorialSearchSpace):
            self._values_iterator = itertools.product(
                *[
                    ExhaustiveOptimizer._dimension_iter(x)
                    for x in self.space.dimensions.values()
                ]
            )
            self._keys = list(self.space.dimensions.keys())
        else:
            raise NotImplementedError(f"Unsupported search space {type(self.space)}.")

    @staticmethod
    def _dimension_iter(dimension: search_space.SearchDimension) -> Iterator[Any]:
        """Iterate over a search dimension.

        Args:
            dimension: The search dimension to iterate over.

        Returns:
            An iterator over the search dimension.
        """
        if isinstance(dimension, search_space.Categorical) or isinstance(
            dimension, search_space.Discrete
        ):
            return iter(dimension.choices)
        elif isinstance(dimension, search_space.Int):
            return iter(range(dimension.lower, dimension.upper + 1))
        elif isinstance(dimension, search_space.Constant):
            return iter([dimension.value])
        else:
            raise ValueError(
                f"Unknown supported dimension type {type(dimension)} for {dimension}"
            )

    def get_parameters(self) -> dict[str, Any] | None:
        """Randomize the parameters in a space.

        Args:
            space: The space to randomize.

        Returns:
            A dictionary of randomized parameters.
        """
        # If there is no next value, we have exhausted the search space.
        try:
            values = next(self._values_iterator)
            return dict(zip(self._keys, values))
        except StopIteration:
            return None
