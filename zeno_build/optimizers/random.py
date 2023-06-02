"""An optimizer using random search."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
from pandas import DataFrame
from zeno import DistillReturn, MetricReturn, ZenoOptions

from zeno_build.experiments import search_space
from zeno_build.optimizers.base import Optimizer

T = TypeVar("T")


class RandomOptimizer(Optimizer):
    """An optimizer using random search."""

    def __init__(
        self,
        space: search_space.SearchSpace,
        distill_functions: list[Callable[[DataFrame, ZenoOptions], DistillReturn]],
        metric: Callable[[DataFrame, ZenoOptions], MetricReturn],
        seed: int | None = None,
        num_trials: int | None = None,
    ):
        """Initialize a random optimizer.

        Args:
            space: The space to search over.
            constants: The constants to use.
            distill_functions: The distill functions to use.
            metric: The metric to use.
            seed: The random seed to use.
            num_trials: The maximum number of trials to run.
        """
        super().__init__(space, distill_functions, metric, num_trials)
        # Set state without side-effects (for single thread)
        saved_state = random.getstate()
        random.seed(seed)
        self._state = random.getstate()
        random.setstate(saved_state)

    @staticmethod
    def _get_parameters_from_space(
        space: search_space.SearchSpace,
    ) -> dict[str, Any] | None:
        if isinstance(space, search_space.CombinatorialSearchSpace):
            params = {}
            for name, dimension in space.dimensions.items():
                if isinstance(dimension, search_space.Categorical) or isinstance(
                    dimension, search_space.Discrete
                ):
                    params[name] = random.choice(dimension.choices)
                elif isinstance(dimension, search_space.Float):
                    params[name] = random.uniform(dimension.lower, dimension.upper)
                elif isinstance(dimension, search_space.Int):
                    params[name] = random.randint(dimension.lower, dimension.upper)
                elif isinstance(dimension, search_space.Constant):
                    params[name] = dimension.value
                else:
                    raise ValueError(f"Unknown search dimension: {dimension}")
            return params
        elif isinstance(space, search_space.CompositeSearchSpace):
            sub_space = space.spaces[
                np.random.choice(len(space.spaces), p=space.weights)
            ]
            return RandomOptimizer._get_parameters_from_space(sub_space)
        else:
            raise NotImplementedError(f"Unsupported search space {type(space)}.")

    def get_parameters(self) -> dict[str, Any] | None:
        """Randomize the parameters in a space.

        Args:
            space: The space to randomize.

        Returns:
            A dictionary of randomized parameters.
        """
        saved_state = random.getstate()
        random.setstate(self._state)
        params = RandomOptimizer._get_parameters_from_space(self.space)
        self._state = random.getstate()
        random.setstate(saved_state)
        return params
