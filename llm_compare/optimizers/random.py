"""An optimizer using random search."""

import json
import logging
import os
import random
from collections.abc import Callable
from typing import Any, TypeVar

from llm_compare import search_space
from llm_compare.evaluators.base import Evaluator
from llm_compare.experiment_run import ExperimentRun
from llm_compare.optimizers.base import Optimizer

T = TypeVar("T")


class RandomOptimizer(Optimizer):
    """An optimizer using random search."""

    def __init__(
        self,
        seed: int | None = None,
    ):
        """Initialize a random optimizer.

        Args:
            seed: The seed to use for randomization.
        """
        if seed is not None:
            random.seed(seed)

    def randomize_params(
        self, space: dict[str, search_space.SearchDimension]
    ) -> dict[str, Any]:
        """Randomize the parameters in a space.

        Args:
            space: The space to randomize.

        Returns:
            A dictionary of randomized parameters.
        """
        params = {}
        for name, dimension in space.items():
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

    def run_sweep(
        self,
        function: Callable[..., list[T]],
        space: dict[str, search_space.SearchDimension],
        constants: dict[str, Any],
        evaluator: Evaluator,
        num_trials: int | None,
        results_dir: str | None = None,
    ) -> list[ExperimentRun]:
        """Run a hyperparameter sweep with Random search.

        Args:
            function: The function to optimize.
            space: The space of hyperparameters to search over.
            constants: Any constants that are fed into the function.
            evaluator: The function used to evaluate the results of a run.
            num_trials: The number of trials to run.
            results_dir: The to save the results to.

        Returns:
            A list of runs.
        """
        if num_trials is None:
            raise ValueError(
                "Random search does not support exhaustive search, set num_trials to "
                "an integer value or use ExhaustiveOptimizer."
            )

        experiment_runs: list[ExperimentRun] = []
        for i in range(num_trials):
            params = self.randomize_params(space)
            logging.getLogger(__name__).info(f"Running with params: {params}")
            results = function(**params, **constants)
            objective = evaluator.evaluate(results)
            current_run = ExperimentRun(params, results, objective)
            experiment_runs.append(current_run)
            if results_dir is not None:
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                with open(os.path.join(results_dir, f"run{i:04d}.json"), "w") as f:
                    json.dump(current_run, f)
        return experiment_runs
