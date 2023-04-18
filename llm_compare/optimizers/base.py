"""Base class for optimizers that run hyperparameter sweeps."""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any, TypeVar

from llm_compare import experiment_run, search_space
from llm_compare.evaluators.base import Evaluator

T = TypeVar("T")


class Optimizer:
    """An optimizer for hyperparameter search."""

    @abc.abstractmethod
    def run_sweep(
        self,
        function: Callable[..., list[T]],
        space: dict[str, search_space.SearchDimension],
        constants: dict[str, Any],
        evaluator: Evaluator,
        num_trials: int | None,
    ) -> list[experiment_run.ExperimentRun]:
        """Run a hyperparameter sweep.

        Args:
            function: The function to optimize.
            space: The space of hyperparameters to search over.
            constants: Any constants that are fed into the function.
            evaluator: The function used to evaluate the results of a run.
            num_trials: The number of trials to run or None to run exhaustively.

        Returns:
            A list of runs.
        """
