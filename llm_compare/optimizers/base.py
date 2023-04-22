"""Base class for optimizers that run hyperparameter sweeps."""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from pandas import DataFrame
from zeno import DistillReturn, MetricReturn, ZenoOptions

from llm_compare import experiment_run, search_space

T = TypeVar("T")


class Optimizer:
    """An optimizer for hyperparameter search."""

    @abc.abstractmethod
    def run_sweep(
        self,
        function: Callable[..., list[T]],
        space: dict[str, search_space.SearchDimension],
        constants: dict[str, Any],
        data: Sequence[Any] | None,
        labels: Sequence[Any] | None,
        distill_functions: list[Callable[[DataFrame, ZenoOptions], DistillReturn]],
        metric: Callable[[DataFrame, ZenoOptions], MetricReturn],
        num_trials: int | None,
        results_dir: str | None = None,
    ) -> list[experiment_run.ExperimentRun]:
        """Run a hyperparameter sweep.

        Args:
            function: The function to optimize.
            space: The space of hyperparameters to search over.
            constants: Any constants that are fed into the function.
            data: The data corresponding to the corpus inputs.
            labels: The labels corresponding to the gold-standard outputs.

            metric: The metric to use for evaluation.
            num_trials: The number of trials to run.
            results_dir: The to save the results to.

        Returns:
            A list of runs.
        """
