"""Base class for optimizers that run hyperparameter sweeps."""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any, TypeVar

from pandas import DataFrame
from zeno import DistillReturn, MetricReturn, ZenoOptions

from zeno_build.experiments import search_space

T = TypeVar("T")


class Optimizer(abc.ABC):
    """An optimizer for hyperparameter search."""

    def __init__(
        self,
        space: search_space.SearchSpace,
        distill_functions: list[Callable[[DataFrame, ZenoOptions], DistillReturn]],
        metric: Callable[[DataFrame, ZenoOptions], MetricReturn],
        num_trials: int | None,
    ):
        """Initialize an optimizer."""
        self.space = space
        self.distill_functions = distill_functions
        self.metric = metric
        self.num_trials = num_trials

    def calculate_metric(
        self, data: list[Any], labels: list[Any], outputs: list[Any]
    ) -> float:
        """Calculate the metric for a set of data, labels, and outputs.

        This must be called only once after get_parameters.

        Args:
            data: The data to use.
            labels: The labels to use.
            outputs: The outputs to use.

        Returns:
            The metric value.
        """
        ops = ZenoOptions(
            data_column="data",
            label_column="labels",
            output_column="outputs",
            id_column="data",
            distill_columns={x.__name__: x.__name__ for x in self.distill_functions},
            data_path="",
            label_path="",
            output_path="",
        )
        df = DataFrame({"data": data, "labels": labels, "outputs": outputs})
        for distill_function in self.distill_functions:
            df[distill_function.__name__] = distill_function(df, ops).distill_output
        return self.metric(df, ops).metric

    def is_complete(
        self,
        output_dir: str,
        include_in_progress: bool = False,
    ) -> bool:
        """Check if the optimizer has completed.

        Args:
            output_dir: The directory where the results of each trial are stored.
            include_in_progress: Whether to include trials that are still running.

        Returns:
            True if the optimizer run has completed.
        """
        if self.num_trials is None:
            return False
        valid_param_files = self.space.get_valid_param_files(
            output_dir=output_dir, include_in_progress=include_in_progress
        )
        return len(valid_param_files) >= self.num_trials

    @abc.abstractmethod
    def get_parameters(self) -> dict[str, Any] | None:
        """Get the next parameters to optimize.

        Returns:
            A dictionary of instantiated parameters, or None
            if there are no more parameters to return.
        """
        ...
