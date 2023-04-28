"""Base class for optimizers that run hyperparameter sweeps."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from pandas import DataFrame
from zeno import DistillReturn, MetricReturn, ZenoOptions

from llm_compare import search_space

T = TypeVar("T")


class Optimizer:
    """An optimizer for hyperparameter search."""

    def __init__(
        self,
        space: dict[str, search_space.SearchDimension],
        constants: dict[str, Any],
        distill_functions: list[Callable[[DataFrame, ZenoOptions], DistillReturn]],
        metric: Callable[[DataFrame, ZenoOptions], MetricReturn],
    ):
        """Initialize an optimizer."""
        self.space = space
        self.constants = constants
        self.distill_functions = distill_functions
        self.metric = metric

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
