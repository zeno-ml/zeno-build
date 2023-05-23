"""An optimizer using the Vizier toolkit."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from pandas import DataFrame
from vizier.service import clients
from vizier.service import pyvizier as vz
from zeno import DistillReturn, MetricReturn, ZenoOptions

from zeno_build.experiments import search_space
from zeno_build.optimizers.base import Optimizer

T = TypeVar("T")


class VizierOptimizer(Optimizer):
    """An optimizer using the Vizier toolkit."""

    def __init__(
        self,
        space: search_space.SearchSpace,
        distill_functions: list[Callable[[DataFrame, ZenoOptions], DistillReturn]],
        metric: Callable[[DataFrame, ZenoOptions], MetricReturn],
        num_trials: int | None = None,
        algorithm: str = "RANDOM_SEARCH",
        owner: str = "zeno-build",
        study_id: str = "zeno-build",
    ):
        """Initialize a Vizier optimizer.

        Args:
            space: The space to search over.
            constants: The constants to use.
            distill_functions: The distill functions to use.
            metric: The metric to use.
            num_trials: The number of trials to run.
            algorithm: The algorithm to use for optimization.
            owner: The owner of the study.
            study_id: The ID of the study.
        """
        if not isinstance(space, search_space.CombinatorialSearchSpace):
            raise NotImplementedError("Only combinatorial search spaces are supported.")

        super().__init__(space, distill_functions, metric, num_trials)
        self.algorithm = algorithm
        self.owner = owner
        self.study_id = study_id
        # Algorithm, search space, and metrics.
        study_config = vz.StudyConfig(algorithm=self.algorithm)
        for name, dimension in space.dimensions.items():
            if isinstance(dimension, search_space.Categorical):
                study_config.search_space.root.add_categorical_param(
                    name, dimension.choices
                )
            elif isinstance(dimension, search_space.Discrete):
                study_config.search_space.root.add_discrete_param(
                    name, dimension.choices
                )
            elif isinstance(dimension, search_space.Float):
                study_config.search_space.root.add_float_param(
                    name, dimension.lower, dimension.upper
                )
            elif isinstance(dimension, search_space.Int):
                study_config.search_space.root.add_int_param(
                    name, dimension.lower, dimension.upper
                )
            else:
                raise ValueError(f"Unknown search space dimension: {dimension}")
        study_config.metric_information.append(
            vz.MetricInformation(metric.__name__, goal=vz.ObjectiveMetricGoal.MAXIMIZE)
        )

        # Setup client and begin optimization. Vizier Service will be implicitly
        # created.
        self.study = clients.Study.from_study_config(
            study_config, owner=self.owner, study_id=self.study_id
        )
        self.last_suggestion: Any | None = None

    def get_parameters(self) -> dict[str, Any]:
        """Get the next set of parameters to try.

        Returns:
            A dictionary of parameters.
        """
        self.last_suggestion = self.study.suggest(count=1)
        return self.last_suggestion.parameters

    def calculate_metric(
        self, data: list[Any], labels: list[Any], outputs: list[Any]
    ) -> float:
        """Calculate the metric for the last set of parameters.

        This must be called only once after get_parameters.

        Args:
            data: The data to use.
            labels: The labels to use.
            outputs: The outputs to use.

        Returns:
            The metric value.
        """
        if self.last_suggestion is None:
            raise ValueError("Must only call calculate_metric after get_parameters.")
        val = super().calculate_metric(data, labels, outputs)
        self.last_suggestion.complete(vz.Measurement({self.metric.__name__: val}))
        self.last_suggestion = None
        return val
