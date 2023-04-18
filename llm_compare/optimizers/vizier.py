"""An optimizer using the Vizier toolkit."""

import json
import os
from collections.abc import Callable
from typing import Any, TypeVar

from vizier.service import clients
from vizier.service import pyvizier as vz

from llm_compare import search_space
from llm_compare.evaluators.base import Evaluator
from llm_compare.experiment_run import ExperimentRun
from llm_compare.optimizers.base import Optimizer

T = TypeVar("T")


class VizierOptimizer(Optimizer):
    """An optimizer using the Vizier toolkit."""

    def __init__(
        self,
        algorithm: str = "RANDOM_SEARCH",
        owner: str = "llm-compare",
        study_id: str = "llm-compare",
    ):
        """Initialize a Vizier optimizer.

        Args:
            algorithm: The algorithm to use for optimization.
            owner: The owner of the study.
            study_id: The ID of the study.
        """
        self.algorithm = algorithm
        self.owner = owner
        self.study_id = study_id

    def run_sweep(
        self,
        function: Callable[..., list[T]],
        space: dict[str, search_space.SearchDimension],
        constants: dict[str, Any],
        evaluator: Evaluator,
        num_trials: int | None,
        results_dir: str | None = None,
    ) -> list[ExperimentRun]:
        """Run a hyperparameter sweep with Vizier.

        Args:
            function: The function to optimize.
            space: The space of hyperparameters to search over.
            constants: Any constants that are fed into the function.
            evaluator: The function used to evaluate the results of a run.
            num_trials: The number of trials to run.
            results_file: The file to save the results to.

        Returns:
            A list of runs.
        """
        if num_trials is None:
            raise ValueError(
                "Vizier does not support exhaustive search, set num_trials to an "
                "integer value."
            )

        # Algorithm, search space, and metrics.
        study_config = vz.StudyConfig(algorithm=self.algorithm)
        for name, dimension in space.items():
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
            vz.MetricInformation(evaluator.name(), goal=vz.ObjectiveMetricGoal.MAXIMIZE)
        )

        # Setup client and begin optimization. Vizier Service will be implicitly
        # created.
        study = clients.Study.from_study_config(
            study_config, owner=self.owner, study_id=self.study_id
        )
        experiment_runs: list[ExperimentRun] = []
        for i in range(num_trials):
            suggestions = study.suggest(count=1)
            for suggestion in suggestions:
                params = suggestion.parameters
                results = function(**params, **constants)
                objective = evaluator.evaluate(results)
                suggestion.complete(vz.Measurement({evaluator.name(): objective}))
                current_run = ExperimentRun(params, results, objective)
                experiment_runs.append(current_run)
                if results_dir is not None:
                    if not os.path.exists(results_dir):
                        os.makedirs(results_dir)
                    with open(os.path.join(results_dir, f"run{i:04d}.json"), "w") as f:
                        json.dump(current_run, f)
        return experiment_runs
