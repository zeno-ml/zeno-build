"""An evaluator that computes the accuracy of a run using the critique service."""

from typing import Any, TypeVar
from inspiredco import critique

from llm_compare.evaluators.base import Evaluator
from llm_compare.evaluators import critique_presets


T = TypeVar("T")


class CritiqueEvaluator(Evaluator):
    """An evaluator that computes the accuracy of a run."""

    def __init__(
        self,
        api_key: str,
        dataset: list[dict[str, Any]] | None,
        preset: str | None = None,
        metric: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        """Initialize the evaluator.

        Args:
            api_key: The API key for the critique service.
            dataset: The dataset to evaluate on, other than the generated outputs. It can be
                set to None if the metric doesn't require anything but the generated outputs.
            preset: The preset to use. If set, metric and config must not be set.
            metric: The metric to use. Required if "preset" is not set.
            config: The configuration for the metric.
        """
        self._client = critique.Critique(api_key)
        self._dataset = dataset
        if preset is not None:
            if metric is not None or config is not None:
                raise ValueError(
                    "If preset is set, metric and config must not be set."
                )
            self._name = preset
            self._metric = critique_presets.critique_presets[preset]["metric"]
            self._config = critique_presets.critique_presets[preset]["config"]
        else:
            if metric is None or config is None:
                raise ValueError("If preset is not set, metric and config must be set.")
            self._name = metric
            self._metric = metric
            self._config = config

    def name(self) -> str:
        """Get the name of the evaluator.

        Returns:
            The name of the evaluator.
        """
        return self._name

    def evaluate(self, predictions: list[T]) -> tuple[float, list[float]]:
        """Evaluate the results of a run.

        Args:
            predictions: The predicted outputs.

        Returns:
            The accuracy of the run.
        """
        if not self._dataset:
            dataset = [{"target": p} for p in predictions]
        else:
            dataset = [{"target": p, **d} for p, d in zip(predictions, self._dataset)]
        response = self._client.evaluate(
            metric=self._metric, config=self._config, dataset=dataset
        )
        overall = response["overall"]["value"]
        examples = [e["value"] for e in response["examples"]]
        return overall, examples
