"""An evaluator that computes the accuracy of a run using the critique service."""

from __future__ import annotations

import json
from typing import Any, TypeVar, cast

from inspiredco import critique

from llm_compare.evaluators import critique_presets
from llm_compare.evaluators.base import Evaluator

T = TypeVar("T")


class CritiqueEvaluator(Evaluator):
    """An evaluator that computes the accuracy of a run."""

    def __init__(
        self,
        api_key: str,
        data: list[str] | None = None,
        labels: list[str] | None = None,
        preset: str | None = None,
        metric: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        """Initialize the evaluator.

        Args:
            api_key: The API key for the critique service.
            data: The input data to evaluate on.
            labels: The gold-standard outputs.
            preset: The preset to use. If set, metric and config must not be set.
            metric: The metric to use. Required if "preset" is not set.
            config: The configuration for the metric.
        """
        self._client = critique.Critique(api_key)

        # Create and validate dataset for passing to Critique
        self._dataset: list[dict[str, Any]] | None = None
        if data is not None and labels is not None:
            if len(data) != len(labels):
                raise ValueError(
                    f"Number of data ({len(data)}) does not match "
                    f"number of labels ({len(labels)})."
                )
            self._dataset = [
                {"source": datum, "references": [label]}
                for datum, label in zip(data, labels)
            ]
        elif data is not None:
            self._dataset = [{"source": datum} for datum in data]
        elif labels is not None:
            self._dataset = [{"references": [label]} for label in labels]

        # Load in preset
        if preset is not None:
            if metric is not None or config is not None:
                raise ValueError("If preset is set, metric and config must not be set.")
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

    def evaluate(self, outputs: list[T]) -> tuple[float, list[float]]:
        """Evaluate the results of a run.

        Args:
            outputs: The predicted outputs.

        Returns:
            The accuracy of the run.
        """
        if not self._dataset:
            dataset = [{"target": cast(str, p)} for p in outputs]
        else:
            dataset = [
                {**d, "target": cast(str, p)} for p, d in zip(outputs, self._dataset)
            ]
        with open("critique_input.json", "w") as f:
            json.dump(
                {"metric": self._metric, "config": self._config, "dataset": dataset}, f
            )
        response = self._client.evaluate(
            metric=self._metric, config=self._config, dataset=dataset
        )
        overall = response["overall"]["value"]
        examples = [e["value"] for e in response["examples"]]
        return overall, examples
