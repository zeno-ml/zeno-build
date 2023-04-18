from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Evaluator:
    """Abstract base class for all evaluators."""

    @abstractmethod
    def name(self) -> str:
        """Get the name of the evaluator.

        Returns:
            The name of the evaluator.
        """

    @abstractmethod
    def evaluate(self, predictions: list[T]) -> float:
        """Evaluate the results of a run.

        Args:
            predictions: The predicted outputs.

        Returns:
            The metric value for the run.
        """
