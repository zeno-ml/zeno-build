"""Base class for all evaluators."""

from abc import abstractmethod
from typing import TypeVar

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
    def evaluate(self, predictions: list[T]) -> tuple[float, list[float]]:
        """Evaluate the results of a run.

        Args:
            predictions: The predicted outputs.

        Returns:
            - The metric value for the entire input.
            - The metric value for each example in the input.
        """
