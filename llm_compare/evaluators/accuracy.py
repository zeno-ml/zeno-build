
from typing import Any, TypeVar

from llm_compare.evaluators.base import Evaluator

T = TypeVar("T")

class AccuracyEvaluator(Evaluator):

    def __init__(self, references: list[T]):
        """Initialize the evaluator.

        Args:
            references: The reference outputs.
        """
        self.references = references

    def name(self) -> str:
        """Get the name of the evaluator.

        Returns:
            The name of the evaluator.
        """
        return "accuracy"

    def evaluate(self, predictions: list[T]) -> float:
        """Evaluate the results of a run.

        Args:
            predictions: The predicted outputs.

        Returns:
            The accuracy of the run.
        """
        if len(self.references) != len(predictions):
            raise ValueError(
                f"Number of references ({len(self.references)}) does not match "
                f"number of predictions ({len(predictions)})."
            )
        return sum(1 if r == p else 0 for r, p in zip(self.references, predictions))/len(self.references)
