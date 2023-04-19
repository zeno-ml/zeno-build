"""Accuracy evaluator."""

from typing import TypeVar

from llm_compare.evaluators.base import Evaluator

T = TypeVar("T")


class AccuracyEvaluator(Evaluator):
    """An evaluator that computes the accuracy of a run."""

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

    def evaluate(self, predictions: list[T]) -> tuple[float, list[float]]:
        """Evaluate the results of a run.

        Args:
            predictions: The predicted outputs.

        Returns:
            The accuracy of the dataset, and the examples.
        """
        if len(self.references) != len(predictions):
            raise ValueError(
                f"Number of references ({len(self.references)}) does not match "
                f"number of predictions ({len(predictions)})."
            )
        examp_acc = [
            1.0 if r == p else 0.0 for r, p in zip(self.references, predictions)
        ]
        return sum(examp_acc) / len(examp_acc), examp_acc
