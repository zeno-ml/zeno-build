"""Data class for a single run of an experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class ExperimentRun(Generic[T]):
    """A single run of an experiment."""

    name: str
    parameters: dict[str, Any]
    predictions: list[T]
    eval_result: float | None = None
