"""Data class for a single run of an experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ExperimentRun(Generic[T]):
    """A single run of an experiment."""

    parameters: dict[str, Any]
    predictions: list[T]
    eval_result: float
    name: str | None = None
