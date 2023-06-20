"""Search space for hyperparameter optimization."""

from __future__ import annotations

import itertools
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class SearchDimension(ABC):
    """A dimension along which a hyperparameter can be searched."""

    @abstractmethod
    def value_in_scope(self, value: Any) -> bool:
        """Check if a value is in the scope of this dimension."""
        ...


@dataclass(frozen=True)
class Constant(SearchDimension, Generic[T]):
    """A constant."""

    value: T

    def value_in_scope(self, value: Any) -> bool:
        """See base class."""
        return value == self.value


@dataclass(frozen=True)
class Categorical(SearchDimension, Generic[T]):
    """A categorical hyperparameter."""

    choices: list[T]

    def value_in_scope(self, value: Any) -> bool:
        """See base class."""
        return value in self.choices


@dataclass(frozen=True)
class Discrete(SearchDimension, Generic[T]):
    """A discrete hyperparameter.

    The difference between a discrete and categorical hyperparameter is that
    the values of a discrete hyperparameter are ordered, while the values of
    a categorical hyperparameter are not.
    """

    choices: list[T]

    def value_in_scope(self, value: Any) -> bool:
        """See base class."""
        return value in self.choices


@dataclass(frozen=True)
class Float(SearchDimension):
    """A float hyperparameter range."""

    lower: float
    upper: float

    def value_in_scope(self, value: Any) -> bool:
        """See base class."""
        return self.lower <= value <= self.upper


@dataclass(frozen=True)
class Int(SearchDimension):
    """An integer hyperparameter range.

    Attributes:
        lower: The lower bound of the range (inclusive).
        upper: The upper bound of the range (inclusive).
    """

    lower: int
    upper: int

    def value_in_scope(self, value: Any) -> bool:
        """See base class."""
        return self.lower <= value <= self.upper


class SearchSpace(ABC):
    """A search space for hyperparameter optimization."""

    @abstractmethod
    def contains_params(self, params: dict[str, Any]) -> bool:
        """Check whether the search space contains the given parameters."""
        ...

    @abstractmethod
    def get_non_constant_dimensions(self) -> list[str]:
        """Get the names of the non-constant dimensions."""
        ...

    def get_valid_param_files(
        self, output_dir: str, include_in_progress: bool
    ) -> list[str]:
        """Get the valid parameter files in the given directory.

        Args:
            output_dir: The directory where the parameter files are stored.
            include_in_progress: Whether to include parameter files for runs that are
                still running.

        Returns:
            Paths to the valid parameter files.
        """
        os.makedirs(output_dir, exist_ok=True)
        param_files = [
            os.path.join(output_dir, x)
            for x in os.listdir(output_dir)
            if x.endswith(".zbp")
        ]
        finished_files = []
        for param_file in param_files:
            fail_file = f"{param_file[:-4]}.zbfail"
            lock_file = f"{param_file[:-4]}.zblock"
            if os.path.exists(fail_file):
                continue
            with open(param_file, "r") as f:
                params = json.load(f)
                if self.contains_params(params):
                    if include_in_progress or not os.path.exists(lock_file):
                        finished_files.append(param_file)
        return finished_files


class CombinatorialSearchSpace(SearchSpace):
    """A search space that includes the cross product of dimensions."""

    def __init__(self, dimensions: dict[str, SearchDimension]):
        """Initialize the search space.

        Args:
            dimensions: The dimensions of the search space.
        """
        self.dimensions = dimensions

    def contains_params(self, params: dict[str, Any]) -> bool:
        """See base class."""
        for k, v in params.items():
            if k not in self.dimensions:
                return False
            if not self.dimensions[k].value_in_scope(v):
                return False
        return True

    def get_non_constant_dimensions(self) -> list[str]:
        """See base class."""
        return [k for k, v in self.dimensions.items() if not isinstance(v, Constant)]


class CompositeSearchSpace(SearchSpace):
    """A search space consisting of multiple search spaces."""

    def __init__(self, spaces: list[SearchSpace], weights: list[float] | None = None):
        """Initialize the search space.

        Args:
            spaces: The search spaces to combine.
            weights: The weights of the search spaces. If None, all search spaces
                have equal weight.
        """
        self.spaces = spaces
        self.weights = weights or [1.0 / len(spaces)] * len(spaces)

    def contains_params(self, params: dict[str, Any]) -> bool:
        """Check if the parameters are contained in the search space."""
        return any([s.contains_params(params) for s in self.spaces])

    def get_non_constant_dimensions(self) -> list[str]:
        """See base class."""
        return sorted(
            list(
                set(
                    itertools.chain.from_iterable(
                        s.get_non_constant_dimensions() for s in self.spaces
                    )
                )
            )
        )
