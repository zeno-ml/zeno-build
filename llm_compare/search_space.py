from abc import ABC
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class SearchDimension(ABC):
    """A dimension along which a hyperparameter can be searched."""


@dataclass(frozen=True)
class Categorical(SearchDimension, Generic[T]):
    """A categorical hyperparameter."""

    choices: list[T]


@dataclass(frozen=True)
class Discrete(SearchDimension, Generic[T]):
    """A discrete hyperparameter.

    The difference between a discrete and categorical hyperparameter is that
    the values of a discrete hyperparameter are ordered, while the values of
    a categorical hyperparameter are not.
    """

    choices: list[T]


@dataclass(frozen=True)
class Float(SearchDimension):
    """A float hyperparameter range."""

    lower: float
    upper: float


@dataclass(frozen=True)
class Int(SearchDimension):
    """An integer hyperparameter range."""

    lower: int
    upper: int
