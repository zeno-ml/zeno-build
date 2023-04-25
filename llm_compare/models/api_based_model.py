"""Config for API-based models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ApiBasedModelConfig:
    """A config for an API-based model.

    Attributes:
        provider: The name of the API provider.
        model: The name of the model.
    """

    provider: str
    model: str
