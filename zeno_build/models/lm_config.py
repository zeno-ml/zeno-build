"""Config for API-based models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LMConfig:
    """A config for a language model.

    Attributes:
        provider: The name of the API provider.
        model: The name of the model.
        cls: The Python class corresponding to the model, mostly for
             Hugging Face transformers.
    """

    provider: str
    model: str
    cls: type | None = None
