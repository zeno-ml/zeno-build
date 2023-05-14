"""Config for language models."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LMConfig:
    """A config for a language model.

    Attributes:
        provider: The name of the API provider.
        model: The name of the model.
        model_cls: The Python class corresponding to the model, mostly for
             Hugging Face transformers.
        tokenizer_cls: The Python class corresponding to the tokenizer, mostly
            for Hugging Face transformers.
        name_replacements: A dictionary mapping from the names of the roles
            (e.g., "system", "assistant", "user") to the names of the
            roles in the model.
        model_loader_kwargs: A dictionary of keyword arguments to pass to the
            model loader.
    """

    provider: str
    model: str
    model_cls: type | None = None
    tokenizer_cls: type | None = None
    name_replacements: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(
            {
                "system": "System",
                "assistant": "Assistant",
                "user": "User",
            }
        )
    )
    model_loader_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
