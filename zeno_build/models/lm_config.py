"""Config for language models."""

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
        system_name: The name of the system in chat.
        user_name: The name of the user in chat.
    """

    provider: str
    model: str
    model_cls: type | None = None
    tokenizer_cls: type | None = None
    name_replacements: dict[str, str] = {
        "system": "System",
        "assistant": "Assistant",
        "user": "User",
    }
