"""Chatbots using API-based services."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass

import datasets

from tasks.chatbot import config as chatbot_config
from zeno_build.cache_utils import get_cache_path
from zeno_build.models.chat_generate import generate_from_chat_prompt


@dataclass(frozen=True)
class ChatExample:
    """A single dialog example.

    Args:
        source: The source utterance by the user.
        context: The utterance preceding the source utterance (by the system).
        reference: A reference utterance demonstrating a "good" response.
    """

    source: str
    reference: str
    context: list[str]


def _build_examples_from_sequence(seq: list[str]) -> Iterable[ChatExample]:
    """Convert a datapoint into dialog examples."""
    stripped_seq = [x.strip() for x in seq]
    for i in range(1, len(stripped_seq)):
        yield ChatExample(
            context=stripped_seq[: i - 1],
            source=stripped_seq[i - 1],
            reference=stripped_seq[i],
        )


def load_data(
    dataset: str | tuple[str, str],
    split: str,
    examples: int | None,
    data_format: str = "sequence",
    data_column: str = "dialog",
) -> list[ChatExample]:
    """Load data from the huggingface library.

    Args:
        dataset: The name of the dataset to load, either:
          - A string, the name of the dataset.
          - A tuple of strings, the name of the dataset and the name of the
            subdataset.
        split: The split of the dataset to load.
        examples: The number of examples to load. If None, load all examples.

    Returns:
        The loaded dataset as dialog examples of context, source, and reference.
    """
    if isinstance(dataset, tuple):
        dname, subdname = dataset
        loaded_data = datasets.load_dataset(dname, subdname, split=split)
    else:
        loaded_data = datasets.load_dataset(dataset, split=split)
    if examples is not None:
        loaded_data = loaded_data.select(range(examples))
    if data_format == "sequence":
        return list(
            itertools.chain.from_iterable(
                _build_examples_from_sequence(x[data_column]) for x in loaded_data
            )
        )
    else:
        raise ValueError(f"Unknown data format {data_format}")


def make_predictions(
    data: list[ChatExample],
    prompt_preset: str,
    model_preset: str,
    temperature: float = 0.3,
    max_tokens: int = 100,
    top_p: float = 1,
    cache_root: str | None = None,
) -> list[str]:
    """Make predictions over a particular dataset.

    Args:
        test_dataset: The test dataset in HuggingFace Datasets format.
        prompt_preset: The prompt to use for the API call.
        model_preset: The model to use for the API call.
        temperature: The temperature to use for sampling.
        max_tokens: The maximum number of tokens to generate.
        top_p: The value to use for top-p sampling.
        cache_root: The location of the cache directory if any

    Returns:
        The predictions in string format.
    """
    # Load from cache if existing
    cache_path: str | None = None
    if cache_root is not None:
        parameters = dict(locals())
        parameters["__name__"] = make_predictions.__name__
        parameters["data_hash"] = hashlib.sha256(
            json.dumps(parameters.pop("data"), default=str).encode("utf-8")
        ).hexdigest()
        for k in ["cache_root", "cache_path"]:
            parameters.pop(k)
        cache_path = get_cache_path(cache_root, parameters, "json")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)

    # Make predictions
    predictions: list[str] = generate_from_chat_prompt(
        [
            {"source": x.source, "context": x.context[-1] if len(x.context) else ""}
            for x in data
        ],
        chatbot_config.prompt_messages[prompt_preset],
        chatbot_config.model_configs[model_preset],
        temperature,
        max_tokens,
        top_p,
    )

    # Dump the cache and return
    if cache_path is not None:
        with open(cache_path, "w") as f:
            json.dump(predictions, f)
    return predictions
