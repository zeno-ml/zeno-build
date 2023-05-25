"""Text summarization using API-based services."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os

import datasets

from examples.summarization import config as summarization_config
from zeno_build.cache_utils import get_cache_path
from zeno_build.models.text_generate import generate_from_text_prompt


def load_data(
    dataset: str | tuple[str, str],
    split: str,
    examples: int | None,
) -> list[dict[str, str]]:
    """Load data from the huggingface library.

    Args:
        dataset: The name of the dataset to load, either:
          - A string, the name of the dataset.
          - A tuple of strings, the name of the dataset and the name of the
            subdataset.
        split: The split of the dataset to load.
        examples: The number of examples to load. If None, load all examples.

    Returns:
        The loaded dataset.
    """
    if isinstance(dataset, tuple):
        dname, subdname = dataset
        loaded_data = datasets.load_dataset(dname, subdname, split=split)
    else:
        loaded_data = datasets.load_dataset(dataset, split=split)
    if examples is not None:
        loaded_data = loaded_data.select(range(examples))
    mapping = summarization_config.dataset_mapping.get(dataset, {})
    data_column = mapping.get("data_column", "text")
    label_column = mapping.get("label_column", "summary")
    return [{"data": x[data_column], "label": x[label_column]} for x in loaded_data]


def make_predictions(
    data: list[str],
    prompt_preset: str,
    model_preset: str,
    temperature: float = 0.3,
    max_tokens: int = 100,
    top_p: float = 1,
    cache_root: str | None = None,
) -> list[str]:
    """Make predictions over a particular dataset.

    Args:
        data: The data to make predictions over.
        prompt_preset: The prompt to use for the API call, as specified in
          prompt_configs.
        model_preset: The model to use for the API call, as specified in
          model_configs.
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

    prompt_template = summarization_config.prompt_text[prompt_preset]
    model_config = summarization_config.model_configs[model_preset]

    # Make predictions
    predictions = asyncio.run(
        generate_from_text_prompt(
            [{"source": x} for x in data],
            prompt_template,
            model_config,
            temperature,
            max_tokens,
            top_p,
        )
    )

    # Cache
    if cache_path is not None:
        with open(cache_path, "w") as f:
            json.dump(predictions, f)
    return predictions
