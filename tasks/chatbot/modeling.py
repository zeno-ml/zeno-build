"""Chatbots using API-based services."""
from __future__ import annotations

import asyncio
import itertools
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import cohere
import datasets
import openai
import tqdm

from llm_compare.cache_utils import get_cache_path
from tasks.chatbot import model_configs, prompt_configs

DATASET_MAPPING: dict[str | tuple[str, str], Any] = {
    "daily_dialog": {
        "data_column": "dialog",
        "data_format": "sequence",
    },
}

cohere_client: cohere.Client | None = None


@dataclass(frozen=True)
class DialogExample:
    """A single dialog example.

    Args:
        source: The source utterance by the user.
        context: The utterance preceding the source utterance (by the system).
        reference: A reference utterance demonstrating a "good" response.
    """

    source: str
    context: str | None
    reference: str | None


def _build_examples_from_sequence(seq: list[str]) -> Iterable[DialogExample]:
    """Convert a datapoint into dialog examples."""
    for i in range(2, len(seq)):
        yield DialogExample(
            context=seq[i - 1],
            source=seq[i],
            reference=seq[i + 1] if i + 1 < len(seq) else None,
        )


def load_data(
    dataset: str | tuple[str, str],
    split: str,
    examples: int | None,
    data_format: str = "sequence",
    data_column: str = "dialog",
) -> list[DialogExample]:
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
    match data_format:
        case "sequence":
            return list(
                itertools.chain.from_iterable(
                    _build_examples_from_sequence(x[data_column]) for x in loaded_data
                )
            )
        case _:
            raise ValueError(f"Unknown data format {data_format}")


async def generate(
    sources: list[str],
    contexts: list[str | None],
    prompt_template: prompt_configs.ChatMessages,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> list[str]:
    """Generate a single example.

    Args:
        source: The source text to consume.
        prompt_template: The template for the prompt.
        provider: The provider to use.
        model: The model to use.
        temperature: The temperature to use.
        max_tokens: The maximum number of tokens to generate.
        top_p: The top p value to use.

    Returns:
        The generated text.
    """
    if len(sources) != len(contexts):
        raise ValueError(
            f"Length of sources and contexts not equal: "
            f"{len(sources)} != {len(contexts)}"
        )
    print(
        f"Generating with {prompt_template=}, {model=}, "
        f"{temperature=}, {max_tokens=}, {top_p=}..."
    )
    if provider == "openai":
        async_responses = [
            openai.Completion.acreate(
                engine=model,
                prompt=prompt_template.to_text_prompt(source, context),
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            for source, context in zip(sources, contexts)
        ]
        responses = await asyncio.gather(*async_responses)
        return [x["choices"][0]["text"] for x in responses]
    elif provider == "openai_chat":
        async_responses = [
            openai.ChatCompletion.acreate(
                model=model,
                messages=prompt_template.to_openai_chat_completion_messages(
                    source, context
                ),
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            for source, context in zip(sources, contexts)
        ]
        responses = await asyncio.gather(*async_responses)
        return [x["choices"][0]["message"]["content"] for x in responses]
    elif provider == "cohere":
        results = []
        for source, context in tqdm.tqdm(
            zip(sources, contexts), "Generating synchronously from Cohere"
        ):
            try:
                assert cohere_client is not None
                prompt = prompt_template.to_text_prompt(source, context)
                response = cohere_client.generate(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    p=top_p,
                )
                results.append(response.generations[0].text)
            except cohere.CohereAPIError as e:
                # Cohere API sometimes rejects queries, if so output a blank line
                print(f"Warning! Cohere API rejected query for {prompt=}: {e.message}")
                results.append("")
        return results
    else:
        raise ValueError("Unknown provider, but you can add your own!")


def make_predictions(
    test_dataset: str,
    prompt_preset: str,
    model_preset: str,
    temperature: float = 0.3,
    max_tokens: int = 100,
    top_p: float = 1,
    test_split: str = "test",
    test_examples: int | None = None,
) -> list[str]:
    """Make predictions over a particular dataset.

    Args:
        test_dataset: The test dataset in HuggingFace Datasets format.
        prompt_preset: The prompt to use for the API call, as specified in
          prompt_configs.
        model_preset: The model to use for the API call, as specified in
          model_configs.
        temperature: The temperature to use for sampling.
        max_tokens: The maximum number of tokens to generate.
        top_p: The value to use for top-p sampling.
        test_split: The split of the test dataset to use.
        test_examples: The number of examples to use from the test dataset.

    Returns:
        The predictions in string format.
    """
    # If we've already used these parameters, load from cache
    parameters = dict(locals())
    parameters["__name__"] = make_predictions.__name__
    cache_path = get_cache_path("chatbot", parameters, ".json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    # Load dataset
    mapping = DATASET_MAPPING.get(test_dataset, {})
    data_format = mapping.get("data_format", "sequence")
    data_column = mapping.get("data_column", "dialog")
    dataset = load_data(
        test_dataset, test_split, test_examples, data_format, data_column
    )

    prompt_template = prompt_configs.prompt_messages[prompt_preset]
    provider = model_configs.model_configs[model_preset]["provider"]
    model = model_configs.model_configs[model_preset]["model"]

    # Make predictions
    predictions = asyncio.run(
        generate(
            [x.source for x in dataset],
            [x.context for x in dataset],
            prompt_template,
            provider,
            model,
            temperature,
            max_tokens,
            top_p,
        )
    )
    with open(cache_path, "w") as f:
        json.dump(predictions, f)
    return predictions
