"""Chatbots using API-based services."""
from __future__ import annotations

import dataclasses
import itertools
import json
import os
import traceback
from collections.abc import Iterable
from typing import Literal

import config as chatbot_config
import datasets

from zeno_build.cache_utils import (
    CacheLock,
    fail_cache,
    get_cache_id_and_path,
    get_cache_path,
)
from zeno_build.models.chat_generate import generate_from_chat_prompt
from zeno_build.prompts.chat_prompt import ChatMessages, ChatTurn


def build_examples_from_sequence(seq: list[str]) -> Iterable[ChatMessages]:
    """Convert a datapoint into dialog examples."""
    stripped_seq = [x.strip() for x in seq]
    stripped_seq = [x if len(x) else "..." for x in stripped_seq]
    for i in range(2, len(stripped_seq) + 1):
        yield ChatMessages(
            messages=[
                ChatTurn(
                    role="assistant" if (i - j) % 2 == 1 else "user",
                    content=y,
                )
                for j, y in enumerate(stripped_seq[:i])
            ],
        )


def build_examples_from_roles_and_contents(
    roles: list[str],
    contents: list[str],
    name_mapping: dict[str, Literal["system", "assistant", "user"]],
) -> Iterable[ChatMessages]:
    """Convert a datapoint into dialog examples."""
    assert len(roles) == len(contents)
    messages = []
    for role, content in zip(roles, contents):
        role = name_mapping[role]
        stripped_content = content.strip()
        if len(stripped_content) == 0:
            stripped_content = "..."
        messages.append(ChatTurn(role=role, content=stripped_content))
        if role == "assistant":
            yield ChatMessages(messages=list(messages))


def process_data(
    dataset: str | tuple[str, str],
    split: str,
    data_format: str = "sequence",
    data_column: str = "dialog",
    output_dir: str = "results",
) -> list[ChatMessages]:
    """Load data from the huggingface library.

    Args:
        dataset: The name of the dataset to load, either:
          - A string, the name of the dataset.
          - A tuple of strings, the name of the dataset and the name of the
            subdataset.
        split: The split of the dataset to load.
        data_format: The format of the data, either:
            - "sequence": A sequence of strings, each string is a message.
            - "dstc11": The format of the DSTC11 dataset.
        data_column: The name of the column containing the data.
        output_dir: The directory to save the processed data to.

    Side effects:
        Writes:
            - The parameters to a 'zbp' file in the output directory
            - The processed data to a 'jsonl' file in the output directory

    Returns:
        The loaded dataset as dialog examples of context and reference.
    """
    # Load from cache and return if existing
    parameters = {k: v for k, v in locals().items() if k != "output_dir"}
    output_path = get_cache_path(output_dir, parameters, "jsonl")
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            return [ChatMessages.from_dict(json.loads(x)) for x in f]

    # Load and standardize from Hugging Face if not in cache
    if isinstance(dataset, tuple):
        dname, subdname = dataset
        loaded_data = datasets.load_dataset(dname, subdname, split=split)
    else:
        loaded_data = datasets.load_dataset(dataset, split=split)
    if data_format == "sequence":
        messages = list(
            itertools.chain.from_iterable(
                build_examples_from_sequence(x[data_column]) for x in loaded_data
            )
        )
    elif data_format == "dstc11":
        messages = list(
            itertools.chain.from_iterable(
                build_examples_from_roles_and_contents(
                    x[data_column]["speaker_role"],
                    x[data_column]["utterance"],
                    name_mapping={
                        "Agent": "assistant",
                        "Customer": "user",
                    },
                )
                for x in loaded_data
            )
        )
    else:
        raise ValueError(f"Unknown data format {data_format}")

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for x in messages:
            print(json.dumps(x.to_dict()), file=f)

    return messages


def make_predictions(
    contexts: list[ChatMessages],
    prompt_preset: str,
    model_preset: str,
    temperature: float = 0.3,
    max_tokens: int = 100,
    top_p: float = 1,
    context_length: int = -1,
    output_dir: str = "results",
    hf_inference_method: str = "huggingface",
) -> tuple[str, list[str]] | None:
    """Make predictions over a particular dataset.

    Args:
        contexts: The previous chat contexts to generate from.
        prompt_preset: The prompt to use for the API call.
        model_preset: The model to use for the API call.
        temperature: The temperature to use for sampling.
        max_tokens: The maximum number of tokens to generate.
        top_p: The value to use for top-p sampling.
        context_length: The maximum length of the context to use. If 0,
            use the full context.
        output_dir: The location of the cache directory if any
        hf_inference_method: The inference method to use for Hugging Face models.
            This can be huggingface or vllm.

    Side effects:
        - Saves the predictions in a '.json' file in the `output_dir` directory
        - Saves the parameters in a '.zbp' file in the `output_dir` directory

    Returns:
        - The system ID of the predictions.
        - The predictions as a list of strings.
    """
    # Load from cache if existing
    parameters = {
        k: v
        for k, v in locals().items()
        if k not in {"contexts", "output_dir", "hf_inference_method"}
    }
    system_id, file_root = get_cache_id_and_path(output_dir, parameters)
    if os.path.exists(f"{file_root}.json"):
        with open(f"{file_root}.json", "r") as f:
            return system_id, json.load(f)

    with CacheLock(file_root) as cache_lock:
        # If the cache is locked, then another process is already generating
        # so just skip this one
        if not cache_lock:
            return None
        # Make predictions
        try:
            # Set the inference method for huggingface models
            my_model = chatbot_config.model_configs[model_preset]
            if my_model.provider == "huggingface":
                my_model = dataclasses.replace(my_model, provider=hf_inference_method)
            # Generate from the chat prompt
            predictions: list[str] = generate_from_chat_prompt(
                contexts,
                chatbot_config.prompt_messages[prompt_preset],
                my_model,
                temperature,
                max_tokens,
                top_p,
                context_length,
            )
        except Exception:
            tb = traceback.format_exc()
            fail_cache(file_root, tb)
            raise

        # Dump the predictions
        with open(f"{file_root}.json", "w") as f:
            json.dump(predictions, f)

    return system_id, predictions
