"""Code Generation using API-based services."""
from __future__ import annotations

import itertools
import json
import os
import traceback
from collections.abc import Iterable
from typing import Literal

import datasets

from examples.code_generation import config as codegen_config
from zeno_build.cache_utils import CacheLock, fail_cache, get_cache_path
from zeno_build.models.code_generate import generate_from_code_prompt


def build_input_from_intents_and_prompts(
    intent: str,
    prompt: str,
) -> str:
    func_head, ret_msg = prompt.split("\n")
    input_text = func_head + "\n\t" + f'"""{intent}"""' + "\n" + ret_msg
    return input_text


def build_test(test_start: str, test: list[str], entry_point: str) -> str:
    return "\n".join(
        [
            test_start,
            "".join(test),
            "",
            f"check({entry_point})",
        ]
    )


def process_data(
    dataset: str | tuple[str, str],
    split: str,
    examples: int,
    data_format: str,
    data_column: str | list[str],
    label_column: str | list[str],
    output_dir: str = "results",
) -> list[dict[str, str]]:
    """Load and process data from the huggingface library.

    Args:
        dataset: The name of the dataset to load, either:
          - A string, the name of the dataset.
          - A tuple of strings, the name of the dataset
            and the name of the subdataset.
        split: The split of the dataset to load.
        data_format: The format of the data, either:
          - "odex": The format of the ODEX dataset.
          - "humaneval": The format of the HumanEval dataset.
        data_column: The name of the column containing the natural language input.
        prompt_column: The name of the column containing function signature.
        output_dir: The directory to save the processed data to.

    Side effects:
        Writes:
          - The parameters to a 'zbp' file in the output directory
          - The processed data to a 'jsonl' file in the output directory

    Returns:
        The loaded dataset as code generation examples of input prompts and reference snippets.
    """
    # Load from cache and return if existing
    parameters = {k: v for k, v in locals().items() if k != "output_dir"}
    output_path = get_cache_path(output_dir, parameters, "jsonl")
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            return [json.loads(x.strip()) for x in f]

    # Load and standardize from Hugging Face if not in cache
    if isinstance(dataset, tuple):
        dname, subdname = dataset
        loaded_data = datasets.load_dataset(dname, subdname, split=split)
    else:
        loaded_data = datasets.load_dataset(dataset, split=split)
    if examples is not None:
        loaded_data = loaded_data.select(range(examples))

    if data_format == "odex":
        intent_column, prompt_column = data_column
        prompts = [
            build_input_from_intents_and_prompts(x[intent_column], x[prompt_column])
            for x in loaded_data
        ]
        start_column, test_column, entry_column = label_column
        labels = [
            build_test(x[start_column], x[test_column], x[entry_column])
            for x in loaded_data
        ]
        suffixes = [x["suffix"] for x in loaded_data]
        # @todo: change to tokenizer map
        data_examples = [
            {"input": p, "label": l, "suffix": s}
            for p, l, s in zip(prompts, labels, suffixes)
        ]
    elif (
        data_format == "humaneval"
    ):
        data_examples = [
            {
                "input": x[data_column],
                "label": x[label_column],
                "suffix": "",
            }
            for x in loaded_data
        ]
    else:
        raise ValueError(f"Unknown data format {data_format}")

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for x in data_examples:
            print(json.dumps(x), file=f)

    return data_examples


def make_predictions(
    data: list[str],
    prompt_preset: str,
    model_preset: str,
    temperature: float = 0.8,
    max_tokens: int = 512,
    top_p: float = 0.95,
    output_dir: str = "results",
) -> list[str] | None:
    # Load from cache if existing
    parameters = {k: v for k, v in locals().items() if k not in {"data", "output_dir"}}
    file_root = get_cache_path(output_dir, parameters)
    if os.path.exists(f"{file_root}.json"):
        with open(f"{file_root}.json", "r") as f:
            return json.load(f)

    prompt_template = codegen_config.prompt_text[prompt_preset]
    model_config = codegen_config.model_configs[model_preset]

    with CacheLock(file_root) as cache_lock:
        # If the cache is locked, then another process is already generating
        # so just skip this one
        if not cache_lock:
            return None
        # Make predictions
        try:
            predictions: list[str] = generate_from_code_prompt(
                [{"source": x.rstrip()} for x in data],
                prompt_template,
                model_config,
                temperature,
                max_tokens,
                top_p,
            )
        except Exception:
            tb = traceback.format_exc()
            fail_cache(file_root, tb)
            raise

        # Dump the predictions
        with open(f"{file_root}.json", "w") as f:
            json.dump(predictions, f)

    return predictions
