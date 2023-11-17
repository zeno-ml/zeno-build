"""Code Generation using API-based services."""
from __future__ import annotations

import json
import os
import traceback

import config as codegen_config
import datasets

from zeno_build.cache_utils import (
    CacheLock,
    fail_cache,
    get_cache_id_and_path,
    get_cache_path,
)
from zeno_build.models.text_generate import generate_from_text_prompt


def build_input_from_intents_and_prompts(
    intent: str,
    prompt: str,
) -> str:
    """Building input text from ODEX intent and prompt.

    Args:
        intent: The natural languag instruction.
        prompt: The function signature.

    Returns:
        The function signature with the intent as a docstring.
    """
    func_head, ret_msg = prompt.split("\n")
    input_text = func_head + "\n\t" + f'"""{intent}"""' + "\n" + ret_msg
    return input_text


def build_test(test_start: str, test: list[str], entry_point: str) -> str:
    """Constructing test cases from function signature, cases, and entry point.

    Args:
        test_start: The checking function signature.
        test: The list of test cases.
        entry_point: The entry point of the function being checked.

    Returns:
        Test cases wrapped into function for execution.
    """
    return "\n".join(
        [
            test_start,
            "".join(test),
            "",
            f"check({entry_point})",
        ]
    )


def _load_single_dataset(
    dataset_preset: str,
) -> list[dict[str, str]]:
    dataset_config = codegen_config.dataset_configs[dataset_preset]

    # Load and standardize from Hugging Face if not in cache
    if isinstance(dataset_config.dataset, tuple):
        dname, subdname = dataset_config.dataset
        loaded_data = datasets.load_dataset(dname, subdname, split=dataset_config.split)
    else:
        loaded_data = datasets.load_dataset(
            dataset_config.dataset, split=dataset_config.split
        )

    if dataset_config.data_format == "odex":
        intent_column, prompt_column = dataset_config.data_column.split()
        prompts = [
            build_input_from_intents_and_prompts(x[intent_column], x[prompt_column])
            for x in loaded_data
        ]
        (
            start_column,
            test_column,
            entry_column,
            canonical_column,
        ) = dataset_config.label_column.split()
        tests = [
            build_test(x[start_column], x[test_column], x[entry_column])
            for x in loaded_data
        ]
        canonicals = [x[canonical_column] for x in loaded_data]
        suffixes = [x["suffix"] for x in loaded_data]

        data_examples = [
            {
                "input": p,
                "test": t,
                "canonical": c,
                "suffix": s,
                "dataset": dataset_preset,
            }
            for p, t, c, s in zip(prompts, tests, canonicals, suffixes)
        ]
    elif dataset_config.data_format == "humaneval":
        test_column, canonical_column = dataset_config.label_column.split()
        data_examples = [
            {
                "input": x[dataset_config.data_column],
                "test": x[test_column],
                "canonical": x[canonical_column],
                "suffix": "",
                "dataset": dataset_preset,
            }
            for x in loaded_data
        ]
    else:
        raise ValueError(f"Unknown data format {dataset_config.data_format}")
    return data_examples


def process_data(
    dataset_preset: str,
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
        The loaded dataset as code examples of input prompts and reference snippets.
    """
    # Load from cache and return if existing
    parameters = {k: v for k, v in locals().items() if k != "output_dir"}
    output_path = get_cache_path(output_dir, parameters, "jsonl")
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            return [json.loads(x.strip()) for x in f]

    if dataset_preset == "all_datasets":
        data_examples = []
        for dataset_preset in codegen_config.dataset_configs.keys():
            data_examples.extend(_load_single_dataset(dataset_preset))
    else:
        data_examples = _load_single_dataset(dataset_preset)

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
) -> tuple[str, list[str]] | None:
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
        output_dir: The directory to save the predictions to.

    Returns:
        A tuple of:
        - The system id
        - The predictions in string format.
    """
    # Load from cache if existing
    parameters = {k: v for k, v in locals().items() if k not in {"data", "output_dir"}}
    system_id, file_root = get_cache_id_and_path(output_dir, parameters)
    if os.path.exists(f"{file_root}.json"):
        with open(f"{file_root}.json", "r") as f:
            return system_id, json.load(f)

    prompt_template = codegen_config.prompt_text[prompt_preset]
    model_config = codegen_config.model_configs[model_preset]

    with CacheLock(file_root) as cache_lock:
        # If the cache is locked, then another process is already generating
        # so just skip this one
        if not cache_lock:
            return None
        # Make predictions
        try:
            predictions: list[str] = generate_from_text_prompt(
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

    return system_id, predictions
