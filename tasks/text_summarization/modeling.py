"""Text summarization using APIs"""

import json
import os
import tqdm
from typing import Any

import openai
import cohere
import datasets

from llm_compare.cache_utils import get_cache_path
from tasks.text_summarization import model_configs, prompt_configs

DATASET_MAPPING: dict[str, Any] = {
}

cohere_client: cohere.Client | None = None
    
def generate_one(
    source: str,
    prompt_template: str,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> str:
    prompt = prompt_template.replace("[X]", source)
    if provider == "openai":
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return response["choices"][0]["text"]
    elif provider == "openai_chat":
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return response["choices"][0]["message"]["content"]
    elif provider == "cohere":
        try:
            assert cohere_client is not None
            response = cohere_client.generate(  
                model=model,  
                prompt=prompt,
                temperature=temperature,  
                max_tokens=max_tokens,
                p=top_p, 
            )
            return response.generations[0].text
        except:
            # Cohere API sometimes rejects queries, if so output a blank line
            print(f"Warning! Cohere API rejected query for {prompt=}")
            return ""
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
        prompt_preset: The prompt to use for the API call, as specified in prompt_configs.
        model_preset: The model to use for the API call, as specified in model_configs.
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
    cache_path = get_cache_path("text_summarization", parameters, "json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    # Load dataset
    mapping = DATASET_MAPPING.get(test_dataset, {})
    input_name = mapping.get("input", "text")
    dataset = datasets.load_dataset(test_dataset, split=test_split)
    if test_examples is not None:
        dataset = dataset.select(range(test_examples))
    inputs = [example[input_name] for example in dataset]

    prompt_template = prompt_configs[prompt_preset]
    provider = model_configs[model_preset]["provider"]
    model = model_configs[model_preset]["model"]

    # Make predictions
    predictions = [
        generate_one(x, prompt_template, provider, model, temperature, max_tokens, top_p)
        for x in tqdm.tqdm(inputs, "Generating predictions")
    ]
    with open(cache_path, "w") as f:
        json.dump(predictions, f)
    return predictions


def get_references(
    test_dataset: str,
    test_split: str = "test",
    test_examples: int | None = None,
) -> list[dict[str, Any]]:
    """Get the reference answers for a particular dataset.

    Args:
        test_dataset: The path to the test dataset.
        test_split: The split of the test dataset to use.
        test_examples: The number of examples to use from the test dataset.

    Returns:
        The references in string format.
    """
    # Load dataset
    mapping = DATASET_MAPPING.get(test_dataset, {})
    output_name = mapping.get("output", "summary")
    input_name = mapping.get("output", "summary")
    dataset = datasets.load_dataset(test_dataset, split=test_split)
    if test_examples is not None:
        dataset = dataset.select(range(test_examples))
    return [{"source": example[input_name], "references": [example[output_name]]} for example in dataset]
