"""A tutorial of how to perform inference with Zeno Build."""

import pandas as pd
from datasets import load_dataset

from zeno_build.evaluation.text_features.exact_match import avg_exact_match, exact_match
from zeno_build.evaluation.text_features.length import input_length, output_length
from zeno_build.experiments.experiment_run import ExperimentRun
from zeno_build.models.lm_config import LMConfig
from zeno_build.models.text_generate import generate_from_text_prompt
from zeno_build.reporting.visualize import visualize


def main():
    """Run the inference example."""
    # Get data from hugging face
    dataset = load_dataset("glue", "sst2", split="validation")
    data = list(dataset["sentence"])
    label_map = dataset.features["label"].names
    labels = [label_map[label] for label in dataset["label"]]
    df = pd.DataFrame({"text": data, "label": labels})

    # Prompt templates for text or chat
    prompt_templates = {
        "huggingface": (
            "Review: {{text}}\n\n"
            "Q: Is this review a negative or positive review?\n\nA: It is a"
        ),
        "openai_chat": (
            "Review: {{text}}\n\n"
            "Please answer with one word. "
            "Is this review a negative or positive review?"
        ),
    }

    # Perform inference
    all_results = []
    for lm_config in [
        LMConfig(provider="openai_chat", model="gpt-3.5-turbo"),
        LMConfig(provider="huggingface", model="gpt2"),
        LMConfig(provider="huggingface", model="gpt2-xl"),
    ]:
        predictions = generate_from_text_prompt(
            [{"text": x} for x in data],
            prompt_template=prompt_templates[lm_config.provider],
            model_config=lm_config,
            temperature=0.0001,
            max_tokens=1,
            top_p=1.0,
            requests_per_minute=400,
        )
        result = ExperimentRun(
            name=lm_config.model,
            parameters={"provider": lm_config.provider, "model": lm_config.model},
            predictions=[x.strip().lower() for x in predictions],
        )
        all_results.append(result)

    functions = [
        output_length,
        input_length,
        exact_match,
        avg_exact_match,
    ]

    visualize(
        df,
        labels,
        all_results,
        "text-classification",
        "text",
        functions,
        zeno_config={"cache_path": "zeno_cache"},
    )


if __name__ == "__main__":
    main()
