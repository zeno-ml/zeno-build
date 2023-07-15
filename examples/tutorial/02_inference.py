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
    dataset = load_dataset("glue", "sst2")
    data = list(dataset["test"]["sentence"])
    label_map = {0: "negative", 1: "positive", -1: "neutral"}
    labels = [label_map[label] for label in dataset["test"]["label"]]
    df = pd.DataFrame({"text": data, "label": labels})

    # Perform inference
    all_results = []
    for lm_config in [
        LMConfig(provider="openai_chat", model="gpt-3.5-turbo"),
        LMConfig(provider="huggingface", model="gpt2"),
    ]:
        predictions = generate_from_text_prompt(
            [{"text": x} for x in data],
            prompt_template=(
                "Review: {{text}}\n\n"
                "Please answer with a single word. "
                "Is this review 'positive', 'negative', or 'neutral'?\n\n"
            ),
            model_config=lm_config,
            temperature=0.0001,
            max_tokens=1,
            top_p=1.0,
        )
        result = ExperimentRun(
            name=lm_config.model,
            parameters={"provider": lm_config.provider, "model": lm_config.model},
            predictions=predictions,
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
