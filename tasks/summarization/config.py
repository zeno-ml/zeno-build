"""Various configuration options for the chatbot task.

This file is intended to be modified. You can go in and change any
of the variables to run different experiments.
"""

from __future__ import annotations

from typing import Any

from zeno_build import search_space
from zeno_build.evaluation.text_features.length import input_length, output_length
from zeno_build.evaluation.text_metrics.critique import (
    avg_chrf,
    avg_length_ratio,
    avg_rouge_1,
    avg_rouge_2,
    avg_rouge_l,
    chrf,
    length_ratio,
    rouge_1,
    rouge_2,
    rouge_l,
)
from zeno_build.models.lm_config import LMConfig

# Define the space of hyperparameters to search over.
space = {
    "prompt_preset": search_space.Categorical(
        ["standard", "tldr", "concise", "complete"]
    ),
    "model_preset": search_space.Categorical(
        ["openai_davinci_003", "openai_gpt_3.5_turbo", "cohere_command_xlarge"]
    ),
    "temperature": search_space.Discrete([0.2, 0.3, 0.4]),
}

# Any constants that are fed into the function
constants: dict[str, Any] = {
    "test_dataset": ("cnn_dailymail", "3.0.0"),
    "test_split": "test",
    "test_examples": 3,
    "max_tokens": 100,
    "top_p": 1.0,
}

# The number of trials to run
num_trials = 2

# The details of each model
model_configs = {
    "openai_davinci_003": LMConfig(provider="openai", model="text-davinci-003"),
    "openai_gpt_3.5_turbo": LMConfig(provider="openai_chat", model="gpt-3.5-turbo"),
    "cohere_command_xlarge": LMConfig(
        provider="cohere", model="command-xlarge-nightly"
    ),
}

# The details of the prompts
prompt_text = {
    "standard": "Summarize the following text:\n{{source}}\n\nSummary:",
    "tldr": "{{source}}\nTL;DR:",
    "concise": "Write a short and concise summary of the following text:\n{{source}}\n\nSummary:",  # noqa: E501
    "complete": "Write a complete summary of the following text:\n{{source}}\n\nSummary:",  # noqa: E501
}

# The functions to use to calculate scores for the hyperparameter sweep
sweep_distill_functions = [chrf]
sweep_metric_function = avg_chrf

# The functions used for Zeno visualization
zeno_distill_and_metric_functions = [
    input_length,
    output_length,
    avg_chrf,
    avg_length_ratio,
    avg_rouge_1,
    avg_rouge_2,
    avg_rouge_l,
    chrf,
    length_ratio,
    rouge_1,
    rouge_2,
    rouge_l,
]

# Some metadata to standardize huggingface datasets
dataset_mapping: dict[str | tuple[str, str], Any] = {
    ("cnn_dailymail", "3.0.0"): {
        "data_column": "article",
        "label_column": "highlights",
    },
}
