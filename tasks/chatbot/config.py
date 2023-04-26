"""Various configuration options for the chatbot task.

This file is intended to be modified. You can go in and change any
of the variables to run different experiments.
"""

from __future__ import annotations

from typing import Any

from llm_compare import search_space
from llm_compare.evaluation.text_features.length import input_length, output_length
from llm_compare.evaluation.text_metrics.critique import (
    avg_chrf,
    avg_length_ratio,
    avg_toxicity,
    chrf,
    length_ratio,
    toxicity,
)
from llm_compare.models.api_based_model import ApiBasedModelConfig
from llm_compare.prompts.chat_prompt import ChatMessages, ChatTurn

# Define the space of hyperparameters to search over.
# Note that "prompt_preset" and "model_preset" are in prompt_configs.py
# and model_configs.py respectively.
space = {
    "prompt_preset": search_space.Categorical(
        ["standard", "friendly", "polite", "cynical"]
    ),
    "model_preset": search_space.Categorical(
        ["openai_davinci_003", "openai_gpt_3.5_turbo", "cohere_command_xlarge"]
    ),
    "temperature": search_space.Discrete([0.2, 0.3, 0.4]),
}

# Any constants that are not searched over
constants: dict[str, Any] = {
    "test_dataset": "daily_dialog",
    "test_split": "validation",
    "test_examples": 40,
    "max_tokens": 100,
    "top_p": 1.0,
}

# The number of trials to run
num_trials = 10

# The details of each model
model_configs = {
    "openai_davinci_003": ApiBasedModelConfig(
        provider="openai", model="text-davinci-003"
    ),
    "openai_gpt_3.5_turbo": ApiBasedModelConfig(
        provider="openai_chat", model="gpt-3.5-turbo"
    ),
    "cohere_command_xlarge": ApiBasedModelConfig(
        provider="cohere", model="command-xlarge-nightly"
    ),
}

# The details of the prompts
prompt_messages: dict[str, ChatMessages] = {
    "standard": ChatMessages(
        messages=[
            ChatTurn(
                role="system",
                content="You are a chatbot tasked with making small-talk with "
                "people.",
            ),
            ChatTurn(role="system", content="{{context}}"),
            ChatTurn(role="user", content="{{source}}"),
        ]
    ),
    "friendly": ChatMessages(
        messages=[
            ChatTurn(
                role="system",
                content="You are a kind and friendly chatbot tasked with making "
                "small-talk with people in a way that makes them feel "
                "pleasant.",
            ),
            ChatTurn(role="system", content="{{context}}"),
            ChatTurn(role="user", content="{{source}}"),
        ]
    ),
    "polite": ChatMessages(
        messages=[
            ChatTurn(
                role="system",
                content="You are an exceedingly polite chatbot that speaks very "
                "formally and tries to not make any missteps in your "
                "responses.",
            ),
            ChatTurn(role="system", content="{{context}}"),
            ChatTurn(role="user", content="{{source}}"),
        ]
    ),
    "cynical": ChatMessages(
        messages=[
            ChatTurn(
                role="system",
                content="You are a cynical chatbot that has a very dark view of the "
                "world and in general likes to point out any possible "
                "problems.",
            ),
            ChatTurn(role="system", content="{{context}}"),
            ChatTurn(role="user", content="{{source}}"),
        ]
    ),
}

# The functions to use to calculate scores for the hyperparameter sweep
sweep_distill_functions = [chrf]
sweep_metric_function = avg_chrf

# The functions used for Zeno visualization
zeno_distill_and_metric_functions = [
    output_length,
    input_length,
    avg_chrf,
    chrf,
    avg_length_ratio,
    length_ratio,
    avg_toxicity,
    toxicity,
]

# Some metadata to standardize huggingface datasets
dataset_mapping: dict[str | tuple[str, str], Any] = {
    "daily_dialog": {
        "data_column": "dialog",
        "data_format": "sequence",
    },
}
