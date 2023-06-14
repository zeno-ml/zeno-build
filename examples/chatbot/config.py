"""Various configuration options for the chatbot task.

This file is intended to be modified. You can go in and change any
of the variables to run different experiments.
"""

from __future__ import annotations

import transformers

from zeno_build.evaluation.text_features.exact_match import avg_exact_match, exact_match
from zeno_build.evaluation.text_features.length import (
    chat_context_length,
    input_length,
    label_length,
    output_length,
)
from zeno_build.evaluation.text_metrics.critique import (
    avg_bert_score,
    avg_chrf,
    avg_length_ratio,
    bert_score,
    chrf,
    length_ratio,
)
from zeno_build.experiments import search_space
from zeno_build.models.dataset_config import DatasetConfig
from zeno_build.models.lm_config import LMConfig
from zeno_build.prompts.chat_prompt import ChatMessages, ChatTurn

# Define the space of hyperparameters to search over if using
# hyperparameter search.
full_space = search_space.CombinatorialSearchSpace(
    {
        "dataset_preset": search_space.Constant("dstc11"),
        "model_preset": search_space.Categorical(
            [
                # "gpt-3.5-turbo",
                # "cohere-command-xlarge",
                "gpt2",
                "gpt2-xl",
                "llama-7b",
                "alpaca-7b",
                "vicuna-7b",
                "mpt-7b-chat",
            ]
        ),
        "prompt_preset": search_space.Categorical(
            ["standard", "friendly", "polite", "cynical"]
        ),
        "temperature": search_space.Discrete([0.2, 0.3, 0.4]),
        "context_length": search_space.Discrete([1, 2, 3, 4]),
        "max_tokens": search_space.Constant(100),
        "top_p": search_space.Constant(1.0),
    }
)

# Specifically, this is the space of hyperparameters used in the Zeno
# chatbot report on the DSTC11 dataset:
# https://github.com/zeno-ml/zeno-build/tree/main/examples/chatbot
# It can be used together with ExhaustiveOptimizer to reproduce the
# results in the report.
report_space = search_space.CompositeSearchSpace(
    [
        # Comparison of models
        search_space.CombinatorialSearchSpace(
            {
                "dataset_preset": search_space.Constant("dstc11"),
                "model_preset": search_space.Categorical(
                    [
                        "gpt-3.5-turbo",
                        "cohere-command-xlarge",
                        "gpt2",
                        "gpt2-xl",
                        "llama-7b",
                        "alpaca-7b",
                        "vicuna-7b",
                        "mpt-7b-chat",
                    ]
                ),
                "prompt_preset": search_space.Constant("standard"),
                "temperature": search_space.Constant(0.3),
                "context_length": search_space.Constant(4),
                "max_tokens": search_space.Constant(100),
                "top_p": search_space.Constant(1.0),
            }
        ),
        # Comparison of prompts
        search_space.CombinatorialSearchSpace(
            {
                "dataset_preset": search_space.Constant("dstc11"),
                "model_preset": search_space.Constant("vicuna-7b"),
                "prompt_preset": search_space.Categorical(
                    ["standard", "friendly", "polite", "cynical", "insurance_standard"]
                ),
                "temperature": search_space.Constant(0.3),
                "context_length": search_space.Constant(4),
                "max_tokens": search_space.Constant(100),
                "top_p": search_space.Constant(1.0),
            }
        ),
        # Comparison of context lengths
        search_space.CombinatorialSearchSpace(
            {
                "dataset_preset": search_space.Constant("dstc11"),
                "model_preset": search_space.Constant("vicuna-7b"),
                "prompt_preset": search_space.Constant("standard"),
                "temperature": search_space.Constant(0.3),
                "context_length": search_space.Discrete([1, 2, 3, 4]),
                "max_tokens": search_space.Constant(100),
                "top_p": search_space.Constant(1.0),
            }
        ),
    ]
)

# The number of trials to run
num_trials = 15

# The details of each dataset
dataset_configs = {
    "dstc11": DatasetConfig(
        dataset="gneubig/dstc11",
        split="validation",
        data_column="turns",
        data_format="dstc11",
    ),
}

# The details of each model
model_configs = {
    "text-davinci-003": LMConfig(provider="openai", model="text-davinci-003"),
    "gpt-3.5-turbo": LMConfig(provider="openai_chat", model="gpt-3.5-turbo"),
    "cohere-command-xlarge": LMConfig(
        provider="cohere", model="command-xlarge-nightly"
    ),
    "gpt2": LMConfig(
        provider="huggingface",
        model="gpt2",
        model_cls=transformers.GPT2LMHeadModel,
    ),
    "gpt2-xl": LMConfig(
        provider="huggingface",
        model="gpt2-xl",
        model_cls=transformers.GPT2LMHeadModel,
    ),
    "llama-7b": LMConfig(
        provider="huggingface",
        model="decapoda-research/llama-7b-hf",
        tokenizer_cls=transformers.LlamaTokenizer,
    ),
    "llama-13b": LMConfig(
        provider="huggingface",
        model="decapoda-research/llama-13b-hf",
        tokenizer_cls=transformers.LlamaTokenizer,
    ),
    "alpaca-7b": LMConfig(
        provider="huggingface",
        model="chavinlo/alpaca-native",
    ),
    "alpaca-13b": LMConfig(
        provider="huggingface",
        model="chavinlo/alpaca-13b",
    ),
    "vicuna-7b": LMConfig(
        provider="huggingface",
        model="eachadea/vicuna-7b-1.1",
        name_replacements={
            "system": "ASSISTANT",
            "assistant": "ASSISTANT",
            "user": "HUMAN",
        },
    ),
    "vicuna-13b": LMConfig(
        provider="huggingface",
        model="eachadea/vicuna-13b-1.1",
        name_replacements={
            "system": "ASSISTANT",
            "assistant": "ASSISTANT",
            "user": "HUMAN",
        },
    ),
    "mpt-7b-chat": LMConfig(
        provider="huggingface",
        model="mosaicml/mpt-7b-chat",
        model_loader_kwargs={"trust_remote_code": True},
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
        ]
    ),
    # The following is purpose-tailored for the DSTC11 insurance dataset
    "insurance_standard": ChatMessages(
        messages=[
            ChatTurn(
                role="system",
                content="You are an agent at the Rivertown Insurance helpdesk that "
                "mainly helps with resolving insurance claims.",
            ),
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
    label_length,
    chat_context_length,
    chrf,
    length_ratio,
    bert_score,
    exact_match,
    avg_chrf,
    avg_length_ratio,
    avg_bert_score,
    avg_exact_match,
]
