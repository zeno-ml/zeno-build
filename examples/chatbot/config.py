"""Various configuration options for the chatbot task.

This file is intended to be modified. You can go in and change any
of the variables to run different experiments.
"""

from __future__ import annotations

import transformers

from zeno_build.evaluation.text_features.clustering import label_clusters
from zeno_build.evaluation.text_features.exact_match import avg_exact_match, exact_match
from zeno_build.evaluation.text_features.length import (
    chat_context_length,
    input_length,
    label_length,
    output_length,
)
from zeno_build.evaluation.text_features.numbers import english_number_count
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

# --- Model Configuration ---

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
    ),
    "gpt2-xl": LMConfig(
        provider="huggingface",
        model="gpt2-xl",
    ),
    # We need to use the transformers library instead of VLLM here
    # because the tokenizer library needs to be set manually
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
    "vicuna-7b-v1.3": LMConfig(
        provider="huggingface",
        model="lmsys/vicuna-7b-v1.3",
        name_replacements={
            "system": "ASSISTANT",
            "assistant": "ASSISTANT",
            "user": "HUMAN",
        },
    ),
    "vicuna-13b-v1.3": LMConfig(
        provider="huggingface",
        model="lmsys/vicuna-13b-v1.3",
        name_replacements={
            "system": "ASSISTANT",
            "assistant": "ASSISTANT",
            "user": "HUMAN",
        },
    ),
    "vicuna-33b-v1.3": LMConfig(
        provider="huggingface",
        model="lmsys/vicuna-33b-v1.3",
        name_replacements={
            "system": "ASSISTANT",
            "assistant": "ASSISTANT",
            "user": "HUMAN",
        },
    ),
    # We need to use huggingface instead of vllm here because we need to
    # set trust_remote_code to True
    "mpt-7b-chat": LMConfig(
        provider="huggingface",
        model="mosaicml/mpt-7b-chat",
        model_loader_kwargs={"trust_remote_code": True},
    ),
}

# These models are used by default in the experiments.
# This can be modified by using the "--models" command line argument.
default_models = [
    "gpt2",
    "gpt2-xl",
    "llama-7b",
    "vicuna-7b",
    "mpt-7b-chat",
]
# The default single model to use in experiments that don't iterate over
# multiple models.
default_single_model = "vicuna-7b"

# --- Dataset Configuration ---

# The details of each dataset
dataset_configs = {
    "dstc11": DatasetConfig(
        dataset="gneubig/dstc11",
        split="validation",
        data_column="turns",
        data_format="dstc11",
    ),
}

# --- Prompt Configuration ---

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
    # The following is purpose-tailored for the DSTC11 insurance dataset
    "insurance_upgrade_1": ChatMessages(
        messages=[
            ChatTurn(
                role="system",
                content="""\n
You are an agent at the Rivertown Insurance helpdesk that helps with resolving insurance
claims.

Make sure you introduce yourself appropriately, example:
> Assistant: Hello. Thank you for calling Rivertown Insurance. How can I help you?

When people provide numbers like their security number, make sure that you repeat the
number back to them to confirm that you have the correct number, example:
> User: Is the account number eight digit or ten digit?
> Assistant: It is eight digit.
> User: Okay. Four five.
> Assistant: Four five.""",
            ),
        ]
    ),
}

default_prompts = list(prompt_messages.keys())
# The default prompt to use in experiments that don't iterate over
# multiple prompts.
default_single_prompt = "standard"

# --- Other Hyperparameters ---

default_temperatures = [0.2, 0.3, 0.4]
default_single_temperature = 0.3

default_context_lengths = [1, 2, 3, 4, 6, 8]
default_single_context_length = 4

default_single_max_tokens = 100
default_single_max_p = 1.0

dataset = "dstc11"

# --- Evaluation/Feature Configuartion ---

# The functions to use to calculate scores for the hyperparameter sweep
sweep_distill_functions = [chrf]
sweep_metric_function = avg_chrf

# The functions used for Zeno visualization
zeno_distill_and_metric_functions = [
    output_length,
    input_length,
    label_length,
    chat_context_length,
    english_number_count,
    label_clusters,
    chrf,
    length_ratio,
    bert_score,
    exact_match,
    avg_chrf,
    avg_length_ratio,
    avg_bert_score,
    avg_exact_match,
]

# --- Experiment Configuration ---

# A bunch of different experiments that could be run. Which ones to run
# is controlled by the "--experiments" command line argument.
experiments = {
    # An exhaustive experiment that tests many different combinations
    "exhaustive": search_space.CombinatorialSearchSpace(
        {
            "model_preset": search_space.Categorical(default_models),
            "prompt_preset": search_space.Categorical(default_prompts),
            "temperature": search_space.Discrete(default_temperatures),
            "context_length": search_space.Discrete(default_context_lengths),
            "max_tokens": search_space.Constant(default_single_max_tokens),
            "top_p": search_space.Constant(default_single_max_p),
        }
    ),
    # An experiment that varies only the model
    "model": search_space.CombinatorialSearchSpace(
        {
            "model_preset": search_space.Categorical(default_models),
            "prompt_preset": search_space.Constant(default_single_prompt),
            "temperature": search_space.Constant(default_single_temperature),
            "context_length": search_space.Constant(default_single_context_length),
            "max_tokens": search_space.Constant(default_single_max_tokens),
            "top_p": search_space.Constant(default_single_max_p),
        }
    ),
    # An experiment that varies only the prompt
    "prompt": search_space.CombinatorialSearchSpace(
        {
            "model_preset": search_space.Constant(default_single_model),
            "prompt_preset": search_space.Categorical(default_prompts),
            "temperature": search_space.Constant(default_single_temperature),
            "context_length": search_space.Constant(default_single_context_length),
            "max_tokens": search_space.Constant(default_single_max_tokens),
            "top_p": search_space.Constant(default_single_max_p),
        }
    ),
    # An experiment that varies only the temperature
    "temperature": search_space.CombinatorialSearchSpace(
        {
            "model_preset": search_space.Constant(default_single_model),
            "prompt_preset": search_space.Constant(default_single_prompt),
            "temperature": search_space.Discrete(default_temperatures),
            "context_length": search_space.Constant(default_single_context_length),
            "max_tokens": search_space.Constant(default_single_max_tokens),
            "top_p": search_space.Constant(default_single_max_p),
        }
    ),
    # An experiment that varies only the context_length
    "context_length": search_space.CombinatorialSearchSpace(
        {
            "model_preset": search_space.Constant(default_single_model),
            "prompt_preset": search_space.Constant(default_single_prompt),
            "temperature": search_space.Constant(default_single_temperature),
            "context_length": search_space.Discrete(default_context_lengths),
            "max_tokens": search_space.Constant(default_single_max_tokens),
            "top_p": search_space.Constant(default_single_max_p),
        }
    ),
}

# The number of trials to run. If set to None, all combinations of experiments will be
# run.
num_trials: int | None = None
