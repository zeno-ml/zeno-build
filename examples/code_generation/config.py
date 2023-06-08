"""Various configuration options for the code generation task.

This file is intended to be modified. You can go in and change any
of the variables to run different experiments.
"""

from __future__ import annotations

import transformers

from zeno_build.evaluation.text_features.exact_match import avg_exact_match, exact_match
from zeno_build.evaluation.text_features.length import (
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
from zeno_build.evaluation.text_metrics.huggingface import (
    execution_accuracy,
    avg_execution_accuracy,
)
from zeno_build.experiments import search_space
from zeno_build.models.dataset_config import DatasetConfig
from zeno_build.models.lm_config import LMConfig


# Define the space of hyperparameters to search over.
space = search_space.CombinatorialSearchSpace(
    {
        "dataset_preset": search_space.Constant("odex"),
        "model_preset": search_space.Categorical(
            [
                # "gpt-3.5-turbo",
                # "cohere-command-xlarge",
                # "gpt2",
                # "gpt2-xl",
                # "llama-7b",
                # "alpaca-7b",
                "codegen-350M-mono"
            ]
        ),
        "prompt_preset": search_space.Categorical(
            ["standard"]
        ),
        "temperature": search_space.Discrete([0.8]),
        "max_tokens": search_space.Constant(512),
        "top_p": search_space.Constant(0.95),
    }
)

# The number of trials to run
num_trials = 1

# The details of each dataset
dataset_configs = {
    "odex": DatasetConfig(
        dataset="neulab/odex",
        split="test",
        data_column=["intent", "prompt"],
        label_column=["test_start", "test", "entry_point"],
        data_format="odex",
    ),
    "odex_lexical": DatasetConfig(
        dataset="neulab/odex",
        split="test",
        data_column=["intent", "prompt"],
        label_column="canonical_solution",
        data_format="odex",
    ),
    "odex-es": DatasetConfig(
        dataset=["neulab/odex", "es"],
        split="test",
        data_column=["intent", "prompt"],
        label_column=["test_start", "test", "entry_point"],
        data_format="odex",
    ), # TODO: add other languages
    "odex-ja": DatasetConfig(
        dataset=["neulab/odex", "ja"],
        split="test",
        data_column=["intent", "prompt"],
        label_column=["test_start", "test", "entry_point"],
        data_format="odex",
    ),
    "odex-ru": DatasetConfig(
        dataset=["neulab/odex", "ru"],
        split="test",
        data_column=["intent", "prompt"],
        label_column=["test_start", "test", "entry_point"],
        data_format="odex",
    ),
    "humaneval": DatasetConfig(
        dataset="openai_humaneval",
        split="test",
        data_column="prompt",
        label_column="test",
        data_format="humaneval",
    ),
    "humaneval_lexical": DatasetConfig(
        dataset="openai_humaneval",
        split="test",
        data_column="prompt",
        label_column="canonical_solution",
        data_format="humaneval",
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
    "codegen-350M-mono": LMConfig(
        provider="huggingface",
        model="Salesforce/codegen-350M-mono",
    )
}

# The details of the prompts
prompt_text = {
    "standard": "Generate Python code solution for the following intent:\n{{source}}\n\n",
}

# The functions to use to calculate scores for the hyperparameter sweep
# sweep_distill_functions = [chrf]
# sweep_metric_function = avg_chrf
sweep_distill_functions = [execution_accuracy]
sweep_metric_function = avg_execution_accuracy

# The functions used for Zeno visualization
zeno_distill_and_metric_functions = [
    output_length,
    input_length,
    label_length,
    chrf,
    length_ratio,
    bert_score,
    exact_match,
    avg_chrf,
    avg_length_ratio,
    avg_bert_score,
    avg_exact_match,
]
