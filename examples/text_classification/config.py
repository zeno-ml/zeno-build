"""Various configuration options for the chatbot task.

This file is intended to be modified. You can go in and change any
of the variables to run different experiments.
"""

from __future__ import annotations

from zeno_build.evaluation.text_features.exact_match import (avg_exact_match,
                                                             exact_match)
from zeno_build.evaluation.text_features.length import (input_length,
                                                        output_length)
from zeno_build.experiments import search_space
from zeno_build.models.dataset_config import DatasetConfig

# Define the space of hyperparameters to search over
space = search_space.CombinatorialSearchSpace(
    {
        "test_dataset_preset": search_space.Constant("imdb_test"),
        "training_dataset_preset": search_space.Categorical(["imdb", "sst2"]),
        "model_preset": search_space.Categorical(
            ["distilbert-base-uncased", "bert-base-uncased"]
        ),
        # "learning_rate": search_space.Float(1e-5, 1e-3),
        "learning_rate": search_space.Discrete([1e-5, 1e-4, 1e-3]),
        "num_train_epochs": search_space.Int(1, 4),
        # "weight_decay": search_space.Float(0.0, 0.01),
        "weight_decay": search_space.Constant(0.0),
        # "bias": search_space.Float(-1.0, 1.0),
        "bias": search_space.Discrete([-0.1, 0.0, 0.1]),
    }
)

# The details of each dataset
dataset_configs = {
    "imdb": DatasetConfig(
        dataset="imdb",
        split="train",
        data_column="text",
        label_column="label",
        label_mapping=["negative", "positive"],
    ),
    "imdb_test": DatasetConfig(
        dataset="imdb",
        split="test",
        data_column="text",
        label_column="label",
        label_mapping=["negative", "positive"],
    ),
    "sst2": DatasetConfig(
        dataset="sst2",
        split="train",
        data_column="sentence",
        label_column="label",
        label_mapping=["negative", "positive"],
    ),
}

# The number of trials to run
# num_trials = 10
num_trials = 100

# The functions to use to calculate scores for the hyperparameter sweep
sweep_distill_functions = [exact_match]
sweep_metric_function = avg_exact_match

# The functions used for Zeno visualization
zeno_distill_and_metric_functions = [
    output_length,
    input_length,
    exact_match,
    avg_exact_match,
]
