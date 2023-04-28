"""Various configuration options for the chatbot task.

This file is intended to be modified. You can go in and change any
of the variables to run different experiments.
"""

from __future__ import annotations

from llm_compare import search_space
from llm_compare.evaluation.classification_metrics import accuracy
from llm_compare.evaluation.text_features.length import input_length, output_length

# Define the space of hyperparameters to search over
space = {
    "training_dataset": search_space.Categorical(["imdb", "sst2"]),
    "base_model": search_space.Categorical(
        ["distilbert-base-uncased", "bert-base-uncased"]
    ),
    "learning_rate": search_space.Float(1e-5, 1e-3),
    "num_train_epochs": search_space.Int(1, 4),
    "weight_decay": search_space.Float(0.0, 0.01),
    "bias": search_space.Float(-1.0, 1.0),
}

# Any constants that are fed into the function
constants = {
    "training_split": "train",
    "training_examples": 50,
    "test_dataset": "imdb",
    "test_split": "test",
    "test_examples": 50,
}

# The number of trials to run
num_trials = 2

# The functions to use to calculate scores for the hyperparameter sweep
sweep_metric_function = accuracy

# The functions used for Zeno visualization
zeno_distill_and_metric_functions = [
    input_length,
    output_length,
    accuracy,
]

# Some metadata to standardize huggingface datasets
dataset_mapping = {
    "imdb": {
        "label_mapping": ["negative", "positive"],
        "input": "text",
    },
    "sst2": {
        "label_mapping": ["negative", "positive"],
        "input": "sentence",
    },
}
