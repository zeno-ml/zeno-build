"""Tests for the exhaustive optimizer."""

import pytest

from zeno_build.evaluation.text_features.length import input_length
from zeno_build.experiments import search_space
from zeno_build.optimizers.exhaustive import ExhaustiveOptimizer


def test_exhaustive_optimizer():
    """Test to make sure outputs from the exhaustive optimizer are as expected."""
    space = search_space.CombinatorialSearchSpace(
        {
            "training_dataset": search_space.Categorical(["imdb", "sst2"]),
            "num_train_epochs": search_space.Int(1, 3),
            "training_split": search_space.Constant("train"),
            "training_examples": search_space.Discrete([100, 1000, 10000]),
        }
    )

    all_params = []
    for training_dataset in ["imdb", "sst2"]:
        for num_train_epochs in [1, 2, 3]:
            for training_examples in [100, 1000, 10000]:
                all_params.append(
                    {
                        "training_dataset": training_dataset,
                        "num_train_epochs": num_train_epochs,
                        "training_split": "train",
                        "training_examples": training_examples,
                    }
                )
    optimizer = ExhaustiveOptimizer(
        space=space, distill_functions=[], metric=input_length
    )

    all_param_choices = []
    while True:
        params = optimizer.get_parameters()
        if params is None:
            break
        all_param_choices.append(params)
    assert all_params == all_param_choices


def test_exhaustive_no_float():
    """Test to make sure the exhaustive optimizer rejects floats."""
    space = search_space.CombinatorialSearchSpace(
        {
            "training_dataset": search_space.Categorical(["imdb", "sst2"]),
            "num_train_epochs": search_space.Int(1, 4),
            "weight_decay": search_space.Float(0.0, 0.01),
            "training_split": search_space.Constant("train"),
            "training_examples": search_space.Discrete([100, 1000, 10000]),
        }
    )

    with pytest.raises(ValueError):
        _ = ExhaustiveOptimizer(space=space, distill_functions=[], metric=input_length)


def test_exhaustive_composite():
    """Test the exhaustive optimizer with composite search spaces."""
    space = search_space.CompositeSearchSpace(
        [
            search_space.CombinatorialSearchSpace(
                {
                    "training_dataset": search_space.Categorical(["a", "b"]),
                    "num_train_epochs": search_space.Int(1, 3),
                }
            ),
            search_space.CombinatorialSearchSpace(
                {
                    "training_dataset": search_space.Categorical(["c", "d"]),
                    "num_train_epochs": search_space.Int(2, 4),
                }
            ),
        ]
    )

    all_params = []
    for training_dataset in ["a", "b"]:
        for num_train_epochs in [1, 2, 3]:
            all_params.append(
                {
                    "training_dataset": training_dataset,
                    "num_train_epochs": num_train_epochs,
                }
            )
    for training_dataset in ["c", "d"]:
        for num_train_epochs in [2, 3, 4]:
            all_params.append(
                {
                    "training_dataset": training_dataset,
                    "num_train_epochs": num_train_epochs,
                }
            )
    optimizer = ExhaustiveOptimizer(
        space=space, distill_functions=[], metric=input_length
    )

    all_param_choices = []
    while True:
        params = optimizer.get_parameters()
        if params is None:
            break
        all_param_choices.append(params)
    assert all_params == all_param_choices
