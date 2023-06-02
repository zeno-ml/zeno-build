"""Tests for the random optimizer."""

import json
import random

from zeno_build.evaluation.text_features.length import input_length
from zeno_build.experiments import search_space
from zeno_build.optimizers.random import RandomOptimizer


def test_random_optimizer():
    """Test to make sure that the outputs from random optimizer are expected.

    1. The optimizer should generate unique parameter settings.
    2. The optimizer should not have side-effects on the random state.
    3. The optimizer should not be affected by the external state.
    """
    # Example from text classification
    space = search_space.CombinatorialSearchSpace(
        {
            "training_dataset": search_space.Categorical(["imdb", "sst2"]),
            "base_model": search_space.Categorical(
                ["distilbert-base-uncased", "bert-base-uncased"]
            ),
            "learning_rate": search_space.Float(1e-5, 1e-3),
            "num_train_epochs": search_space.Int(1, 4),
            "weight_decay": search_space.Float(0.0, 0.01),
            "bias": search_space.Float(-1.0, 1.0),
            "training_split": search_space.Constant("train"),
            "training_examples": search_space.Constant(50),
            "test_dataset": search_space.Constant("imdb"),
            "test_split": search_space.Constant("test"),
            "test_examples": search_space.Constant(50),
        }
    )

    optimizer = RandomOptimizer(space=space, distill_functions=[], metric=input_length)

    total_number = 100
    all_param_choices = []
    random.seed(123)
    external_state = random.getstate()
    for _ in range(total_number):
        random.seed(123)
        all_param_choices.append(optimizer.get_parameters())
        assert external_state == random.getstate()
    unique_param_jsons = set([json.dumps(x) for x in all_param_choices])
    assert total_number == len(unique_param_jsons)


def test_random_composite():
    """Check the composite space with weighting.

    This uses random sampling at a 0.2/0.8 ratio, which
    means that the first space should be sampled about 20%
    of the time.
    """
    space = search_space.CompositeSearchSpace(
        [
            search_space.CombinatorialSearchSpace({"a": search_space.Constant(1)}),
            search_space.CombinatorialSearchSpace({"a": search_space.Constant(2)}),
        ],
        weights=[0.2, 0.8],
    )

    optimizer = RandomOptimizer(space=space, distill_functions=[], metric=input_length)
    NUM_TRIALS = 10000
    num_first = 0
    for _ in range(NUM_TRIALS):
        if optimizer.get_parameters()["a"] == 1:
            num_first += 1
    assert 0.1 < num_first / NUM_TRIALS < 0.3
