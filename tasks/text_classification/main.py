"""The main entry point for performing comparison on text classification.

If you want to adapt this, look at:
1. The "space" variable in this file which defines the training data,
    model, and hyperparameters to search over.
2. The "constants" variable in this file which defines the test data.
3. The "modeling.py" file which defines the function that trains and
    makes predictions.
4. The "num_trials" variable in this file which defines the number of
    trials to run.
"""

import argparse
import json
import os

from llm_compare import search_space
from llm_compare.evaluators import accuracy
from llm_compare.optimizers import standard

from . import modeling


def text_classification_main(
    results_dir: str,
):
    """Run the text classification experiment."""
    # Define the space of hyperparameters to search over
    space = {
        "training_dataset": search_space.Categorical(["imdb", "ag_news"]),
        "base_model": search_space.Categorical(["distilbert-base-uncased"]),
        "learning_rate": search_space.Float(1e-5, 1e-3),
        "num_train_epochs": search_space.Int(0, 4),
        "weight_decay": search_space.Float(0.0, 0.01),
        "bias": search_space.Float(-1.0, 1.0),
    }

    # Any constants that are fed into the function
    constants = {
        "test_dataset": "imdb",
        "training_split": "train",
        "validation_split": "validation",
        "test_split": "test",
    }

    # Get the reference answers and create an evaluator for accuracy
    references = modeling.get_references(
        constants["test_dataset"], constants["test_split"]
    )
    evaluator = accuracy.AccuracyEvaluator(references)

    # Run the hyperparameter sweep
    optimizer = standard.StandardOptimizer()
    result = optimizer.run_sweep(
        function=modeling.train_and_predict,
        space=space,
        constants=constants,
        evaluator=evaluator,
        num_trials=10,
    )
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(result, f)

    # Print the best result
    raise NotImplementedError("Perform analysis/visualization on the results.")


if __name__ == "__main__":

    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="The directory to store the results in.",
    )
    args = parser.parse_args()

    text_classification_main(
        results_dir=args.results_dir,
    )
