"""The main entry point for performing comparison on text classification."""

import argparse
import json
import os
from dataclasses import asdict

import modeling

from llm_compare import search_space
from llm_compare.evaluation import classification_metrics
from llm_compare.evaluators import accuracy
from llm_compare.optimizers import standard
from llm_compare.visualize import visualize


def text_classification_main(
    results_dir: str,
):
    """Run the text classification experiment."""
    # Make results dir if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

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
        "test_dataset": "imdb",
        "training_split": "train",
        "test_split": "test",
    }

    # Get the reference answers and create an evaluator for accuracy
    references = modeling.get_references(
        constants["test_dataset"], constants["test_split"]
    )

    if os.path.exists(os.path.join(results_dir, "all_runs.json")):
        with open(os.path.join(results_dir, "all_runs.json"), "r") as f:
            serialized_results = json.load(f)
    else:
        evaluator = accuracy.AccuracyEvaluator(references)
        with open(os.path.join(results_dir, "references.json"), "w") as f:
            json.dump(references, f)

        # Run the hyperparameter sweep and print out results
        optimizer = standard.StandardOptimizer()
        result = optimizer.run_sweep(
            function=modeling.train_and_predict,
            space=space,
            constants=constants,
            evaluator=evaluator,
            num_trials=10,
            results_dir=results_dir,
        )

        # Print out results
        serialized_results = [asdict(x) for x in result]
        with open(os.path.join(results_dir, "all_runs.json"), "w") as f:
            json.dump(serialized_results, f)

    dataset = modeling.get_dataset(constants["test_dataset"], constants["test_split"])

    visualize(
        dataset,
        references,
        serialized_results,
        "text-classification",
        "text",
        [classification_metrics.accuracy],
    )


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
