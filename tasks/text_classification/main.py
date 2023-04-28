"""The main entry point for performing comparison on text classification."""

import argparse
import json
import os
from dataclasses import asdict

import modeling

from llm_compare.experiment_run import ExperimentRun
from llm_compare.optimizers import standard
from llm_compare.visualize import visualize
from tasks.text_classification import config as classification_config


def text_classification_main(
    results_dir: str,
    cached_runs: str | None = None,
):
    """Run the text classification experiment."""
    # Make results dir if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load the necessary data, either from HuggingFace or a cached file
    test_dataset = (classification_config.constants.pop("test_dataset"),)
    dataset = modeling.load_data(
        test_dataset,
        classification_config.constants.pop("test_split"),
        examples=classification_config.constants.pop("test_examples"),
    )
    with open(os.path.join(results_dir, "examples.json"), "w") as f:
        json.dump([asdict(x) for x in dataset], f)
    labels = modeling.get_labels(test_dataset, dataset)

    # Run the hyperparameter sweep and print out results
    if cached_runs is not None:
        with open(cached_runs, "r") as f:
            serialized_results = json.load(f)
        results = [ExperimentRun(**x) for x in serialized_results]
    else:
        optimizer = standard.StandardOptimizer()
        results = optimizer.run_sweep(
            function=modeling.train_and_predict,
            space=classification_config.space,
            constants=classification_config.constants,
            data=None,
            labels=labels,
            distill_functions=[],
            metric=classification_config.sweep_metric_function,
            num_trials=classification_config.num_trials,
            results_dir=results_dir,
        )

        # Print out results
        serialized_results = [asdict(x) for x in results]
        with open(os.path.join(results_dir, "all_runs.json"), "w") as f:
            json.dump(serialized_results, f)

    visualize(
        dataset,
        labels,
        results,
        "text-classification",
        "text",
        classification_config.zeno_distill_and_metric_functions,
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
    parser.add_argument(
        "--cached_runs",
        type=str,
        default=None,
        help="A path to a json file with cached runs.",
    )
    args = parser.parse_args()

    text_classification_main(
        results_dir=args.results_dir,
        cached_runs=args.cached_runs,
    )
