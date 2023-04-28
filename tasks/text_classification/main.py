"""The main entry point for performing comparison on text classification."""

from __future__ import annotations

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
    test_dataset = classification_config.constants["test_dataset"]
    data = modeling.load_data(
        test_dataset,
        classification_config.constants.pop("test_split"),
        examples=classification_config.constants.pop("test_examples"),
    )
    with open(os.path.join(results_dir, "examples.json"), "w") as f:
        json.dump(list(data), f)
    labels = modeling.get_labels(data, test_dataset)

    # Run the hyperparameter sweep and print out results
    results: list[ExperimentRun] = []
    if cached_runs is not None:
        with open(cached_runs, "r") as f:
            serialized_results = json.load(f)
        results = [ExperimentRun(**x) for x in serialized_results]
    else:
        # Perform the hyperparameter sweep
        optimizer = standard.StandardOptimizer(
            space=classification_config.space,
            constants=classification_config.constants,
            distill_functions=classification_config.sweep_distill_functions,
            metric=classification_config.sweep_metric_function,
        )
        for _ in range(classification_config.num_trials):
            parameters = optimizer.get_parameters()
            predictions = modeling.train_and_predict(
                data=data,
                test_dataset=test_dataset,
                training_dataset=parameters["training_dataset"],
                base_model=parameters["base_model"],
                learning_rate=parameters["learning_rate"],
                num_train_epochs=parameters["num_train_epochs"],
                weight_decay=parameters["weight_decay"],
                bias=parameters["bias"],
                training_split=parameters["training_split"],
                training_examples=parameters["training_examples"],
            )
            eval_result = optimizer.calculate_metric(test_dataset, labels, predictions)
            run = ExperimentRun(
                parameters=parameters,
                predictions=predictions,
                eval_result=eval_result,
            )
            results.append(run)

        # Print out results
        serialized_results = [asdict(x) for x in results]
        with open(os.path.join(results_dir, "all_runs.json"), "w") as f:
            json.dump(serialized_results, f)

    visualize(
        test_dataset.to_pandas(),
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
