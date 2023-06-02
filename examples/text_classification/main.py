"""The main entry point for performing comparison on text classification."""

from __future__ import annotations

import argparse
import json
import logging
import os

import datasets
import pandas as pd

from examples.text_classification import config as text_classification_config
from examples.text_classification.modeling import load_data, train_and_predict
from zeno_build.experiments import search_space
from zeno_build.experiments.experiment_run import ExperimentRun
from zeno_build.optimizers import standard
from zeno_build.reporting import reporting_utils
from zeno_build.reporting.visualize import visualize


def text_classification_main(
    results_dir: str,
    do_prediction: bool = True,
    do_visualization: bool = True,
):
    """Run the text classification experiment."""
    # Get the dataset configuration
    test_dataset_dim = text_classification_config.space.dimensions[
        "test_dataset_preset"
    ]
    if not isinstance(test_dataset_dim, search_space.Constant):
        raise ValueError("All experiments must be run on a single dataset.")
    test_dataset_preset = test_dataset_dim.value

    models_dir = os.path.join(results_dir, "models")
    predictions_dir = os.path.join(results_dir, "predictions")

    # Load the necessary data
    test_dataset_config = text_classification_config.dataset_configs[
        test_dataset_preset
    ]
    test_dataset: datasets.Dataset = load_data(test_dataset_preset)

    # Organize the data into labels (output) and context (input)
    test_data: list[str] = [x[test_dataset_config.data_column] for x in test_dataset]
    test_labels: list[str] = [
        test_dataset_config.label_mapping[x[test_dataset_config.label_column]]
        for x in test_dataset
    ]

    if do_prediction:
        # Perform the hyperparameter sweep
        optimizer = standard.StandardOptimizer(
            space=text_classification_config.space,
            distill_functions=text_classification_config.sweep_distill_functions,
            metric=text_classification_config.sweep_metric_function,
            num_trials=text_classification_config.num_trials,
        )

        while not optimizer.is_complete(predictions_dir, include_in_progress=True):
            parameters = optimizer.get_parameters()
            if parameters is None:
                break
            predictions = train_and_predict(
                test_data=test_dataset,
                test_dataset_preset=test_dataset_preset,
                training_dataset_preset=parameters["training_dataset_preset"],
                model_preset=parameters["model_preset"],
                learning_rate=parameters["learning_rate"],
                num_train_epochs=parameters["num_train_epochs"],
                weight_decay=parameters["weight_decay"],
                bias=parameters["bias"],
                models_dir=models_dir,
                predictions_dir=predictions_dir,
            )
            if predictions is None:
                print(f"*** Skipped run for {parameters=} ***")
                continue
            eval_result = optimizer.calculate_metric(
                test_data, test_labels, predictions
            )
            print("*** Iteration complete. ***")
            print(f"Parameters: {parameters}")
            print(f"Eval: {eval_result}")
            print("***************************")

    if do_visualization:
        param_files = text_classification_config.space.get_valid_param_files(
            predictions_dir, include_in_progress=False
        )
        if len(param_files) < text_classification_config.num_trials:
            logging.getLogger().warning(
                "Not enough completed but performing visualization anyway."
            )
        results: list[ExperimentRun] = []
        for param_file in param_files:
            assert param_file.endswith(".zbp")
            with open(param_file, "r") as f:
                loaded_parameters = json.load(f)
            with open(f"{param_file[:-4]}.json", "r") as f:
                predictions = json.load(f)
            name = reporting_utils.parameters_to_name(
                loaded_parameters, text_classification_config.space
            )
            results.append(
                ExperimentRun(
                    parameters=loaded_parameters, predictions=predictions, name=name
                )
            )

        # Perform the visualization
        df = pd.DataFrame(
            {
                "text": test_data,
                "label": test_labels,
            }
        )
        visualize(
            df,
            test_labels,
            results,
            "text-classification",
            "text",
            text_classification_config.zeno_distill_and_metric_functions,
        )


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="The directory to store the results in.",
    )
    parser.add_argument(
        "--skip-prediction",
        action="store_true",
        help="Skip prediction and just do visualization.",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualization and just do prediction.",
    )
    args = parser.parse_args()

    if args.skip_prediction and args.skip_visualization:
        raise ValueError(
            "Cannot specify both --skip-prediction and --skip-visualization."
        )

    text_classification_main(
        results_dir=args.results_dir,
        do_prediction=not args.skip_prediction,
        do_visualization=not args.skip_visualization,
    )
