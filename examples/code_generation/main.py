"""The main entry point for performing comparison on code lm."""

from __future__ import annotations

import argparse
import json
import logging
import os

import config as codegen_config
import pandas as pd
from modeling import make_predictions, process_data

from zeno_build.experiments import search_space
from zeno_build.experiments.experiment_run import ExperimentRun
from zeno_build.optimizers import standard
from zeno_build.reporting import reporting_utils
from zeno_build.reporting.visualize import visualize


def codegen_main(
    results_dir: str,
    do_prediction: bool = True,
    do_visualization: bool = True,
):
    """Run the chatbot experiment."""
    # Get the dataset configuration
    dataset_preset = codegen_config.space.dimensions["dataset_preset"]
    if not isinstance(dataset_preset, search_space.Constant):
        raise ValueError("All experiments must be run on a single dataset.")

    # Define the directories for storing data and predictions
    data_dir = os.path.join(results_dir, "data")
    predictions_dir = os.path.join(results_dir, "predictions")

    # Load and standardize the format of the necessary data. The resulting
    # processed data will be stored in the `results_dir/data` directory
    # both for browsing and for caching for fast reloading on future runs.
    inputs_and_labels: list[dict[str, str]] = process_data(
        dataset_preset=dataset_preset.value,
        output_dir=data_dir,
    )

    # Organize the data into labels (output) and context (input)
    inputs: list[str] = [d["input"] for d in inputs_and_labels]
    tests: list[str] = [d["test"] for d in inputs_and_labels]
    datasets: list[str] = [d["dataset"] for d in inputs_and_labels]
    suffixes: list[str] = [d["suffix"] for d in inputs_and_labels]

    if do_prediction:
        # Perform the hyperparameter sweep
        optimizer = standard.StandardOptimizer(
            space=codegen_config.space,
            distill_functions=codegen_config.sweep_distill_functions,
            metric=codegen_config.sweep_metric_function,
            num_trials=codegen_config.num_trials,
        )

        while not optimizer.is_complete(predictions_dir, include_in_progress=True):
            parameters = optimizer.get_parameters()
            if parameters is None:
                break
            id_and_predictions = make_predictions(
                data=inputs,
                prompt_preset=parameters["prompt_preset"],
                model_preset=parameters["model_preset"],
                temperature=parameters["temperature"],
                max_tokens=parameters["max_tokens"],
                top_p=parameters["top_p"],
                output_dir=predictions_dir,
            )
            if id_and_predictions is None:
                print(f"*** Skipped run for {parameters=} ***")
                continue
            # Run or read the evaluation result
            id, predictions = id_and_predictions
            predictions = [p + s for p, s in zip(predictions, suffixes)]
            if os.path.exists(f"{predictions_dir}/{id}.eval"):
                with open(f"{predictions_dir}/{id}.eval", "r") as f:
                    eval_result = float(next(f).strip())
            else:
                eval_result = optimizer.calculate_metric(inputs, tests, predictions)
                with open(f"{predictions_dir}/{id}.eval", "w") as f:
                    f.write(f"{eval_result}")
            print("*** Iteration complete. ***")
            print(f"Parameters: {parameters}")
            print(f"Eval: {eval_result}")
            print("***************************")

    if do_visualization:
        param_files = codegen_config.space.get_valid_param_files(
            predictions_dir, include_in_progress=False
        )
        if len(param_files) < codegen_config.num_trials:
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
                loaded_parameters, codegen_config.space
            )
            results.append(
                ExperimentRun(
                    parameters=loaded_parameters, predictions=predictions, name=name
                )
            )
        results.sort(key=lambda x: x.name)

        # Perform the visualization
        df = pd.DataFrame(
            {
                "data": inputs,
                "labels": tests,
                "dataset": datasets,
            }
        )
        visualize(
            df,
            tests,
            results,
            "code-generation",
            "data",
            codegen_config.zeno_distill_and_metric_functions,
            zeno_config={"cache_path": os.path.join(results_dir, "zeno_cache")},
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

    codegen_main(
        results_dir=args.results_dir,
        do_prediction=not args.skip_prediction,
        do_visualization=not args.skip_visualization,
    )
