"""The main entry point for performing comparison on summarizations."""

from __future__ import annotations

import argparse
import json
import logging
import os

import config as summarization_config
import pandas as pd
from modeling import load_data, make_predictions

from zeno_build.experiments import search_space
from zeno_build.experiments.experiment_run import ExperimentRun
from zeno_build.optimizers import standard
from zeno_build.reporting import reporting_utils
from zeno_build.reporting.visualize import visualize


def summarization_main(
    results_dir: str,
    do_prediction: bool = True,
    do_visualization: bool = True,
):
    """Run the summarization experiment."""
    # Get the dataset configuration
    dataset_preset = summarization_config.space.dimensions["dataset_preset"]
    if not isinstance(dataset_preset, search_space.Constant):
        raise ValueError("All experiments must be run on a single dataset.")
    dataset_config = summarization_config.dataset_configs[dataset_preset.value]

    data_dir = os.path.join(results_dir, "data")
    predictions_dir = os.path.join(results_dir, "predictions")

    # Load and standardize the format of the necessary data. The resulting
    # processed data will be stored in the `results_dir/data` directory
    # both for browsing and for caching for fast reloading on future runs.
    data_and_labels: list[dict[str, str]] = load_data(
        dataset=dataset_config.dataset,
        split=dataset_config.split,
        data_format=dataset_config.data_format,
        data_column=dataset_config.data_column,
        output_dir=data_dir,
    )

    # Organize the data into labels (output) and context (input)
    data: list[str] = [d["data"] for d in data_and_labels]
    labels: list[str] = [d["label"] for d in data_and_labels]

    if do_prediction:
        # Perform the hyperparameter sweep
        optimizer = standard.StandardOptimizer(
            space=summarization_config.space,
            distill_functions=summarization_config.sweep_distill_functions,
            metric=summarization_config.sweep_metric_function,
            num_trials=summarization_config.num_trials,
        )

        while not optimizer.is_complete(predictions_dir, include_in_progress=True):
            parameters = optimizer.get_parameters()
            if parameters is None:
                break
            predictions = make_predictions(
                data=data,
                dataset_preset=parameters["dataset_preset"],
                prompt_preset=parameters["prompt_preset"],
                model_preset=parameters["model_preset"],
                temperature=parameters["temperature"],
                max_tokens=parameters["max_tokens"],
                top_p=parameters["top_p"],
                context_length=parameters["context_length"],
                output_dir=predictions_dir,
            )
            if predictions is None:
                print(f"*** Skipped run for {parameters=} ***")
                continue
            eval_result = optimizer.calculate_metric(data, labels, predictions)
            print("*** Iteration complete. ***")
            print(f"Parameters: {parameters}")
            print(f"Eval: {eval_result}")
            print("***************************")

    if do_visualization:
        param_files = summarization_config.space.get_valid_param_files(
            predictions_dir, include_in_progress=False
        )
        if len(param_files) < summarization_config.num_trials:
            logging.getLogger().warning(
                "Not enough completed but performing visualization anyway."
            )
        results: list[ExperimentRun] = []
        for param_file in param_files:
            assert param_file.endswith(".zbp")
            with open(param_file, "r") as f:
                loaded_parameters = json.load(f)
            with open(f"{param_file[:-4]}.jsonl", "r") as f:
                predictions = [json.loads(x) for x in f.readlines()]
            name = reporting_utils.parameters_to_name(
                loaded_parameters, summarization_config.space
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
                "data": data,
                "label": labels,
            }
        )
        visualize(
            df,
            labels,
            results,
            "text-classification",
            "data",
            summarization_config.zeno_distill_and_metric_functions,
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

    summarization_main(
        results_dir=args.results_dir,
        do_prediction=not args.skip_prediction,
        do_visualization=not args.skip_visualization,
    )
