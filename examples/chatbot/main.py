"""The main entry point for performing comparison on chatbots."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict

import pandas as pd

from examples.chatbot import config as chatbot_config
from examples.chatbot.modeling import make_predictions, process_data
from zeno_build.experiments import search_space
from zeno_build.experiments.experiment_run import ExperimentRun
from zeno_build.optimizers import exhaustive
from zeno_build.prompts.chat_prompt import ChatMessages
from zeno_build.reporting import reporting_utils
from zeno_build.reporting.visualize import visualize


def chatbot_main(
    results_dir: str,
    do_prediction: bool = True,
    do_visualization: bool = True,
):
    """Run the chatbot experiment."""
    # Get the dataset configuration
    dataset_preset = chatbot_config.report_space.spaces[0].dimensions["dataset_preset"]
    if not isinstance(dataset_preset, search_space.Constant):
        raise ValueError("All experiments must be run on a single dataset.")
    dataset_config = chatbot_config.dataset_configs[dataset_preset.value]

    # Define the directories for storing data and predictions
    data_dir = os.path.join(results_dir, "data")
    predictions_dir = os.path.join(results_dir, "predictions")

    # Load and standardize the format of the necessary data. The resulting
    # processed data will be stored in the `results_dir/data` directory
    # both for browsing and for caching for fast reloading on future runs.
    contexts_and_labels: list[ChatMessages] = process_data(
        dataset=dataset_config.dataset,
        split=dataset_config.split,
        data_format=dataset_config.data_format,
        data_column=dataset_config.data_column,
        output_dir=data_dir,
    )

    # Organize the data into labels (output) and context (input)
    labels: list[str] = []
    contexts: list[ChatMessages] = []
    for x in contexts_and_labels:
        labels.append(x.messages[-1].content)
        contexts.append(ChatMessages(x.messages[:-1]))

    if do_prediction:
        # Perform the hyperparameter sweep
        optimizer = exhaustive.ExhaustiveOptimizer(
            space=chatbot_config.report_space,
            distill_functions=chatbot_config.sweep_distill_functions,
            metric=chatbot_config.sweep_metric_function,
            num_trials=chatbot_config.num_trials,
        )

        while not optimizer.is_complete(predictions_dir, include_in_progress=True):
            # Get parameters
            parameters = optimizer.get_parameters()
            if parameters is None:
                break
            # Get the run ID and resulting predictions
            id_and_predictions = make_predictions(
                contexts=contexts,
                dataset_preset=parameters["dataset_preset"],
                prompt_preset=parameters["prompt_preset"],
                model_preset=parameters["model_preset"],
                temperature=parameters["temperature"],
                max_tokens=parameters["max_tokens"],
                top_p=parameters["top_p"],
                context_length=parameters["context_length"],
                output_dir=predictions_dir,
            )
            if id_and_predictions is None:
                print(f"*** Skipped run for {parameters=} ***")
                continue
            # Run or read the evaluation result
            id, predictions = id_and_predictions
            if os.path.exists(f"{predictions_dir}/{id}.eval"):
                with open(f"{predictions_dir}/{id}.eval", "r") as f:
                    eval_result = float(next(f).strip())
            else:
                eval_result = optimizer.calculate_metric(contexts, labels, predictions)
                with open(f"{predictions_dir}/{id}.eval", "w") as f:
                    f.write(f"{eval_result}")
            # Print out the results
            print("*** Iteration complete. ***")
            print(f"Eval: {eval_result}, Parameters: {parameters}")
            print("***************************")

    if do_visualization:
        param_files = chatbot_config.report_space.get_valid_param_files(
            predictions_dir, include_in_progress=False
        )
        if len(param_files) < chatbot_config.num_trials:
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
                loaded_parameters, chatbot_config.report_space
            )
            results.append(
                ExperimentRun(
                    parameters=loaded_parameters, predictions=predictions, name=name
                )
            )

        # Perform the visualization
        df = pd.DataFrame(
            {
                "messages": [[asdict(y) for y in x.messages] for x in contexts],
                "label": labels,
            }
        )
        visualize(
            df,
            labels,
            results,
            "openai-chat",
            "messages",
            chatbot_config.zeno_distill_and_metric_functions,
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

    chatbot_main(
        results_dir=args.results_dir,
        do_prediction=not args.skip_prediction,
        do_visualization=not args.skip_visualization,
    )
