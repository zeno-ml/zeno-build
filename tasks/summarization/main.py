"""The main entry point for performing comparison on text summarization."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import cohere
import modeling
import openai
import pandas as pd

from llm_compare.experiment_run import ExperimentRun
from llm_compare.models import global_models
from llm_compare.optimizers import standard
from llm_compare.visualize import visualize
from tasks.summarization import config as summarization_config


def summarization_main(
    results_dir: str,
    cached_data: str | None = None,
    cached_runs: str | None = None,
):
    """Run the summarization experiment."""
    # Set all API keys
    openai.api_key = os.environ["OPENAI_API_KEY"]
    global_models.cohere_client = cohere.Client(os.environ["COHERE_API_KEY"])

    # Make results dir if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load the necessary data, either from HuggingFace or a cached file
    if cached_data is None:
        data_and_labels = modeling.load_data(
            summarization_config.constants.pop("test_dataset"),
            summarization_config.constants.pop("test_split"),
            examples=summarization_config.constants.pop("test_examples"),
        )
        with open(os.path.join(results_dir, "examples.json"), "w") as f:
            json.dump(data_and_labels, f)
    else:
        with open(cached_data, "r") as f:
            data_and_labels = json.load(f)
    data = [x["data"] for x in data_and_labels]
    labels = [x["label"] for x in data_and_labels]
    df = pd.DataFrame({"source": data})

    # Run the hyperparameter sweep and print out results
    results: list[ExperimentRun] = []
    if cached_runs is not None:
        with open(cached_runs, "r") as f:
            serialized_results = json.load(f)
        results = [ExperimentRun(**x) for x in serialized_results]
    else:
        # Perform the hyperparameter sweep
        optimizer = standard.StandardOptimizer(
            space=summarization_config.space,
            constants=summarization_config.constants,
            distill_functions=summarization_config.sweep_distill_functions,
            metric=summarization_config.sweep_metric_function,
        )
        for i in range(summarization_config.num_trials):
            parameters = optimizer.get_parameters()
            predictions = modeling.make_predictions(
                data=data,
                prompt_preset=parameters["prompt_preset"],
                model_preset=parameters["model_preset"],
                temperature=parameters["temperature"],
                max_tokens=parameters["max_tokens"],
                top_p=parameters["top_p"],
            )
            eval_result = optimizer.calculate_metric(data, labels, predictions)
            run = ExperimentRun(
                parameters=parameters,
                predictions=predictions,
                eval_result=eval_result,
            )
            results.append(run)

        serialized_results = [asdict(x) for x in results]
        with open(os.path.join(results_dir, "all_runs.json"), "w") as f:
            json.dump(serialized_results, f)

    visualize(
        df,
        labels,
        results,
        "text-classification",
        "source",
        summarization_config.zeno_distill_and_metric_functions,
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
        "--cached_data",
        type=str,
        default=None,
        help="A path to a json file with the cached data.",
    )
    parser.add_argument(
        "--cached_runs",
        type=str,
        default=None,
        help="A path to a json file with cached runs.",
    )
    args = parser.parse_args()

    summarization_main(
        results_dir=args.results_dir,
        cached_data=args.cached_data,
        cached_runs=args.cached_runs,
    )
