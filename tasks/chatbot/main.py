"""The main entry point for performing comparison on chatbots."""

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
from tasks.chatbot import config as chatbot_config
from tasks.chatbot.modeling import ChatExample


def chatbot_main(
    results_dir: str,
    cached_data: str | None = None,
    cached_runs: str | None = None,
):
    """Run the chatbot experiment."""
    # Set all API keys
    openai.api_key = os.environ["OPENAI_API_KEY"]
    global_models.cohere_client = cohere.Client(os.environ["COHERE_API_KEY"])

    # Make results dir if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load the necessary data, either from HuggingFace or a cached file
    if cached_data is None:
        data = modeling.load_data(
            chatbot_config.constants.pop("test_dataset"),
            chatbot_config.constants.pop("test_split"),
            examples=chatbot_config.constants.pop("test_examples"),
        )
        with open(os.path.join(results_dir, "examples.json"), "w") as f:
            json.dump([asdict(x) for x in data], f)
    else:
        with open(cached_data, "r") as f:
            data = [ChatExample(**x) for x in json.load(f)]
    labels = [x.reference for x in data]
    df = pd.DataFrame({"source": [x.source for x in data]})

    # Run the hyperparameter sweep and print out results
    if cached_runs is not None:
        with open(cached_runs, "r") as f:
            serialized_results = json.load(f)
        results = [ExperimentRun(**x) for x in serialized_results]
    else:
        optimizer = standard.StandardOptimizer()
        results = optimizer.run_sweep(
            function=modeling.make_predictions,
            space=chatbot_config.space,
            constants=chatbot_config.constants,
            data=data,
            labels=labels,
            distill_functions=chatbot_config.sweep_distill_functions,
            metric=chatbot_config.sweep_metric_function,
            num_trials=chatbot_config.num_trials,
            results_dir=results_dir,
        )

        serialized_results = [asdict(x) for x in results]
        with open(os.path.join(results_dir, "all_runs.json"), "w") as f:
            json.dump(serialized_results, f)

    # Perform the visualization
    visualize(
        df,
        labels,
        results,
        "text-classification",
        "source",
        chatbot_config.zeno_distill_and_metric_functions,
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

    chatbot_main(
        results_dir=args.results_dir,
        cached_data=args.cached_data,
        cached_runs=args.cached_runs,
    )
