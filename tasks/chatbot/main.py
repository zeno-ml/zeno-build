"""The main entry point for performing comparison on chatbots."""

import argparse
import json
import os
from dataclasses import asdict
from typing import Any

import cohere
import modeling
import openai
import pandas as pd

from llm_compare import search_space
from llm_compare.evaluation.text_features.length import input_length, output_length
from llm_compare.evaluation.text_metrics.critique import (
    avg_chrf,
    avg_length_ratio,
    avg_toxicity,
    chrf,
    length_ratio,
    toxicity,
)
from llm_compare.experiment_run import ExperimentRun
from llm_compare.optimizers import standard
from llm_compare.visualize import visualize


def chatbot_main(
    results_dir: str,
):
    """Run the chatbot experiment."""
    # Set all API keys
    openai.api_key = os.environ["OPENAI_API_KEY"]
    modeling.cohere_client = cohere.Client(os.environ["COHERE_API_KEY"])

    # Make results dir if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Define the space of hyperparameters to search over.
    # Note that "prompt_preset" and "model_preset" are in prompt_configs.py
    # and model_configs.py respectively.
    space = {
        "prompt_preset": search_space.Categorical(
            ["standard", "friendly", "polite", "cynical"]
        ),
        "model_preset": search_space.Categorical(
            ["openai_davinci_003", "openai_gpt_3.5_turbo", "cohere_command_xlarge"]
        ),
        "temperature": search_space.Discrete([0.2, 0.3, 0.4]),
    }

    # Any constants that are fed into the function
    constants: dict[str, Any] = {
        "test_dataset": "daily_dialog",
        "test_split": "test",
        "test_examples": 40,
        "max_tokens": 100,
        "top_p": 1.0,
    }

    # Get the necessary data
    data = modeling.load_data(
        constants["test_dataset"],
        constants["test_split"],
        examples=constants["test_examples"],
    )
    serialized_data = [asdict(x) for x in data]
    with open(os.path.join(results_dir, "examples.json"), "w") as f:
        json.dump(serialized_data, f)
    labels = [x.reference for x in data]

    if os.path.exists(os.path.join(results_dir, "all_runs.json")):
        with open(os.path.join(results_dir, "all_runs.json"), "r") as f:
            serialized_results = json.load(f)
        results = [ExperimentRun(**x) for x in serialized_results]
    else:
        # Run the hyperparameter sweep and print out results
        optimizer = standard.StandardOptimizer()
        results = optimizer.run_sweep(
            function=modeling.make_predictions,
            space=space,
            constants=constants,
            data=data,
            labels=labels,
            distill_functions=[chrf],
            metric=avg_chrf,
            num_trials=10,
            results_dir=results_dir,
        )

        # Print out results
        serialized_results = [asdict(x) for x in results]
        with open(os.path.join(results_dir, "all_runs.json"), "w") as f:
            json.dump(serialized_results, f)

    dataset = modeling.load_data(
        constants["test_dataset"], constants["test_split"], constants["test_examples"]
    )
    dataframe = pd.DataFrame(
        {
            "source": [x.source for x in dataset],
        }
    )

    visualize(
        dataframe,
        labels,
        results,
        "text-classification",
        "source",
        [
            output_length,
            input_length,
            avg_chrf,
            chrf,
            avg_length_ratio,
            length_ratio,
            avg_toxicity,
            toxicity,
        ],
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

    chatbot_main(
        results_dir=args.results_dir,
    )
