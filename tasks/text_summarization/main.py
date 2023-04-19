"""The main entry point for performing comparison on text summarization."""

import argparse
import json
import os
from dataclasses import asdict
from typing import Any

import openai
import cohere

from llm_compare import search_space
from llm_compare.evaluators import critique
from llm_compare.optimizers import standard

from . import modeling


def text_summarization_main(
    results_dir: str,
):
    """Run the text summarization experiment."""
    # Set all API keys
    openai.api_key = os.environ["OPENAI_API_KEY"]
    modeling.cohere_client = cohere.Client(os.environ["COHERE_API_KEY"])
    inspiredco_api_key = os.environ["INSPIREDCO_API_KEY"]

    # Make results dir if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Define the space of hyperparameters to search over.
    # Note that "prompt_preset" and "model_preset" are in prompt_configs.py
    # and model_configs.py respectively.
    space = {
        "prompt_preset": search_space.Categorical(["standard", "tldr", "concise", "complete"]),
        "model_preset": search_space.Categorical(["openai_davinci_003", "openai_gpt_3.5_turbo", "cohere_command_xlarge"]),
        "temperature": search_space.Discrete([0.2, 0.3, 0.4]),
    }

    # Any constants that are fed into the function
    constants: dict[str, Any] = {
        "test_dataset": "cnn_dailymail",
        "test_split": "test",
        "test_examples": 3,
        "max_tokens": 100,
        "top_p": 1.0,
    }

    # Get the reference answers and create an evaluator for accuracy
    references = modeling.get_references(
        constants["test_dataset"], constants["test_split"], test_examples=constants["test_examples"]
    )
    evaluator = critique.CritiqueEvaluator(
        api_key=os.environ["CRITIQUE_API_KEY"],
        dataset=references,
        preset="ROUGE-1",

    )
    with open(os.path.join(results_dir, "references.json"), "w") as f:
        json.dump(references, f)

    # Run the hyperparameter sweep and print out results
    optimizer = standard.StandardOptimizer()
    result = optimizer.run_sweep(
        function=modeling.make_predictions,
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

    # Print the best result
    raise NotImplementedError("Perform analysis/visualization on the results.")


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

    text_summarization_main(
        results_dir=args.results_dir,
    )
