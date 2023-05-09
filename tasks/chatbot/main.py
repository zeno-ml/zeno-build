"""The main entry point for performing comparison on chatbots."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import cohere
import openai
import pandas as pd

from tasks.chatbot import config as chatbot_config
from tasks.chatbot.modeling import load_data, make_predictions
from zeno_build.experiments.experiment_run import ExperimentRun
from zeno_build.models import global_models
from zeno_build.optimizers import standard
from zeno_build.prompts.chat_prompt import ChatMessages
from zeno_build.reporting.visualize import visualize


def chatbot_main(
    results_dir: str,
    cached_data: str | None = None,
    cached_runs: str | None = None,
    do_visualization: bool = True,
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
        data = load_data(
            chatbot_config.constants.pop("test_dataset"),
            chatbot_config.constants.pop("test_split"),
            examples=chatbot_config.constants.pop("test_examples"),
        )
        with open(os.path.join(results_dir, "examples.json"), "w") as f:
            json.dump([asdict(x) for x in data], f)
    else:
        with open(cached_data, "r") as f:
            data = [ChatMessages(**x) for x in json.load(f)]
    # Organize the data into source and context
    labels, contexts = [], []
    for x in data:
        labels.append(x.messages[-1].content)
        contexts.append(x.messages[:-1])
    df = pd.DataFrame({"context": contexts, "label": labels})

    # Run the hyperparameter sweep and print out results
    results: list[ExperimentRun] = []
    if cached_runs is not None:
        with open(cached_runs, "r") as f:
            serialized_results = json.load(f)
        results = [ExperimentRun(**x) for x in serialized_results]

    else:
        # Perform the hyperparameter sweep
        optimizer = standard.StandardOptimizer(
            space=chatbot_config.space,
            constants=chatbot_config.constants,
            distill_functions=chatbot_config.sweep_distill_functions,
            metric=chatbot_config.sweep_metric_function,
        )
        for _ in range(chatbot_config.num_trials):
            parameters = optimizer.get_parameters()
            setting_name = " ".join(
                [
                    parameters[k]
                    if isinstance(parameters[k], str)
                    else f"{k}={parameters[k]}"
                    for k in chatbot_config.space.keys()
                ]
            )
            predictions = make_predictions(
                data=data,
                prompt_preset=parameters["prompt_preset"],
                model_preset=parameters["model_preset"],
                temperature=parameters["temperature"],
                max_tokens=parameters["max_tokens"],
                top_p=parameters["top_p"],
                context_length=parameters["context_length"],
                cache_root=os.path.join(results_dir, "cache"),
            )
            eval_result = optimizer.calculate_metric(data, labels, predictions)
            run = ExperimentRun(
                parameters=parameters,
                predictions=predictions,
                eval_result=eval_result,
                name=setting_name,
            )
            results.append(run)

        serialized_results = [asdict(x) for x in results]
        with open(os.path.join(results_dir, "all_runs.json"), "w") as f:
            json.dump(serialized_results, f)

    # Perform the visualization
    if do_visualization:
        visualize(
            df,
            labels,
            results,
            "openai-chat",
            "context",
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
    parser.add_argument(
        "--skip_visualization",
        action="store_true",
        help="Whether to skip the visualization step.",
    )
    args = parser.parse_args()

    chatbot_main(
        results_dir=args.results_dir,
        cached_data=args.cached_data,
        cached_runs=args.cached_runs,
        do_visualization=not args.skip_visualization,
    )
