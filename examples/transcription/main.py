"""The main entry point for performing comparison on transcription models."""

from __future__ import annotations

import argparse
import json
import logging
import os

import pandas as pd

import examples.transcription.config as transcription_config
from examples.transcription.modeling import make_predictions
from zeno_build.experiments.experiment_run import ExperimentRun
from zeno_build.optimizers import standard
from zeno_build.reporting import reporting_utils
from zeno_build.reporting.visualize import visualize


def transcription_main(
    input_metadata: str,
    results_dir: str,
    do_prediction: bool = True,
    do_visualization: bool = True,
) -> None:
    """Run the analysis of transcription models."""
    # Load data from input dir.
    metadata = pd.read_csv(input_metadata)
    audio_paths = (
        transcription_config.data_source
        + metadata[transcription_config.data_source_column]
    ).tolist()
    labels = metadata[transcription_config.label_column].tolist()

    # Define the directories for storing predictions
    predictions_dir = os.path.join(results_dir, "predictions")

    if do_prediction:
        # Perform the hyperparameter sweep
        optimizer = standard.StandardOptimizer(
            space=transcription_config.space,
            distill_functions=transcription_config.sweep_distill_functions,
            metric=transcription_config.sweep_metric_function,
            num_trials=transcription_config.num_trials,
        )
        while not optimizer.is_complete(predictions_dir, include_in_progress=True):
            parameters = optimizer.get_parameters()
            if parameters is None:
                break
            predictions = make_predictions(
                audio_paths=audio_paths,
                model_name=parameters["model_preset"],
                output_dir=predictions_dir,
            )
            if predictions is None:
                print(f"*** Skipped run for {parameters=} ***")
                continue
            eval_result = optimizer.calculate_metric(audio_paths, labels, predictions)
            print("*** Iteration complete. ***")
            print(f"Parameters: {parameters}")
            print(f"Eval: {eval_result}")
            print("***************************")

    if do_visualization:
        param_files = transcription_config.space.get_valid_param_files(
            predictions_dir, include_in_progress=False
        )
        if len(param_files) < transcription_config.num_trials:
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
                loaded_parameters, transcription_config.space
            )
            results.append(
                ExperimentRun(
                    parameters=loaded_parameters, predictions=predictions, name=name
                )
            )

        visualize(
            metadata,
            labels,
            results,
            "audio-transcription",
            "id",
            transcription_config.zeno_distill_and_metric_functions,
            zeno_config={
                "cache_path": os.path.join(results_dir, "zeno_cache"),
                "data_path": transcription_config.data_source,
            },
        )


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-metadata",
        type=str,
        help="Metadata file pointing to input audio files.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="The directory to store the results in.",
    )
    args = parser.parse_args()

    transcription_main(
        input_metadata=args.input_metadata,
        results_dir=args.results_dir,
    )
