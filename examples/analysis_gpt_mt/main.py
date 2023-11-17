"""The main entry point for performing comparison on analysis_gpt_mts."""

from __future__ import annotations

import argparse
import os

import config
import pandas as pd
from modeling import GptMtInstance, process_data, process_output

from zeno_build.experiments import search_space
from zeno_build.experiments.experiment_run import ExperimentRun
from zeno_build.reporting.visualize import visualize


def analysis_gpt_mt_main(
    input_dir: str,
    results_dir: str,
) -> None:
    """Run the analysis of GPT-MT experiment."""
    # Get the dataset configuration
    lang_pair_preset = config.main_space.dimensions["lang_pairs"]
    if not isinstance(lang_pair_preset, search_space.Constant):
        raise ValueError(
            "All experiments must be run on a single set of language pairs."
        )
    lang_pairs = config.lang_pairs[lang_pair_preset.value]

    # Load and exhaustiveize the format of the necessary data.
    test_data: list[GptMtInstance] = process_data(
        input_dir=input_dir,
        lang_pairs=lang_pairs,
    )

    results: list[ExperimentRun] = []
    model_presets = config.main_space.dimensions["model_preset"]
    if not isinstance(model_presets, search_space.Categorical):
        raise ValueError("The model presets must be a categorical parameter.")
    for model_preset in model_presets.choices:
        output = process_output(
            input_dir=input_dir,
            lang_pairs=lang_pairs,
            model_preset=model_preset,
        )
        results.append(
            ExperimentRun(model_preset, {"model_preset": model_preset}, output)
        )

    # Perform the visualization
    df = pd.DataFrame(
        {
            "data": [x.data for x in test_data],
            "label": [x.label for x in test_data],
            "lang_pair": [x.lang_pair for x in test_data],
            "doc_id": [x.doc_id for x in test_data],
        }
    )
    labels = [x.label for x in test_data]
    visualize(
        df,
        labels,
        results,
        "text-classification",
        "data",
        config.zeno_distill_and_metric_functions,
        zeno_config={"cache_path": os.path.join(results_dir, "zeno_cache")},
    )


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        help="The directory of the GPT-MT repo.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="The directory to store the results in.",
    )
    args = parser.parse_args()

    analysis_gpt_mt_main(
        input_dir=args.input_dir,
        results_dir=args.results_dir,
    )
