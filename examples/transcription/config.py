"""Config for analyzing transcription models."""

from __future__ import annotations

from zeno_build.evaluation.audio_metrics.error import avg_wer, wer
from zeno_build.experiments import search_space

# The search space for the main experiments
space = search_space.CombinatorialSearchSpace(
    {
        "data_source_preset": search_space.Constant(
            "https://zenoml.s3.amazonaws.com/accents/"
        ),
        "data_source_column": search_space.Constant("id"),
        "label_column": search_space.Constant("label"),
        "model_preset": search_space.Categorical(
            [
                "tiny.en",
                "tiny",
                # "base",
                # "base.en",
                # "small",
                # "small.en",
                # "medium",
                # "medium.en",
                # "large",
            ]
        ),
    }
)

# The number of trials to run
num_trials = 2

sweep_distill_functions = [wer]
sweep_metric_function = avg_wer

# The functions used for Zeno visualization
zeno_distill_and_metric_functions = [wer, avg_wer]
