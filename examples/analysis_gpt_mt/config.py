"""Config for analyzing GPT-MT."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from examples.analysis_gpt_mt.modeling import remove_leading_language
from zeno_build.evaluation.text_features.capitalization import input_capital_char_ratio
from zeno_build.evaluation.text_features.exact_match import avg_exact_match, exact_match
from zeno_build.evaluation.text_features.frequency import output_max_word_freq
from zeno_build.evaluation.text_features.length import (
    doc_context_length,
    input_length,
    label_length,
    output_length,
)
from zeno_build.evaluation.text_metrics.critique import (
    avg_bert_score,
    avg_chrf,
    avg_comet,
    avg_length_ratio,
    bert_score,
    chrf,
    comet,
    length_ratio,
)
from zeno_build.experiments import search_space

lang_pairs: dict[str, list[str]] = {
    # All language pairs used in any experiment
    "all_lang_pairs": [
        "csen",
        "deen",
        "defr",
        "encs",
        "ende",
        "enha",
        "enis",
        "enja",
        "enru",
        "enuk",
        "enzh",
        "frde",
        "haen",
        "isen",
        "jaen",
        "ruen",
        "uken",
        "zhen",
    ],
    # Language pairs used in the experiments on a limited number of language pairs
    "limited_lang_pairs": [
        "deen",
        "defr",
        "ende",
        "enru",
        "enzh",
        "frde",
        "ruen",
        "zhen",
    ],
}

# The search space for the main experiments
main_space = search_space.CombinatorialSearchSpace(
    {
        "lang_pairs": search_space.Constant("all_lang_pairs"),
        "model_preset": search_space.Categorical(
            [
                "text-davinci-003-zeroshot",
                "text-davinci-003-RR-1-shot",
                "text-davinci-003-RR-5-shot",
                "text-davinci-003-QR-1-shot",
                "text-davinci-003-QR-5-shot",
                "gpt-3.5-turbo-0301-zeroshot",
                "gpt-4-0314-zeroshot",
                "gpt-4-0314-zeroshot-postprocess",
                "MS-Translator",
                "google-cloud",
                "wmt-best",
            ]
        ),
    }
)


@dataclass(frozen=True)
class GptMtConfig:
    """Config for gpt-MT models."""

    path: str
    base_model: str
    prompt_strategy: str | None = None
    prompt_shots: int | None = None
    post_processors: list[Callable[[str], str]] | None = None


# The details of each model
model_configs = {
    "text-davinci-003-RR-1-shot": GptMtConfig(
        "text-davinci-003/RR/1-shot", "text-davinci-003", "RR", 1
    ),
    "text-davinci-003-RR-5-shot": GptMtConfig(
        "text-davinci-003/RR/5-shot", "text-davinci-003", "RR", 5
    ),
    "text-davinci-003-QR-1-shot": GptMtConfig(
        "text-davinci-003/QR/1-shot", "text-davinci-003", "QR", 1
    ),
    "text-davinci-003-QR-5-shot": GptMtConfig(
        "text-davinci-003/QR/5-shot", "text-davinci-003", "QR", 5
    ),
    "text-davinci-003-zeroshot": GptMtConfig(
        "text-davinci-003/zeroshot", "text-davinci-003", None, 0
    ),
    "gpt-3.5-turbo-0301-zeroshot": GptMtConfig(
        "gpt-3.5-turbo-0301/zeroshot", "gpt-3.5-turbo-0301", None, 0
    ),
    "gpt-4-0314-zeroshot": GptMtConfig("gpt-4-0314/zeroshot", "gpt-4-0314", None, 0),
    "gpt-4-0314-zeroshot-postprocess": GptMtConfig(
        "gpt-4-0314/zeroshot", "gpt-4-0314", None, 0, [remove_leading_language]
    ),
    "MS-Translator": GptMtConfig("MS-Translator", "MS-Translator"),
    "google-cloud": GptMtConfig("google-cloud", "google-cloud"),
    "wmt-best": GptMtConfig("wmt-best", "wmt-best"),
}

sweep_distill_functions = [chrf]
sweep_metric_function = avg_chrf

# The functions used for Zeno visualization
zeno_distill_and_metric_functions = [
    output_length,
    input_length,
    label_length,
    doc_context_length,
    input_capital_char_ratio,
    output_max_word_freq,
    chrf,
    comet,
    length_ratio,
    bert_score,
    exact_match,
    avg_chrf,
    avg_comet,
    avg_length_ratio,
    avg_bert_score,
    avg_exact_match,
]
