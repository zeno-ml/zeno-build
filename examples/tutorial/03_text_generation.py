"""A tutorial of how to perform inference with Zeno Build."""

import pandas as pd
from datasets import load_dataset

from zeno_build.evaluation.text_features.length import input_length, output_length
from zeno_build.evaluation.text_metrics.critique import (
    avg_bert_score,
    avg_chrf,
    bert_score,
    chrf,
)
from zeno_build.experiments.experiment_run import ExperimentRun
from zeno_build.models.lm_config import LMConfig
from zeno_build.models.text_generate import generate_from_text_prompt
from zeno_build.reporting.visualize import visualize


def main():
    """Run the inference example."""
    # Get the first 250 translations from hugging face
    dataset = load_dataset("ted_multi", split="validation")
    srcs, trgs = [], []
    src_language, trg_language = "fr", "en"
    for datum in dataset:
        if (
            src_language not in datum["translations"]["language"]
            or trg_language not in datum["translations"]["language"]
        ):
            continue
        src_index = datum["translations"]["language"].index(src_language)
        trg_index = datum["translations"]["language"].index(trg_language)
        srcs.append(datum["translations"]["translation"][src_index])
        trgs.append(datum["translations"]["translation"][trg_index])
        if len(srcs) >= 250:
            break
    df = pd.DataFrame({"text": srcs, "label": trgs})

    # Prompt templates for text or chat
    prompt_template = (
        "Translate this sentence into English:\n\n" "Sentence: {{text}}\n" "English: "
    )

    # Perform inference
    all_results = []
    for lm_config in [
        # LMConfig(provider="openai_chat", model="gpt-3.5-turbo"),
        LMConfig(provider="huggingface", model="gpt2"),
        LMConfig(provider="huggingface", model="gpt2-xl"),
    ]:
        predictions = generate_from_text_prompt(
            [{"text": x} for x in srcs],
            prompt_template=prompt_template,
            model_config=lm_config,
            temperature=0.0001,
            max_tokens=200,
            top_p=1.0,
            requests_per_minute=400,
        )
        result = ExperimentRun(
            name=lm_config.model,
            parameters={"provider": lm_config.provider, "model": lm_config.model},
            predictions=[x.strip().split("\n")[0] for x in predictions],
        )
        all_results.append(result)

    functions = [
        output_length,
        input_length,
        chrf,
        avg_chrf,
        bert_score,
        avg_bert_score,
    ]

    visualize(
        df,
        trgs,
        all_results,
        "text-classification",
        "text",
        functions,
        zeno_config={"cache_path": "zeno_cache"},
    )


if __name__ == "__main__":
    main()
