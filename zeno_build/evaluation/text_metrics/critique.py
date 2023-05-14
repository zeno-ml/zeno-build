"""Text metrics using InspiredCo's Critique API."""
import logging
import os

import tqdm
from inspiredco.critique import Critique
from inspiredco.critique_utils.exceptions import CritiqueError
from pandas import DataFrame
from zeno import DistillReturn, MetricReturn, ZenoOptions, distill, metric

from zeno_build.prompts.chat_prompt import ChatMessages


def call_critique(
    df: DataFrame,
    ops: ZenoOptions,
    metric_name: str,
    config: dict = None,
    batch_size: int = 20000,
) -> MetricReturn:
    """Call Critique.

    Args:
        metric_name: Name of the metric
        df: Zeno DataFrame
        ops: Zeno options
        config: Metric configuration

    Returns:
        MetricReturn: Metric results
    """
    eval_dict = df[[ops.output_column, ops.label_column, ops.data_column]].to_dict(
        "records"
    )
    for d in eval_dict:
        d["references"] = [d.pop(ops.label_column)]
        d["target"] = d.pop(ops.output_column)
        if len(d["references"][0]) == 0:
            raise ValueError(f"Empty reference at {d}")
        data = d.pop(ops.data_column)
        if isinstance(data, str):
            d["source"] = data
        elif isinstance(data, ChatMessages):
            d["source"] = (
                data.messages[-1].content if len(data.messages) >= 1 else "..."
            )
            d["context"] = (
                data.messages[-2].content if len(data.messages) >= 2 else "..."
            )

    client = Critique(api_key=os.environ["INSPIREDCO_API_KEY"])

    # evaluate batch by batch
    all_results = []
    for i in tqdm.tqdm(
        range(0, len(eval_dict), batch_size), desc=f"Evaluating {metric_name}"
    ):
        # Allow up to 3 retries
        for j in range(3):
            try:
                result = client.evaluate(
                    metric=metric_name,
                    config=config,
                    dataset=eval_dict[i : i + batch_size],
                )
                for r in result["examples"]:
                    all_results.append(round(r["value"], 6))
                break
            except CritiqueError:
                if j == 2:
                    raise
                else:
                    logging.warning(f"Error evaluating {metric_name}, retrying...")

    return DistillReturn(distill_output=all_results)


@distill
def bert_score(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """BERT score (with the bert-base-uncased model).

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: BERT scores
    """
    # NOTE: It is necessary to mention "ops.output_column" in this function
    # to work-around a hack in Zeno (as of v0.4.11):
    # https://github.com/zeno-ml/zeno/blob/5c064e74b5276173fa354c4a546ce0d762d8f4d7/zeno/backend.py#L187  # noqa: E501
    return call_critique(
        df, ops, "bert_score", {"model": "bert-base-uncased"}, batch_size=1000
    )


@distill
def sentence_bleu(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Sentence-level BLEU score (with add-1 smoothing).

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: Sentence-level BLEU scores
    """
    # NOTE: It is necessary to mention "ops.output_column" in this function
    # to work-around a hack in Zeno (as of v0.4.11):
    # https://github.com/zeno-ml/zeno/blob/5c064e74b5276173fa354c4a546ce0d762d8f4d7/zeno/backend.py#L187  # noqa: E501
    return call_critique(
        df, ops, "bleu", {"smooth_method": "add_k", "smooth-value": 1.0}
    )


@distill
def chrf(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """CHRF score.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: CHRF scores
    """
    # NOTE: It is necessary to mention "ops.output_column" in this function
    # to work-around a hack in Zeno (as of v0.4.11):
    # https://github.com/zeno-ml/zeno/blob/5c064e74b5276173fa354c4a546ce0d762d8f4d7/zeno/backend.py#L187  # noqa: E501
    return call_critique(df, ops, "chrf", {})


@distill
def length_ratio(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Length ratio.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: Length ratios
    """
    # NOTE: It is necessary to mention "ops.output_column" in this function
    # to work-around a hack in Zeno (as of v0.4.11):
    # https://github.com/zeno-ml/zeno/blob/5c064e74b5276173fa354c4a546ce0d762d8f4d7/zeno/backend.py#L187  # noqa: E501
    return call_critique(df, ops, "length_ratio", {})


@distill
def rouge_1(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """ROUGE-1 score.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: ROUGE-1 scores
    """
    # NOTE: It is necessary to mention "ops.output_column" in this function
    # to work-around a hack in Zeno (as of v0.4.11):
    # https://github.com/zeno-ml/zeno/blob/5c064e74b5276173fa354c4a546ce0d762d8f4d7/zeno/backend.py#L187  # noqa: E501
    return call_critique(df, ops, "rouge", {"variety": "rouge_1"})


@distill
def rouge_2(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """ROUGE-2 score.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: ROUGE-2 scores
    """
    # NOTE: It is necessary to mention "ops.output_column" in this function
    # to work-around a hack in Zeno (as of v0.4.11):
    # https://github.com/zeno-ml/zeno/blob/5c064e74b5276173fa354c4a546ce0d762d8f4d7/zeno/backend.py#L187  # noqa: E501
    return call_critique(df, ops, "rouge", {"variety": "rouge_2"})


@distill
def rouge_l(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """ROUGE-L score.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: ROUGE-L scores
    """
    # NOTE: It is necessary to mention "ops.output_column" in this function
    # to work-around a hack in Zeno (as of v0.4.11):
    # https://github.com/zeno-ml/zeno/blob/5c064e74b5276173fa354c4a546ce0d762d8f4d7/zeno/backend.py#L187  # noqa: E501
    return call_critique(df, ops, "rouge", {"variety": "rouge_l"})


@distill
def toxicity(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Toxicity score.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: Toxicity scores
    """
    # NOTE: It is necessary to mention "ops.output_column" in this function
    # to work-around a hack in Zeno (as of v0.4.11):
    # https://github.com/zeno-ml/zeno/blob/5c064e74b5276173fa354c4a546ce0d762d8f4d7/zeno/backend.py#L187  # noqa: E501
    return call_critique(
        df, ops, "detoxify", {"model": "unitary/toxic-bert"}, batch_size=1000
    )


@distill
def coherence(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Coherence score.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: Coherence scores
    """
    # NOTE: It is necessary to mention "ops.output_column" in this function
    # to work-around a hack in Zeno (as of v0.4.11):
    # https://github.com/zeno-ml/zeno/blob/5c064e74b5276173fa354c4a546ce0d762d8f4d7/zeno/backend.py#L187  # noqa: E501
    return call_critique(
        df,
        ops,
        "uni_eval",
        {"task": "dialog", "evaluation_aspect": "coherence"},
        batch_size=150,
    )


@metric
def avg_bert_score(df: DataFrame, ops: ZenoOptions) -> MetricReturn:
    """Average BERT score.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        MetricReturn: Average BERT score
    """
    if len(df) == 0:
        return MetricReturn(metric=0)
    return MetricReturn(metric=df[ops.distill_columns["bert_score"]].fillna(0).mean())


@metric
def avg_sentence_bleu(df: DataFrame, ops: ZenoOptions) -> MetricReturn:
    """Average Sentence-level BLEU score.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        MetricReturn: Average Sentence-level BLEU score
    """
    if len(df) == 0:
        return MetricReturn(metric=0)
    return MetricReturn(
        metric=df[ops.distill_columns["sentence_bleu"]].fillna(0).mean()
    )


@metric
def avg_chrf(df: DataFrame, ops: ZenoOptions) -> MetricReturn:
    """Average CHRF score.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        MetricReturn: Average CHRF score
    """
    if len(df) == 0:
        return MetricReturn(metric=0)
    return MetricReturn(metric=df[ops.distill_columns["chrf"]].fillna(0).mean())


@metric
def avg_length_ratio(df: DataFrame, ops: ZenoOptions) -> MetricReturn:
    """Average length ratio.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        MetricReturn: Average length ratio
    """
    if len(df) == 0:
        return MetricReturn(metric=0)
    return MetricReturn(metric=df[ops.distill_columns["length_ratio"]].fillna(0).mean())


@metric
def avg_rouge_1(df: DataFrame, ops: ZenoOptions) -> MetricReturn:
    """Average ROUGE-1 score.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        MetricReturn: Average ROUGE-1 score
    """
    if len(df) == 0:
        return MetricReturn(metric=0)
    return MetricReturn(metric=df[ops.distill_columns["rouge_1"]].fillna(0).mean())


@metric
def avg_rouge_2(df: DataFrame, ops: ZenoOptions) -> MetricReturn:
    """Average ROUGE-2 score.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        MetricReturn: Average ROUGE-2 score
    """
    if len(df) == 0:
        return MetricReturn(metric=0)
    return MetricReturn(metric=df[ops.distill_columns["rouge_2"]].fillna(0).mean())


@metric
def avg_rouge_l(df: DataFrame, ops: ZenoOptions) -> MetricReturn:
    """Average ROUGE-L score.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        MetricReturn: Average ROUGE-L score
    """
    if len(df) == 0:
        return MetricReturn(metric=0)
    return MetricReturn(metric=df[ops.distill_columns["rouge_l"]].fillna(0).mean())


@metric
def avg_toxicity(df: DataFrame, ops: ZenoOptions) -> MetricReturn:
    """Average toxicity score.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        MetricReturn: Average toxicity score
    """
    if len(df) == 0:
        return MetricReturn(metric=0)
    return MetricReturn(metric=df[ops.distill_columns["toxicity"]].fillna(0).mean())


@metric
def avg_coherence(df: DataFrame, ops: ZenoOptions) -> MetricReturn:
    """Average coherence score.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        MetricReturn: Average coherence score
    """
    if len(df) == 0:
        return MetricReturn(metric=0)
    return MetricReturn(metric=df[ops.distill_columns["coherence"]].fillna(0).mean())
