"""Text metrics using InspiredCo's Critique API."""
import os

from inspiredco.critique import Critique
from pandas import DataFrame
from zeno import DistillReturn, MetricReturn, ZenoOptions, distill, metric


def call_critique(
    df: DataFrame,
    ops: ZenoOptions,
    metric_name: str,
    config: dict = None,
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
    eval_dict = df[[ops.output_column, ops.label_column]].to_dict("records")
    for d in eval_dict:
        d["references"] = [d.pop(ops.label_column)]
        d["target"] = d.pop(ops.output_column)

    client = Critique(api_key=os.environ["INSPIREDCO_API_KEY"])
    result = client.evaluate(
        metric=metric_name,
        config=config,
        dataset=eval_dict,
    )

    return DistillReturn(
        distill_output=[round(r["value"], 6) for r in result["examples"]]
    )


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
    return call_critique(df, ops, "bert_score", {"model": "bert-base-uncased"})


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
    return call_critique(df, ops, "detoxify", {"model": "unitary/toxic-bert"})


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
