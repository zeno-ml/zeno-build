"""Text metrics using InspiredCo's Critique API."""
import os

from inspiredco.critique import Critique
from pandas import DataFrame
from zeno import DistillReturn, MetricReturn, ZenoOptions, distill, metric

client = Critique(api_key=os.environ["INSPIREDCO_API_KEY"])


@distill
def bert_score(df, ops):
    """BERT score.

    Args:
        df (DataFrame): Zeno DataFrame
        ops (ZenoOptions): Zeno options

    Returns:
        DistillReturn: BERT scores
    """
    eval_dict = df[["source", ops.output_column, "reference"]].to_dict("records")
    for d in eval_dict:
        d["references"] = [d.pop("reference")]
        d["target"] = d.pop(ops.output_column)

    result = client.evaluate(
        metric="bert_score", config={"model": "bert-base-uncased"}, dataset=eval_dict
    )

    return DistillReturn(
        distill_output=[round(r["value"], 6) for r in result["examples"]]
    )


@distill
def bleu(df, ops):
    """BLEU score.

    Args:
        df (DataFrame): Zeno DataFrame
        ops (ZenoOptions): Zeno options

    Returns:
        DistillReturn: BLEU scores
    """
    eval_dict = df[[ops.output_column, "reference"]].to_dict("records")
    for d in eval_dict:
        d["references"] = [d.pop("reference")]
        d["target"] = d.pop(ops.output_column)

    result = client.evaluate(
        metric="bleu",
        config={"smooth_method": "add_k", "smooth-value": 1.0},
        dataset=eval_dict,
    )

    return DistillReturn(
        distill_output=[round(r["value"], 6) for r in result["examples"]]
    )


@distill
def chrf(df, ops):
    """CHRF score.

    Args:
        df (DataFrame): Zeno DataFrame
        ops (ZenoOptions): Zeno options

    Returns:
        DistillReturn: CHRF scores
    """
    eval_dict = df[[ops.output_column, "reference"]].to_dict("records")
    for d in eval_dict:
        d["references"] = [d.pop("reference")]
        d["target"] = d.pop(ops.output_column)

    result = client.evaluate(
        metric="chrf",
        config={},
        dataset=eval_dict,
    )

    return DistillReturn(
        distill_output=[round(r["value"], 6) for r in result["examples"]]
    )


@distill
def length_ratio(df, ops):
    """Length ratio.

    Args:
        df (DataFrame): Zeno DataFrame
        ops (ZenoOptions): Zeno options

    Returns:
        DistillReturn: Length ratios
    """
    eval_dict = df[[ops.output_column, "reference"]].to_dict("records")
    for d in eval_dict:
        d["references"] = [d.pop("reference")]
        d["target"] = d.pop(ops.output_column)

    result = client.evaluate(
        metric="length_ratio",
        config={},
        dataset=eval_dict,
    )

    return DistillReturn(
        distill_output=[round(r["value"], 6) for r in result["examples"]]
    )


@distill
def rouge(df: DataFrame, ops: ZenoOptions):
    """ROUGE score.

    Args:
        df (DataFrame): Zeno DataFrame
        ops (ZenoOptions): Zeno options

    Returns:
        DistillReturn: ROUGE scores
    """
    eval_dict = df[[ops.output_column, ops.label_column]].to_dict("records")
    for d in eval_dict:
        d["references"] = [d.pop(ops.label_column)]
        d["target"] = d.pop(ops.output_column)

    result = client.evaluate(
        metric="rouge",
        config={"variety": "rouge_1"},
        dataset=eval_dict,
    )

    return DistillReturn(
        distill_output=[round(r["value"], 6) for r in result["examples"]]
    )


@metric
def avg_bert_score(df, ops: ZenoOptions):
    """Average BERT score.

    Args:
        df (DataFrame): Zeno DataFrame
        ops (ZenoOptions): Zeno options

    Returns:
        MetricReturn: Average BERT score
    """
    if len(df) == 0:
        return MetricReturn(metric=0)
    return MetricReturn(metric=df[ops.distill_columns["bert_score"]].fillna(0).mean())


@metric
def avg_bleu(df, ops: ZenoOptions):
    """Average BLEU score.

    Args:
        df (DataFrame): Zeno DataFrame
        ops (ZenoOptions): Zeno options

    Returns:
        MetricReturn: Average BLEU score
    """
    if len(df) == 0:
        return MetricReturn(metric=0)
    return MetricReturn(metric=df[ops.distill_columns["bleu"]].fillna(0).mean())


@metric
def avg_chrf(df, ops: ZenoOptions):
    """Average CHRF score.

    Args:
        df (DataFrame): Zeno DataFrame
        ops (ZenoOptions): Zeno options

    Returns:
        MetricReturn: Average CHRF score
    """
    if len(df) == 0:
        return MetricReturn(metric=0)
    return MetricReturn(metric=df[ops.distill_columns["chrf"]].fillna(0).mean())


@metric
def avg_length_ratio(df, ops: ZenoOptions):
    """Average length ratio.

    Args:
        df (DataFrame): Zeno DataFrame
        ops (ZenoOptions): Zeno options

    Returns:
        MetricReturn: Average length ratio
    """
    if len(df) == 0:
        return MetricReturn(metric=0)
    return MetricReturn(metric=df[ops.distill_columns["length_ratio"]].fillna(0).mean())


@metric
def avg_rouge(df, ops: ZenoOptions):
    """Average ROUGE score.

    Args:
        df (DataFrame): Zeno DataFrame
        ops (ZenoOptions): Zeno options

    Returns:
        MetricReturn: Average ROUGE score
    """
    if len(df) == 0:
        return MetricReturn(metric=0)
    return MetricReturn(metric=df[ops.distill_columns["rouge"]].fillna(0).mean())
