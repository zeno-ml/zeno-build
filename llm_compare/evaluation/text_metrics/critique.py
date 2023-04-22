"""Text metrics using InspiredCo's Critique API."""
import os

from inspiredco.critique import Critique
from pandas import DataFrame
from zeno import DistillReturn, MetricReturn, ZenoOptions, distill, metric

client = Critique(api_key=os.environ["INSPIREDCO_API_KEY"])


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
    """BERT score.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: BERT scores
    """
    eval_dict = df[[ops.data_column, ops.output_column, ops.label_column]].to_dict(
        "records"
    )
    for d in eval_dict:
        d["references"] = [d.pop(ops.label_column)]
        d["target"] = d.pop(ops.output_column)

    result = client.evaluate(
        metric="bert_score", config={"model": "bert-base-uncased"}, dataset=eval_dict
    )

    return DistillReturn(
        distill_output=[round(r["value"], 6) for r in result["examples"]]
    )


@distill
def bleu(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """BLEU score.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: BLEU scores
    """
    eval_dict = df[[ops.output_column, ops.label_column]].to_dict("records")
    for d in eval_dict:
        d["references"] = [d.pop(ops.label_column)]
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
def chrf(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """CHRF score.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: CHRF scores
    """
    eval_dict = df[[ops.output_column, ops.label_column]].to_dict("records")
    for d in eval_dict:
        d["references"] = [d.pop(ops.label_column)]
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
def length_ratio(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Length ratio.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: Length ratios
    """
    eval_dict = df[[ops.output_column, ops.label_column]].to_dict("records")
    for d in eval_dict:
        d["references"] = [d.pop(ops.label_column)]
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
def avg_bleu(df: DataFrame, ops: ZenoOptions) -> MetricReturn:
    """Average BLEU score.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        MetricReturn: Average BLEU score
    """
    if len(df) == 0:
        return MetricReturn(metric=0)
    return MetricReturn(metric=df[ops.distill_columns["bleu"]].fillna(0).mean())


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
