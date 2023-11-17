"""Ground truth and prediction are identical."""

from pandas import DataFrame
from zeno import DistillReturn, MetricReturn, ZenoOptions, distill, metric


@distill
def exact_match(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Whether the ground truth and prediction are identical.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: Whether the ground truth and prediction are identical
    """
    return DistillReturn(distill_output=df[ops.label_column] == df[ops.output_column])


@metric
def avg_exact_match(df: DataFrame, ops: ZenoOptions) -> float:
    """Average exact matches.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        float: Average exact matches
    """
    if len(df) == 0:
        return MetricReturn(metric=0.0)
    return MetricReturn(metric=df[ops.distill_columns["exact_match"]].mean())
