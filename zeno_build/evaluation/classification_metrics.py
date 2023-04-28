"""Classification metrics for Zeno."""

from pandas import DataFrame
from zeno import MetricReturn, ZenoOptions, metric


@metric
def accuracy(df: DataFrame, ops: ZenoOptions) -> MetricReturn:
    """Calculate the accuracy of a model.

    Args:
        df: DataFrame from Zeno
        ops: Options from Zeno

    Returns:
        MetricReturn: accuracy value
    """
    if len(df) == 0:
        return MetricReturn(metric=0.0)
    return MetricReturn(metric=(df[ops.label_column] == df[ops.output_column]).mean())
