"""Functions for measuring the error of audio models."""

import math

import pandas as pd
from jiwer import wer as jiwer_wer
from zeno import DistillReturn, MetricReturn, ZenoOptions, distill, metric


@distill
def wer(df: pd.DataFrame, ops: ZenoOptions):
    """Calculate the word error rate.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        MetricReturn: The word error rate.
    """
    return DistillReturn(
        distill_output=df.apply(
            lambda x: jiwer_wer(x[ops.label_column], x[ops.output_column]), axis=1
        )
    )


@metric
def avg_wer(df, ops: ZenoOptions):
    """Calculate the average word error rate.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        MetricReturn: The average word error rate.
    """
    avg = df[ops.distill_columns["wer"]].mean()
    if pd.isnull(avg) or math.isnan(avg):
        return MetricReturn(metric=0)
    return MetricReturn(metric=avg)
