"""Capitalization-related features."""
from collections import Counter
from typing import Any

from pandas import DataFrame
from zeno import DistillReturn, ZenoOptions, distill


def _count_max_word_freq(string: Any) -> int:
    """Count max word frequency in a single string."""
    if not isinstance(string, str):
        raise TypeError(f"Input must be a string but got {type(string)} for {string}.")
    tokens = string.split()
    if not tokens:
        return 0
    else:
        return max(Counter(tokens).values())


@distill
def input_max_word_freq(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Max frequency of words in the input.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        Max frequency of words in input
    """
    return DistillReturn(distill_output=df[ops.data_column].apply(_count_max_word_freq))


@distill
def output_max_word_freq(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Max frequency of words in the output.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        Max frequency of words in output
    """
    return DistillReturn(
        distill_output=df[ops.output_column].apply(_count_max_word_freq)
    )


@distill
def label_max_word_freq(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Max frequency of words in the labels.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        Max frequency of words in labels
    """
    return DistillReturn(
        distill_output=df[ops.label_column].apply(_count_max_word_freq)
    )
