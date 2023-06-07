"""Capitalization-related features."""
from pandas import DataFrame
from zeno import DistillReturn, ZenoOptions, distill


@distill
def input_capital_char_ratio(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Ratio of capital letters in the input.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        Ratio of capital characters in input
    """
    return DistillReturn(
        distill_output=df[ops.data_column].str.count(r"[A-Z]")
        / df[ops.data_column].str.len()
    )


@distill
def output_capital_char_ratio(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Ratio of capital letters in the output.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        Ratio of capital characters in output
    """
    return DistillReturn(
        distill_output=df[ops.output_column].str.count(r"[A-Z]")
        / df[ops.output_column].str.len()
    )


@distill
def label_capital_char_ratio(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Ratio of capital letters in the labels.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        Ratio of capital characters in labels
    """
    return DistillReturn(
        distill_output=df[ops.label_column].str.count(r"[A-Z]")
        / df[ops.label_column].str.len()
    )
