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
        DistillReturn: Lengths of inputs
    """
    return DistillReturn(
        distill_output=df[ops.data_column].str.count(r"[A-Z]")
        / df[ops.data_column].str.len()
    )


@distill
def label_capital_char_ratio(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Ratio of capital letters in the input.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: Lengths of inputs
    """
    return DistillReturn(
        distill_output=df[ops.label_column].str.count(r"[A-Z]")
        / df[ops.label_column].str.len()
    )
