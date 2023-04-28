"""Length-related features."""
from pandas import DataFrame
from zeno import DistillReturn, ZenoOptions, distill


@distill
def output_length(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Length of model output.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: Lengths of outputs
    """
    return DistillReturn(distill_output=df[ops.output_column].str.len())


@distill
def input_length(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Length of model input.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: Lengths of inputs
    """
    return DistillReturn(distill_output=df[ops.data_column].str.len())
