"""Ground truth and prediction are identical."""

from pandas import DataFrame
from zeno import DistillReturn, ZenoOptions, distill


@distill
def perfect_match(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Whether the ground truth and prediction are identical.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: Whether the ground truth and prediction are identical
    """
    return DistillReturn(distill_output=df[ops.label_column] == df[ops.output_column])
