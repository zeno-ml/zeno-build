"""Length-related features."""
from pandas import DataFrame
from zeno import DistillReturn, ZenoOptions, distill

from zeno_build.prompts.chat_prompt import ChatMessages


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


@distill
def chat_context_length(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Length of the input context (e.g. for chatbots).

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn:
    """
    data = df[ops.data_column]
    if not isinstance(data, ChatMessages):
        raise ValueError(
            f"Expected {ops.data_column} column to be ChatMessages, but got "
            f"{type(data)} instead."
        )
    return DistillReturn(distill_output=len(data.messages))
