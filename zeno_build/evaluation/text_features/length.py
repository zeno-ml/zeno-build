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
def label_length(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Length of the gold-standard label.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: Lengths of labels
    """
    return DistillReturn(distill_output=df[ops.label_column].str.len())


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
    chat_context_lengths = []
    for data in df[ops.data_column]:
        if isinstance(data, ChatMessages):
            chat_context_lengths.append(len(data.messages))
        elif isinstance(data, list):
            chat_context_lengths.append(len(data))
        else:
            raise ValueError(
                f"Expected {ops.data_column} column to be ChatMessages, or list "
                f"but got {type(data)} instead."
            )
    return DistillReturn(distill_output=chat_context_lengths)


@distill
def doc_context_length(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Length of the previous document context, e.g. for document-level translation.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: The document context
    """
    doc_context_lengths = []
    # Count the number of times `doc_id` has appeared so far
    doc_ids: dict[str, int] = {}
    for x in df["doc_id"]:
        doc_ids[x] = doc_ids.get(x, 0) + 1
        doc_context_lengths.append(doc_ids[x])
    return DistillReturn(distill_output=doc_context_lengths)
