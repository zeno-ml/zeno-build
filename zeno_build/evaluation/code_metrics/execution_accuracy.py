"""Evaluating code execution accuracy."""

from pandas import DataFrame
from zeno import DistillReturn, MetricReturn, ZenoOptions, distill, metric

from zeno_build.evaluation.code_metrics import execution_accuracy_utils


@distill
def execution_accuracy(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Evaluate execution accuracy of code.

    Note that there are some expectations about the content of df:
    - df[ops.data_column]: A column containing code prefixes.
    - df[ops.output_column]: A column containing generated code.
    - df[ops.label_column]: A column containing lists of tests to be executed.

    Args:
        df: A dataframe containing the necessary inputs.
        ops: The options for the inputs.

    Side effect:
        Adds a column "execution_accuracy_message" with the message from
        the execution accuracy evaluator.

    Returns:
        The results of pass@1 execution accuracy.
    """
    import os

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    eval_dict = df[[ops.output_column, ops.label_column, ops.data_column]].to_dict(
        "records"
    )

    predictions: list[list[str]] = []
    tests: list[str] = []
    for d in eval_dict:
        predictions.append([d.get(ops.data_column) + d.get(ops.output_column)])
        tests.append(d.get(ops.label_column))

    pass_at_k, results = execution_accuracy_utils.compute_execution_accuracy(
        predictions=predictions,
        tests=tests,
        k=[1],
    )

    df["execution_accuracy_messages"] = [
        r[0].error_message or r[0].success_value for r in results
    ]
    return DistillReturn(distill_output=pass_at_k["pass@1"].tolist())


@metric
def avg_execution_accuracy(df: DataFrame, ops: ZenoOptions) -> MetricReturn:
    """Average execution accuracy.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        MetricReturn: Average execution accuracy
    """
    if len(df) == 0:
        return MetricReturn(metric=0.0)
    return MetricReturn(
        metric=df[ops.distill_columns["execution_accuracy"]].fillna(0).mean()
    )
