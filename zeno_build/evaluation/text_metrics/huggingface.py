"""Evaluating code execution accuracy."""

import evaluate
import tqdm
from pandas import DataFrame
from zeno import DistillReturn, MetricReturn, ZenoOptions, distill, metric


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

    for d in eval_dict:
        d["references"] = d.get(ops.label_column)
        d["target"] = [d.get(ops.data_column) + d.get(ops.output_column)]
        if len(d["references"]) == 0 or len(d["references"][0]) == 0:
            raise ValueError(f"Empty references at {d}")

    metric_name = "code_eval"
    eval_metric = evaluate.load(metric_name)

    # evaluate all outputs
    all_results = []
    all_messages = []
    for i in tqdm.tqdm(
        range(0, len(eval_dict)),
        desc=f"Evaluating {metric_name}",
    ):
        predictions = [eval_dict[i]["target"]]
        references = [eval_dict[i]["references"]]

        pass_at_k, results = eval_metric.compute(
            predictions=predictions,
            references=references,
            k=[1],
        )
        all_results.append(pass_at_k["pass@1"])
        all_messages.append(results[0])
    # Save the messages for possible future reference
    df["execution_accuracy_messages"] = all_messages
    return DistillReturn(distill_output=all_results)


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
