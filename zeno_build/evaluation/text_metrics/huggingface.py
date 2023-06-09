"""Evaluating code execution accuracy."""

import tqdm
import evaluate
from pandas import DataFrame
from zeno import ZenoOptions, DistillReturn, MetricReturn, distill, metric


def call_evaluate(
    df: DataFrame,
    ops: ZenoOptions,
    metric_name: str,
    config: dict,
    batch_size: int = 1,
) -> DistillReturn:
    """Call HuggingFace evaluate.
    
    Args:
        df: Zeno Dataframe
        ops: Zeno Options
        metric_name: Name of the metric
        config: Metric configuration
    
    Returns:
        MetricReturn: Metric results
    """
    eval_dict = df[[ops.output_column, ops.label_column, ops.data_column]].to_dict(
        "records"
    )

    for d in eval_dict:
        d["references"] = d.get(ops.label_column)
        d["target"] = [d.get(ops.output_column)]
        # d["target"] = [d.get(ops.data_column) + d.pop(ops.output_column)]
        if len(d["references"][0]) == 0:
            raise ValueError(f"Empty referencea at {d}")
    
    eval_metric = evaluate.load(metric_name)

    # evaluate batch by batch
    all_results = []
    for i in tqdm.tqdm(
        range(0, len(eval_dict), batch_size), desc=f"Evaluating {metric_name}",
    ):
        pass_at_k, results = eval_metric.compute(
            predictions=[d["target"] for d in eval_dict[i: i + batch_size]],
            references=[d["references"] for d in eval_dict[i: i + batch_size]],
            **config
        )
        all_results.append(round(pass_at_k["pass@1"], 6))
    
    return DistillReturn(distill_output=all_results)



@distill
def execution_accuracy(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    ops.output_column = "outputs"

    import os
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return call_evaluate(df, ops, "code_eval", {"k": [1]})


@metric
def avg_execution_accuracy(df: DataFrame, ops: ZenoOptions) -> MetricReturn:
    if len(df) == 0:
        return MetricReturn(metric=0.0)
    return MetricReturn(metric=sum(df[ops.distill_columns["execution_accuracy"]]) / len(df))