"""Run Zeno to visualize the results of a parameter search run."""
from operator import itemgetter

import pandas as pd
from zeno import (
    MetricReturn,
    ModelReturn,
    ZenoOptions,
    ZenoParameters,
    metric,
    model,
    zeno,
)


@metric
def accuracy(df: pd.DataFrame, ops: ZenoOptions):
    """Calculate the accuracy of a model.

    Args:
        df (pd.DataFrame): DataFrame from Zeno
        ops (ZenoOptions): Options from Zeno

    Returns:
        MetricReturn: accuracy value
    """
    if len(df) == 0:
        return MetricReturn(metric=0.0)
    return MetricReturn(metric=(df[ops.label_column] == df[ops.output_column]).mean())


def visualize(
    df: pd.DataFrame, references, results, view: str, data_column: str
) -> None:
    """Run Zeno to visualize the results of a parameter search run.

    Args:
        df (pd.DataFrame): DataFrame with the data to visualize
        references (_type_): List of ground truth labels
        results (_type_): List of dictionaries with model outputs
        view (str): The Zeno view to use for the data
        data_column (str): The column in the DataFrame with the data
    """

    model_names = []
    model_results = {}
    for res in results:
        # Hash model params to represent in Zeno. W&B uses random names.
        name = str(hash("_".join([str(r) for r in res["parameters"].values()])))

        # Prevent duplicate runs being added to Zeno
        if name not in model_names:
            model_names.append(name)
            model_results[name] = res

    @model
    def get_model(name):
        mod = model_results[name]["predictions"]

        def pred(df, ops):
            return ModelReturn(model_output=itemgetter(*df["index"].to_list())(mod))

        return pred

    # Print a table mapping hashes to parameters.
    for name in model_names:
        print(name, model_results[name]["parameters"])

    df["label"] = references
    config = ZenoParameters(
        metadata=df,
        view=view,
        models=model_names,
        functions=[get_model, accuracy],
        data_column=data_column,
        label_column="label",
        batch_size="25000",
    )

    zeno(config)
