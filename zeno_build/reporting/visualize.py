"""Run Zeno to visualize the results of a parameter search run."""
from collections.abc import Callable
from operator import itemgetter
from typing import Any

import pandas as pd
from zeno import ModelReturn, ZenoParameters, model, zeno

from zeno_build.experiments.experiment_run import ExperimentRun


def visualize(
    df: pd.DataFrame,
    labels: list[Any],
    results: list[ExperimentRun],
    view: str,
    data_column: str,
    functions: list[Callable],
    zeno_config: dict = {},
) -> None:
    """Run Zeno to visualize the results of a parameter search run.

    Args:
        df: DataFrame with the data to visualize. Must contain "data_column" column.
        labels: List of ground truth labels
        results: List of dictionaries with model outputs
        view: The Zeno view to use for the data
        data_column: The column in the DataFrame with the data
        functions: List of functions to use in Zeno
        zeno_config: Zeno configuration parameters
    """
    if len(df) != len(labels):
        raise ValueError("Length of data and labels must be equal.")
    if data_column not in df.columns:
        raise ValueError(f"Data column {data_column} not in DataFrame.")
    model_results: dict[str, ExperimentRun] = {x.name: x for x in results}

    @model
    def get_model(name):
        mod = model_results[name].predictions

        def pred(df, ops):
            return ModelReturn(model_output=itemgetter(*df["index"].to_list())(mod))

        return pred

    # Print a table mapping hashes to parameters.
    for name in model_results:
        print(name, model_results[name].parameters)

    df["label"] = labels
    config = ZenoParameters(
        metadata=df,
        view=view,
        models=list(model_results.keys()),
        functions=[get_model] + functions,
        data_column=data_column,
        label_column="label",
        batch_size=100000,
        multiprocessing=False,
    )
    config = config.copy(update=zeno_config)

    zeno(config)
