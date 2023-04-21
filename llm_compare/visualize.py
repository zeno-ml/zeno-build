"""Run Zeno to visualize the results of a parameter search run."""
from collections.abc import Callable
from operator import itemgetter
from typing import Any

import pandas as pd
from zeno import ModelReturn, ZenoParameters, model, zeno

from llm_compare.experiment_run import ExperimentRun


def visualize(
    df: pd.DataFrame,
    references: list[Any],
    results: list[ExperimentRun],
    view: str,
    data_column: str,
    functions: list[Callable],
) -> None:
    """Run Zeno to visualize the results of a parameter search run.

    Args:
        df: DataFrame with the data to visualize
        references: List of ground truth labels
        results: List of dictionaries with model outputs
        view: The Zeno view to use for the data
        data_column: The column in the DataFrame with the data
        functions: List of functions to use in Zeno
    """
    model_results: dict[str, ExperimentRun] = {}
    for res in results:
        # Hash model params to represent in Zeno. W&B uses random names.
        name = str(hash("_".join([f"{k}={v}" for k, v in res.parameters.items()])))

        # Prevent duplicate runs being added to Zeno
        if name not in model_results:
            model_results[name] = res

    @model
    def get_model(name):
        mod = model_results[name].predictions

        def pred(df, ops):
            return ModelReturn(model_output=itemgetter(*df["index"].to_list())(mod))

        return pred

    # Print a table mapping hashes to parameters.
    for name in model_results:
        print(name, model_results[name].parameters)

    df["label"] = references
    config = ZenoParameters(
        metadata=df,
        view=view,
        models=list(model_results.keys()),
        functions=[get_model] + functions,
        data_column=data_column,
        label_column="label",
        batch_size="25000",
    )

    zeno(config)
