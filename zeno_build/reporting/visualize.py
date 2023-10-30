"""Run Zeno to visualize the results of a parameter search run."""
from collections.abc import Callable
from operator import itemgetter
from typing import Any

import pandas as pd
from zeno import ModelReturn, ZenoParameters, model
from zeno.classes.base import ZenoColumnType
from zeno_build.experiments.experiment_run import ExperimentRun
from zeno.backend import ZenoBackend
from zeno_client import ZenoClient, ZenoMetric

import os
import time

def visualize(
    df: pd.DataFrame,
    labels: list[Any],
    results: list[ExperimentRun],
    view: str,
    data_column: str,
    functions: list[Callable],
    zeno_config: dict = {},
    project_name:str = ""
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
    if "ZENO_CLIENT_KEY" not in os.environ:
        raise ValueError(
            "ZENO_CLIENT_KEY environment variable must be set to visualize results."
        )

    zeno_client = ZenoClient(os.environ["ZENO_CLIENT_KEY"])
    if project_name == "":
        project_name = time.strftime("%Y-%m-%d %H-%M-%S")
        print(f"Creating project {project_name}")

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

    zeno_instance = ZenoBackend(config)

    # wait for the backend to complete processing
    zeno_thread = zeno_instance.start_processing()
    zeno_thread.join()

    project = zeno_client.create_project(
                    name=project_name,
                    view=view,
                    metrics=[ZenoMetric(id=idx,name=metric, type='mean', columns=[metric[4:]]) for idx,metric in enumerate(zeno_instance.metric_functions.keys())])

    # upload the dataset
    dataset_columns = ["index", data_column, "label"]
    dataset_df = zeno_instance.df[dataset_columns]
    for predistill_function in zeno_instance.predistill_functions.keys():
        dataset_df[predistill_function] = zeno_instance.df[ZenoColumnType.PREDISTILL+predistill_function]
    project.upload_dataset(dataset_df, id_column="index", data_column=data_column, label_column="label")

    # upload the model results
    for model_name in model_results:
        df_model = pd.DataFrame({"index": dataset_df["index"]})
        df_model["output"] = zeno_instance.df[ZenoColumnType.OUTPUT+"output" + model_name]
        for postdistill_function in zeno_instance.postdistill_functions.keys():
            df_model[postdistill_function] = zeno_instance.df[ZenoColumnType.POSTDISTILL + postdistill_function + model_name]
        project.upload_system(df_model, name=model_name, id_column="index", output_column="output")
