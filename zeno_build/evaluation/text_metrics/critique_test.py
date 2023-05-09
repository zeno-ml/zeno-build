"""Unit tests for the Critique-reliant evaluators."""

import os
from unittest import mock

import pandas as pd
import pytest
from zeno import DistillReturn, MetricReturn, ZenoOptions

from zeno_build.evaluation.text_metrics.critique import avg_bert_score, bert_score

example_df = pd.DataFrame(
    {
        "id": [0, 1],
        "input": ["hello world", "how are you?"],
        "output": ["goodbye world", "i am fine"],
        "label": ["hola world", "i am ok"],
    }
)


example_ops = ZenoOptions(
    id_column="id",
    data_column="input",
    label_column="label",
    output_column="output",
    distill_columns={"bert_score": "bert_score"},
    data_path="",
    label_path="",
    output_path="",
)


@pytest.fixture(autouse=True)
def mock_settings_env_vars():
    """Mock the environment variables that are used to set the Critique API key."""
    with mock.patch.dict(os.environ, {"INSPIREDCO_API_KEY": "mock"}):
        yield


def test_mock_bert_score_distill():
    """Test bert_score with a mocked call to Critique.

    This can be used as a representative for other Critique distill functions.
    """
    with mock.patch("inspiredco.critique.Critique.evaluate") as mock_evaluate:
        mock_evaluate.return_value = {
            "value": 0.3,
            "examples": [{"value": 0.3}, {"value": 0.4}],
        }
        actual_distill = bert_score(example_df, example_ops)
        expected_distill = DistillReturn(distill_output=[0.3, 0.4])
        assert isinstance(actual_distill, DistillReturn)
        assert expected_distill.distill_output == actual_distill.distill_output


def test_mock_avg_bert_score_metric():
    """Test avg_bert_score with a mocked call to Critique.

    This can be used as a representative for other Critique metrics.
    """
    with mock.patch("inspiredco.critique.Critique.evaluate") as mock_evaluate:
        # Note that the metric takes the average of the distilled values, not "value"
        my_df = pd.DataFrame(example_df)
        mock_evaluate.return_value = {
            "value": 0.3,
            "examples": [{"value": 0.3}, {"value": 0.4}],
        }
        my_df["bert_score"] = bert_score(my_df, example_ops).distill_output
        actual_metric = avg_bert_score(my_df, example_ops)
        expected_metric = MetricReturn(metric=0.35)
        assert isinstance(actual_metric, MetricReturn)
        assert expected_metric.metric == actual_metric.metric
