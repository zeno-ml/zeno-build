"""Tests for the distill functions based on sequence length."""

import pandas as pd

from zeno import DistillReturn, ZenoOptions
from zeno_build.evaluation.text_features.length import input_length, output_length

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
    distill_columns={},
    data_path="",
    label_path="",
    output_path="",
)


def test_input_length():
    """Test the input length function."""
    actual_result = input_length(example_df, example_ops)
    expected_result = DistillReturn(distill_output=[11, 12])
    assert isinstance(actual_result, DistillReturn)
    assert all(expected_result.distill_output == actual_result.distill_output)


def test_output_length():
    """Test the output length function."""
    actual_result = output_length(example_df, example_ops)
    expected_result = DistillReturn(distill_output=[13, 9])
    assert isinstance(actual_result, DistillReturn)
    assert all(expected_result.distill_output == actual_result.distill_output)
