"""Tests for the distill functions based on sequence capitalization."""

import pandas as pd

from zeno import DistillReturn, ZenoOptions
from zeno_build.evaluation.text_features.capitalization import (
    input_capital_char_ratio, label_capital_char_ratio,
    output_capital_char_ratio)

example_df = pd.DataFrame(
    {
        "id": [0, 1],
        "input": ["HELLO WORLD", "How are you?"],
        "output": ["Goodbye world", "I am FINE"],
        "label": ["Hola world", "I am ok"],
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


def test_input_capital_char_ratio():
    """Test the input capitalization function."""
    actual_result = input_capital_char_ratio(example_df, example_ops)
    expected_result = DistillReturn(distill_output=[10 / 11, 1 / 12])
    assert isinstance(actual_result, DistillReturn)
    assert all(expected_result.distill_output == actual_result.distill_output)


def test_output_capital_char_ratio():
    """Test the output capitalization function."""
    actual_result = output_capital_char_ratio(example_df, example_ops)
    expected_result = DistillReturn(distill_output=[1 / 13, 5 / 9])
    assert isinstance(actual_result, DistillReturn)
    assert all(expected_result.distill_output == actual_result.distill_output)


def test_label_capital_char_ratio():
    """Test the label capitalization function."""
    actual_result = label_capital_char_ratio(example_df, example_ops)
    expected_result = DistillReturn(distill_output=[1 / 10, 1 / 7])
    assert isinstance(actual_result, DistillReturn)
    assert all(expected_result.distill_output == actual_result.distill_output)
