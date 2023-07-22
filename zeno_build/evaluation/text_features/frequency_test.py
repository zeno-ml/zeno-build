"""Tests for the distill functions based on sequence frequency."""

import pandas as pd

from zeno import DistillReturn, ZenoOptions
from zeno_build.evaluation.text_features.frequency import (
    input_max_word_freq, label_max_word_freq, output_max_word_freq)

example_df = pd.DataFrame(
    {
        "id": [0, 1],
        "input": ["hello hello", "a b c a b c a b"],
        "output": ["yes", "no no"],
        "label": ["", "a b r a c a d a b r a c a d"],
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


def test_input_max_word_freq():
    """Test the input frequency function."""
    actual_result = input_max_word_freq(example_df, example_ops)
    expected_result = DistillReturn(distill_output=[2, 3])
    assert isinstance(actual_result, DistillReturn)
    assert all(expected_result.distill_output == actual_result.distill_output)


def test_output_max_word_freq():
    """Test the output frequency function."""
    actual_result = output_max_word_freq(example_df, example_ops)
    expected_result = DistillReturn(distill_output=[1, 2])
    assert isinstance(actual_result, DistillReturn)
    assert all(expected_result.distill_output == actual_result.distill_output)


def test_label_max_word_freq():
    """Test the label frequency function."""
    actual_result = label_max_word_freq(example_df, example_ops)
    expected_result = DistillReturn(distill_output=[0, 6])
    assert isinstance(actual_result, DistillReturn)
    assert all(expected_result.distill_output == actual_result.distill_output)
