"""Tests for the distill functions based on sequence capitalization."""

import pandas as pd

from zeno import DistillReturn, ZenoOptions
from zeno_build.evaluation.text_features.numbers import (
    digit_count,
    english_number_count,
)

example_df = pd.DataFrame(
    {
        "id": [0, 1, 2],
        "label": ["one, two, three", "It's as easy as 1234", "ONE TWO THREE"],
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


def test_digit_count():
    """Test the digit count function."""
    actual_result = digit_count(example_df, example_ops)
    expected_result = DistillReturn(distill_output=[0, 4, 0])
    assert isinstance(actual_result, DistillReturn)
    assert all(expected_result.distill_output == actual_result.distill_output)


def test_english_number_count():
    """Test the English number count function."""
    actual_result = english_number_count(example_df, example_ops)
    expected_result = DistillReturn(distill_output=[3, 0, 3])
    assert isinstance(actual_result, DistillReturn)
    assert expected_result.distill_output == actual_result.distill_output
