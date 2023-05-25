"""Tests of search spaces."""

import json
import tempfile

from zeno_build.experiments import search_space


def test_constant_value_in_scope():
    """Test that a constant value is in scope."""
    constant = search_space.Constant(1)
    assert constant.value_in_scope(1)
    assert not constant.value_in_scope(2)


def test_categorical_value_in_scope():
    """Test that a categorical value is in scope."""
    categorical = search_space.Categorical([1, 2, 3])
    assert categorical.value_in_scope(1)
    assert not categorical.value_in_scope(4)


def test_discrete_value_in_scope():
    """Test that a discrete value is in scope."""
    discrete = search_space.Discrete([1, 2, 3])
    assert discrete.value_in_scope(1)
    assert not discrete.value_in_scope(4)


def test_float_value_in_scope():
    """Test that a float value is in scope."""
    float_ = search_space.Float(0.0, 1.0)
    assert float_.value_in_scope(0.5)
    assert not float_.value_in_scope(1.5)


def test_int_value_in_scope():
    """Test that an int value is in scope."""
    int_ = search_space.Int(0, 1)
    assert int_.value_in_scope(0)
    assert int_.value_in_scope(1)
    assert not int_.value_in_scope(2)


def test_get_valid_param_files():
    """Test that get_valid_param_files works."""
    space = search_space.CombinatorialSearchSpace(
        {"a": search_space.Constant(1), "b": search_space.Categorical([1, 2, 3])}
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create valid and invalid parameter files.
        with open(f"{temp_dir}/valid1.zbp", "w") as file:
            json.dump({"a": 1, "b": 1}, file)
        with open(f"{temp_dir}/valid2.zbp", "w") as file:
            json.dump({"a": 1, "b": 3}, file)
        with open(f"{temp_dir}/constant_invalid.zbp", "w") as file:
            json.dump({"a": 2, "b": 1}, file)
        with open(f"{temp_dir}/categorical_invalid.zbp", "w") as file:
            json.dump({"a": 1, "b": 5}, file)

        # Check that get_valid_param_files returns the correct files.
        actual_output = sorted(space.get_valid_param_files(temp_dir, False))
        expected_output = [f"{temp_dir}/valid1.zbp", f"{temp_dir}/valid2.zbp"]
        assert actual_output == expected_output


def test_get_valid_param_with_locks_and_fails():
    """Test that get_valid_param_files works."""
    space = search_space.CombinatorialSearchSpace(
        {"a": search_space.Constant(1), "b": search_space.Categorical([1, 2, 3])}
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create valid and invalid parameter files.
        with open(f"{temp_dir}/valid1.zbp", "w") as file:
            json.dump({"a": 1, "b": 1}, file)
        with open(f"{temp_dir}/valid2.zbp", "w") as file:
            json.dump({"a": 1, "b": 3}, file)
        with open(f"{temp_dir}/valid2.zblock", "w") as file:
            pass
        with open(f"{temp_dir}/valid3.zbp", "w") as file:
            json.dump({"a": 1, "b": 1}, file)
        with open(f"{temp_dir}/valid3.zbfail", "w") as file:
            pass
        with open(f"{temp_dir}/valid4.zbp", "w") as file:
            json.dump({"a": 1, "b": 3}, file)
        with open(f"{temp_dir}/constant_invalid.zbp", "w") as file:
            json.dump({"a": 2, "b": 1}, file)
        with open(f"{temp_dir}/categorical_invalid.zbp", "w") as file:
            json.dump({"a": 1, "b": 5}, file)

        # Check that get_valid_param_files returns the correct files without locks
        actual_output_no_progress = sorted(space.get_valid_param_files(temp_dir, False))
        expected_output_no_progress = [
            f"{temp_dir}/valid1.zbp",
            f"{temp_dir}/valid4.zbp",
        ]
        assert actual_output_no_progress == expected_output_no_progress
        # Check that get_valid_param_files returns the correct files with locks
        actual_output_in_progress = sorted(space.get_valid_param_files(temp_dir, True))
        expected_output_in_progress = [
            f"{temp_dir}/valid1.zbp",
            f"{temp_dir}/valid2.zbp",
            f"{temp_dir}/valid4.zbp",
        ]
        assert actual_output_in_progress == expected_output_in_progress
