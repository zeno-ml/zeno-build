"""Tests for the cache_utils package."""

import os
import tempfile

from zeno_build import cache_utils


def test_get_cache_path():
    """Test that get_cache_path() returns the correct path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # First lookup of the path
        first_path = cache_utils.get_cache_path(temp_dir, {"test_param": "test_value"})
        existant_files = os.listdir(temp_dir)
        assert len(existant_files) == 1
        assert existant_files[0].endswith(".zbp")
        assert existant_files[0][:-4] == first_path.split("/")[-1]

        # Second lookup of the path
        second_path = cache_utils.get_cache_path(
            temp_dir, {"test_param": "test_value2"}
        )
        existant_files = os.listdir(temp_dir)
        assert len(existant_files) == 2
        assert second_path != first_path

        # Third lookup of the path, identical to first
        third_path = cache_utils.get_cache_path(temp_dir, {"test_param": "test_value"})
        existant_files = os.listdir(temp_dir)
        assert len(existant_files) == 2
        assert third_path == first_path
