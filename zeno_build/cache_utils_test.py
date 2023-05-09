"""Tests for the cache_utils package."""

import os
import tempfile

from zeno_build import cache_utils


def test_get_cache_path():
    """Test that get_cache_path() returns the correct path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        my_path = cache_utils.get_cache_path(temp_dir, {"test_param": "test_value"})
        existant_files = os.listdir(temp_dir)
        assert len(existant_files) == 1
        assert existant_files[0].endswith(".zbp")
        assert existant_files[0][:-4] == my_path.split("/")[-1]
