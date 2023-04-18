"""Tests for the cache_utils package."""

import os
import tempfile
from unittest import mock

from llm_compare import cache_utils


def test_get_cache_root_default():
    """Test that get_cache_root() returns the correct value."""
    # mock os.path.expanduser
    with mock.patch("os.path.expanduser") as mock_expanduser:
        mock_expanduser.return_value = "/home/user"
        assert cache_utils.get_cache_root() == "/home/user/.cache/llm_compare"


def test_get_cache_root_environ():
    """Test that get_cache_root() returns the correct value with environmental vars."""
    with mock.patch.dict(os.environ, {"LLM_COMPARE_CACHE": "/tmp/llm_compare_cache"}):
        assert cache_utils.get_cache_root() == "/tmp/llm_compare_cache"


def test_clear_task_cache():
    """Test that clear_task_cache() calls shutil.rmtree with the correct arguments."""
    with mock.patch("shutil.rmtree") as mock_rmtree, mock.patch.dict(
        os.environ, {"LLM_COMPARE_CACHE": "/tmp/llm_compare_cache"}
    ):
        cache_utils.clear_task_cache("test_task")
        mock_rmtree.assert_called_once_with("/tmp/llm_compare_cache/test_task")


def test_get_cache_path():
    """Test that get_cache_path() returns the correct path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with mock.patch(
            "llm_compare.cache_utils.get_cache_root"
        ) as mock_get_cache_root:
            mock_get_cache_root.return_value = temp_dir
            my_path = cache_utils.get_cache_path(
                "test_task", {"test_param": "test_value"}
            )
            existant_files = os.listdir(os.path.join(temp_dir, "test_task"))
            assert len(existant_files) == 1
            assert existant_files[0].endswith(".llmcp")
            assert existant_files[0][:-6] == my_path.split("/")[-1]
