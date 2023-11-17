"""Tests of unicode tokenizer."""

from zeno_build.evaluation.text_tokenizers.unicode import (
    DEFAULT_MERGE_SYMBOL,
    detokenize,
    tokenize,
)


def test_unicode_tokenizer_basic():
    """Test basic tokenization."""
    dms = DEFAULT_MERGE_SYMBOL
    assert tokenize("Hello, world!") == f"Hello {dms}, world {dms}!"


def test_unicode_tokenizer_without_merge():
    """Test basic tokenization without the merge character."""
    assert tokenize("Hello, world!", merge_symbol="") == "Hello , world !"


def test_unicode_detokenize_equal():
    """Test detokenization.

    Read in the current file and make sure that each line detokenizes to itself.
    """
    with open(__file__, "r") as f:
        for line in f:
            if "▁" not in line:
                assert detokenize(tokenize(line)) == line
