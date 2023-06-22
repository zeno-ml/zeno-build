# Copyright 2023 Zeno Build Team
#
# This is an adaption of "A simple, reversible, language-agnostic tokenizer",
# published first in "Spell Once, Summon Anywhere: A Two-Level Open-Vocabulary
# Language Model" ( https://arxiv.org/abs/1804.08205 ), and obtainable at
# https://sjmielke.com/papers/tokenize/ .
#
# Copyright 2018 Sebastian J. Mielke
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
"""Tokenizer based on unicode character classes."""

import unicodedata

DEFAULT_MERGE_SYMBOL = "▁"


def _is_weird(c):
    return not (
        unicodedata.category(c)[0] in "LMN" or c.isspace()
    )  # Caution: python's isalnum(c) does not accept Marks (category M*)!


def tokenize(
    instring: str,
    merge_symbol: str = DEFAULT_MERGE_SYMBOL,
) -> str:
    """Tokenize on unicode categories, but keep weird characters separate.

    Args:
        instring: The string to tokenize.
        merge_symbol: The symbol to use for merges.

    Returns:
        A list of tokens.
    """
    # Walk through the string!
    outsequence = []
    for i in range(len(instring)):
        c = instring[i]
        c_p = instring[i - 1] if i > 0 else c
        c_n = instring[i + 1] if i < len(instring) - 1 else c

        # Is it a letter (i.e. Unicode category starts with 'L')?
        # Or alternatively, is it just whitespace?
        # So if it's not weird, just copy.
        if not _is_weird(c):
            outsequence.append(c)
        # Otherwise it should be separated!
        else:
            # Was there something non-spacey before?
            # Then we have to introduce a new space and a merge marker.
            if not c_p.isspace():
                outsequence.append(" " + merge_symbol)
            # Copy character itself
            outsequence.append(c)

            # Is there something non-spacey after?
            # Then we have to introduce a new space and a merge marker.
            # If, however the next character would just want to merge left
            # anyway, no need to do it now.
            if not c_n.isspace() and not _is_weird(c_n):
                outsequence.append(merge_symbol + " ")

    return "".join(outsequence)


def detokenize(
    instring: str,
    merge_symbol: str = DEFAULT_MERGE_SYMBOL,
) -> str:
    """Detokenize a string tokenized with tokenize()."""
    # Walk through the string!
    outsequence = []
    i = 0
    while i < len(instring):
        c = instring[i]
        c_n = instring[i + 1] if i < len(instring) - 1 else c
        c_nn = instring[i + 2] if i < len(instring) - 2 else c

        # It could be one of the spaces we introduced
        if c + c_n == " " + merge_symbol and _is_weird(c_nn):
            i += 2
        elif _is_weird(c) and c_n + c_nn == merge_symbol + " ":
            outsequence.append(c)
            i += 3
        else:
            outsequence.append(c)
            i += 1

    return "".join(outsequence)
