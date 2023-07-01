"""Length-related features."""
from pandas import DataFrame
from zeno import DistillReturn, ZenoOptions, distill

from zeno_build.evaluation.text_tokenizers.unicode import tokenize


@distill
def digit_count(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Number of digits in the output.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: Number of digits in the output
    """
    return DistillReturn(distill_output=df[ops.label_column].str.count(r"[0-9]"))


@distill
def english_number_count(df: DataFrame, ops: ZenoOptions) -> DistillReturn:
    """Number of English number words in the output.

    Args:
        df: Zeno DataFrame
        ops: Zeno options

    Returns:
        DistillReturn: Number of English number words in the output
    """
    english_number_words = {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eightteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
        "trillion",
        "quadrillion",
        "quintillion",
    }
    number_of_numbers = [
        sum(
            1 if y.lower() in english_number_words else 0
            for y in tokenize(x, merge_symbol="").split()
        )
        for x in df[ops.label_column]
    ]
    return DistillReturn(distill_output=number_of_numbers)
