"""Configurations for datasets."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    """A config for a dataset.

    Attributes:
        dataset: The name of the dataset, or a tuple of the dataset name and
            the version.
        split: The name of the split.
        data_column: The name of the column containing the data.
        data_format: The format of the data, if applicable.
    """

    dataset: str | tuple[str, str]
    split: str
    data_column: str
    data_format: str | None = None
