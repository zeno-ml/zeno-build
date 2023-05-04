"""A simple text classification pipeline in HuggingFace Transformers."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import datasets
import transformers

from tasks.text_classification import config as classification_config
from zeno_build.cache_utils import get_cache_path


def train_model(
    training_dataset: str | tuple[str, str],
    base_model: str,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 3,
    weight_decay: float = 0.01,
    training_split: str = "train",
    training_examples: int | None = None,
    cache_root: str | None = None,
) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """Train a model on a text classification task.

    Args:
        training_dataset: The path to the training dataset, either as string or tuple.
        base_model: The name of the base model to use.
        learning_rate: The learning rate to use
        num_train_epoch: The number of epochs to train for
        weight_decay: The weight decay parameter to use
        training_split: The training split to use
        training_examples: The number of training examples to use
        cache_root: The root of the cache directory, if any

    Returns:
        The trained model and tokenizer.
    """
    # Load from cache if existing
    cache_path: str | None = None
    if cache_root is not None:
        parameters = dict(locals())
        parameters["__name__"] = train_model.__name__
        cache_path = get_cache_path(cache_root, parameters)
        if os.path.exists(cache_path):
            tokenizer = transformers.AutoTokenizer.from_pretrained(cache_path)
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                cache_path
            )
            return tokenizer, model

    # Load tokenizer and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2
    )

    # Load dataset
    dataset = load_data(training_dataset, training_split, examples=training_examples)
    mapping = classification_config.dataset_mapping.get(training_dataset, {})

    # Tokenize data
    input_name = mapping.get("input", "text")

    def tokenize_function(examples):
        return tokenizer(examples[input_name], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Define training settings
    training_args = transformers.TrainingArguments(
        output_dir=cache_path,
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
    )

    # Train the model
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )
    trainer.train()

    # Save the model
    if cache_path is not None:
        tokenizer.save_pretrained(cache_path)
        model.save_pretrained(cache_path)

    return model, tokenizer


def make_predictions(
    data: datasets.Dataset,
    test_dataset: str | tuple[str, str],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    bias: float = 0.0,
) -> list[str]:
    """Make predictions over a particular dataset.

    Args:
        data: The data from the test dataset.
        test_dataset: The path to the test dataset.
        model: The model to evaluate.
        tokenizer: The tokenizer to use.
        bias: The bias to apply to the first class.
        test_split: The split of the test dataset to use.

    Returns:
        The predictions in string format.
    """
    # Load dataset
    mapping = classification_config.dataset_mapping.get(test_dataset, {})

    # Tokenize data
    input_name = mapping.get("input", "text")

    def tokenize_function(examples):
        return tokenizer(examples[input_name], padding="max_length", truncation=True)

    tokenized_datasets = data.map(tokenize_function, batched=True)

    # Make predictions
    trainer = transformers.Trainer(model=model)
    predictions = trainer.predict(tokenized_datasets)

    # Convert predictions to labels
    labels: Sequence[str] = mapping.get("label_mapping", data.features["label"].names)
    predictions.predictions[:, 0] += bias
    return [
        labels[prediction] for prediction in predictions.predictions.argmax(axis=-1)
    ]


def load_data(
    dataset: str | tuple[str, str], split: str, examples: int | None = None
) -> datasets.Dataset:
    """Get the full dataset for task.

    Args:
        dataset: The path to the test dataset.
        split: The split of the test dataset to use.
        examples: The number of examples to use.

    Returns:
        The dataset.
    """
    if isinstance(dataset, tuple):
        dname, subdname = dataset
        loaded_data = datasets.load_dataset(dname, subdname, split=split)
    else:
        loaded_data = datasets.load_dataset(dataset, split=split)
    if examples is not None:
        loaded_data = loaded_data.select(range(examples))
    return loaded_data


def get_labels(dataset: datasets.Dataset, dataset_name: str) -> list[str]:
    """Get the labels for a particular dataset.

    Args:
        dataset: The dataset to get the labels for.
        dataset_name: The dataset to get the labels for.

    Returns:
        The labels in string format.
    """
    # Load dataset
    mapping = classification_config.dataset_mapping.get(dataset_name, {})

    # Convert labels to strings
    label_mapping: Sequence[str] = mapping.get(
        "label_mapping", dataset.features["label"].names
    )
    return [label_mapping[x["label"]] for x in dataset]


def train_and_predict(
    data: datasets.Dataset,
    test_dataset: str,
    training_dataset: str,
    base_model: str,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 3,
    weight_decay: float = 0.01,
    bias: float = 0.0,
    training_split: str = "train",
    training_examples: int | None = None,
    cache_root: str | None = None,
) -> list[str]:
    """Train and make predictions.

    Args:
        data: The test data in huggingface dataset format.
        test_dataset: The name of the testing dataset.
        training_dataset: The name of the training dataset.
        base_model: The name of the base model to use.
        learning_rate: The learning rate to use.
        num_train_epochs: The number of training epochs.
        weight_decay: The weight decay to use.
        bias: The bias to apply to the first class at inference time.
        training_split: The split of the training dataset to use.
        training_examples: The number of examples to use from the training dataset, or
            None to use all of them.
        cache_root: The root of the cache directory, if any

    Returns:
        The predicted labels in string format.
    """
    # Load from cache if existing
    cache_path: str | None = None
    if cache_root is not None:
        parameters = dict(locals())
        parameters["__name__"] = train_and_predict.__name__
        parameters.pop(
            "data"
        )  # We assume that knowing the name `test_dataset` is enough
        cache_path = get_cache_path(cache_root, parameters, extension="json")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)

    # Train the model
    model, tokenizer = train_model(
        training_dataset=training_dataset,
        base_model=base_model,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        training_split=training_split,
        training_examples=training_examples,
        cache_root=cache_root,
    )

    # Make predictions
    predictions = make_predictions(
        data=data,
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        bias=bias,
    )

    # Save the results
    if cache_path is not None:
        with open(cache_path, "w") as f:
            json.dump(predictions, f)

    return predictions
