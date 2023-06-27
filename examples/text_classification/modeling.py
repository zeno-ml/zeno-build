"""A simple text classification pipeline in HuggingFace Transformers."""

from __future__ import annotations

import json
import os
import traceback
from collections.abc import Sequence

import config as text_classification_config
import datasets
import transformers

from zeno_build.cache_utils import CacheLock, fail_cache, get_cache_path


def train_model(
    training_dataset_preset: str,
    model_preset: str,
    learning_rate: float,
    num_train_epochs: int,
    weight_decay: float,
    models_dir: str,
) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """Train a model on a text classification task.

    Args:
        training_dataset: The path to the training dataset, either as string or tuple.
        model_preset: The name of the base model to use.
        learning_rate: The learning rate to use
        num_train_epoch: The number of epochs to train for
        weight_decay: The weight decay parameter to use
        training_split: The training split to use
        training_examples: The number of training examples to use
        cache_root: The root of the cache directory, if any

    Returns:
        The trained model and tokenizer.
    """
    parameters = {k: v for k, v in locals().items() if k != "models_dir"}
    output_path = get_cache_path(models_dir, parameters)
    if os.path.exists(output_path):
        if os.path.exists(output_path):
            tokenizer = transformers.AutoTokenizer.from_pretrained(output_path)
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                output_path
            )
        return model, tokenizer

    # Load tokenizer and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_preset)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_preset, num_labels=2
    )

    # Load dataset
    dataset = load_data(training_dataset_preset)
    data_column = text_classification_config.dataset_configs[
        training_dataset_preset
    ].data_column

    def tokenize_function(examples):
        return tokenizer(examples[data_column], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Define training settings
    training_args = transformers.TrainingArguments(
        output_dir=output_path,
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
    )

    # Train the model
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    trainer.train()

    # Save the model
    if output_path is not None:
        tokenizer.save_pretrained(output_path)
        model.save_pretrained(output_path)

    return model, tokenizer


def make_predictions(
    test_data: datasets.Dataset,
    data_column: str,
    label_mapping: list[str],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    bias: float = 0.0,
) -> list[str]:
    """Make predictions over a particular dataset.

    Args:
        test_data: The test data to use.
        data_column: The name of the column containing the data.
        label_mapping: The mapping from integers to labels.
        model: The model to evaluate.
        tokenizer: The tokenizer to use.
        bias: The bias to apply to the first class.

    Returns:
        The predictions in string format.
    """
    trainer = transformers.Trainer(model=model)

    def tokenize_function(examples):
        return tokenizer(examples[data_column], padding="max_length", truncation=True)

    tokenized_dataset = test_data.map(tokenize_function, batched=True)
    predictions = trainer.predict(tokenized_dataset)

    # Convert predictions to labels
    predictions.predictions[:, 0] += bias
    return [
        label_mapping[prediction]
        for prediction in predictions.predictions.argmax(axis=-1)
    ]


def load_data(dataset_preset: str) -> datasets.Dataset:
    """Load data from the huggingface library.

    Args:
        dataset: The name of the dataset to load, either:
          - A string, the name of the dataset.
          - A tuple of strings, the name of the dataset and the name of the
            subdataset.
        split: The split of the dataset to load.
        data_column: The name of the column containing the data.
        label_column: The name of the column containing the labels.

    Returns:
        The loaded dataset as dialog examples of context and reference.
    """
    config = text_classification_config.dataset_configs[dataset_preset]
    if isinstance(config.dataset, tuple):
        dname, subdname = config.dataset
        return datasets.load_dataset(dname, subdname, split=config.split)
    else:
        return datasets.load_dataset(config.dataset, split=config.split)


def get_labels(dataset: datasets.Dataset, dataset_name: str) -> list[str]:
    """Get the labels for a particular dataset.

    Args:
        dataset: The dataset to get the labels for.
        dataset_name: The dataset to get the labels for.

    Returns:
        The labels in string format.
    """
    # Load dataset
    mapping = text_classification_config.dataset_mapping.get(dataset_name, {})

    # Convert labels to strings
    label_mapping: Sequence[str] = mapping.get(
        "label_mapping", dataset.features["label"].names
    )
    return [label_mapping[x["label"]] for x in dataset]


def train_and_predict(
    test_data: list[str],
    test_dataset_preset: str,
    training_dataset_preset: str,
    model_preset: str,
    learning_rate: float,
    num_train_epochs: int,
    weight_decay: float,
    bias: float,
    models_dir: str,
    predictions_dir: str,
) -> list[str] | None:
    """Train and make predictions.

    Args:
        test_data: The data from the test dataset.
        test_dataset_preset: The name of the test dataset.
        training_dataset_preset: The name of the training dataset.
        model_preset: The name of the model to use.
        learning_rate: The learning rate to use.
        num_train_epochs: The number of training epochs.
        weight_decay: The weight decay to use.
        bias: The bias to apply to the first class.
        models_dir: The directory to save the models to.
        predictions_dir: The directory to save the predictions to.

    Returns:
        The predicted labels in string format, or None if this run is
        skipped.
    """
    # Load from cache if existing
    parameters = {
        k: v
        for k, v in locals().items()
        if k not in {"test_data", "models_dir", "predictions_dir"}
    }
    file_root = get_cache_path(predictions_dir, parameters)
    if os.path.exists(f"{file_root}.json"):
        with open(f"{file_root}.json", "r") as f:
            return json.load(f)

    with CacheLock(file_root) as cache_lock:
        # If the cache is locked, then another process is already generating
        # so just skip this one
        if not cache_lock:
            return None

        try:
            # Train the model
            model, tokenizer = train_model(
                training_dataset_preset=training_dataset_preset,
                model_preset=model_preset,
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                weight_decay=weight_decay,
                models_dir=models_dir,
            )

            # Make predictions
            predictions = make_predictions(
                test_data=test_data,
                data_column=text_classification_config.dataset_configs[
                    test_dataset_preset
                ].data_column,
                label_mapping=text_classification_config.dataset_configs[
                    training_dataset_preset
                ].label_mapping,
                model=model,
                tokenizer=tokenizer,
                bias=bias,
            )
        except Exception:
            tb = traceback.format_exc()
            fail_cache(file_root, tb)
            raise

        # Save the results
        with open(f"{file_root}.json", "w") as f:
            json.dump(predictions, f)

    return predictions
