"""A simple text classification pipeline in HuggingFace Transformers."""

import json
import os

from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          PreTrainedModel, PreTrainedTokenizer, Trainer,
                          TrainingArguments)

from llm_compare.cache_utils import get_cache_path


def train_model(
    training_dataset: str,
    base_model: str,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 3,
    weight_decay: float = 0.01,
    training_split: str = "train",
    validation_split: str = "validation",
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Train a model on a text classification task.

    Args:
        training_dataset: The path to the training dataset.
        base_model: The name of the base model to use.

    Returns:
        The trained model and tokenizer.
    """
    # If the model is already trained, recover it from the cache
    parameters = dict(locals())
    parameters["__name__"] = train_model.__name__
    model_dir = get_cache_path("text_classification", parameters)
    if os.path.exists(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        return tokenizer, model

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)

    # Load dataset
    dataset = load_dataset(training_dataset, split=training_split)

    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Define training settings
    training_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets[training_split],
        eval_dataset=tokenized_datasets[validation_split],
    )
    trainer.train()

    # Save the model
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)

    return model, tokenizer


def make_predictions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_dataset: str,
    bias: float = 0.0,
    test_split: str = "test",
) -> list[str]:
    """Make predictions over a particular dataset.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer to use.
        test_dataset: The path to the test dataset.
        bias: The bias to apply to the first class.
        test_split: The split of the test dataset to use.

    Returns:
        The predictions in string format.
    """
    # Load dataset
    dataset = load_dataset(test_dataset, split=test_split)

    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Make predictions
    trainer = Trainer(model=model)
    predictions = trainer.predict(tokenized_datasets[test_split])

    # Convert predictions to labels
    labels = dataset.features["label"].names
    predictions.predictions[:, 0] += bias
    return [
        labels[prediction] for prediction in predictions.predictions.argmax(axis=-1)
    ]


def get_references(test_dataset: str, test_split: str = "test") -> list[str]:
    """Get the reference answers for a particular dataset.

    Args:
        test_dataset: The path to the test dataset.
        test_split: The split of the test dataset to use.

    Returns:
        The references in string format.
    """
    # Load dataset
    dataset = load_dataset(test_dataset, split=test_split)

    # Convert labels to strings
    labels = dataset.features["label"].names
    return [labels[label] for label in dataset["label"]]


def train_and_predict(
    training_dataset: str,
    test_dataset: str,
    base_model: str,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 3,
    weight_decay: float = 0.01,
    bias: float = 0.0,
    training_split: str = "train",
    validation_split: str = "validation",
    test_split: str = "test",
) -> list[str]:
    """Train and make predictions."""
    # If the experiment is already finished, recover it from the cache
    parameters = dict(locals())
    parameters["__name__"] = train_model.__name__
    result_file = get_cache_path("text_classification", parameters, extension="json")
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            return json.load(f)

    # Train the model
    model, tokenizer = train_model(
        training_dataset,
        base_model,
        learning_rate,
        num_train_epochs,
        weight_decay,
        training_split,
        validation_split,
    )

    # Make predictions
    predictions = make_predictions(
        model,
        tokenizer,
        test_dataset,
        bias,
        test_split,
    )

    # Save the results
    with open(result_file, "w") as f:
        json.dump(predictions, f)

    return predictions
