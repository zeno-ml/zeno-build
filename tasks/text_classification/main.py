from llm_compare import optimizers, search_space
from llm_compare.evaluators import accuracy

from . import modeling


def text_classification_main():

    # Define the space of hyperparameters to search over
    space = {
        "training_dataset": search_space.Categorical(["imdb", "ag_news"]),
        "base_model": search_space.Categorical(["distilbert-base-uncased"]),
        "learning_rate": search_space.Float(1e-5, 1e-3),
        "num_train_epochs": search_space.Int(0, 4),
        "weight_decay": search_space.Float(0.0, 0.01),
        "bias": search_space.Float(-1.0, 1.0),
    }

    # Any constants that are fed into the function
    constants = {
        "test_dataset": "imdb",
        "training_split": "train",
        "validation_split": "validation",
        "test_split": "test",
    }

    # Get the reference answers and create an evaluator for accuracy
    references = modeling.get_references(
        constants["test_dataset"], constants["test_split"]
    )
    evaluator = accuracy.AccuracyEvaluator(references)

    # Run the hyperparameter sweep
    optimizer = optimizers.StandardOptimizer()
    result = optimizer.run_sweep(
        function=modeling.train_and_predict,
        space=space,
        constants=constants,
        evaluator=evaluator,
    )


if __name__ == "__main__":
    text_classification_main()
