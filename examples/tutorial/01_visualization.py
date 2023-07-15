"""A tutorial of how to perform visualization with Zeno Build."""

import pandas as pd

from zeno_build.evaluation.text_features.exact_match import avg_exact_match, exact_match
from zeno_build.evaluation.text_features.length import input_length, output_length
from zeno_build.experiments.experiment_run import ExperimentRun
from zeno_build.reporting.visualize import visualize


def main():
    """Run the visualization example."""
    data = ["What is 5+5?", "What is 3+2?", "What is 6-5?", "What is 12-2?"]
    labels = ["10", "5", "1", "10"]
    operation_type = ["addition", "addition", "subtraction", "subtraction"]
    df = pd.DataFrame({"text": data, "label": labels, "operation_type": operation_type})

    result1 = ExperimentRun(
        name="math_dunce",
        parameters={"model": "math_dunce", "skill": 3},
        predictions=["5", "4", "1", "5"],
    )
    result2 = ExperimentRun(
        name="math_master",
        parameters={"model": "math_master", "skill": 9},
        predictions=["10", "5", "1", "14"],
    )

    functions = [
        output_length,
        input_length,
        exact_match,
        avg_exact_match,
    ]

    visualize(
        df,
        labels,
        [result1, result2],
        "text-classification",
        "text",
        functions,
        zeno_config={"cache_path": "zeno_cache"},
    )


if __name__ == "__main__":
    main()
