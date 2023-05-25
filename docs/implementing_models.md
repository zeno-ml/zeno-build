# Implementing Models in Zeno Build

To implement models in Zeno Build, we suggest that you build on the an
existing example in the [examples](/examples/) directory. The parts relevant
to model implementation are usually the `modeling.py` and `main.py` files.

We will illustrate these by taking examples of the
[chatbot][/examples/chatbot/] task, which builds a chatbot and evaluates
it on a dataset of customer service conversations.

## `main.py`

The `main.py` file is the main entrypoint for the experiment. It specifies
the whole experimental flow which basically follows the following pseudo-code:

```python
def main():
    # Load the test data 
    load_data()
    # Loop over hyperparameter settings
    while not optimizer.is_complete():
        # Generate predictions
        make_predictions()
    # Evaluate and visualize the predictions
    visualize()
```

Let's look at a concrete example from the
[chatbot main.py file](/examples/chatbot/main.py).

### Preliminaries

We start with a function definition specifying the output directory
where results are written to `results_dir`,
and whether to do prediction and visualization. You can
opt to do only prediction (to just generate outputs), or only
visualization (if you already have outputs and want to visualize them).

```python
def chatbot_main(
    results_dir: str,
    do_prediction: bool = True,
    do_visualization: bool = True,
):
```

### Loading Testing Data

Next we load the test data. The concepts of `search_space` and
`chatbot_config` that will be covered in the
[next section on specifying parameters](specifying_parameters.md)

```python
    dataset_preset = chatbot_config.space.dimensions["dataset_preset"]
    if not isinstance(dataset_preset, search_space.Constant):
        raise ValueError("All experiments must be run on a single dataset.")
    dataset_config = chatbot_config.dataset_configs[dataset_preset.value]
```

## `modeling.py`

## Next Steps

Next,
