# Implementing Models in Zeno Build

To implement models in Zeno Build, we suggest that you build on the an
existing example in the [examples](/examples/) directory. The parts relevant
to model implementation are usually the `modeling.py` and `main.py` files.

We will illustrate these by taking examples of the
[chatbot](/examples/chatbot/) task, which builds a chatbot and evaluates
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
[next section on specifying parameters](specifying_parameters.md),
but the following two lines essentially acquire the dataset
configuration that can be used to read in data.

```python
    dataset_preset = chatbot_config.space.dimensions["dataset_preset"]
    if not isinstance(dataset_preset, search_space.Constant):
        raise ValueError("All experiments must be run on a single dataset.")
    dataset_config = chatbot_config.dataset_configs[dataset_preset.value]
```

Then we actually do the data loading by calling the `process_data()` function
that is implemented in `modeling.py`. The details of this function will be
task specific.

```python
    contexts_and_labels: list[ChatMessages] = process_data(
        dataset=dataset_config.dataset,
        split=dataset_config.split,
        data_format=dataset_config.data_format,
        data_column=dataset_config.data_column,
        output_dir=data_dir,
    )
```

For the chatbot task, we split this into inputs (the chat context)
and labels (the next message in the conversation).

```python
    labels: list[str] = []
    contexts: list[ChatMessages] = []
    for x in contexts_and_labels:
        labels.append(x.messages[-1].content)
        contexts.append(ChatMessages(x.messages[:-1]))
```

### Making Predictions for Different Parameters

Next, if we are doing predictions,

```python
    if do_prediction:
```

we first initialize the optimizer, which will propose different
parameter settings for us to try.

```python
        optimizer = standard.StandardOptimizer(
            space=chatbot_config.space,
            distill_functions=chatbot_config.sweep_distill_functions,
            metric=chatbot_config.sweep_metric_function,
            num_trials=chatbot_config.num_trials,
        )
```

We then loop over the different parameter settings

```python
        while not optimizer.is_complete(predictions_dir, include_in_progress=True):
            parameters = optimizer.get_parameters()
```

and make predictions for the models we want to compare. `make_predictions()`
is also implemented in `modeling.py`, and is task-specific. Note that
internally `make_predictions()` writes out the predictions to
`predictions_dir`.

```python
            predictions = make_predictions(
                contexts=contexts,
                dataset_preset=parameters["dataset_preset"],
                prompt_preset=parameters["prompt_preset"],
                model_preset=parameters["model_preset"],
                temperature=parameters["temperature"],
                max_tokens=parameters["max_tokens"],
                top_p=parameters["top_p"],
                context_length=parameters["context_length"],
                output_dir=predictions_dir,
            )
```

We finally evaluate the output given a metric function and print the result:

```python
            eval_result = optimizer.calculate_metric(contexts, labels, predictions)
```

### Visualizing and Exploring the Results

Finally, if we're visualizing the results for exploration

```python
    if do_visualization:
```

we read in the parameter files (ending in `.zbp`) and the prediction files
(ending in something like `.json` or `.jsonl`):

```python
        results: list[ExperimentRun] = []
        for param_file in param_files:
            assert param_file.endswith(".zbp")
            with open(param_file, "r") as f:
                parameters = json.load(f)
            with open(f"{param_file[:-4]}.jsonl", "r") as f:
                predictions = [json.loads(x) for x in f.readlines()]
```

We then turn the parameters into a readable name, and add to the results.

```python
            name = reporting_utils.parameters_to_name(parameters, chatbot_config.space)
            results.append(
                ExperimentRun(parameters=parameters, predictions=predictions, name=name)
            )
```

Finally, we turn them into a dataframe and visualize them. Some of the settings
here, such as the format of the data and which settings we use are task dependent.

```python
        df = pd.DataFrame(
            {
                "messages": [[asdict(y) for y in x.messages] for x in contexts],
                "label": labels,
            }
        )
        visualize(
            df,
            labels,
            results,
            "openai-chat",
            "messages",
            chatbot_config.zeno_distill_and_metric_functions,
        )
```

When the visualization is run, it will calculate feature or metric functions
for a while and when it displays "processing complete", you can open the
web page (usually `https://localhost:8000`) to see the results as in the figure
below!

![Chatbot visualization](/docs/images/chatbot_visualization.png)

## `modeling.py`

Unlike `main.py`, which is relatively constant across tasks, `modeling.py`
is task-specific. It contains the functions that are used to load data and
make predictions.

### Loading Data

Here are two example functions to load data for very different styles of tasks:

* [Chatbot](https://github.com/zeno-ml/zeno-build/blob/fa64cbd592c8d0c87e2041282837d0239930b349/examples/chatbot/modeling.py#L53-L119)
* [Text Classification](https://github.com/zeno-ml/zeno-build/blob/fa64cbd592c8d0c87e2041282837d0239930b349/examples/text_classification/modeling.py#L129-L149)

Here the chatbot example returns a special class `ChatMessages`, and
the text classification example returns a Hugging Face `Dataset`. This
demonstrates how you can flexibly change the functions based on the
task specification.

### Making Predictions

The general template for the prediction function follows a common
pattern. Here is an example function profile from the chatbot task:

```python
def make_predictions(
    contexts: list[ChatMessages],
    dataset_preset: str,
    prompt_preset: str,
    model_preset: str,
    temperature: float = 0.3,
    max_tokens: int = 100,
    top_p: float = 1,
    context_length: int = -1,
    output_dir: str = "results",
) -> list[str] | None:
```

First, we get parameters that we would like to consider when
saving the model or displaying its name. We ignore parameters that
correspond to data or local file systems, like `context`

```python
    # Load from cache if existing
    parameters = {
        k: v for k, v in locals().items() if k not in {"contexts", "output_dir"}
    }
```

Then we try to find if the experiment has been run already, and simply
return the results if so.

```python
    file_root = get_cache_path(output_dir, parameters)
    if os.path.exists(f"{file_root}.json"):
        with open(f"{file_root}.json", "r") as f:
            return json.load(f)
```

Next, `CacheLock` creates a file lock to prevent multiple processes
from running the same experiment (in the case of running multiple
experiments in parallel). If the cache is locked, we simply return
`None`, indicating that the experiment should be skipped.

```python
    with CacheLock(file_root) as cache_lock:
        if not cache_lock:
            return None
```

We then try to make predictions. Here we're using the Zeno Build
`generate_from_chat_prompt()` function that allows you to generate
text from large language models, but you can also implement your
own custom functions if you want.
If the prediction fails, we call `fail_cache()`
which writes out a file (`.zbfail`) that indicates that this
particular experiment has failed.

```python
        try:
            predictions: list[str] = generate_from_chat_prompt(
                contexts,
                chatbot_config.prompt_messages[prompt_preset],
                chatbot_config.model_configs[model_preset],
                temperature,
                max_tokens,
                top_p,
                context_length,
            )
        except Exception:
            tb = traceback.format_exc()
            fail_cache(file_root, tb)
            raise
```

Then we write out the predictions to a file (`.json`), and return.

```python
        with open(f"{file_root}.json", "w") as f:
            json.dump(predictions, f)

    return predictions
```

## Next Steps

Next, we move on to describing how we can [specifying parameters](/docs/specifying_parameters.md)
