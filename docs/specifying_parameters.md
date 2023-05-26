# Specifying Parameters

Running a single experiment with Zeno Build can test many different models
and parameter settings. Exactly which parameters are tested is specified in
the `config.py` file in each example directory.

Here we will provide an example using for the [chatbot](/examples/chatbot/)
task.

## Search Space

The most important parameter is the `search_space` parameter, which specifies
the search space of hyperparameters to test. This is specified in the
`config.py` file as follows:

```python
space = search_space.CombinatorialSearchSpace(
    {
        "dataset_preset": search_space.Constant("dstc11"),
        "model_preset": search_space.Categorical(
            ["gpt2-xl", "llama-7b", "alpaca-7b", "vicuna-7b", "mpt-7b-chat"]
        ),
        "prompt_preset": search_space.Categorical(
            ["standard", "friendly", "polite", "cynical"]
        ),
        "temperature": search_space.Float(0.2, 0.4),
        "context_length": search_space.Discrete([1, 2, 3, 4]),
        "max_tokens": search_space.Constant(100),
        "top_p": search_space.Constant(1.0),
    }
)
```

Here the `search_space` parameter is a `CombinatorialSearchSpace` object,
which indicates that we should search over the cross product of all the
parameters specified in the dictionary. Each parameter in the dictionary
can be of a variety of types:

- `search_space.Constant`: A constant value that is not searched over.
- `search_space.Categorical`: A categorical value, such as a list of strings.
- `search_space.Discrete`: A discrete value, such as a list of integers.
  The difference between `Categorical` and `Discrete`
  is that there is an inherent ordering in `Discrete` values, but not
  in `Categorical` values.
- `search_space.Float`: A float value, such as a float between 0 and 1.
- `search_space.Int`: An integer value, such as an integer between 0 and 10.

## Number of Trials

There is also a `num_trials` parameter, which specifies the number of unique
parameter settings to try out.

```python
num_trials = 10
```

## Data, Model, Prompt Presets

Each task is slightly different, so the parameters that are searched over
may be different. But most tasks can have a `dataset_preset` and
`model_preset` parameter to specify the data and the model. And when using
a language model by providing it with a prompt, a `prompt_preset` parameter
can be used to specify the prompt.

### Data Configurations

Data configurations can specify the dataset name (usually from
[Hugging Face Datasets](https://huggingface.co/datasets)), the split of
that dataset, that column in the dataset to use (and also the `label_column`
if we have a separate column for labels), and the format of the data.
Here are some examples for chatbots:

```python
dataset_configs = {
    "dstc11": DatasetConfig(
        dataset="gneubig/dstc11",
        split="validation",
        data_column="turns",
        data_format="dstc11",
    ),
    "daily_dialog": DatasetConfig(
        dataset="daily_dialog",
        split="validation",
        data_column="dialog",
        data_format="sequence",
    ),
}
```

### Model Configurations

Model configurations can specify the provider of the model, such as OpenAI,
Cohere, or Hugging Face, the name of the model. In some cases extra auxiliary
parameters such as the model class or tokenizer class may be necessary.

```python
model_configs = {
    "text-davinci-003": LMConfig(provider="openai", model="text-davinci-003"),
    "gpt-3.5-turbo": LMConfig(provider="openai_chat", model="gpt-3.5-turbo"),
    "cohere-command-xlarge": LMConfig(
        provider="cohere", model="command-xlarge-nightly"
    ),
    "gpt2": LMConfig(
        provider="huggingface",
        model="gpt2",
        model_cls=transformers.GPT2LMHeadModel,
    ),
    "llama-7b": LMConfig(
        provider="huggingface",
        model="decapoda-research/llama-7b-hf",
        tokenizer_cls=transformers.LlamaTokenizer,
    ),
    ...
}
```

### Prompt Configurations

Prompt configurations can specify the type of prompt to use. Here are
a few examples:

```python
prompt_messages: dict[str, ChatMessages] = {
    "standard": ChatMessages(
        messages=[
            ChatTurn(
                role="system",
                content="You are a chatbot tasked with making small-talk with "
                "people.",
            ),
        ]
    ),
    "friendly": ChatMessages(
        messages=[
            ChatTurn(
                role="system",
                content="You are a kind and friendly chatbot tasked with making "
                "small-talk with people in a way that makes them feel "
                "pleasant.",
            ),
        ]
    ),
    ...
}
```

## Distill and Metric Functions

Finally, we define `distill` and `metric` functions. `distill` functions are
essentially [Zeno](https://zenoml.com)'s terminology for "feature" functions,
that calculate a feature
vector for a given input. `metric` functions are functions that calculate an
evaluation metric for an entire dataset, sometimes taking the output of a
`distill` function as input.
Zeno Build provides a library of various distill and metric functions
in the [evaluation](../zeno_build/evaluation/) directory, so you can
browse through the full library there.

We use distill/metric functions at two points in
the experimentation process.

First, we use them while making a sweep over the various parameter values,
evaluating a metric value after the end of every experiment. Here is an
example of using the [chrf](https://aclanthology.org/W15-3049/) string
overlap metric. We specify both the distill function `chrf`, which calculates
the `chrf` for each example, and the metric function `avg_chrf`, which
calculates the average `chrf` for the entire dataset.

```python
sweep_distill_functions = [chrf]
sweep_metric_function = avg_chrf
```

Second, we use them in Zeno visualization for
[exploring results](exploring_results.md), specifying both

- Distill functions used to slice and dice the data such as `output_length`,
  `input_length`, `label_length`, `chat_context_length`.
- Distill functions used to calculate metrics such as `chrf`, `length_ratio`,
  `bert_score`, `exact_match`.
- Metric functions used to calculate average metrics such as `avg_chrf`,
    `avg_length_ratio`, `avg_bert_score`, `avg_exact_match`.

```python
zeno_distill_and_metric_functions = [
    output_length,
    input_length,
    label_length,
    chat_context_length,
    chrf,
    length_ratio,
    bert_score,
    exact_match,
    avg_chrf,
    avg_length_ratio,
    avg_bert_score,
    avg_exact_match,
]
```

## Next Steps

Now that we've implemented our model and defined the hyperparameters,
we can move on to [running experiments](running_experiments.md).
