# Important Concepts

This page outlines some important concepts in Zeno Build, particularly useful
for modifying or implementing new [tasks](/tasks/README.md).

## Test and Training Sets

In Zeno Build, you need to have a test set to evaluate your system. This is
usually loaded from [Hugging Face datasets](https://huggingface.co/datasets).

In addition, if you're performing model training (e.g. with
[Hugging Face](https://huggingface.co/)), you will
need to load a training set that can be done in a similar way. Alternatively,
if the example just uses an API-based model such as
[OpenAI](https://openai.com/), you may not need a training set. See some
examples below:

- **Without training**: [chatbots](chatbot/) and [summarization](summarization/)
- **With training**: [text classification](text_classification/)

## Data, Label, and Output

Zeno Build (and [Zeno](https://zenoml.com)) defines the following for each example:

- `data`: This is the raw data for a particular instance. It is usually the
  input to the machine learning model.
- `label`: This is the label for a particular instance. It is typically the
  ground truth that we'd like the output to match.
- `output`: This is the output/prediction of a machine learning model that we'd
  like to evaluate and analyze.

## Search Spaces and Constants

When testing out models, you may want to both _search different parameters_,
and _keep some parameters constant_. Zeno Build handles this by defining two
variables `space` and `constant` in the `config.py` file. Some examples from
the [chatbots](chatbot/) example are shown below:

```python
# The space of hyperparameters to search over
space = {
    "prompt_preset": search_space.Categorical(
        ["standard", "friendly", "polite", "cynical"]
    ),
    "model_preset": search_space.Categorical(
        ["openai_davinci_003", "openai_gpt_3.5_turbo", "cohere_command_xlarge"]
    ),
    "temperature": search_space.Discrete([0.2, 0.3, 0.4]),
}

# Any constants that are not searched over
constants: dict[str, Any] = {
    "test_dataset": "daily_dialog",
    "test_split": "validation",
    "test_examples": 40,
    "max_tokens": 100,
    "top_p": 1.0,
}
```

This will result in us searching over four different prompts and three
different model presets (also defined in `config.py`), as well as three
different temperature values. Other things
such as the number of test examples, number of tokens, an the test dataset are
all kept constant.

## Hyperparameter Sweeps

Given this search space, Zeno Build can iteratively run multiple experiments
for you. In doing so it will appropriately optimize your hyperparameters using
a number of different strategies.

- **Random Search**: It will randomly pick configurations and test them out.
- **Google Vizier**: This uses
  [Google Vizier](https://github.com/google/vizier) to perform intelligent
  hyperparameter search more efficiently.

All available optimizers can be found in the
[optimizers directory](/zeno_build/optimizers/).

## Distill Functions, Metrics, and Evaluation

Zeno Build implements a number of different methods that you can use to
calculate features of your outputs. There are two important concepts:

- **Distill Functions**: These calculate a feature for each example. These can
  be used for evaluating individual examples or exploring data in the
  visualization interface described below.
- **Metric Functions**: These calculate a single score for the entire test
  dataset, and are usually used for evaluating the quality of the system as a
  whole. They are often calculated by averaging the value of distill functions.

Zeno Build implements these distill and metric functions for various quality
measures. For instance, you can evaluate:

- **Accuracy** for text classification tasks.
- **Text generation quality metrics** such as fluency, factuality, toxicity,
  etc. for text generation tasks.

All evaluation methods can be found in the
[evaluation](/zeno_build/evaluation/) directory, with some examples below:

## Visualization

Finally, Zeno Build allows you to build reports to better understand your
output quality using the [Zeno](https://zenoml.com) visualization toolkit.
This is called by using the `visualize()` function at the end of the
training run, which takes in the various metrics.

Generally when Zeno Build finishes, you can click over to
[http://localhost:8000](http://localhost:8000) to view the visualizations on
your local machine and play around with them.

Take a look at the [Zeno documentation](https://zenoml.com/docs/intro)
to see more details!
