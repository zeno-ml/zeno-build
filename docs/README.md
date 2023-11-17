# Zeno Build

[Zeno Build](https://github.com/zeno-ml/zeno-build) is a tool for developers
who want to quickly build, compare, and iterate on applications using large
language models.

![Zeno Build Overview](images/zeno-build-overview.png)

Zeno Build allows you to:

1. **Easily prototype** LLM-based applications through a unified wrapper
   interface over open-source and API-based models.
2. **Specify the space of experiments** you want to to run.
3. **Run experiments** to train and evaluate these models, including:
   1. **Hyperparameter optimization** to find the best hyperparameters for your
      task.
   2. **Evaluation of outputs** using state-of-the-art model-based metrics for
      evaluation of generated text.
4. **Explore the results** of these experiments, using a comprehensive visual
    report that allows you to slice-and-dice data and uncover insights that
    can feed back to better model, data, and prompt engineering.

## Tutorial

Getting started with Zeno Build is easy!
Click over to our the [tutorial](tutorial/) directory to get started with
some simple exercises that demonstrate the main concepts.

## End-to-end Examples

In addition to our simple tutorial above, we also have a number of end-to-end
[examples](../examples/) of how you can use `zeno-build` to implement and run
a full set of experiments. You can take a look at the example directory to see
all of the tasks that we have examples for, but most of them follow the same
general pattern. See below for a comprehensive explanation (specifically based)
on our chatbot example.

* [Implementing Models](implementing_models.md)
* [Specifying Experimental Parameters](specifying_parameters.md)
* [Running Experiments](running_experiments.md)
* [Exploring Results](exploring_results.md)
