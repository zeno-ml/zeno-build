# Zeno Build

[Zeno Build](https://github.com/zeno-ml/zeno-build) is a tool for developers
who want to quickly build, compare, and iterate on applications using large
language models.

![Zeno Build Overview](/docs/images/zeno-build-overview.png)

Zeno Build allows you to:

1. **Relatively easily implement** LLM-based apps through a unified wrapper
   interface over open-source and API-based models.
2. **Specify the space of experiments** you want to to run.
3. **Run experiments** to train and evaluate these models, including:
   1. **Hyperparameter optimization** to find the best hyperparameters for your
      task.
   2. **Evaluate outputs** using state-of-the-art model-based metrics for
      evaluation of generated text.
4. **Explore the results** of these experiments, using a comprehensive visual
    report that allows you to slice-and-dice data and uncover insights that
    can feed back to better model, data, and prompt engineering.

In order to demonstrate how each of the above features work, we have a number
of [examples](/examples/) of how you can use `zeno-build` for different tasks.

Read on for more details:

* [Implementing Models](/docs/implementing_models.md)
* [Specifying Experimental Parameters](/docs/specifying_parameters.md)
* [Running Experiments](/docs/running_experiments.md)
* [Exploring Results](/docs/exploring_results.md)

To do so, we recommend that you first take a look at the existing
examples. Most of them consist of four main files:

* `README.md` describing the features of the tasks, models, and hyperparameters.
* `modeling.py` implementing the core modeling strategy for the task. You can
  change this file to implement new modeling strategies.
* `config.py` describing the hyperparameters used in the experiments. You can
  modify this this file to change the hyperparameter settings that you want to
  experiment with.
* `main.py` implementing the main experimental logic, including the entire flow
  of initialization, experimental runs, and visualization.

If you're looking to dive deeper, you should also click over to our [important
concepts](/docs/concepts.md) doc, and browse through the code.
