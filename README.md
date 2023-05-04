# Zeno Build

[![PyPI version](https://badge.fury.io/py/zeno-build.svg)](https://badge.fury.io/py/zeno-build)
![Github Actions CI tests](https://github.com/zeno-ml/zeno-build/actions/workflows/ci.yml/badge.svg)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Discord](https://img.shields.io/discord/1086004954872950834)](https://discord.gg/km62pDKAkE)

**Zeno Build** is a tool for developers who want to quickly build, compare, and
iterate on applications using large language models.

![Zeno Build Overview](/docs/images/zeno-build-overview.png)

It provides:

- **Simple examples** of code to build LLM-based apps. The examples are
  architecture agnostic, we don't care if you are using
  [OpenAI](https://openai.com/),
  [LangChain](https://github.com/hwchase17/langchain), or [Hugging
  Face](https://huggingface.co).
- **Experiment management** and **hyperparameter optimization** code, so you can
  quickly kick off experiments using a bunch of different settings and compare
  the results.
- **Evaluation** of LLM outputs, so you can check if your outputs are correct,
  fluent, factual, interesting, or "good" by whatever definition of good you
  prefer! Use these insights to compare models and iteratively improve your
  application with model, data, or prompt engineering.

Sound interesting? Read on!

## Getting Started

To get started with `zeno-build`, install the package from PyPI:

```bash
pip install zeno-build
```

Next, _start building_! Browse the [tasks/](tasks/) directory, where we have a
bunch of examples of how you can use `zeno-build` for different tasks, such as
[chatbots](tasks/chatbot/), [text summarization](tasks/summarization/), or [text
classification](tasks/text_classification/). Check out the [Zeno Build
Concepts](tasks/CO) doc for more details on the different aspects of the
library.

Each of the examples include code for running experiments and evaluating the
results. `zeno-build` will produce a comprehensive report with the
[Zeno](https://zenoml.com/) ML analysis platform. To give you a flavor of what
these reports will look like, check out a few of our pre-made reports below:

- [Zeno Chatbot Report](TODO): A report comparing different methods for creating
  chatbots, including API-based models such as ChatGPT, Claude, and Cohere, with
  open-source models such as Vicuna, Alpaca, and Flan-T5.
- [Zeno Summarization Report](TODO): A report comparing different methods for
  text summarization, including GPT-3, Flan-T5, and Pegasus.
- [Zeno Sentiment Analysis Report](TODO): A report comparing different
  pre-trained models for sentiment analysis across a variety of datasets.

## Building Your Own Apps (and Contributing Back)

Each of the examples in the [tasks/](tasks/) directory is specifically designed
to be self-contained and easy to modify. To get started building your own apps,
we suggest that you first click into the directory and read the general README,
find the closest example to what you're trying to do, copy the example to the
new directory, and start hacking!

If you build something cool, **we'd love for you to contribute it back**. We
welcome pull requests of both new task examples, new reports for existing tasks,
and new functionality for the core `zeno_build` library. If this is of interest
to you, please click through to our [contributing doc](contributing.md) doc to
learn more.

## Get in Touch

If you have any questions, feature requests, bug reports, etc., we recommend
getting in touch via the github [issues
page](https://github.com/zeno-ml/zeno-build/issues) or
[discord](https://discord.gg/km62pDKAkE), where the community can discuss and/or
implement your suggestions!
