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

Next, _start building_! Browse to the [docs](docs/) directory to get a
primer or to the [examples/](examples/) directory, where we
have a bunch of examples of how you can use `zeno-build` for different tasks,
such as [chatbots](examples/chatbot/),
[text summarization](examples/summarization/), or [text
classification](examples/text_classification/).

Each of the examples include code for running experiments and evaluating the
results. `zeno-build` will produce a comprehensive report with the
[Zeno](https://zenoml.com/) AI evaluation platform.

## Interactive Demos/Reports

Using Zeno Build, we have generated reports and online browsing demos of
state-of-the-art systems for different popular generative AI tasks.
Check out our pre-made reports below:

- **Chatbots** ([Report](examples/chatbot/report/),
  [Browser](https://zeno-ml-chatbot-report.hf.space/)):
  A report comparing different methods
  for creating chatbots, including API-based models such as ChatGPT and Cohere,
  with open-source models such as Vicuna, Alpaca, and MPT.
- **Translation** ([Report](examples/analysis_gpt_mt/report/),
  [Browser](https://zeno-ml-translation-report.hf.space/)):
  A report comparing GPT-based methods, Microsoft Translator, and the best system
  from the Conference on Machine Translation.

## Building Your Own Apps (and Contributing Back)

Each of the examples in the [examples/](examples/) directory is specifically designed
to be self-contained and easy to modify. To get started building your own apps,
we suggest that you first click into the directory and read the general README,
find the closest example to what you're trying to do, copy the example to the
new directory, and start hacking!

If you build something cool, **we'd love for you to contribute it back**. We
welcome pull requests of both new examples, new reports for existing examples,
and new functionality for the core `zeno_build` library. If this is of interest
to you, please click through to our [contributing doc](contributing.md) doc to
learn more.

## Get in Touch

If you have any questions, feature requests, bug reports, etc., we recommend
getting in touch via the github [issues
page](https://github.com/zeno-ml/zeno-build/issues) or
[discord](https://discord.gg/km62pDKAkE), where the community can discuss and/or
implement your suggestions!
