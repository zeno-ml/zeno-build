# Zeno Build

[![PyPI version](https://badge.fury.io/py/zeno-build.svg)](https://badge.fury.io/py/zeno-build)
![Github Actions CI tests](https://github.com/zeno-ml/zeno-build/actions/workflows/ci.yml/badge.svg)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Discord](https://img.shields.io/discord/1086004954872950834)](https://discord.gg/km62pDKAkE)
[![Open Zeno](https://img.shields.io/badge/%20-Open_Zeno-612593.svg?labelColor=white&logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iMzMiIGhlaWdodD0iMzMiIHZpZXdCb3g9IjAgMCAzMyAzMyIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTMyIDE1Ljc4NDJMMTYuNDg2MiAxNS43ODQyTDE2LjQ4NjIgMC4yNzA0MDFMMjQuMzAyIDguMDg2MTdMMzIgMTUuNzg0MloiIGZpbGw9IiM2MTI1OTMiLz4KPHBhdGggZD0iTTE1Ljc5MTcgMTUuODMxMUw4LjAzNDc5IDguMDc0MjJMMTUuNzkxNyAwLjMxNzMyOEwxNS43OTE3IDE1LjgzMTFaIiBmaWxsPSIjNjEyNTkzIiBmaWxsLW9wYWNpdHk9IjAuOCIvPgo8cGF0aCBkPSJNMTQuODY1NSAxNS44MzExTDcuNTk0ODUgMTUuODMxMUw3LjU5NDg1IDguNTYwNDJMMTQuODY1NSAxNS44MzExWiIgZmlsbD0iIzYxMjU5MyIgZmlsbC1vcGFjaXR5PSIwLjYiLz4KPHBhdGggZD0iTTYuMTEyOSAxNS44MzExTDMuMjQxNyAxNS44MzExTDMuMjQxNyAxMi44NjcyTDYuMTEyOSAxNS44MzExWiIgZmlsbD0iIzZBMUI5QSIgZmlsbC1vcGFjaXR5PSIwLjQiLz4KPHBhdGggZD0iTTIuNzMyMjggMTUuODMxTDEuNTE1NSAxNC42MTQzTDIuNzQyNzEgMTMuMzg3TDIuNzMyMjggMTUuODMxWiIgZmlsbD0iIzZBMUI5QSIgZmlsbC1vcGFjaXR5PSIwLjMiLz4KPHBhdGggZD0iTTIuMDM3NiAxNS43ODQyTDEuMTU3NzEgMTUuNzg0MkwxLjE1NzcxIDE0Ljk1MDZMMi4wMzc2IDE1Ljc4NDJaIiBmaWxsPSIjNkExQjlBIiBmaWxsLW9wYWNpdHk9IjAuMiIvPgo8cGF0aCBkPSJNMC44MzM1NjggMTUuNzg0MUwwLjUwOTM5OSAxNS40NkwwLjgzMzU2NyAxNS4xMzU4TDAuODMzNTY4IDE1Ljc4NDFaIiBmaWxsPSIjNjEyNTkzIiBmaWxsLW9wYWNpdHk9IjAuMSIvPgo8cGF0aCBkPSJNMC4xMDYxODcgMTUuNzk0NEwwLjMwMTAyNSAxNS41OTk2TDAuNDk1ODYzIDE1Ljc5NDRIMC4xMDYxODdaIiBmaWxsPSIjNjEyNTkzIiBmaWxsLW9wYWNpdHk9IjAuMSIvPgo8cGF0aCBkPSJNNi45NTIxMyAxNS44MjQ4TDMuNjQwOTkgMTIuNTEzN0w2Ljk2OTYzIDkuMTg1MDNMNi45NTIxMyAxNS44MjQ4WiIgZmlsbD0iIzYxMjU5MyIgZmlsbC1vcGFjaXR5PSIwLjUiLz4KPHBhdGggZD0iTTAuMjk0MjM1IDE2LjQ3OTVMMTUuODA4IDE2LjQ3OTVMMTUuODA4IDMxLjk5MzNMNy45OTIyMyAyNC4xNzc1TDAuMjk0MjM1IDE2LjQ3OTVaIiBmaWxsPSIjNjEyNTkzIi8+CjxwYXRoIGQ9Ik0xNi40OTU2IDE3LjI0MzZMMjMuODUwNyAyNC41ODVMMTYuNDk1NiAzMS45NEwxNi40OTU2IDE3LjI0MzZaIiBmaWxsPSIjNjEyNTkzIiBmaWxsLW9wYWNpdHk9IjAuOCIvPgo8cGF0aCBkPSJNMTYuNTMyNiAxNi40Nzk1TDI0LjQ1MTUgMTYuNDc5NUwyNC40NTE1IDI0LjAyOEwxNi41MzI2IDE2LjQ3OTVaIiBmaWxsPSIjNjEyNTkzIiBmaWxsLW9wYWNpdHk9IjAuNiIvPgo8cGF0aCBkPSJNMjYuMTgxMyAxNi40MzI2TDI5LjA1MjUgMTYuNDMyNkwyOS4wNTI1IDE5LjM5NjRMMjYuMTgxMyAxNi40MzI2WiIgZmlsbD0iIzZBMUI5QSIgZmlsbC1vcGFjaXR5PSIwLjQiLz4KPHBhdGggZD0iTTI5LjU2MTkgMTYuNDMyNkwzMC43Nzg3IDE3LjY0OTRMMjkuNTUxNSAxOC44NzY2TDI5LjU2MTkgMTYuNDMyNloiIGZpbGw9IiM2QTFCOUEiIGZpbGwtb3BhY2l0eT0iMC4zIi8+CjxwYXRoIGQ9Ik0zMC4yNTY2IDE2LjQ3OTVMMzEuMTM2NSAxNi40Nzk1TDMxLjEzNjUgMTcuMzEzMUwzMC4yNTY2IDE2LjQ3OTVaIiBmaWxsPSIjNkExQjlBIiBmaWxsLW9wYWNpdHk9IjAuMiIvPgo8cGF0aCBkPSJNMzEuNDYwNiAxNi40Nzk1TDMxLjc4NDggMTYuODAzN0wzMS40NjA2IDE3LjEyNzlMMzEuNDYwNiAxNi40Nzk1WiIgZmlsbD0iIzYxMjU5MyIgZmlsbC1vcGFjaXR5PSIwLjEiLz4KPHBhdGggZD0iTTMyLjE4OCAxNi40NjkyTDMxLjk5MzIgMTYuNjY0MUwzMS43OTgzIDE2LjQ2OTJIMzIuMTg4WiIgZmlsbD0iIzYxMjU5MyIgZmlsbC1vcGFjaXR5PSIwLjEiLz4KPHBhdGggZD0iTTI1LjM0MjEgMTYuNDM4OUwyOC42NTMyIDE5Ljc1TDI1LjMyNDYgMjMuMDc4NkwyNS4zNDIxIDE2LjQzODlaIiBmaWxsPSIjNjEyNTkzIiBmaWxsLW9wYWNpdHk9IjAuNSIvPgo8L3N2Zz4K)](https://hub.zenoml.com)

**Zeno Build** is a collection of examples using **Zeno** to evaluate generative AI models.
Use it to get started with common evaluation setups.

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
