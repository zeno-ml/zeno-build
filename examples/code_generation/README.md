# Code Generation

This is an example of usin zeno-build to test the code generation abilities of LLMs.
In the example here we test several text and code-specific generation models,
including API-based models like:
[OpenAI's ChatGPT](https://openai.com/blog/gpt-3-apps/), and
[Cohere's text generation models](https://cohere.ai/generate),
as well as publicly available models such as
[GPT-2](https://huggingface.co/gpt2),
[LLaMa](https://huggingface.co/decapoda-research/llama-7b-hf),
[Alpaca](https://huggingface.co/chavinlo/alpaca-native),
[StarCoder](https://huggingface.co/bigcode/starcoder),
and [CodeGen](https://huggingface.co/Salesforce/codegen2-16B).

Evaluation of the models is done with the
[Inspired Cognition Critique](https://docs.inspiredco.ai/critique/) tool
for text generation evaluation.
We demonstrate the case for Code LMs on examples from the
[HumanEval](https://github.com/openai/human-eval)
and [ODEX](https://github.com/zorazrw/odex) datasets.

But you can
swap in whatever models, prompts, metrics, and data that you would like to try on
other tasks too! The result of running Zeno Build will be an interface where you
can browse and explore the results. See an example below:

* [Browsing Interface](https://zeno-ml-codegen-report.hf.space) (To Do)
* [Textual Summary](report/) (To Do)

## Setup

To run this example, you'll need to install the requirements.
First install the `zeno-build` package:

```bash
pip install zeno-build
```

Then install the requirements for this example:

```bash
pip install -r requirements.txt
```

Finally, you'll want to set an API key for the Inspired Cognition Critique
service, which is used to evaluate the outputs. You can do this by getting
the necessary API key:

* [Inspired Cognition API Key](https://dashboard.inspiredco.ai)

Then setting it as environment variables in whatever environment you use to
run the example.

```bash
INSPIREDCO_API_KEY=...
```

In addition, if you want to generate results for the API-based models (this
is off by default, but can be turned on in config.py) you'll
need to set the following environment variables:

* [OpenAI API Key](https://openai.com/blog/openai-api/)
* [Cohere API Key](https://cohere.ai/)

```bash
OPENAI_API_KEY=...
COHERE_API_KEY=...
```

## Run the Example

To run the example, run the following command:

```bash
python main.py
```

This will run ten training runs with various hyperparameters for:

* `dataset`: two different data benchmarks: HumanEval and ODEX
* `model`: starcoder, codegen, gpt2, llama, alpaca, chatgpt, or cohere
* `evaluation`: lexical-based and execution-based evaluation

The results will be saved to the `results` directory, and a report of the
comparison will be displayed using [Zeno](https://zenoml.com/).

## Modification

If you want to make modifications to the example, there are two main ways:

1. All of the hyperparameters are included in `config.py`, you can modify this
   file directly to run different experiments or comparisons.
2. You can also feel free to modify the `modeling.py` file to implement different
   modeling strategies.
