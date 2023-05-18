# Chatbots

This is an example of using zeno-build to test the creation of chatbots.
In the example here we test several text generation models, including API-based
models like
[OpenAI's ChatGPT](https://openai.com/blog/gpt-3-apps/), and
[Cohere's text generation models](https://cohere.ai/generate),
as well as publicly available models such as
[GPT-2](https://huggingface.co/gpt2),
[LLaMa](https://huggingface.co/decapoda-research/llama-7b-hf),
[Alpaca](https://huggingface.co/chavinlo/alpaca-native),
[Vicuna](https://huggingface.co/eachadea/vicuna-7b-1.1),
and [MPT-Chat](https://huggingface.co/mosaicml/mpt-7b-chat).

Evaluation of the models is done with the
[Inspired Cognition Critique](https://docs.inspiredco.ai/critique/)
tool for text generation evaluation. We demonstrate the case for chatbots
on examples from the
[DSTC 11 customer service dataset](https://github.com/amazon-science/dstc11-track2-intent-induction).

But you can
swap in whatever models, prompts, metrics, and data that you would like to try on
other tasks too! The result of running Zeno Build will be an interface where you
can browse and explore the results. See an example below:

* [Browsing Interface](https://zeno-ml-chatbot-report.hf.space)
* [Textual Summary](report/)

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

Finally, you'll want to set three API keys used by the various APIs
included in this example. You can do this by creating a file called
`.env` in this directory with the following contents:

You can get the necessary API keys here:

* [OpenAI API Key](https://openai.com/blog/openai-api/)
* [Cohere API Key](https://cohere.ai/)
* [Inspired Cognition API Key](https://dashboard.inspiredco.ai)

Then set them as environment variables in whatever environment you use to
run the example.

```bash
OPENAI_API_KEY=...
COHERE_API_KEY=...
INSPIREDCO_API_KEY=...
```

## Run the Example

To run the example, run the following command:

```bash
python main.py
```

This will run ten training runs with various hyperparameters for:

* `prompt_template`: four different prompts (found in [prompt_configs.py](prompt_configs.py))
* `model`: openai davinci-003 and gpt-3.5-turbo, and cohere command-xlarge
* `temperature`: between 0.2, 0.3, or 0.4

The results will then be saved to `results.json`, and a visual
comparison will be displayed using [Zeno](https://zenoml.com/).

## Modification

If you want to make modifications to the example, there are two main ways:

1. All of the hyperparameters are included in `config.py`, you can modify this
   file directly to run different experiments or comparisons.
2. You can also feel free to modify the `modeling.py` file to implement different
   modeling strategies.
