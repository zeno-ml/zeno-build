# Text Summarization

In the example here we test two different companies' text generation models,
[OpenAI's GPT-3](https://openai.com/blog/gpt-3-apps/), and
[Cohere's text generation models](https://cohere.ai/generate). Evaluation of
the models is done with the
[Inspired Cognition Critique](https://docs.inspiredco.ai/critique/)
tool for text generation evaluation. We demonstrate the case for text summarization
on 100 examples from the
[CNN-DailyMail dataset](https://huggingface.co/datasets/cnn_dailymail). But you can
swap in whatever models, prompts, metrics, and data that you would like to try on
other tasks too!

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

* `prompt`: four different prompts (found in [prompt_configs.py](prompt_configs.py))
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
