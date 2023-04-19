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

## Install Requirements

To run this example, you'll need to install the requirements.
First install the `llm-compare` package:

```bash
pip install llm-compare
```

Then install the requirements for this example:

```bash
pip install -r requirements.txt
```

## Run the Example

To run the example, run the following command:

```bash
python main.py
```

This will run ten training runs with various hyperparameters for:

* `model`: 
* `learning_rate`: between 1e-5 and 1e-3,
* `num_train_epochs`: between 1 and 4,
* `weight_decay`: between 0.0 and 0.01,
* `bias`: between -1.0 and 1.0,

The results will then be saved to `results.json`, and a visual
comparison will be displayed using [Zeno](https://zenoml.com/).

## Modification

You may want to modify the example for your own purposes. For
instance:

1. In `main.py` you can modify `space` variable, which defines the training data,
   base model, and hyperparameters to search over.
2. In `main.py` you can modify the `constants` variable in this file which
   defines the test data.
3. In `main.py` the `num_trials` variable defines the number of trials to run.
4. In `modeling.py` you can modify various design decisions about the model.
