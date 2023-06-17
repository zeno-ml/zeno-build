# GPT-MT Analysis with Zeno Build

Machine translation is one of the most important and classic language-related
tasks, and a huge investment has been put into it over the years. Due to this
fact, we now have great products like
[Google Translate](https://translate.google.com/),
[Bing Translator](https://www.bing.com/translator), and
[DeepL](https://www.deepl.com/en/translator) that are created specifically
for this task. On the other hand, recently general-purpose language models
such as [ChatGPT](https://chat.openai.com/) have been released as
general-purpose tools that can handle many different tasks?

So how do special-purpose MT models stack up against the GPT language models?
Luckily, some good folks at Microsoft
[ran an extensive set of experiments](https://github.com/microsoft/gpt-MT),
generating different results from GPT and comparing them against existing
methods. This Zeno Build example demonstrates how we can use Zeno Build to
visualize and examine the system outputs provided by these experiments.

The result of running Zeno Build will be an interface where you
can browse and explore the results. See an example below:

* [Browsing Interface](https://zeno-ml-gpt-mt-report.hf.space)
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

Finally, you'll want to set an API key for the
[Inspired Cognition Critique](https://docs.inspiredco.ai/critique/)
service, which is used to evaluate the outputs. You can do this by getting
the necessary API key:

* [Inspired Cognition API Key](https://dashboard.inspiredco.ai)

Then setting it as environment variables in whatever environment you use to
run the example.

```bash
INSPIREDCO_API_KEY=...
```

## Run the Example

To run the example, first clone the GPT-MT library (we clone a fork that containing more translation systems).

```bash
git clone git@github.com:zwhe99/gpt-MT.git
```

Then run the following command to perform evaluation and analysis:

```bash
python main.py --input-dir gpt-MT --results-dir results
```

The results will be saved to the `results` directory, and a report of the
comparison will be displayed using [Zeno](https://zenoml.com/).
Once the evalaution is finished you will be able to view the results at
[https://localhost:8000](https://localhost:8000).
You can then go in and explore the results, making slices, reports, etc.
Alternatively, you can view the
[ready-made hosted report](https://zeno-ml-gpt-mt-report.hf.space).
