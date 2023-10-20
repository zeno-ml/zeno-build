# Open LLM Leaderboard

This example downloads data from the [Open LLM Leaderboard][1] and ingests it
into a Zeno Project.
There are four tasks in the leaderboard, for each task, there is one notebook
to upload your data.

You can configure which models to upload data for.

## Setup

To run this example, you'll need to install the requirements.

```bash
pip install -r requirements.txt
```

You also need to add an environment variable named `ZENO_API_KEY` that contains
your API key to be able to upload data to Zeno.

Then, simply run any of the notebooks.

[1]: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
