# Open LLM Leaderboard

Use Zeno to visualize the data and model outputs of the [Open LLM Leaderboard][1]!
There is a notebook for uploading the raw task data and model results for each of the four tasks in the leaderboard.
The notebooks re-use the outputs from the leaderboard so you don't have to run any inference to explore the results.

You can pick which model results to upload by passing in the org/model strings into the notebooks from
the [Leaderboard Details](https://huggingface.co/datasets/open-llm-leaderboard/details/tree/main) dataset.

> Explore our [example report](https://hub.zenoml.com/report/a13x/What%20does%20the%20OpenLLM%20Leaderboard%20measure%3F) to get an idea of what the resulting data will look like.

## Setup

To run this example, first install the requirements:

```bash
pip install -r requirements.txt
```

You'll then need to get an API key from Zeno Hub. Create an account at [https://hub.zenoml.com](https://hub.zenoml.com) and navigate 
to [your account page](https://hub.zenoml.com/account) to get an API key.
Add this key as an environment variable, `ZENO_API_KEY`.

You can now run the notebooks to create a Zeno Project for any of the four benchmark datasets.

[1]: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
