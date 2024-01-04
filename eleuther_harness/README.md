# EleutherAI Harness

Use Zeno to visualize the data from the [Eleuther LM Evaluation Harness][1]!
There is a notebook for running harness tasks on diferent models and uploading
the results to Zeno.

You can configure tasks, models, and more directly in the notebook, get a task
list at:
[Task List](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md)

## Setup

To run this example, first install the requirements:

```bash
pip install -r requirements.txt
```

You'll then need to get an API key from Zeno Hub.
Create an account at [https://hub.zenoml.com](https://hub.zenoml.com) and navigate
to [your account page](https://hub.zenoml.com/account) to get an API key.
Add this key as an environment variable, `ZENO_API_KEY`.

You can now run the notebook to create a Zeno Project for any of the
harness tasks.

[1]: https://github.com/EleutherAI/lm-evaluation-harness
