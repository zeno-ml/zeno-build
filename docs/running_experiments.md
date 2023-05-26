# Running Experiments

Once you have [implemented your models](implementing_models.md) and
[specified your parameters](specifying_parameters.md), running experiments
is actually relatively simple.

You can just run the `main.py` file in your example directory, and it will
run the experiments and generate a report. For example, for chatbots:

```bash
python examples/chatbot/main.py --results-dir examples/chatbot/results
```

As with any experimentation, this may require GPUs or API keys if you're
using local models or API-based models respectively.

## Parallelizing Experiments

Because experiments may take a long time to run, Zeno Build also supports
simple parallelization of experiments by simply running multiple instances
of `main.py` in parallel.

For example, if you have 4 GPUs that can each be used for an experiment,
you can run 4 experiments in parallel by running the following:

```bash
mkdir -p examples/chatbot/results

python examples/chatbot/main.py --results-dir examples/chatbot/results \
    --skip-visualization \
    &> examples/chatbot/results/run1.log &

python examples/chatbot/main.py --results-dir examples/chatbot/results \
    --skip-visualization \
    &> examples/chatbot/results/run2.log &

python examples/chatbot/main.py --results-dir examples/chatbot/results \
    --skip-visualization \
    &> examples/chatbot/results/run3.log &

python examples/chatbot/main.py --results-dir examples/chatbot/results \
    --skip-visualization \
    &> examples/chatbot/results/run4.log &
```

You could also spread these across multiple machines, etc., as long as
they have access to a shared filesystem.
Here we are using the `--skip-visualization` argument because we want
to wait until all the experiments are done before visualizing the results.

Finally, once you have monitored the logs and confirmed that all the
experiments have finished, you can run the visualization:

```bash
python examples/chatbot/main.py --results-dir examples/chatbot/results \
    --skip-prediction
```

## Next Steps

Once the experiments are done, we can move on to
[exploring results](exploring_results.md).
