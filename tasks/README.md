# Zeno Build Tasks

Here, you can find all of the tasks supported by `zeno-build`.

## Tasks

The tasks supported by default are:

- [Chatbots](chatbot/) that chat with users and return appropriate responses.
- [Summarization](summarization/) systems that take longer articles and reduce
  them in length.
- [Text Classification](text_classification/) systems that detect the sentiment
  or topic of text.

If you want to jump in and run any of the tasks, click over to the appropriate
directory, read the README to set up your environment, and get
started!

## Build Your Own Task

Zeno Build is designed to make it easy for you to build your own new tasks and
applications. To do so, we recommend that you first take a look at the existing
tasks. Most of them consist of four main files:

- `README.md` describing the features of the tasks, models, and hyperparameters.
- `modeling.py` implementing the core modeling strategy for the task. You can
  change this file to implement new modeling strategies.
- `config.py` describing the hyperparameters used in the experiments. You can
  modify this this file to change the hyperparameter settings that you want to
  experiment with.
- `main.py` implementing the main experimental logic, including the entire flow
  of initialization, experimental runs, and visualization.

If you're looking to dive deeper, you should also click over to our [important
concepts](CONCEPTS.md) doc, and browse through the code.

## Contributing Back

If you built something with Zeno Build, please contribute it back! We'd love a
pull request to the main repo with your new examples.

If you have any questions, please [get in touch](../README.md#get-in-touch).
