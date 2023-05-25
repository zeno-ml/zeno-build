# Zeno Build Examples

Here, you can find examples supported by `zeno-build`.

## Examples

The examples supported right now are:

- [Chatbots](chatbot/) that chat with users and return appropriate responses.
- [Summarization](summarization/) systems that take longer articles and reduce
  them in length.
- [Text Classification](text_classification/) systems that detect the sentiment
  or topic of text.

If you want to jump in and run any of the examples, click over to the appropriate
directory, read the README to set up your environment, and get
started!

## Build Your Own Example

Zeno Build is designed to make it easy for you to build your own new examples and
applications. To get started, we suggest that you do one or both of the following:

- Browse the [docs](../docs/) to get a sense of how the library works.
- Find an example in this directory that is close to what you want, browse it,
  then copy the example to a new directory, and start hacking!

If you opt to dive in to the latter approach, most of the examples consist of
four main files and you should be able to get started by modifying these:

- `README.md` describing the features of the tasks, models, and hyperparameters.
- `modeling.py` implementing the core modeling strategy for the task. You can
  change this file to implement new modeling strategies.
- `config.py` describing the hyperparameters used in the experiments. You can
  modify this this file to change the hyperparameter settings that you want to
  experiment with.
- `main.py` implementing the main experimental logic, including the entire flow
  of initialization, experimental runs, and visualization.

## Contributing Back

If you built something with Zeno Build, please contribute it back! We'd love a
pull request to the main repo with your new examples.

If you have any questions, please [get in touch](../README.md#get-in-touch).
