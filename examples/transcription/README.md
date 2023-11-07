# Audio Transcription

Audio transcription is an essential task for applications such as voice assistants,
podcast search, and video captioning. There are numerous open-source and commercial
tools for audio transcription, and it can be difficult to know which one to use.
[OpenAI's Whisper](https://github.com/openai/whisper) API is often people's
go-to choice,but there are nine different models to choose from with different
sizes, speeds, and cost.

In this example, we'll use Zeno to compare the performance of the different
models on the [Speech Accent Archive](https://accent.gmu.edu/) dataset.
The dataset has over 2,000 people from around the world reading the same
paragraph in English.We'll use the dataset to evaluate the performance of
the different models on different accents and English fluency levels.

The result of running Zeno Build will be an interface where you
can browse and explore the results. See an example below:

- [Browsing Interface](https://hub.zenoml.com/project/cabreraalex/Audio%20Transcription%20Accents/explore)
- [Textual Summary](https://hub.zenoml.com/report/cabreraalex/Audio%20Transcription%20Report)

## Setup

To run this example, you'll need to install the requirements.

```bash
pip install -r requirements.txt
```

This example also requires the `ffmpeg` library to be installed. You can test
if it is installed by running `ffmpeg --help`. If it is not found, you should
install it through your package manager. For example, if you are using conda,
you can just run the following (and other managers such as `brew` and `apt` also
work).

```bash
conda install ffmpeg
```

## Run the Example

Follow `transcription.ipynb` to run inference and generate a Zeno project.
