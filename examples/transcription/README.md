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

- [Browsing Interface](https://zeno-ml-transcription-report.hf.space)
- [Textual Summary](report/)

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

This example also requires the `ffmpeg` library to be installed. You can test
if it is installed by running `ffmpeg --help`. If it is not found, you should
install it through your package manager. For example, if you are using conda,
you can just run the following (and other managers such as `brew` and `apt` also
work).

```bash
conda install ffmpeg
```

## Run the Example

Run the following command to perform evaluation and analysis:

```bash
python main.py --input-metadata speech_accent_archive.csv --results-dir results
```

The results will be saved to the `results` directory, and a report of the
comparison will be displayed using [Zeno](https://zenoml.com/).
Once the evalaution is finished you will be able to view the results at
[https://localhost:8000](https://localhost:8000).
You can then go in and explore the results, making slices, reports, etc.
Alternatively, you can view the
[ready-made hosted report](https://zeno-ml-transcription-report.hf.space).
