"""Presets for Critique."""

critique_presets = {
    "ROUGE-1": {
        "metric": "rouge",
        "config": {"variety": "rouge_1"},
    },
    "ROUGE-2": {
        "metric": "rouge",
        "config": {"variety": "rouge_2"},
    },
    "ROUGE-L": {
        "metric": "rouge",
        "config": {"variety": "rouge_l"},
    },
    "BERTScore (bert-base-uncased)": {
        "metric": "bert_score",
        "config": {"model": "bert-base-uncased"},
    },
    "UniEval (Relevance)": {
        "metric": "uni_eval",
        "config": {"task": "summarization", "evaluation_aspect": "relevance"},
    },
    "UniEval (Consistency)": {
        "metric": "uni_eval",
        "config": {"task": "summarization", "evaluation_aspect": "consistency"},
    },
    "UniEval (Coherence)": {
        "metric": "uni_eval",
        "config": {"task": "summarization", "evaluation_aspect": "coherence"},
    },
    "UniEval (Fluency)": {
        "metric": "uni_eval",
        "config": {"task": "summarization", "evaluation_aspect": "fluency"},
    },
    "BartScore (Coverage)": {
        "metric": "bart_score",
        "config": {"model": "facebook/bart-large-cnn", "variety": "reference_target_bidirectional"},
    },
    "Length Ratio": {
        "metric": "length_ratio",
        "config": {},
    },
}