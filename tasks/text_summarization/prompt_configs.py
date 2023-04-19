"""Specify the prompts you want to use."""

prompt_configs = {
    "standard": "Summarize the following text:\n[X]\n\nSummary:",
    "tldr": "[X]\nTL;DR:",
    "concise": "Write a short and concise summary of the following text:\n[X]\n\nSummary:",  # noqa: E501
    "complete": "Write a complete summary of the following text:\n[X]\n\nSummary:",
}
