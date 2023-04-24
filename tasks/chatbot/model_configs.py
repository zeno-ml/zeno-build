"""Specifies which models can be used for building a chatbot."""

model_configs = {
    "openai_davinci_003": {
        "provider": "openai",
        "model": "text-davinci-003",
    },
    "openai_gpt_3.5_turbo": {
        "provider": "openai_chat",
        "model": "gpt-3.5-turbo",
    },
    "cohere_command_xlarge": {
        "provider": "cohere",
        "model": "command-xlarge-nightly",
    },
}
