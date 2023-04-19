
# Specify which models you want to use
model_configs = {
    "cohere_command_xlarge": {
        "provider": "cohere",
        "config": {
            "model": "command-xlarge-nightly",
            "temperature": 0.3,
            "max_tokens": 100,
            "top_p": 1,
        }
    },
    "openai_davinci_003": {
        "provider": "openai",
        "config": {
            "model": "text-davinci-003",
            "temperature": 0.3,
            "max_tokens": 100,
            "top_p": 1,
        }
    },
    "openai_gpt_3.5_turbo": {
        "provider": "openai_chat",
        "config": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.3,
            "max_tokens": 100,
            "top_p": 1,
        }
    },
}