"""Tools to generate from prompts."""
import asyncio

from zeno_build.models import lm_config
from zeno_build.models.providers.cohere_utils import generate_from_cohere
from zeno_build.models.providers.huggingface_utils import generate_from_huggingface
from zeno_build.models.providers.openai_utils import (
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
)
from zeno_build.prompts import chat_prompt


def generate_from_chat_prompt(
    full_contexts: list[chat_prompt.ChatMessages],
    prompt_template: chat_prompt.ChatMessages,
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 150,
) -> list[str]:
    """Generate from a list of chat-style prompts.

    Args:
        variables: The variables to be replaced in the prompt template.
        prompt_template: The template for the prompt.
        api_based_model_config: The API-based model configuration.
        temperature: The temperature to use.
        max_tokens: The maximum number of tokens to generate.
        top_p: The top p value to use.
        context_length: The length of the context to use.
        requests_per_minute: Limit on the number of OpenAI requests per minute

    Returns:
        The generated text.
    """
    print(
        f"Generating with {prompt_template=}, {model_config.model=}, "
        f"{temperature=}, {max_tokens=}, {top_p=}, {context_length=}..."
    )
    if model_config.provider == "openai":
        return asyncio.run(
            generate_from_openai_completion(
                full_contexts,
                prompt_template,
                model_config,
                temperature,
                max_tokens,
                top_p,
                context_length,
                requests_per_minute,
            )
        )
    elif model_config.provider == "openai_chat":
        return asyncio.run(
            generate_from_openai_chat_completion(
                full_contexts,
                prompt_template,
                model_config,
                temperature,
                max_tokens,
                top_p,
                context_length,
                requests_per_minute,
            )
        )
    elif model_config.provider == "cohere":
        return asyncio.run(
            generate_from_cohere(
                full_contexts,
                prompt_template,
                model_config,
                temperature,
                max_tokens,
                top_p,
                context_length,
                requests_per_minute,
            )
        )
    elif model_config.provider == "huggingface":
        return generate_from_huggingface(
            full_contexts,
            prompt_template,
            model_config,
            temperature,
            max_tokens,
            top_p,
            context_length,
        )
    else:
        raise ValueError("Unknown provider, but you can add your own!")
