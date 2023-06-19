"""Tools to generate from prompts."""
import asyncio

from zeno_build.models import lm_config
from zeno_build.models.providers.cohere_utils import generate_code_from_cohere
from zeno_build.models.providers.huggingface_utils import generate_code_from_huggingface
from zeno_build.models.providers.openai_utils import (
    generate_code_from_openai_chat_completion,
    generate_code_from_openai_completion,
)


def generate_from_code_prompt(
    variables: list[dict[str, str]],
    prompt_template: str,
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
    requests_per_minute: int = 50,
) -> list[str]:
    """Generate from a list of code prompts.

    Args:
        variables: The variables to be replaced in the prompt template.
        prompt_template: The template for the prompt.
        model_config: The API-based model configuration.
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
        f"{temperature=}, {max_tokens=}, {top_p=}..."
    )
    if model_config.provider == "huggingface":
        return generate_code_from_huggingface(
            variables,
            prompt_template,
            model_config,
            temperature,
            max_tokens,
            top_p,
        )
    elif model_config.provider == "openai":
        return asyncio.run(
            generate_code_from_openai_completion(
                variables,
                prompt_template,
                model_config,
                temperature,
                max_tokens,
                top_p,
                requests_per_minute,
            )
        )
    elif model_config.provider == "openai_chat":
        return asyncio.run(
            generate_code_from_openai_chat_completion(
                variables,
                prompt_template,
                model_config,
                temperature,
                max_tokens,
                top_p,
                requests_per_minute,
            )
        )
    elif model_config.provider == "cohere":
        return asyncio.run(
            generate_code_from_cohere(
                variables,
                prompt_template,
                model_config,
                temperature,
                max_tokens,
                top_p,
                requests_per_minute,
            )
        )
    else:
        raise NotImplementedError(f"Unsupported provider: {model_config.provider}")
