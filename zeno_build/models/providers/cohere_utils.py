"""Tools to generate from Cohere prompts."""
import logging
import os

import aiolimiter
import cohere
from tqdm.asyncio import tqdm_asyncio

from zeno_build.models import global_models, lm_config
from zeno_build.prompts import chat_prompt


async def _throttled_cohere_acreate(
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> str:
    async with limiter:
        assert global_models.cohere_client is not None
        try:
            response = global_models.cohere_client.generate(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                p=top_p,
            )
            return response.generations[0].text
        except cohere.CohereAPIError as e:
            # Cohere API sometimes rejects queries, if so output a blank line
            logging.getLogger(__name__).warn(
                f"Warning! Cohere API rejected query for {prompt=}: {e.message}"
            )
            return ""


async def generate_from_cohere(
    full_contexts: list[chat_prompt.ChatMessages],
    prompt_template: chat_prompt.ChatMessages,
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int,
) -> list[str]:
    """Generate outputs from the Cohere API.

    COHERE_API_KEY must be set in order for this function to work.

    Args:
        full_contexts: The full contexts to generate from.
        prompt_template: The prompt template to use.
        model_config: The model configuration.
        temperature: The temperature to use.
        max_tokens: The maximum number of tokens to generate.
        top_p: The top-p value to use.
        context_length: The context length to use.
        requests_per_minute: The number of requests per minute to make.

    Returns:
        The generated outputs.
    """
    if "COHERE_API_KEY" not in os.environ:
        raise ValueError(
            "COHERE_API_KEY environment variable must be set when using the "
            "Cohere API."
        )
    global_models.cohere_client = cohere.Client(os.environ["COHERE_API_KEY"])
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_cohere_acreate(
            model=model_config.model,
            prompt=prompt_template.to_text_prompt(
                full_context=full_context.limit_length(context_length),
                name_replacements=model_config.name_replacements,
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for full_context in full_contexts
    ]
    return await tqdm_asyncio.gather(*async_responses)
