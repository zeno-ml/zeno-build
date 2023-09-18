"""Tools to generate from litellm prompts."""

import asyncio
import logging
from typing import Any

import aiolimiter
from tqdm.asyncio import tqdm_asyncio

from zeno_build.models import lm_config
from zeno_build.prompts import chat_prompt


async def _throttled_litellm_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    n: int,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    try:
        from litellm import acompletion, error
    except ImportError:
        raise ImportError(
            "Please `pip install cohere` to perform cohere-based inference"
        )
    ERROR_ERRORS_TO_MESSAGES = {
        error.InvalidRequestError: "litellm API Invalid Request: Prompt was filtered",
        error.RateLimitError: "litellm API rate limit exceeded. Sleeping for 10 seconds.",  # noqa E501
        error.APIConnectionError: "litellm API Connection Error: Error Communicating with litellm",  # noqa E501
        error.Timeout: "litellm APITimeout Error: litellm Timeout",
        error.ServiceUnavailableError: "litellm service unavailable error: {e}",
        error.APIError: "litellm API error: {e}",
    }
    async with limiter:
        for _ in range(3):
            try:
                return await acompletion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=n,
                )
            except tuple(ERROR_ERRORS_TO_MESSAGES.keys()) as e:
                if isinstance(e, (error.ServiceUnavailableError, error.APIError)):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                elif isinstance(e, error.InvalidRequestError):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": "Invalid Request: Prompt was filtered"
                                }
                            }
                        ]
                    }
                else:
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                await asyncio.sleep(10)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_litellm_completion(
    full_contexts: list[chat_prompt.ChatMessages],
    prompt_template: chat_prompt.ChatMessages,
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    n: int,
    requests_per_minute: int = 150,
) -> list[list[str]]:
    """Generate from litellm Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        prompt_template: Prompt template to use.
        model_config: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        n: Number of responses to generate for each API call.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_litellm_completion_acreate(
            model=model_config.model,
            messages=prompt_template.to_openai_chat_completion_messages(
                full_context=full_context.limit_length(context_length),
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            limiter=limiter,
        )
        for full_context in full_contexts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    # Note: will never be none because it's set, but mypy doesn't know that.
    all_responses = []
    for x in responses:
        all_responses.append([x["choices"][i]["message"]["content"] for i in range(n)])
    return all_responses
