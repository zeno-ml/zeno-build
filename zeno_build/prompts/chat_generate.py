"""Tools to generate from prompts."""

import asyncio

import openai
import tqdm

from zeno_build.models import api_based_model, global_models
from zeno_build.prompts import chat_prompt


async def generate_from_chat_prompt(
    variables: list[dict[str, str]],
    prompt_template: chat_prompt.ChatMessages,
    model_config: api_based_model.ApiBasedModelConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> list[str]:
    """Generate from a list of chat-style prompts.

    Args:
        variables: The variables to be replaced in the prompt template.
        prompt_template: The template for the prompt.
        api_based_model_config: The API-based model configuration.
        temperature: The temperature to use.
        max_tokens: The maximum number of tokens to generate.
        top_p: The top p value to use.

    Returns:
        The generated text.
    """
    print(
        f"Generating with {prompt_template=}, {model_config.model=}, "
        f"{temperature=}, {max_tokens=}, {top_p=}..."
    )
    if model_config.provider == "openai":
        async_responses = [
            openai.Completion.acreate(
                engine=model_config.model,
                prompt=prompt_template.to_text_prompt(vars),
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            for vars in variables
        ]
        responses = await asyncio.gather(*async_responses)
        return [x["choices"][0]["text"] for x in responses]
    elif model_config.provider == "openai_chat":
        async_responses = [
            openai.ChatCompletion.acreate(
                model=model_config.model,
                messages=prompt_template.to_openai_chat_completion_messages(vars),
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            for vars in variables
        ]
        responses = await asyncio.gather(*async_responses)
        return [x["choices"][0]["message"]["content"] for x in responses]
    elif model_config.provider == "cohere":
        import cohere

        results = []
        for vars in tqdm.tqdm(variables, "Generating synchronously from Cohere"):
            try:
                assert global_models.cohere_client is not None
                prompt = prompt_template.to_text_prompt(vars)
                response = global_models.cohere_client.generate(
                    model=model_config.model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    p=top_p,
                )
                results.append(response.generations[0].text)
            except cohere.CohereAPIError as e:
                # Cohere API sometimes rejects queries, if so output a blank line
                print(f"Warning! Cohere API rejected query for {prompt=}: {e.message}")
                results.append("")
        return results
    else:
        raise ValueError("Unknown provider, but you can add your own!")
