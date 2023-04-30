"""Generate from a textual prompt."""

import asyncio

import openai
import tqdm

from zeno_build.models import global_models, lm_config
from zeno_build.prompts.prompt_utils import replace_variables


async def generate_from_text_prompt(
    variables: list[dict[str, str]],
    prompt_template: str,
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> list[str]:
    """Generate from a textual prompt.

    Args:
        variables: The source set of variables to consume.
        prompt_template: The template for the prompt.
        model_config: Configuration of the model.
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
                prompt=replace_variables(prompt_template, vars),
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
                messages=[
                    {
                        "role": "user",
                        "content": replace_variables(prompt_template, vars),
                    },
                ],
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
                prompt = replace_variables(prompt_template, vars)
                assert global_models.cohere_client is not None
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
        raise ValueError("Unknown model_config.provider, but you can add your own!")
