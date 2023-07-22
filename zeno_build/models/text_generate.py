"""Generate from a textual prompt."""

import asyncio

import tqdm

from zeno_build.models import global_models, lm_config
from zeno_build.models.providers.huggingface_utils import generate_from_huggingface
from zeno_build.models.providers.openai_utils import (
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
)
from zeno_build.prompts.chat_prompt import ChatMessages, ChatTurn
from zeno_build.prompts.prompt_utils import replace_variables


def generate_from_text_prompt(
    variables: list[dict[str, str]],
    prompt_template: str,
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
    requests_per_minute: int = 150,
) -> list[str]:
    """Generate from a textual prompt.

    Args:
        variables: The source set of variables to consume.
        prompt_template: The template for the prompt.
        model_config: Configuration of the model.
        temperature: The temperature to use.
        max_tokens: The maximum number of tokens to generate.
        top_p: The top p value to use.
        requests_per_minute: Limit on the number of OpenAI requests per minute.

    Returns:
        The generated text.
    """
    print(
        f"Generating with {prompt_template=}, {model_config.model=}, "
        f"{temperature=}, {max_tokens=}, {top_p=}..."
    )
    if model_config.provider == "huggingface":
        prompts = [replace_variables(prompt_template, vars) for vars in variables]
        return generate_from_huggingface(
            prompts,
            model_config,
            temperature,
            max_tokens,
            top_p,
        )
    elif model_config.provider == "openai":
        response_per_api_call = 1
        prompts = [replace_variables(prompt_template, vars) for vars in variables]
        return asyncio.run(
            generate_from_openai_completion(
                prompts,
                model_config,
                temperature,
                max_tokens,
                response_per_api_call,
                top_p,
                requests_per_minute,
            )
        )
    elif model_config.provider == "openai_chat":
        full_contexts = [
            ChatMessages(
                [
                    ChatTurn(
                        role="user",
                        content=replace_variables(prompt_template, vars),
                    )
                ]
            )
            for vars in variables
        ]
        return asyncio.run(
            generate_from_openai_chat_completion(
                full_contexts=full_contexts,
                prompt_template=ChatMessages([]),
                model_config=model_config,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=1,
                context_length=1,
                requests_per_minute=requests_per_minute,
            )
        )
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
