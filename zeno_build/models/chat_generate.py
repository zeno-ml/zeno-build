"""Tools to generate from prompts."""

import logging
import re

import aiolimiter
import openai
import torch
import tqdm
import transformers

from zeno_build.models import global_models, lm_config
from zeno_build.prompts import chat_prompt


async def _generate_from_openai_completion(
    variables: list[dict[str, str]],
    prompt_template: chat_prompt.ChatMessages,
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
    requests_per_minute: int = 300,
) -> list[str]:
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    responses = []
    for vars in tqdm.tqdm(variables, "dispatching openai requests"):
        async with limiter:
            response = await openai.Completion.acreate(
                engine=model_config.model,
                prompt=prompt_template.to_text_prompt(
                    vars,
                    system_name=model_config.system_name,
                    user_name=model_config.user_name,
                ),
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            responses.append(response)
    return [x["choices"][0]["text"] for x in responses]


async def _generate_from_openai_chat_completion(
    variables: list[dict[str, str]],
    prompt_template: chat_prompt.ChatMessages,
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
    requests_per_minute: int = 300,
) -> list[str]:
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    responses = []
    for vars in tqdm.tqdm(variables, "dispatching openai requests"):
        async with limiter:
            response = await openai.ChatCompletion.acreate(
                engine=model_config.model,
                prompt=prompt_template.to_openai_chat_completion_messages(vars),
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            responses.append(response)
    return [x["choices"][0]["message"]["content"] for x in responses]


async def _cohere_acreate(
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> str:
    import cohere

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


async def _generate_from_cohere(
    variables: list[dict[str, str]],
    prompt_template: chat_prompt.ChatMessages,
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
    requests_per_minute: int,
) -> list[str]:
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    responses = []
    for vars in tqdm.tqdm(variables, "dispatching cohere requests"):
        async with limiter:
            response = await _cohere_acreate(
                model=model_config.model,
                prompt=prompt_template.to_text_prompt(
                    vars,
                    system_name=model_config.system_name,
                    user_name=model_config.user_name,
                ),
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            responses.append(response)
    return responses


def _generate_from_huggingface(
    variables: list[dict[str, str]],
    prompt_template: chat_prompt.ChatMessages,
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> list[str]:
    # Load model
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cls = (
        model_config.model_cls
        if model_config.model_cls is not None
        else transformers.AutoModelForCausalLM
    )
    tokenizer_cls = (
        model_config.tokenizer_cls
        if model_config.tokenizer_cls is not None
        else transformers.AutoTokenizer
    )
    model: transformers.PreTrainedModel = model_cls.from_pretrained(
        model_config.model
    ).to(torch_device)
    tokenizer: transformers.PreTrainedTokenizer = tokenizer_cls.from_pretrained(
        model_config.model
    )
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    gen_config = transformers.GenerationConfig(
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Create the prompts
    filled_prompts: list[str] = [
        prompt_template.to_text_prompt(
            vars,
            system_name=model_config.system_name,
            user_name=model_config.user_name,
        )
        for vars in variables
    ]
    # Process in batches
    results = []
    batch_size = 8
    for i in tqdm.trange(0, len(filled_prompts), batch_size):
        batch_prompts = filled_prompts[i : i + batch_size]
        encoded_prompts = tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(torch_device)
        with torch.no_grad():
            outputs = model.generate(**encoded_prompts, generation_config=gen_config)
        outputs = outputs[:, encoded_prompts["input_ids"].shape[-1] :]
        results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    # Post-processing to get only the system utterance
    results = [
        re.split(rf"\n\n({model_config.user_name}|{model_config.system_name}):", x)[
            0
        ].strip()
        for x in results
    ]
    return results


async def generate_from_chat_prompt(
    variables: list[dict[str, str]],
    prompt_template: chat_prompt.ChatMessages,
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
    requests_per_minute: int = 50,
) -> list[str]:
    """Generate from a list of chat-style prompts.

    Args:
        variables: The variables to be replaced in the prompt template.
        prompt_template: The template for the prompt.
        api_based_model_config: The API-based model configuration.
        temperature: The temperature to use.
        max_tokens: The maximum number of tokens to generate.
        top_p: The top p value to use.
        requests_per_minute: Limit on the number of OpenAI requests per minute

    Returns:
        The generated text.
    """
    print(
        f"Generating with {prompt_template=}, {model_config.model=}, "
        f"{temperature=}, {max_tokens=}, {top_p=}..."
    )
    if model_config.provider == "openai":
        return await _generate_from_openai_completion(
            variables,
            prompt_template,
            model_config,
            temperature,
            max_tokens,
            top_p,
            requests_per_minute,
        )
    elif model_config.provider == "openai_chat":
        return await _generate_from_openai_chat_completion(
            variables,
            prompt_template,
            model_config,
            temperature,
            max_tokens,
            top_p,
            requests_per_minute,
        )
    elif model_config.provider == "cohere":
        return await _generate_from_cohere(
            variables,
            prompt_template,
            model_config,
            temperature,
            max_tokens,
            top_p,
            requests_per_minute,
        )
    elif model_config.provider == "huggingface":
        return _generate_from_huggingface(
            variables,
            prompt_template,
            model_config,
            temperature,
            max_tokens,
            top_p,
        )
    else:
        raise ValueError("Unknown provider, but you can add your own!")
