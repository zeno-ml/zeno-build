"""Tools to generate from prompts."""

import asyncio

import openai
import torch
import tqdm
import transformers

from zeno_build.models import global_models, lm_config
from zeno_build.prompts import chat_prompt


async def generate_from_chat_prompt(
    variables: list[dict[str, str]],
    prompt_template: chat_prompt.ChatMessages,
    model_config: lm_config.LMConfig,
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
    system_name = "System"
    user_name = "User"
    if model_config.provider == "openai":
        async_responses = [
            openai.Completion.acreate(
                engine=model_config.model,
                prompt=prompt_template.to_text_prompt(
                    vars, system_name=system_name, user_name=user_name
                ),
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

        results: list[str] = []
        for vars in tqdm.tqdm(variables, "Generating synchronously from Cohere"):
            try:
                assert global_models.cohere_client is not None
                prompt = prompt_template.to_text_prompt(
                    vars, system_name=system_name, user_name=user_name
                )
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
    elif model_config.provider == "huggingface":
        # Load model
        model_class = (
            model_config.model_cls
            if model_config.model_cls is not None
            else transformers.AutoModel
        )
        model: transformers.PreTrainedModel = model_class.from_pretrained(
            model_config.model
        )
        if not model.can_generate():
            raise ValueError(f"Model {model_config} cannot generate.")
        tokenizer_class = (
            model_config.tokenizer_cls
            if model_config.tokenizer_cls is not None
            else transformers.AutoTokenizer
        )
        tokenizer: transformers.PreTrainedTokenizer = tokenizer_class.from_pretrained(
            model_config.model
        )
        tokenizer.padding_side = "left"
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
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
                vars, system_name=system_name, user_name=user_name
            )
            for vars in variables
        ]
        # Process in batches
        results = []
        batch_size = 8
        for i in tqdm.trange(0, len(filled_prompts), batch_size):
            batch_prompts = filled_prompts[i : i + batch_size]
            encoded_prompts = tokenizer(
                batch_prompts, padding=True, return_tensors="pt"
            )

            with torch.no_grad():
                outputs = model.generate(
                    **encoded_prompts, generation_config=gen_config
                )
            outputs = outputs[:, encoded_prompts["input_ids"].shape[-1] :]
            results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        # Post-processing to get only the system utterance
        results = [x.split(f"\n\n{user_name}:")[0].strip() for x in results]
        return results
    else:
        raise ValueError("Unknown provider, but you can add your own!")
