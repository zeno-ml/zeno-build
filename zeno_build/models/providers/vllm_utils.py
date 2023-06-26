"""Tools to generate from huggingface."""

import re

import torch
import tqdm
import vllm

from zeno_build.models import lm_config
from zeno_build.prompts import chat_prompt
from zeno_build.prompts.prompt_utils import replace_variables


def generate_from_vllm(
    full_contexts: list[chat_prompt.ChatMessages],
    prompt_template: chat_prompt.ChatMessages,
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
) -> list[str]:
    """Generate outputs from a huggingface model.

    Args:
        full_contexts: The full contexts to generate from.
        prompt_template: The prompt template to use.
        model_config: The model configuration.
        temperature: The temperature to use.
        max_tokens: The maximum number of tokens to generate.
        top_p: The top-p value to use.
        context_length: The context length to use.

    Returns:
        The generated outputs.
    """
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
        model_config.model,
        **model_config.model_loader_kwargs,
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
            full_context=full_context.limit_length(context_length),
            name_replacements=model_config.name_replacements,
        )
        for full_context in full_contexts
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
    results = [re.split("\n\n", x)[0].strip() for x in results]
    return results


def text_generate_from_vllm(
    variables: list[dict[str, str]],
    prompt_template: str,
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> list[str]:
    """Generate outputs from a huggingface model.

    Args:
        variables: The variables to be replaced in the prompt template.
        prompt_template: The prompt template to use.
        model_config: The model configuration.
        temperature: The temperature to use.
        max_tokens: The maximum number of tokens to generate.
        top_p: The top-p value to use.

    Returns:
        The generated outputs.
    """
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
        model_config.model,
        **model_config.model_loader_kwargs,
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
    # Generate
    filled_prompts = [replace_variables(prompt_template, vars) for vars in variables]
    # Process in batches
    results = []
    batch_size = 8
    for i in tqdm.trange(0, len(filled_prompts), batch_size):
        batch = filled_prompts[i : i + batch_size]
        encoded_prompts = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        ).to(torch_device)
        with torch.no_grad():
            outputs = model.generate(**encoded_prompts, generation_config=gen_config)
        outputs = outputs[:, encoded_prompts["input_ids"].shape[-1] :]
        results.extend(
            tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                truncate_before_pattern=["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
            )
        )
    # Post-processing to get only the system utterance
    results = [re.split("\n\n", x)[0].strip() for x in results]
    return results
