"""Tools to generate from huggingface."""

import re

import torch
import tqdm
import transformers

from zeno_build.models import lm_config


def generate_from_huggingface(
    prompts: list[str],
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> list[str]:
    """Generate outputs from a huggingface model.

    Args:
        prompts: The prompts to generate from.
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
    # Process in batches
    results = []
    batch_size = 8
    for i in tqdm.trange(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        encoded_prompts = tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(torch_device)
        with torch.no_grad():
            outputs = model.generate(**encoded_prompts, generation_config=gen_config)
        outputs = outputs[:, encoded_prompts["input_ids"].shape[-1] :]
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded_outputs)
    # Post-processing to get only the system utterance
    results = [re.split("\n\n", x)[0].strip() for x in results]
    return results
