"""Tools to generate from vLLM.

More info about vLLM here:
> https://github.com/vllm-project/vllm

Note that vllm isn't installed by default, largely because it is
somewhat less mature than other libraries and has many library
dependencies, notably GPU. You will need to install vLLM to use
this inference library.
"""

import re

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
    """Generate outputs from a VLLM model.

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
    # Import vllm
    try:
        import vllm
    except:
        raise ImportError(
            "Please `pip install vllm` to perform vllm-based inference"
        )
    # Load model
    llm = vllm.LLM(model=model_config.model)
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
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
    results = llm.generate(filled_prompts, sampling_params)
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
    # Import vllm
    try:
        import vllm
    except:
        raise ImportError(
            "Please `pip install vllm` to perform vllm-based inference"
        )
    # Load model
    llm = vllm.LLM(model=model_config.model)
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    # Generate
    filled_prompts = [replace_variables(prompt_template, vars) for vars in variables]
    # Process in batches
    results = llm.generate(filled_prompts, sampling_params)
    # Post-processing to get only the system utterance
    results = [re.split("\n\n", x)[0].strip() for x in results]
    return results
