"""Tools to generate from vLLM.

More info about vLLM here:
> https://github.com/vllm-project/vllm

Note that vllm isn't installed by default, largely because it is
somewhat less mature than other libraries and has many library
dependencies, notably GPU. You will need to install vLLM to use
this inference library.
"""

import re

import torch.cuda

from zeno_build.models import lm_config


def generate_from_vllm(
    prompts: list[str],
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> list[str]:
    """Generate outputs from a VLLM model.

    Args:
        prompts: The prompts to use.
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
    except ImportError:
        raise ImportError("Please `pip install vllm` to perform vllm-based inference")
    # Load model
    num_gpus = torch.cuda.device_count()
    llm = vllm.LLM(
        model=model_config.model,
        tensor_parallel_size=num_gpus,
    )
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    # Process in batches
    results = llm.generate(prompts, sampling_params)
    # Post-processing to get only the system utterance
    results = [
        re.split("\n\n", x.outputs[0].text)[0].replace("</s>", "").strip()
        for x in results
    ]
    return results
