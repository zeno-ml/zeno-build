from zeno_build.models.lm_config import LMConfig
from zeno_build.models.text_generate import generate_from_text_prompt, multiple_generate_from_text_prompt

model = "gpt-3.5-turbo"
model_provider = {"gpt-3.5-turbo": "openai_chat"}
lm_config = LMConfig(provider=model_provider[model], model=model)
prompt_templates = {
    "openai_chat": ("Please rephrase this in the style of Donald Trump: {{text}}")
}

texts = [
    "With Donald Trump skipping the first 2024 Republican presidential primary debate, brawled for second-place status Wednesday night.",
    "Vivek Ramaswamy, the 38-year-old entrepreneur and first-time candidate, was the central figure for much of the night.",
]

predictions = multiple_generate_from_text_prompt(
    [{"text": x} for x in texts],
    prompt_template=prompt_templates[lm_config.provider],
    model_config=lm_config,
    temperature=1,
    max_tokens=50,
    top_p=1.0,
    num_responses=5,
    requests_per_minute=100,
)

print(predictions)

single_predictions = generate_from_text_prompt(
    [{"text": x} for x in texts],
    prompt_template=prompt_templates[lm_config.provider],
    model_config=lm_config,
    temperature=1,
    max_tokens=50,
    top_p=1.0,
    requests_per_minute=100,
)

print(single_predictions)
