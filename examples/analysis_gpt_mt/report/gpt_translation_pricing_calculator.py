"""A script to calculate estimated cost for translation."""

# Calculation for GPT-series models
for model, input_cost, output_cost in [
    ("gpt-3.5-turbo", 1.5, 1.5),
    ("text-davinci-003", 20, 20),
    ("gpt-4", 30, 60),
]:
    for shots in [0, 1, 5]:
        input_tokens = (1.5 + shots) / 2.5
        output_tokens = (1 + shots) / 2.5
        price = input_tokens * input_cost + output_tokens * output_cost
        print(f"{model} {shots}-shot: {price}")
