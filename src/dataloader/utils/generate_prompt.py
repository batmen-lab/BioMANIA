
def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            f"{example['input']}\n{example['instruction']}\n### Response:"
        )
    return (
        f"{example['instruction']}\n### Response:"
    )
