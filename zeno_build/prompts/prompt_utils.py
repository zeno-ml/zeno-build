"""Various utilities to make prompting easier."""


def replace_variables(s: str, variables: dict[str, str]) -> str:
    """Replace variables in a string."""
    for k, v in variables.items():
        s = s.replace("{{" + k + "}}", v)
    return s
