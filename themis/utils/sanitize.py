import re

from themis.core.completion.utils import GenerationConfig
from themis.utils.logger import console, logger


def extract_code_block(output: str) -> str:
    r"""Extract the code block from the output."""
    output = output.strip().split("</think>")[-1].strip()
    # FIXME: Handle for reasoning LLM
    if "```" not in output:
        return output
    try:
        pattern = r"```(.*?)\n([\s\S]*?)\n```"
        result = re.findall(pattern, output)
        return result[0][1]
    except Exception as e:
        logger.error(f"Error processing output: {e}")
        return output


def parse_sampling_params(sampling_params: list[str]) -> GenerationConfig:
    params = {}
    console.print("[blue]Setting sampling parameters:[/blue]")

    for param in sampling_params:
        key, value = param.split("=", 1)

        # Try to convert to appropriate types
        if value.lower() == "true":
            params[key] = True
        elif value.lower() == "false":
            params[key] = False
        elif value.isdigit():
            params[key] = int(value)
        elif is_float(value):
            params[key] = float(value)
        else:
            params[key] = value
        console.print(f"  [cyan]{key}:[/cyan] {value}")
    return GenerationConfig(**params)


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False
