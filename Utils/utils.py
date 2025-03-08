import re
from typing import Any, List

from math_verify import parse, verify


def separate_steps(steps: List[str], mode: str = "join") -> Any:
    delimiter = "\n\n"
    if mode == "join":
        if not isinstance(steps, list):
            raise TypeError("For 'join' mode, 'steps' must be a list of strings.")
        return delimiter.join(steps)
    elif mode == "split":
        if not isinstance(steps, str):
            raise TypeError("For 'split' mode, 'steps' must be a string.")
        return steps.split(delimiter)
    else:
        raise ValueError("Mode should be either 'join' or 'split'.")


def check_correctness(generated_response: str, expected_answer: str) -> bool:
    sentences = re.split(
        r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", generated_response.strip()
    )
    last_sentence = sentences[-1] if sentences else ""
    gold, answer = parse(f"${expected_answer}$"), parse(last_sentence)
    return verify(gold, answer)
