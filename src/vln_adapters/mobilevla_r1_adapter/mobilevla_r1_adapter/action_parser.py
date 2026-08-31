from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import re

from .contracts import EXPECTED_VECTOR_LENGTH, UNSUPPORTED_BEHAVIORS, VELOCITY_INDICES


class ActionParseError(ValueError):
    def __init__(self, message: str, reason: str = "parser_failure"):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class ParsedAction:
    vector: tuple[float, ...]
    velocity: tuple[float, float, float]


_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def _unsupported_behavior(text: str) -> str | None:
    lowered = text.lower()
    for behavior in UNSUPPORTED_BEHAVIORS:
        if re.search(rf"\b{re.escape(behavior)}\b", lowered):
            return behavior
    return None


def parse_action(text: str, expected_length: int = EXPECTED_VECTOR_LENGTH) -> ParsedAction:
    if not isinstance(text, str) or not text.strip():
        raise ActionParseError("empty model response")
    blocks = _ANSWER.findall(text)
    if len(blocks) != 1:
        if len(blocks) == 0:
            behavior = _unsupported_behavior(text)
            if behavior:
                raise ActionParseError(f"unsupported behavior: {behavior}", "unsupported_behavior")
            raise ActionParseError("response must contain exactly one closed <answer> block")
        raise ActionParseError("response contains multiple <answer> blocks")
    answer = blocks[0].strip()
    behavior = _unsupported_behavior(answer)
    if behavior:
        raise ActionParseError(f"unsupported behavior: {behavior}", "unsupported_behavior")
    try:
        value = ast.literal_eval(answer)
    except (SyntaxError, ValueError) as error:
        raise ActionParseError("<answer> must be a Python-style numeric list") from error
    if not isinstance(value, list) or len(value) != expected_length:
        size = len(value) if isinstance(value, list) else "not-a-list"
        raise ActionParseError(f"expected one {expected_length}-element list, got {size}")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ActionParseError("action vector contains a non-numeric element")
    vector = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in vector):
        raise ActionParseError("action vector contains NaN or Inf")
    velocity = tuple(vector[index] for index in VELOCITY_INDICES)
    return ParsedAction(vector=vector, velocity=velocity)


def is_model_stop(velocity, epsilon: float = 1.0e-4) -> bool:
    return all(abs(float(value)) <= epsilon for value in velocity)

