from __future__ import annotations

from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class ParsedAction:
    kind: str
    value: float = 0.0


_NUMBER = r"(\d+(?:\.\d+)?)"
_PATTERNS = {
    "forward": re.compile(rf"\bmove\s+forward\s+{_NUMBER}\s*(?:cm|centimeters?)\b", re.IGNORECASE),
    "left": re.compile(rf"\bturn\s+left\s+{_NUMBER}\s*degrees?\b", re.IGNORECASE),
    "right": re.compile(rf"\bturn\s+right\s+{_NUMBER}\s*degrees?\b", re.IGNORECASE),
}
_STOP = re.compile(r"\bstop\b", re.IGNORECASE)


def parse_action(text, allowed_forward_cm=(25, 50, 75), allowed_turn_degrees=(15, 30, 45)):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("NaVILA action is empty")
    matches = []
    if _STOP.search(text):
        matches.append(ParsedAction("stop"))
    for kind, pattern in _PATTERNS.items():
        for match in pattern.finditer(text):
            value = float(match.group(1))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"NaVILA {kind} value must be finite and positive")
            matches.append(ParsedAction(kind, value))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one supported NaVILA action, found {len(matches)} in {text!r}")
    action = matches[0]
    allowed = allowed_forward_cm if action.kind == "forward" else allowed_turn_degrees
    if action.kind != "stop" and not any(math.isclose(action.value, float(value)) for value in allowed):
        raise ValueError(f"Unsupported NaVILA {action.kind} step {action.value}; allowed={list(allowed)}")
    return action
