from __future__ import annotations

from .contracts import ActionPlan


ALLOWED_ACTIONS = frozenset({"forward", "left", "right", "stop"})
SAFE_TRAILING_PUNCTUATION = ".,;:!?"


def parse_action_sequence(text: str, max_predictions: int = 4, max_execute: int = 2) -> ActionPlan:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Uni-NaVid action output is empty")
    tokens = text.strip().lower().split()
    tokens[-1] = tokens[-1].rstrip(SAFE_TRAILING_PUNCTUATION)
    if not tokens[-1]:
        raise ValueError("Uni-NaVid final action token is empty")
    values = tuple(tokens)
    if len(values) > max_predictions:
        raise ValueError(f"Uni-NaVid returned {len(values)} actions; maximum is {max_predictions}")
    unknown = [token for token in values if token not in ALLOWED_ACTIONS]
    if unknown:
        raise ValueError(f"Unsupported Uni-NaVid action token(s): {unknown}")
    stop_index = values.index("stop") if "stop" in values else len(values)
    normalized = values[: stop_index + 1] if stop_index < len(values) else values
    selected = normalized[:max_execute]
    movement = tuple(action for action in selected if action != "stop")
    return ActionPlan(
        actions=normalized,
        executed_actions=movement,
        stop_after_trajectory="stop" in selected,
    )
