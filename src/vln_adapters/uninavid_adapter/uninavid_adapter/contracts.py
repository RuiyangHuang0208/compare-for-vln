from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionPlan:
    actions: tuple[str, ...]
    executed_actions: tuple[str, ...]
    stop_after_trajectory: bool


def response_is_current(response, episode_id: str, generation: int) -> bool:
    try:
        return str(response["episode_id"]) == episode_id and int(response["generation"]) == generation
    except (KeyError, TypeError, ValueError):
        return False
