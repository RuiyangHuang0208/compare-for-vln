import pytest

from uninavid_adapter.action_parser import parse_action_sequence
from uninavid_adapter.contracts import response_is_current


def test_official_four_action_output_executes_first_two():
    plan = parse_action_sequence("forward left right forward")
    assert plan.actions == ("forward", "left", "right", "forward")
    assert plan.executed_actions == ("forward", "left")
    assert not plan.stop_after_trajectory


def test_stop_truncates_later_actions_and_is_deferred_after_motion():
    plan = parse_action_sequence("forward stop right")
    assert plan.actions == ("forward", "stop")
    assert plan.executed_actions == ("forward",)
    assert plan.stop_after_trajectory


def test_immediate_stop():
    plan = parse_action_sequence("stop")
    assert not plan.executed_actions
    assert plan.stop_after_trajectory


def test_safe_trailing_punctuation_is_removed_only_at_the_end():
    assert parse_action_sequence("forward left.").executed_actions == ("forward", "left")


@pytest.mark.parametrize("text", ["", "FORWARD, left", "forward jump", "forward left right forward stop"])
def test_invalid_output_is_rejected(text):
    with pytest.raises(ValueError):
        parse_action_sequence(text)


def test_late_response_from_previous_episode_generation_is_discarded():
    response = {"episode_id": "episode_a", "generation": 3}
    assert response_is_current(response, "episode_a", 3)
    assert not response_is_current(response, "episode_a", 4)
    assert not response_is_current(response, "episode_b", 3)
