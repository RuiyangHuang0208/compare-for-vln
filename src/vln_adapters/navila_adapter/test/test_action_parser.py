import pytest

from navila_adapter.action_parser import ParsedAction, parse_action


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("stop", ParsedAction("stop", 0.0)),
        ("The next action is move forward 50 cm", ParsedAction("forward", 50.0)),
        ("The next action is turn left 30 degree", ParsedAction("left", 30.0)),
        ("turn right 45 degrees", ParsedAction("right", 45.0)),
    ],
)
def test_official_action_sentences(text, expected):
    assert parse_action(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "walk ahead",
        "move forward 10 cm",
        "turn left -15 degree",
        "move forward nan cm",
        "stop then move forward 25 cm",
        "move forward 25 cm then move forward 50 cm",
    ],
)
def test_unknown_or_unsafe_output_is_rejected(text):
    with pytest.raises(ValueError):
        parse_action(text)
