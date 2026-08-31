import math

import pytest

from mobilevla_r1_adapter.action_parser import ActionParseError, parse_action


VECTOR = [0.345, 0.0, -0.397, -0.01, 2.0, 0.5, 0.0, 0.0, 0.17, 0.0, 0.0, 0.19]


def response(answer):
    return f"<think>There are 99 doors at 3.5 meters.</think>\n<answer>{answer}</answer>"


def test_exact_official_vector_only_exports_three_velocity_fields():
    parsed = parse_action(response(VECTOR))
    assert parsed.velocity == pytest.approx((0.345, 0.0, -0.397))
    assert parsed.vector[3:] == pytest.approx(VECTOR[3:])


@pytest.mark.parametrize(
    "text",
    [
        "",
        "The velocity is 0.2 0.0 0.1",
        "<think>[0.2, 0.0, 0.1]</think>",
        "<answer>[0.2, 0.0, 0.1]",
        "<answer></answer>",
        "<answer>[0.2, 0.0, 0.1]</answer>",
        "<answer>[0.2, 0.0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 'bad']</answer>",
        "<answer>[nan, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</answer>",
        "<answer>[1e999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</answer>",
        f"<answer>{VECTOR}</answer><answer>{VECTOR}</answer>",
    ],
)
def test_rejects_unsafe_or_ambiguous_output(text):
    with pytest.raises(ActionParseError):
        parse_action(text)


@pytest.mark.parametrize("behavior", ["jump", "dance", "hello", "stretch", "sit", "lie down"])
def test_unsupported_behaviors_are_stop_reasons(behavior):
    with pytest.raises(ActionParseError) as caught:
        parse_action(f"<think>safe</think><answer>{behavior}</answer>")
    assert caught.value.reason == "unsupported_behavior"


def test_finite_exponents_are_accepted():
    parsed = parse_action(response([1e-2] + [0.0] * 11))
    assert math.isclose(parsed.velocity[0], 0.01)

