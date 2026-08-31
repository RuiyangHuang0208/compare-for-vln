import builtins

from mobilevla_r1_adapter.official_inference import SYSTEM_PROMPT, load_official_module, strip_prompt_echo


def test_strip_prompt_echo_leaves_only_generated_answer():
    instruction = "Walk toward the chair"
    generated = "<think>turn slightly</think><answer>[0.2, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0]</answer>"
    decoded = f"chat prefix {SYSTEM_PROMPT}\n{instruction} assistant {generated}"
    cleaned = strip_prompt_echo(decoded, instruction)
    assert cleaned == f"assistant {generated}"
    assert cleaned.count("<answer>") == 1


def test_strip_prompt_echo_preserves_non_echoed_generation():
    generated = "<answer>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</answer>"
    assert strip_prompt_echo(generated, "Stop") == generated


def test_official_module_missing_optional_import_is_contained(tmp_path):
    (tmp_path / "inference.py").write_text(
        "def function(value: Optional[str]) -> Optional[str]:\n    return value\n", encoding="utf-8"
    )
    had_optional = hasattr(builtins, "Optional")
    previous = getattr(builtins, "Optional", None)
    module = load_official_module(tmp_path)
    assert module.function("ok") == "ok"
    assert hasattr(builtins, "Optional") is had_optional
    if had_optional:
        assert builtins.Optional is previous
