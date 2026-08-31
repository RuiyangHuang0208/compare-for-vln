import csv

from vln_evaluation.evaluator import append_summary_row, watchdog_expired


def test_summary_schema_expansion_preserves_existing_rows(tmp_path):
    path = tmp_path / "summary.csv"
    append_summary_row(str(path), {"model_name": "old", "success": "False"})
    append_summary_row(
        str(path),
        {
            "model_name": "navila",
            "success": "True",
            "comparison_track": "rgb_only",
            "model_inputs": '["rgb","instruction"]',
        },
    )
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {
            "model_name": "old",
            "success": "False",
            "comparison_track": "",
            "model_inputs": "",
        },
        {
            "model_name": "navila",
            "success": "True",
            "comparison_track": "rgb_only",
            "model_inputs": '["rgb","instruction"]',
        },
    ]


def test_watchdog_uses_live_simulation_clock_before_wall_time():
    assert not watchdog_expired(50.0, 26.0, 100.1, 0.1, 2.0, 5.0)
    assert watchdog_expired(50.0, 50.1, 100.1, 0.1, 2.0, 5.0)
    assert watchdog_expired(50.0, 26.0, 100.1, 6.0, 2.0, 5.0)
