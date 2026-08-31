import math
import json

from vln_evaluation.metrics import MetricsAccumulator, classify_failure


def test_failure_attribution_separates_model_and_system_outcomes():
    assert classify_failure(False, model_stop_requested=True) == "model_behavior"
    assert classify_failure(False, model_stop_requested=True, physical_collision_count=1) == "collision_or_scene_contact"
    assert classify_failure(False, watchdog_requested=True) == "inconclusive_timeout"
    assert classify_failure(True, model_stop_requested=True) == "success"


def test_metrics_contract():
    metrics = MetricsAccumulator("dummy", "test")
    metrics.start({
        "episode_id": "e1",
        "scene": "hospital",
        "spawn": [0, 0],
        "goal": [2, 0],
        "camera_horizontal_fov_degrees": 120.0,
    })
    metrics.add_pose(0, 0, 0.0)
    metrics.add_pose(1, 0, 1.0)
    metrics.add_pose(2, 0, 2.0)
    metrics.add_latency(0.1)
    metrics.add_latency(0.3)
    metrics.add_collision({"is_pedestrian": True})
    metrics.set_stuck(True)
    metrics.set_stuck(False)
    result = metrics.finalize(True, 0.1, duration=4.0)
    assert result["path_length"] == 2.0
    assert result["spl"] == 1.0
    assert result["physical_collision_count"] == 1
    assert result["pedestrian_collision_count"] == 1
    assert result["stuck_count"] == 1
    assert result["recovery_count"] == 1
    assert result["comparison_track"] == "untracked"
    assert result["model_inputs"] == []
    assert result["camera_horizontal_fov_degrees"] == 120.0
    assert result["failure_attribution"] is None
    assert math.isclose(result["p50_inference_latency"], 0.2)
    json.dumps(result, allow_nan=False)


def test_missing_latency_is_standard_json_null():
    metrics = MetricsAccumulator("dummy", "test")
    metrics.start({"episode_id": "e2", "spawn": [0, 0], "goal": [1, 0]})
    metrics.add_pose(0, 0, 0.0)
    result = metrics.finalize(False, 1.0, duration=1.0)
    assert result["mean_inference_latency"] is None
    assert result["p95_inference_latency"] is None
    assert result["failure_attribution"] is None
    assert "NaN" not in json.dumps(result, allow_nan=False)
