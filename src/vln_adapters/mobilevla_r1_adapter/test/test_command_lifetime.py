import pytest

from mobilevla_r1_adapter.contracts import validate_runtime_contract


def test_real_runtime_refuses_unverified_duration():
    with pytest.raises(ValueError, match="command_duration_s"):
        validate_runtime_contract(
            history_frames=8,
            depth_frames=1,
            pointcloud_points=2048,
            expected_vector_length=12,
            command_duration_s=0.0,
            allow_stub=False,
        )


def test_stub_can_exercise_interface_without_claiming_real_semantics():
    validate_runtime_contract(
        history_frames=8,
        depth_frames=1,
        pointcloud_points=2048,
        expected_vector_length=12,
        command_duration_s=0.0,
        allow_stub=True,
    )

