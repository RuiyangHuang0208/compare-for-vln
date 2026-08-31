import numpy as np
import pytest

from mobilevla_r1_adapter.pointcloud_from_depth import PointCloudError, pointcloud_from_depth


def test_metric_plane_projection_uses_camera_info_and_is_deterministic():
    depth = np.full((3, 3), 2.0, np.float32)
    intrinsics = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
    first = pointcloud_from_depth(depth, intrinsics, max_points=9, normalize=False, seed=7)
    second = pointcloud_from_depth(depth, intrinsics, max_points=9, normalize=False, seed=7)
    assert np.array_equal(first, second)
    assert first.shape == (9, 3)
    assert any(np.allclose(point, (0.0, 0.0, 2.0)) for point in first)
    assert np.allclose(first[:, 2], 2.0)


def test_sampling_normalization_and_invalid_depth():
    depth = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
    k = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
    a = pointcloud_from_depth(depth, k, max_points=16, normalize=True, seed=3)
    b = pointcloud_from_depth(depth, k, max_points=16, normalize=True, seed=3)
    assert np.array_equal(a, b)
    assert a.shape == (16, 3) and np.isfinite(a).all()
    with pytest.raises(PointCloudError):
        pointcloud_from_depth(np.zeros((2, 2), np.float32), k)

