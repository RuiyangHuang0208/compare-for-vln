import numpy as np

from mobilevla_r1_adapter.sensor_sync import (
    ApproximateSensorSynchronizer,
    StampedValue,
    sample_rgb_history,
)


def item(stamp, episode="e"):
    return StampedValue(stamp, episode, stamp)


def test_synchronizes_same_episode_within_slop():
    sync = ApproximateSensorSynchronizer(10, 0.05, 0.25)
    assert sync.add("rgb", item(10.00), 10.02) is None
    assert sync.add("depth", item(10.02), 10.02) is None
    bundle = sync.add("camera_info", item(10.01), 10.02)
    assert bundle is not None
    assert bundle.depth.value == 10.02


def test_rejects_stale_misaligned_and_cross_episode_data():
    stale = ApproximateSensorSynchronizer(10, 0.05, 0.25)
    assert stale.add("rgb", item(1.0), 2.0) is None
    assert stale.last_rejection_reason == "stale_or_future_sensor"
    mismatch = ApproximateSensorSynchronizer(10, 0.01, 0.25)
    mismatch.add("rgb", item(2.0), 2.0)
    mismatch.add("depth", item(2.1), 2.1)
    assert mismatch.add("camera_info", item(2.1), 2.1) is None
    assert mismatch.last_rejection_reason == "sensor_time_mismatch"
    episodes = ApproximateSensorSynchronizer(10, 0.05, 0.25)
    episodes.add("rgb", item(3.0, "old"), 3.0)
    episodes.add("depth", item(3.0, "new"), 3.0)
    assert episodes.add("camera_info", item(3.0, "new"), 3.0) is None
    assert episodes.last_rejection_reason == "sensor_episode_mismatch"


def test_reports_missing_modality_after_sync_window():
    sync = ApproximateSensorSynchronizer(10, 0.05, 0.25)
    sync.add("rgb", item(4.0), 4.0)
    sync.add("camera_info", item(4.0), 4.0)
    assert sync.pending_failure(4.04) is None
    assert sync.pending_failure(4.06) == "missing_depth"


def test_history_uniform_sampling_current_frame_last_and_padding():
    frames = [np.full((1, 1, 3), value, np.uint8) for value in range(10)]
    sampled = sample_rgb_history(frames, 8)
    assert [int(frame[0, 0, 0]) for frame in sampled] == [0, 1, 2, 3, 5, 6, 7, 9]
    padded = sample_rgb_history(frames[:2], 8)
    assert [int(frame[0, 0, 0]) for frame in padded] == [0, 1, 1, 1, 1, 1, 1, 1]
