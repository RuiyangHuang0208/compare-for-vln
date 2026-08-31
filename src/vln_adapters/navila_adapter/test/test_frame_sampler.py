import numpy as np

from navila_adapter.contracts import sample_episode_frames


def frame(value):
    return np.full((4, 6, 3), value, dtype=np.uint8)


def values(frames):
    return [int(item[0, 0, 0]) for item in frames]


def test_short_history_is_left_padded_and_current_is_last():
    current = frame(9)
    sampled = sample_episode_frames([frame(3), current], 8)
    assert len(sampled) == 8
    assert values(sampled) == [0, 0, 0, 0, 0, 0, 3, 9]
    assert all(item.shape == (512, 512, 3) for item in sampled[:6])
    assert all(item.shape == (4, 6, 3) for item in sampled[6:])
    assert sampled[-1] is current


def test_entire_episode_history_is_uniformly_sampled():
    history = [frame(index) for index in range(20)]
    sampled = sample_episode_frames(history, 8)
    assert values(sampled) == [0, 2, 5, 8, 10, 13, 16, 19]
    assert sampled[-1] is history[-1]


def test_empty_history_returns_exact_black_frame_count():
    sampled = sample_episode_frames([], 8, width=6, height=4)
    assert len(sampled) == 8
    assert all(item.shape == (4, 6, 3) and not item.any() for item in sampled)
