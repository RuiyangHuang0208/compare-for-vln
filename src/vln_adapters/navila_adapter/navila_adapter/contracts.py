from __future__ import annotations

import numpy as np


def validate_rgb(image):
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise ValueError(f"RGB frame must be uint8 HxWx3, got shape={value.shape} dtype={value.dtype}")
    return np.ascontiguousarray(value)


def sample_episode_frames(images, num_frames, width=512, height=512):
    """Match NaVILA's official full-history sampling and 512x512 black padding."""
    if not isinstance(num_frames, int) or num_frames <= 0:
        raise ValueError("num_frames must be a positive integer")
    frames = [validate_rgb(image) for image in images]
    if frames and any(frame.shape[:2] != frames[-1].shape[:2] for frame in frames):
        raise ValueError("All NaVILA RGB frames must have the same shape")
    if not width or not height:
        raise ValueError("black padding width and height must be positive")
    frame_width, frame_height = int(width), int(height)
    if not frames:
        return [np.zeros((frame_height, frame_width, 3), dtype=np.uint8) for _ in range(num_frames)]
    if len(frames) < num_frames:
        # Upstream sample_and_pad_images() creates 512x512 PIL RGB frames even
        # when the live observation has another resolution. process_images()
        # subsequently applies the checkpoint image processor to every frame.
        padding = [
            np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            for _ in range(num_frames - len(frames))
        ]
        sampled = padding + frames
    elif num_frames == 1:
        sampled = [frames[-1]]
    else:
        indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
        sampled = [frames[index] for index in indices] + [frames[-1]]
    if len(sampled) != num_frames or sampled[-1] is not frames[-1]:
        raise RuntimeError("NaVILA sampler contract violation")
    return [np.ascontiguousarray(frame) for frame in sampled]
