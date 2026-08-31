from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class StampedValue:
    stamp_s: float
    episode_id: str
    value: object


@dataclass(frozen=True)
class SensorBundle:
    rgb: StampedValue
    depth: StampedValue
    camera_info: StampedValue

    @property
    def stamp_s(self):
        return max(self.rgb.stamp_s, self.depth.stamp_s, self.camera_info.stamp_s)


class ApproximateSensorSynchronizer:
    def __init__(self, queue_size: int, slop_s: float, maximum_age_s: float):
        if queue_size <= 0 or slop_s < 0.0 or maximum_age_s <= 0.0:
            raise ValueError("invalid synchronization limits")
        self.queues = {name: deque(maxlen=queue_size) for name in ("rgb", "depth", "camera_info")}
        self.slop_s = float(slop_s)
        self.maximum_age_s = float(maximum_age_s)
        self.last_rejection_reason = None

    def clear(self):
        for values in self.queues.values():
            values.clear()
        self.last_rejection_reason = None

    def add(self, kind: str, value: StampedValue, now_s: float) -> SensorBundle | None:
        if kind not in self.queues:
            raise KeyError(kind)
        if now_s - value.stamp_s > self.maximum_age_s or value.stamp_s - now_s > self.slop_s:
            self.last_rejection_reason = "stale_or_future_sensor"
            return None
        self.last_rejection_reason = None
        self.queues[kind].append(value)
        return self._latest_bundle(now_s)

    def pending_failure(self, now_s: float) -> str | None:
        if self.last_rejection_reason:
            return self.last_rejection_reason
        populated = [queue[-1] for queue in self.queues.values() if queue]
        if not populated:
            return None
        oldest_arrival = min(item.stamp_s for item in populated)
        if now_s - oldest_arrival > self.slop_s:
            missing = [name for name, queue in self.queues.items() if not queue]
            if missing:
                return "missing_" + "_and_".join(missing)
        return None

    def _latest_bundle(self, now_s: float) -> SensorBundle | None:
        if any(not queue for queue in self.queues.values()):
            return None
        anchor = max(queue[-1].stamp_s for queue in self.queues.values())
        selected = {}
        for name, queue in self.queues.items():
            candidate = min(queue, key=lambda item: abs(item.stamp_s - anchor))
            if abs(candidate.stamp_s - anchor) > self.slop_s:
                self.last_rejection_reason = "sensor_time_mismatch"
                return None
            selected[name] = candidate
        episode_ids = {item.episode_id for item in selected.values()}
        if len(episode_ids) != 1 or now_s - anchor > self.maximum_age_s:
            self.last_rejection_reason = (
                "sensor_episode_mismatch" if len(episode_ids) != 1 else "stale_sensor_bundle"
            )
            return None
        bundle = SensorBundle(**selected)
        self.clear()
        return bundle


def sample_rgb_history(frames, count: int):
    """Match the official NavCoT loader: uniform history, then repeat the last frame."""
    if count <= 0 or not frames:
        raise ValueError("history requires a positive count and at least one frame")
    import numpy as np

    available = list(frames)
    while len(available) < count:
        available.append(available[-1])
    indices = np.linspace(0, len(available) - 1, num=count, dtype=int)
    return [available[index] for index in indices]
