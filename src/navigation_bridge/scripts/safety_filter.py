"""Apply simulation command limits and fail-closed stop semantics."""

from __future__ import annotations

import numpy as np


class SafetyFilter:
    def __init__(self, max_forward_speed=0.4, max_yaw_rate=0.4):
        self.max_forward_speed = float(max_forward_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self.emergency_stop = False

    def stop(self):
        self.emergency_stop = True

    def resume(self):
        self.emergency_stop = False

    def apply(self, command):
        value = np.asarray(command, dtype=np.float32)
        if self.emergency_stop or value.shape != (3,) or not np.isfinite(value).all():
            return np.zeros(3, dtype=np.float32)
        return np.array(
            (
                np.clip(value[0], 0.0, self.max_forward_speed),
                0.0,
                np.clip(value[2], -self.max_yaw_rate, self.max_yaw_rate),
            ),
            dtype=np.float32,
        )
