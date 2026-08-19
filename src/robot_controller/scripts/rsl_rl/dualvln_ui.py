"""Isaac Sim UI for observable DualVLN inference and control state."""

from __future__ import annotations

import math

import omni.ui as ui


class DualVlnStatusWindow:
    """Show model inputs, outputs, and controller telemetry without exposing hidden reasoning."""

    def __init__(self, instruction: str, desired_speed: float):
        self.window = ui.Window(
            "DualVLN Monitor",
            width=480,
            height=360,
            visible=True,
            dock_preference=ui.DockPreference.RIGHT_TOP,
        )
        with self.window.frame:
            with ui.VStack(spacing=7, style={"margin": 10}):
                ui.Label("DualVLN simulation monitor", height=24, style={"font_size": 18})
                ui.Separator(height=2)
                ui.Label("Instruction", height=20, style={"font_size": 14})
                self._instruction = ui.Label(
                    instruction or "Waiting for instruction...",
                    height=54,
                    word_wrap=True,
                )
                ui.Separator(height=2)
                self._state = self._make_row("State")
                self._model_stage = self._make_row("Model stage")
                self._model_output = self._make_row("Observable output", height=42, word_wrap=True)
                self._frame = self._make_row("Frame")
                self._inference = self._make_row("Inference")
                self._speed = self._make_row("Desired speed")
                self._command = self._make_row("Command [vx, vy, wz]")
                self._remaining = self._make_row("Path remaining")

        self.set_state("READY" if instruction else "WAITING", "No inference request")
        self.set_result("-", "No model output", -1, math.nan)
        self.set_telemetry(desired_speed, (0.0, 0.0, 0.0), math.inf)

    @staticmethod
    def _make_row(name: str, height: int = 22, word_wrap: bool = False):
        with ui.HStack(height=height):
            ui.Label(name, width=150)
            return ui.Label("-", word_wrap=word_wrap)

    def set_instruction(self, instruction: str):
        self._instruction.text = instruction or "Waiting for instruction..."

    def set_state(self, state: str, detail: str = ""):
        self._state.text = f"{state}: {detail}" if detail else state

    def set_result(
        self,
        stage: str,
        output: str,
        frame_id: int,
        inference_s: float,
    ):
        self._model_stage.text = stage
        self._model_output.text = output
        self._frame.text = "-" if frame_id < 0 else str(frame_id)
        self._inference.text = "-" if not math.isfinite(inference_s) else f"{inference_s:.2f} s"

    def set_telemetry(self, desired_speed: float, command, remaining: float):
        self._speed.text = f"{desired_speed:.2f} m/s"
        self._command.text = f"[{command[0]:.2f}, {command[1]:.2f}, {command[2]:.2f}]"
        self._remaining.text = "-" if not math.isfinite(remaining) else f"{remaining:.2f} m"

    def close(self):
        if self.window is not None:
            self.window.visible = False
            self.window.destroy()
            self.window = None
