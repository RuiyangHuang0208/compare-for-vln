from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import sys
import time
from typing import Optional

import numpy as np
from PIL import Image

from .contracts import EXPECTED_VECTOR_LENGTH, OFFICIAL_FIELDS


SYSTEM_PROMPT = (
    "You are a Unitree Go2 robot dog, given an instruction and observation, output exactly one list "
    "[x_vel_cmd, y_vel_cmd, yaw_vel_cmd, body_height_cmd, step_frequency_cmd, gait1, gait2, gait3, "
    "footswing_height_cmd, pitch_cmd, roll_cmd, stance_width_cmd] where all values are floats. "
    "Please provide your reasoning in <think></think> tags and your final answer in <answer></answer> "
    "tags with exactly 12 numerical values."
)


def load_official_module(repository: Path):
    """Load upstream inference.py while containing its missing Optional import."""
    inference_path = repository / "inference.py"
    spec = importlib.util.spec_from_file_location("mobilevla_r1_official_inference", inference_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official MobileVLA-R1 inference module: {inference_path}")
    module = importlib.util.module_from_spec(spec)
    sentinel = object()
    previous = getattr(builtins, "Optional", sentinel)
    builtins.Optional = Optional
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is sentinel:
            del builtins.Optional
        else:
            builtins.Optional = previous
    return module


def strip_prompt_echo(output: str, instruction: str) -> str:
    """Remove the user prompt echoed by the upstream full-sequence decoder."""
    marker = f"{SYSTEM_PROMPT}\n{instruction}"
    marker_index = output.find(marker)
    if marker_index >= 0:
        output = output[marker_index + len(marker) :]
    return output.strip()


class StubMobileVLAR1:
    metadata = {
        "variant": "stub",
        "implementation": "deterministic safety test; not MobileVLA-R1 inference",
        "history_frames": 8,
        "depth_frames": 1,
        "pointcloud_points": 2048,
        "expected_vector_length": EXPECTED_VECTOR_LENGTH,
        "official_fields": list(OFFICIAL_FIELDS),
    }

    def infer(self, _rgb, _depth, _pointcloud, instruction, _generation):
        vector = [0.2, 0.0, 0.0] + [0.0] * 9
        response = f"<think>Stub response for interface testing: {instruction}</think>\n<answer>{vector}</answer>"
        return response, 0.0

    def close(self):
        return None

    def reset(self):
        return None


class OfficialMobileVLAR1:
    def __init__(self, repository, model_path, lora_path, device, use_flash_attention=True):
        repository = Path(repository).resolve()
        sys.path.insert(0, str(repository))
        module = load_official_module(repository)
        self.runtime = module.NaVILAImageInference(
            model_path=str(model_path),
            lora_path=str(lora_path) if lora_path else None,
            device=device,
            use_flash_attn=use_flash_attention,
            depth_scale=1000.0,
            pointcloud_points=2048,
        )
        self.metadata = {
            "variant": "official",
            "implementation": "AIGeeksGroup/MobileVLA-R1 inference.py::NaVILAImageInference",
            "history_frames": 8,
            "depth_frames": 1,
            "pointcloud_points": 2048,
            "expected_vector_length": EXPECTED_VECTOR_LENGTH,
            "official_fields": list(OFFICIAL_FIELDS),
            "conversation_mode": self.runtime.conv_mode,
        }

    def infer(self, rgb, depth, pointcloud, instruction, generation):
        import torch

        torch.manual_seed(int(generation))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(generation))
        rgb_tensor = torch.stack(
            [self.runtime.load_image_from_pil(Image.fromarray(frame, mode="RGB")).squeeze(0) for frame in rgb]
        )
        depth_tensor = torch.from_numpy(np.asarray(depth, dtype=np.float32))[None, None]
        point_tensor = torch.from_numpy(np.asarray(pointcloud, dtype=np.float32))[None]
        started = time.perf_counter()
        response = self.runtime.generate_response(
            image_input=rgb_tensor,
            question=f"{SYSTEM_PROMPT}\n{instruction}",
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            depth_input=depth_tensor,
            point_cloud=point_tensor,
        )
        return strip_prompt_echo(response, instruction), time.perf_counter() - started

    def close(self):
        import torch

        del self.runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset(self):
        # generate() uses per-call KV cache; no episode history is retained by the model wrapper.
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
