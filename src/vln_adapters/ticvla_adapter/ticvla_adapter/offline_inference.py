#!/usr/bin/env python3
"""Run one real TIC-VLA prediction and save its 30x2 local trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from ticvla_adapter.inference_server import load_formal_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--instruction", default="Walk forward along the corridor and stop at the target.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--history-length", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    image = Path(args.image).resolve()
    if not image.is_file():
        raise FileNotFoundError(f"Offline RGB image not found: {image}")
    if args.history_length < 2:
        raise ValueError("history-length must include at least one delayed and one current frame")

    model_args = SimpleNamespace(
        repository=args.repository,
        base_model=args.base_model,
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        action_horizon=args.action_horizon,
    )
    started = time.perf_counter()
    model = load_formal_model(model_args)
    load_latency = time.perf_counter() - started
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        response, waypoints, prompt = model.predict(
            delayed_image_paths=[str(image)] * (args.history_length - 1),
            current_image_path=str(image),
            instruction=args.instruction,
            robot_state=torch.zeros(5, dtype=torch.float32),
            time_delay=0.0,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_latency = time.perf_counter() - started
    values = waypoints.detach().float().cpu().numpy()
    expected = (1, args.action_horizon, 2)
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(f"Expected finite {expected} trajectory, got {values.shape}")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "implementation": "ticvla.models.ticvla.TICVLA",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "base_model": str(Path(args.base_model).resolve()),
        "image": str(image),
        "instruction": args.instruction,
        "coordinate_frame": "base_link",
        "trajectory_semantics": "30 local (x, y) positions; no cumsum",
        "shape": list(values.shape),
        "finite": bool(np.isfinite(values).all()),
        "model_load_latency_seconds": load_latency,
        "inference_latency_seconds": inference_latency,
        "generated_response": str(response),
        "prompt": str(prompt),
        "waypoints": values[0].tolist(),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[TICVLA OFFLINE] PASS shape={values.shape} finite=true "
        f"latency={inference_latency:.3f}s output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
