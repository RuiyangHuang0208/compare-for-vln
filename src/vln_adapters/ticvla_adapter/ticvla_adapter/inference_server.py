#!/usr/bin/env python3
"""HTTP model process using only ticvla.models.ticvla.TICVLA."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

from flask import Flask, jsonify, request
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5802)
    parser.add_argument("--action-horizon", type=int, default=30)
    return parser.parse_args()


def load_formal_model(args):
    import sys

    repository = Path(args.repository).resolve()
    dynanav = repository / "DynaNav"
    if not (dynanav / "ticvla.py").is_file():
        raise FileNotFoundError(f"Not a TIC-VLA repository: {repository}")
    for value, name in ((args.base_model, "base model"), (args.checkpoint, "checkpoint")):
        if not Path(value).exists():
            raise FileNotFoundError(f"Missing TIC-VLA {name}: {value}")
    # The public benchmark loads DynaNav/ticvla.py, whose predict_async() keeps
    # the latest VLM model state while decoding fresh actions from every image.
    # DynaNav targets Isaac Sim 5.0's Transformers stack. The current pinned
    # InternVL implementation still expects the older torch_dtype keyword.
    import transformers
    from transformers.cache_utils import DynamicCache
    from types import SimpleNamespace

    original_from_pretrained = transformers.AutoModel.from_pretrained

    def compatible_from_pretrained(*model_args, **model_kwargs):
        if "dtype" in model_kwargs and "torch_dtype" not in model_kwargs:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        return original_from_pretrained(*model_args, **model_kwargs)

    transformers.AutoModel.from_pretrained = compatible_from_pretrained
    if not hasattr(DynamicCache, "layers"):
        DynamicCache.layers = property(
            lambda cache: [
                SimpleNamespace(keys=keys, values=values)
                for keys, values in cache.to_legacy_cache()
            ]
        )
    sys.path.insert(0, str(dynanav))
    from ticvla import TICVLA

    model = TICVLA(
        model_path=str(Path(args.base_model).resolve()),
        device=args.device,
        num_action_chunks=args.action_horizon,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    remapped = {
        (key[len("model.") :] if key.startswith("model.") else key): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(remapped, strict=True)
    print(
        f"[TICVLA SERVER] DynaNav checkpoint loaded strictly: tensors={len(remapped)}",
        flush=True,
    )
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    model = model.to(torch.device(args.device), dtype=dtype).eval()
    return model


def create_app(model, horizon):
    app = Flask(__name__)
    request_dir = tempfile.mkdtemp(prefix="ticvla_server_")
    state = {
        "requests": 0,
        "action_started": None,
        "action_latency": 0.0,
        "generation_starts": [],
    }

    def action_pre_hook(_module, _args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        state["action_started"] = time.perf_counter()

    def action_hook(_module, _args, _output):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        state["action_latency"] = time.perf_counter() - state["action_started"]

    model.action_expert.register_forward_pre_hook(action_pre_hook)
    model.action_expert.register_forward_hook(action_hook)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ready",
                "implementation": "DynaNav/ticvla.py::TICVLA.predict_async",
                "horizon": horizon,
            }
        )

    @app.post("/reset")
    def reset():
        model.reset_episode_state()
        state["generation_starts"].clear()
        state["requests"] = 0
        shutil.rmtree(request_dir, ignore_errors=True)
        os.makedirs(request_dir, exist_ok=True)
        return jsonify({"status": "reset", "implementation": "DynaNav predict_async"})

    @app.post("/step")
    def step():
        started = time.perf_counter()
        try:
            metadata = json.loads(request.form["json"])
            ordered = sorted(
                ((key, value) for key, value in request.files.items() if key.startswith("image_")),
                key=lambda item: int(item[0].split("_")[1]),
            )
            if len(ordered) < 1:
                raise ValueError("TIC-VLA requires at least one current image")
            request_id = int(state["requests"])
            paths = []
            for index, (_, storage) in enumerate(ordered):
                path = os.path.join(request_dir, f"request_{request_id:06d}_frame_{index:03d}.jpg")
                storage.save(path)
                paths.append(path)

            pose = np.asarray(metadata["current_pose"], dtype=np.float64)
            velocity = np.asarray(metadata["velocity"], dtype=np.float64)
            if pose.shape != (3,) or velocity.shape != (3,) or not np.isfinite(pose).all() or not np.isfinite(velocity).all():
                raise ValueError("current_pose and velocity must be finite [x,y,yaw]")
            current_step = int(metadata["sim_step"])
            dx = dy = time_delay = 0.0
            if state["generation_starts"]:
                reference = state["generation_starts"][0]
                delta = pose[:2] - reference["pose"][:2]
                c, s = math.cos(reference["pose"][2]), math.sin(reference["pose"][2])
                dx = c * delta[0] + s * delta[1]
                dy = -s * delta[0] + c * delta[1]
                time_delay = max(0.0, (current_step - reference["step"]) / 30.0)
            robot_state = torch.tensor(
                [velocity[0], velocity[1], 0.0, velocity[2], dx, dy], dtype=torch.float32
            )
            half_yaw = 0.5 * pose[2]
            robot_pose = {
                "position": [float(pose[0]), float(pose[1]), 0.0],
                "quaternion": [float(math.cos(half_yaw)), 0.0, 0.0, float(math.sin(half_yaw))],
            }
            with torch.inference_mode():
                response_text, waypoints, generation_step, cache_ready, generation_pose = model.predict_async(
                    image_paths=paths,
                    delayed_image_paths=paths,
                    instruction=str(metadata["instruction"]),
                    robot_state=robot_state,
                    current_step=current_step,
                    current_robot_pose=robot_pose,
                    time_delay=time_delay,
                    previous_waypoints_text=str(metadata.get("previous_waypoints_text", "")),
                    # Match the public DynaNav high-level prompt. B2-W remains
                    # the separate SRU-ONNX executor downstream.
                    robot_type="wheeled robot",
                )
            if generation_step is not None and generation_pose is not None:
                generation_position = generation_pose.get("position")
                generation_quaternion = generation_pose.get("quaternion")
                if generation_position is not None and generation_quaternion is not None:
                    quaternion = np.asarray(generation_quaternion, dtype=np.float64)
                    yaw = math.atan2(
                        2.0 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
                        1.0 - 2.0 * (quaternion[2] ** 2 + quaternion[3] ** 2),
                    )
                    state["generation_starts"].append(
                        {
                            "step": int(generation_step),
                            "pose": np.asarray(
                                [generation_position[0], generation_position[1], yaw], dtype=np.float64
                            ),
                        }
                    )
                    del state["generation_starts"][:-2]
            values = waypoints.detach().float().cpu().numpy()
            if values.shape != (1, horizon, 2) or not np.isfinite(values).all():
                raise ValueError(f"DynaNav TIC-VLA returned invalid shape {values.shape}")
            total = time.perf_counter() - started
            action_latency = float(state["action_latency"])
            state["requests"] += 1
            return jsonify(
                {
                    "waypoints": values[0].tolist(),
                    "total_latency": total,
                    "vlm_latency": max(0.0, total - action_latency),
                    "action_latency": action_latency,
                    "kv_cache_available": bool(cache_ready),
                    "response_text": response_text or "",
                }
            )
        except Exception as error:
            app.logger.exception("TIC-VLA inference failed")
            return jsonify({"error": f"{type(error).__name__}: {error}"}), 500

    return app


def main():
    args = parse_args()
    model = load_formal_model(args)
    print(
        f"[TICVLA SERVER] Ready at http://{args.host}:{args.port}; "
        "implementation=DynaNav/ticvla.py::TICVLA.predict_async; output=30x2 local XY (no cumsum)",
        flush=True,
    )
    create_app(model, args.action_horizon).run(host=args.host, port=args.port, threaded=False, use_reloader=False)


if __name__ == "__main__":
    main()
