#!/usr/bin/env python3
"""Simulation-only HTTP inference service for InternVLA-N1-DualVLN."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
INTERNNAV_ROOT = WORKSPACE_ROOT / "third_party" / "InternNav"
os.environ.setdefault("INTERNNAV_MINIMAL_IMPORT", "1")

import sys

sys.path.insert(0, str(INTERNNAV_ROOT))

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(WORKSPACE_ROOT / "checkpoints" / "dualvln"))
    parser.add_argument("--aux-checkpoint-root", default=str(WORKSPACE_ROOT / "checkpoints" / "dualvln"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5801)
    parser.add_argument("--resize-w", type=int, default=384)
    parser.add_argument("--resize-h", type=int, default=384)
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--plan-step-gap", type=int, default=4)
    parser.add_argument("--attention-implementation", default="sdpa")
    return parser.parse_args()


def create_app(agent):
    app = Flask(__name__)
    lock = threading.Lock()
    state = {"requests": 0, "started": time.time()}

    @app.get("/health")
    def health():
        return jsonify({"status": "ready", "requests": state["requests"], "uptime_s": time.time() - state["started"]})

    @app.post("/reset")
    def reset():
        with lock:
            agent.reset()
        return jsonify({"status": "reset"})

    @app.post("/step")
    def step():
        started = time.time()
        try:
            metadata = json.loads(request.form["json"])
            instruction = str(metadata["instruction"]).strip()
            if not instruction:
                raise ValueError("instruction is empty")
            rgb = np.asarray(Image.open(request.files["image"].stream).convert("RGB"))
            depth = np.asarray(Image.open(request.files["depth"].stream), dtype=np.float32) / 10000.0
            if rgb.shape[:2] != depth.shape:
                raise ValueError(f"RGB/depth shapes are not aligned: {rgb.shape} vs {depth.shape}")
            intrinsic = np.asarray(metadata["intrinsics"], dtype=np.float32).reshape(3, 3)
            capture_pose = np.asarray(metadata["capture_pose"], dtype=np.float32)
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:2, 3] = capture_pose[:2]
            c, s = np.cos(capture_pose[2]), np.sin(capture_pose[2])
            camera_pose[:2, :2] = ((c, -s), (s, c))

            with lock:
                if metadata.get("reset", False):
                    agent.reset()
                with torch.inference_mode():
                    output = agent.step(rgb, depth, camera_pose, instruction, intrinsic=intrinsic, look_down=False)
                    if output.output_action == [5]:
                        output = agent.step(rgb, depth, camera_pose, instruction, intrinsic=intrinsic, look_down=True)
                state["requests"] += 1

            payload = {"inference_s": time.time() - started}
            if output.output_action is not None:
                payload["discrete_action"] = list(output.output_action)
                payload["stop"] = output.output_action == [0]
            elif output.output_trajectory is not None:
                payload["trajectory"] = np.asarray(output.output_trajectory).tolist()
                if output.output_pixel is not None:
                    payload["pixel_goal"] = list(output.output_pixel)
            else:
                raise RuntimeError("DualVLN returned neither an action nor a trajectory")
            return jsonify(payload)
        except Exception as exc:
            app.logger.exception("DualVLN inference failed")
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    return app


def main():
    args = parse_args()
    os.environ["INTERNNAV_AUX_CHECKPOINT_ROOT"] = str(Path(args.aux_checkpoint_root).resolve())
    model_args = argparse.Namespace(
        device=args.device,
        model_path=str(Path(args.model_path).resolve()),
        resize_w=args.resize_w,
        resize_h=args.resize_h,
        num_history=args.num_history,
        plan_step_gap=args.plan_step_gap,
        attn_implementation=args.attention_implementation,
    )
    print(f"[DUALVLN SERVER] Loading {model_args.model_path} on {args.device} ({model_args.attn_implementation})", flush=True)
    agent = InternVLAN1AsyncAgent(model_args)
    agent.reset()
    print(f"[DUALVLN SERVER] Ready at http://{args.host}:{args.port}", flush=True)
    create_app(agent).run(host=args.host, port=args.port, threaded=False, use_reloader=False)


if __name__ == "__main__":
    main()
