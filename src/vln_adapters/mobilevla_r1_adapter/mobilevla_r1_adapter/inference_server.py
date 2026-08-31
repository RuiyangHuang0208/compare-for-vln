from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import threading
import time

from flask import Flask, jsonify, request
import numpy as np

from .contracts import (
    DEPTH_FRAMES,
    EXPECTED_VECTOR_LENGTH,
    HISTORY_RGB_FRAMES,
    POINTCLOUD_POINTS,
    checkpoint_status,
)
from .official_inference import OfficialMobileVLAR1, StubMobileVLAR1


def parse_args():
    parser = argparse.ArgumentParser(description="Independent MobileVLA-R1 inference service")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default="")
    parser.add_argument("--lora", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5806)
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--no-flash-attention", action="store_true")
    return parser.parse_args()


def validate_bundle(bundle):
    rgb = np.asarray(bundle["rgb"], dtype=np.uint8)
    depth = np.asarray(bundle["depth"], dtype=np.float32)
    pointcloud = np.asarray(bundle["pointcloud"], dtype=np.float32)
    if rgb.ndim != 4 or rgb.shape[0] != HISTORY_RGB_FRAMES or rgb.shape[-1] != 3:
        raise ValueError(f"RGB history must be 8xHxWx3, got {rgb.shape}")
    if depth.ndim != 2 or not np.isfinite(depth).any():
        raise ValueError(f"depth must be a finite HxW map, got {depth.shape}")
    if pointcloud.shape != (POINTCLOUD_POINTS, 3) or not np.isfinite(pointcloud).all():
        raise ValueError(f"point cloud must be finite 2048x3, got {pointcloud.shape}")
    return rgb, depth, pointcloud


def create_app(runtime):
    app = Flask(__name__)
    lock = threading.Lock()
    state = {
        "episode_id": None,
        "generation": None,
        "request_count": 0,
        "last_sensor_stamp": None,
    }

    @app.get("/health")
    def health():
        return jsonify({"status": "ready", **runtime.metadata})

    @app.post("/reset")
    def reset():
        metadata = request.get_json(force=True)
        with lock:
            runtime.reset()
            state["episode_id"] = str(metadata["episode_id"])
            state["generation"] = int(metadata["generation"])
            state["last_sensor_stamp"] = None
        return jsonify({"status": "reset", "episode_id": state["episode_id"], "generation": state["generation"]})

    @app.post("/step")
    def step():
        service_started = time.perf_counter()
        try:
            metadata = json.loads(request.form["json"])
            with lock:
                if str(metadata["episode_id"]) != state["episode_id"]:
                    raise ValueError("request episode_id does not match active episode")
                if int(metadata["generation"]) != state["generation"]:
                    raise ValueError("request generation is stale")
                if not str(metadata.get("instruction", "")).strip():
                    raise ValueError("instruction is empty")
                sensor_stamp = float(metadata["sensor_stamp"])
                if not np.isfinite(sensor_stamp):
                    raise ValueError("sensor_stamp is not finite")
                if (
                    state["last_sensor_stamp"] is not None
                    and sensor_stamp <= state["last_sensor_stamp"]
                ):
                    raise ValueError("sensor bundle is stale or duplicated")
                archive = np.load(io.BytesIO(request.files["bundle"].read()), allow_pickle=False)
                rgb, depth, pointcloud = validate_bundle(archive)
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                except ImportError:
                    torch = None
                raw_response, inference_latency = runtime.infer(
                    rgb,
                    depth,
                    pointcloud,
                    str(metadata["instruction"]),
                    int(metadata["generation"]),
                )
                peak = int(torch.cuda.max_memory_allocated()) if torch is not None and torch.cuda.is_available() else 0
                state["request_count"] += 1
                state["last_sensor_stamp"] = sensor_stamp
            return jsonify(
                {
                    "request_id": str(metadata["request_id"]),
                    "episode_id": state["episode_id"],
                    "generation": state["generation"],
                    "raw_response": str(raw_response),
                    "inference_latency": float(inference_latency),
                    "service_latency": time.perf_counter() - service_started,
                    "peak_gpu_memory_bytes": peak,
                    "history_frames": HISTORY_RGB_FRAMES,
                    "depth_frames": DEPTH_FRAMES,
                    "pointcloud_points": POINTCLOUD_POINTS,
                }
            )
        except Exception as error:
            try:
                import torch

                if isinstance(error, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                    return jsonify({"error": f"CUDA OOM: {error}"}), 507
            except ImportError:
                pass
            app.logger.exception("MobileVLA-R1 inference failed")
            return jsonify({"error": f"{type(error).__name__}: {error}"}), 500

    return app


def main():
    args = parse_args()
    repository = Path(args.repository).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not (repository / "inference.py").is_file() or not (repository / "action_extractor.py").is_file():
        raise FileNotFoundError(f"official MobileVLA-R1 source is missing: {repository}")
    if args.stub:
        runtime = StubMobileVLAR1()
    else:
        status = checkpoint_status(checkpoint)
        if not status["complete"]:
            raise SystemExit(
                "checkpoint is not a complete RGB-D+Point MobileVLA-R1 model; "
                f"refusing RGB-only fallback: {status}"
            )
        model_path = Path(args.base_model).expanduser().resolve() if args.base_model else checkpoint
        lora_path = Path(args.lora).expanduser().resolve() if args.lora else None
        runtime = OfficialMobileVLAR1(
            repository,
            model_path,
            lora_path,
            args.device,
            use_flash_attention=not args.no_flash_attention,
        )
    print(f"[MOBILEVLA-R1 SERVER] Ready at http://{args.host}:{args.port}; {runtime.metadata}", flush=True)
    try:
        create_app(runtime).run(host=args.host, port=args.port, threaded=False, use_reloader=False)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
