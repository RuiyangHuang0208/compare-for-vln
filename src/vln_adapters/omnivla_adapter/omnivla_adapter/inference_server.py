from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time

from flask import Flask, jsonify, request
import numpy as np
from PIL import Image

from .contracts import (
    ACTION_SHAPE,
    CHECKPOINT_VARIANT,
    LANGUAGE_ONLY_MODALITY_ID,
    RESUME_STEP,
    checkpoint_status,
    validate_language_only_contract,
)
from .official_inference import OfficialOmniVLA, StubOmniVLA


def parse_args():
    parser = argparse.ArgumentParser(description="Independent full OmniVLA inference service")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5805)
    parser.add_argument("--stub", action="store_true", help="Use deterministic trajectory without loading OmniVLA")
    return parser.parse_args()


def create_app(runtime):
    app = Flask(__name__)
    lock = threading.Lock()
    is_stub = runtime.metadata["variant"] == "stub"
    state = {
        "episode_id": None,
        "generation": None,
        "request_count": 0,
        "language_only_leakage_verified": is_stub,
        "language_only_leakage_max_abs_difference": 0.0 if is_stub else None,
    }

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ready",
                **runtime.metadata,
                "language_only_leakage_verified": state["language_only_leakage_verified"],
                "language_only_leakage_max_abs_difference": state[
                    "language_only_leakage_max_abs_difference"
                ],
            }
        )

    @app.post("/reset")
    def reset():
        metadata = request.get_json(force=True)
        with lock:
            state["episode_id"] = str(metadata["episode_id"])
            state["generation"] = int(metadata["generation"])
        return jsonify(
            {"status": "reset", "episode_id": state["episode_id"], "generation": state["generation"]}
        )

    @app.post("/step")
    def step():
        started = time.perf_counter()
        try:
            metadata = json.loads(request.form["json"])
            validate_language_only_contract(str(metadata["goal_profile"]), int(metadata["modality_id"]))
            if not str(metadata.get("instruction", "")).strip():
                raise ValueError("instruction must be non-empty")
            image = np.asarray(Image.open(request.files["image"].stream).convert("RGB"), dtype=np.uint8)
            with lock:
                # Validate identity while holding the same lock used by
                # /reset and model inference.  Without this, a reset could
                # occur between the check and inference, allowing a late
                # request from the previous episode to run against the new
                # server state.
                if str(metadata["episode_id"]) != state["episode_id"]:
                    return jsonify(
                        {
                            "error": "stale episode request",
                            "active_episode_id": state["episode_id"],
                            "active_generation": state["generation"],
                        }
                    ), 409
                if int(metadata["generation"]) != state["generation"]:
                    return jsonify(
                        {
                            "error": "stale generation request",
                            "active_episode_id": state["episode_id"],
                            "active_generation": state["generation"],
                        }
                    ), 409
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                except ImportError:
                    torch = None
                if not state["language_only_leakage_verified"]:
                    passed, difference, raw, model_latency = runtime.check_language_only_leakage(
                        image, str(metadata["instruction"])
                    )
                    state["language_only_leakage_max_abs_difference"] = difference
                    if not passed:
                        raise RuntimeError(
                            f"language-only modality leakage check failed: max_abs_difference={difference}"
                        )
                    state["language_only_leakage_verified"] = True
                else:
                    raw, model_latency = runtime.infer(image, str(metadata["instruction"]))
                peak = (
                    int(torch.cuda.max_memory_allocated())
                    if torch is not None and torch.cuda.is_available()
                    else 0
                )
                state["request_count"] += 1
            raw = np.asarray(raw, dtype=np.float32)
            if raw.shape != ACTION_SHAPE or not np.isfinite(raw).all():
                raise ValueError(f"invalid OmniVLA trajectory shape/values: {raw.shape}")
            return jsonify(
                {
                    "request_id": str(metadata["request_id"]),
                    "episode_id": state["episode_id"],
                    "generation": state["generation"],
                    "variant": runtime.metadata["variant"],
                    "resume_step": runtime.metadata["resume_step"],
                    "modality_id": LANGUAGE_ONLY_MODALITY_ID,
                    "raw_trajectory_shape": list(raw.shape),
                    "raw_trajectory_8x4": raw.tolist(),
                    "inference_latency": float(model_latency),
                    "service_latency": time.perf_counter() - started,
                    "peak_gpu_memory_bytes": peak,
                    "language_only_leakage_verified": state["language_only_leakage_verified"],
                    "language_only_leakage_max_abs_difference": state[
                        "language_only_leakage_max_abs_difference"
                    ],
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
            app.logger.exception("OmniVLA inference failed")
            return jsonify({"error": f"{type(error).__name__}: {error}"}), 500

    return app


def main():
    args = parse_args()
    repository = Path(args.repository).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not (repository / "inference" / "run_omnivla.py").is_file():
        raise FileNotFoundError(f"official OmniVLA source is missing: {repository}")
    if args.stub:
        runtime = StubOmniVLA()
    else:
        status = checkpoint_status(checkpoint)
        if not status["complete"]:
            raise FileNotFoundError(
                f"incomplete {CHECKPOINT_VARIANT} step {RESUME_STEP} checkpoint; missing={status['missing']}"
            )
        runtime = OfficialOmniVLA(repository, checkpoint, args.device)
    print(
        f"[OMNIVLA SERVER] Ready at http://{args.host}:{args.port}; "
        f"implementation={runtime.metadata['implementation']}",
        flush=True,
    )
    try:
        create_app(runtime).run(host=args.host, port=args.port, threaded=False, use_reloader=False)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
