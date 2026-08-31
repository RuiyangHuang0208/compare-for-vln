#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time

from flask import Flask, jsonify, request
from PIL import Image
import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]


def parse_args():
    parser = argparse.ArgumentParser(description="NaVILA deterministic text-action inference service")
    parser.add_argument("--repository", default=str(WORKSPACE_ROOT / "third_party" / "NaVILA"))
    parser.add_argument("--checkpoint", default=str(WORKSPACE_ROOT / "checkpoints" / "vln" / "navila"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5803)
    return parser.parse_args()


def load_official_model(repository, checkpoint, device, dtype):
    import sys

    sys.path.insert(0, str(Path(repository).resolve()))
    from llava.model.builder import load_pretrained_model

    checkpoint = str(Path(checkpoint).resolve())
    model_name = os.path.basename(os.path.normpath(checkpoint))
    tokenizer, model, image_processor, _context_len = load_pretrained_model(checkpoint, model_name)
    model = model.to(device).eval()
    num_frames = int(model.config.num_video_frames)
    if num_frames <= 0:
        raise ValueError(f"checkpoint returned invalid num_video_frames={num_frames}")
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype)
    if torch_dtype is None:
        raise ValueError(f"unsupported dtype={dtype!r}; expected float16 or bfloat16")
    return tokenizer, model, image_processor, num_frames, torch_dtype


def create_app(tokenizer, model, image_processor, num_frames, torch_dtype):
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token

    app = Flask(__name__)
    inference_lock = threading.Lock()
    state = {"requests": 0, "started": time.time()}

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ready",
                "implementation": "llava.model.builder.load_pretrained_model",
                "num_video_frames": num_frames,
                "requests": state["requests"],
                "uptime_s": time.time() - state["started"],
            }
        )

    @app.post("/reset")
    def reset():
        return jsonify({"status": "reset"})

    @app.post("/step")
    def step():
        if not inference_lock.acquire(blocking=False):
            return jsonify({"error": "NaVILA inference service is busy"}), 429
        started = time.perf_counter()
        request_id = ""
        episode_id = ""
        try:
            metadata = json.loads(request.form["json"])
            instruction = str(metadata["instruction"]).strip()
            request_id = str(metadata["request_id"])
            episode_id = str(metadata["episode_id"])
            if not instruction or not request_id or not episode_id:
                raise ValueError("instruction, request_id, and episode_id must be non-empty")
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            ordered = sorted(
                ((key, value) for key, value in request.files.items() if key.startswith("image_")),
                key=lambda item: int(item[0].split("_")[1]),
            )
            if len(ordered) != num_frames:
                raise ValueError(f"checkpoint requires exactly {num_frames} RGB frames, got {len(ordered)}")
            images = [Image.open(storage.stream).convert("RGB") for _, storage in ordered]
            interleaved_images = "<image>\n" * (len(images) - 1)
            question = (
                "Imagine you are a robot programmed for navigation tasks. You have been given a video "
                f"of historical observations {interleaved_images}, and current observation <image>\n. "
                f'Your assigned task is: "{instruction}" Analyze this series of images to decide your next action, '
                "which could be turning left or right by a specific degree, moving forward a certain distance, "
                "or stop if the task is completed."
            )
            conv = conv_templates["llama_3"].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            images_tensor = process_images(images, image_processor, model.config).to(
                model.device, dtype=torch_dtype
            )
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=32,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    pad_token_id=tokenizer.eos_token_id,
                )
            raw_action = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if stop_str and raw_action.endswith(stop_str):
                raw_action = raw_action[: -len(stop_str)].strip()
            latency = time.perf_counter() - started
            state["requests"] += 1
            peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            return jsonify(
                {
                    "request_id": request_id,
                    "episode_id": episode_id,
                    "raw_action": raw_action,
                    "inference_s": latency,
                    "peak_memory_bytes": int(peak_memory),
                }
            )
        except torch.cuda.OutOfMemoryError as error:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            app.logger.exception("NaVILA CUDA OOM")
            return jsonify(
                {
                    "request_id": request_id,
                    "episode_id": episode_id,
                    "error": f"CUDAOutOfMemoryError: {error}",
                    "inference_s": time.perf_counter() - started,
                }
            ), 507
        except Exception as error:
            app.logger.exception("NaVILA inference failed")
            return jsonify(
                {
                    "request_id": request_id,
                    "episode_id": episode_id,
                    "error": f"{type(error).__name__}: {error}",
                    "inference_s": time.perf_counter() - started,
                }
            ), 500
        finally:
            inference_lock.release()

    return app


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    with open(checkpoint / "config.json", encoding="utf-8") as stream:
        checkpoint_config = json.load(stream)
    print(
        f"[NAVILA SERVER] Loading {checkpoint} model_type={checkpoint_config.get('model_type')} "
        f"num_video_frames={checkpoint_config.get('num_video_frames')} on {args.device}",
        flush=True,
    )
    tokenizer, model, image_processor, num_frames, torch_dtype = load_official_model(
        args.repository, checkpoint, args.device, args.dtype
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print(f"[NAVILA SERVER] Ready at http://{args.host}:{args.port}; RGB frames={num_frames}", flush=True)
    create_app(tokenizer, model, image_processor, num_frames, torch_dtype).run(
        host=args.host, port=args.port, threaded=True, use_reloader=False
    )


if __name__ == "__main__":
    main()
