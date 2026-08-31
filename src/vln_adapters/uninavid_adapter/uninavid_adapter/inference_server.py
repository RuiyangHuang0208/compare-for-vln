#!/usr/bin/env python3
"""Persistent Uni-NaVid HTTP inference process using the official model implementation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time

from flask import Flask, jsonify, request
import numpy as np
from PIL import Image
import torch


PROMPT = (
    "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical "
    "observations and an image of the current observation <image>. Your assigned task is: '{}'. Analyze this "
    "series of images to determine your next four actions. The predicted action should be one of the following: "
    "forward, left, right, or stop."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eva-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5804)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser.parse_args()


def resolved_checkpoint(checkpoint: Path, repository: Path, eva_checkpoint: Path):
    config_path = checkpoint / "config.json"
    processor = repository / "uninavid" / "processor" / "clip-patch14-224"
    for value, label in ((config_path, "config.json"), (processor, "image processor"), (eva_checkpoint, "EVA checkpoint")):
        if not value.exists():
            raise FileNotFoundError(f"Missing Uni-NaVid {label}: {value}")
    runtime = Path(tempfile.mkdtemp(prefix="uninavid_resolved_checkpoint_"))
    for source in checkpoint.iterdir():
        if source.name != "config.json":
            os.symlink(source.resolve(), runtime / source.name, target_is_directory=source.is_dir())
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # The released checkpoint records the base LLaMA type, while Uni-NaVid
    # registers its multimodal implementation as ``llava``. Correct this only
    # in the temporary runtime config; upstream files stay untouched.
    config["model_type"] = "llava"
    config["architectures"] = ["LlavaLlamaAttForCausalLM"]
    config["mm_vision_tower"] = str(eva_checkpoint.resolve())
    config["image_processor"] = str(processor.resolve())
    (runtime / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return runtime


class OfficialUniNaVid:
    def __init__(self, args):
        repository = Path(args.repository).resolve()
        checkpoint = Path(args.checkpoint).resolve()
        if not (repository / "uninavid" / "model" / "builder.py").is_file():
            raise FileNotFoundError(f"Not an official Uni-NaVid repository: {repository}")
        self.runtime_checkpoint = resolved_checkpoint(
            checkpoint, repository, Path(args.eva_checkpoint).resolve()
        )
        sys.path.insert(0, str(repository))
        from uninavid.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from uninavid.conversation import SeparatorStyle, conv_templates
        from uninavid.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
        from uninavid.model.builder import load_pretrained_model

        self.constants = (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX)
        self.SeparatorStyle = SeparatorStyle
        self.conv_templates = conv_templates
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.tokenizer_image_token = tokenizer_image_token
        model_path = str(self.runtime_checkpoint)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, None, model_name, device=args.device
        )
        self.device = torch.device(args.device)
        self.temperature = float(args.temperature)
        self.max_new_tokens = int(args.max_new_tokens)
        self.model.config.run_type = "eval"
        self.reset_cache()

    def close(self):
        shutil.rmtree(self.runtime_checkpoint, ignore_errors=True)

    def reset_cache(self):
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0

    def infer(self, frames, instruction):
        DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX = self.constants
        prompt_text = PROMPT.format(instruction)
        question = prompt_text.replace(DEFAULT_IMAGE_TOKEN, "").replace("\n", "")
        qs = prompt_text
        special = {
            name: self.tokenizer(token, return_tensors="pt").input_ids[0][1:].to(self.device)
            for name, token in {
                "image_start": "<image_special>",
                "image_end": "</image_special>",
                "video_start": "<video_special>",
                "video_end": "</video_special>",
                "navigation": "[Navigation]",
                "separator": "<image_sep>",
            }.items()
        }
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs.replace("<image>", "")
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs.replace("<image>", "")
        conv = self.conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        token_prompt = self.tokenizer_image_token(
            conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).to(self.device)
        output_parts = []
        indices = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]
        while indices.numel() > 0:
            index = indices[0]
            output_parts.extend(
                [
                    token_prompt[:index], special["video_start"], special["separator"],
                    token_prompt[index:index + 1], special["video_end"], special["image_start"],
                    special["image_end"], special["navigation"],
                ]
            )
            token_prompt = token_prompt[index + 1:]
            indices = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]
        if token_prompt.numel():
            output_parts.append(token_prompt)
        input_ids = torch.cat(output_parts).unsqueeze(0)
        stop_string = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        stopping = self.KeywordsStoppingCriteria([stop_string], self.tokenizer, input_ids)
        batch = np.asarray(frames)
        self.model.get_model().new_frames = len(frames)
        video = self.image_processor.preprocess(batch, return_tensors="pt")["pixel_values"].half().to(self.device)
        with torch.inference_mode():
            self.model.update_prompt([[question]])
            output_ids = self.model.generate(
                input_ids,
                images=[video],
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping],
            )
        text = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        if text.endswith(stop_string):
            text = text[:-len(stop_string)].strip()
        return text


def create_app(runtime):
    app = Flask(__name__)
    lock = threading.Lock()
    state = {"episode_id": None, "generation": None, "requests": 0}

    @app.get("/health")
    def health():
        return jsonify({"status": "ready", "implementation": "official Uni-NaVid", "online_cache": True})

    @app.post("/reset")
    def reset():
        metadata = request.get_json(force=True)
        with lock:
            runtime.reset_cache()
            state["episode_id"] = str(metadata["episode_id"])
            state["generation"] = int(metadata["generation"])
        return jsonify({"status": "reset", "episode_id": state["episode_id"], "generation": state["generation"]})

    @app.post("/step")
    def step():
        started = time.perf_counter()
        try:
            metadata = json.loads(request.form["json"])
            ordered = sorted(
                ((key, value) for key, value in request.files.items() if key.startswith("image_")),
                key=lambda item: int(item[0].split("_")[1]),
            )
            if not ordered:
                raise ValueError("Uni-NaVid step requires at least one new RGB frame")
            request_episode = str(metadata["episode_id"])
            request_generation = int(metadata["generation"])
            if request_episode != state["episode_id"] or request_generation != state["generation"]:
                # A delayed request can arrive after a reset.  Make this a
                # recoverable conflict so the adapter can reinitialize the
                # online cache instead of retrying the same stale request.
                return jsonify(
                    {
                        "error": "request episode/generation does not match initialized cache",
                        "episode_id": state["episode_id"],
                        "generation": state["generation"],
                    }
                ), 409
            frames = [np.asarray(Image.open(storage.stream).convert("RGB")) for _, storage in ordered]
            with lock:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                action = runtime.infer(frames, str(metadata["instruction"]))
                peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                state["requests"] += 1
            return jsonify(
                {
                    "request_id": str(metadata["request_id"]),
                    "episode_id": state["episode_id"],
                    "generation": state["generation"],
                    "raw_action": action,
                    "latency": time.perf_counter() - started,
                    "peak_gpu_memory_bytes": int(peak),
                    "new_frames": len(frames),
                }
            )
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            return jsonify({"error": f"CUDA OOM: {error}"}), 507
        except Exception as error:
            app.logger.exception("Uni-NaVid inference failed")
            return jsonify({"error": f"{type(error).__name__}: {error}"}), 500

    return app


def main():
    args = parse_args()
    runtime = OfficialUniNaVid(args)
    print(f"[UNINAVID SERVER] Ready at http://{args.host}:{args.port}; online RGB cache enabled", flush=True)
    try:
        create_app(runtime).run(host=args.host, port=args.port, threaded=False, use_reloader=False)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
