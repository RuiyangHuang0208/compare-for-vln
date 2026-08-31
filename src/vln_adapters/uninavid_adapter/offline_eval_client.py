#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Run official Uni-NaVid images through the persistent service.")
    parser.add_argument("test_case")
    parser.add_argument("--server", default="http://127.0.0.1:5804")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.test_case).resolve()
    instruction = json.loads((root / "instruction.json").read_text(encoding="utf-8"))["instruction"]
    images = sorted((root / "images").glob("*.jpg"), key=lambda path: int(path.stem))
    if not images:
        raise FileNotFoundError(f"No JPG images in {root / 'images'}")
    episode_id = "offline_vln_1"
    generation = 1
    reset = requests.post(
        args.server.rstrip("/") + "/reset",
        json={"episode_id": episode_id, "generation": generation},
        timeout=30.0,
    )
    reset.raise_for_status()
    records = []
    for index, path in enumerate(images):
        request_id = f"offline-{index:03d}"
        metadata = {
            "episode_id": episode_id,
            "generation": generation,
            "request_id": request_id,
            "instruction": instruction,
        }
        started = time.perf_counter()
        with open(path, "rb") as stream:
            response = requests.post(
                args.server.rstrip("/") + "/step",
                data={"json": json.dumps(metadata)},
                files={"image_0": (path.name, stream, "image/jpeg")},
                timeout=120.0,
            )
        response.raise_for_status()
        result = response.json()
        result["client_latency"] = time.perf_counter() - started
        result["frame"] = path.name
        records.append(result)
        print(
            f"frame={path.name} raw_action={result['raw_action']!r} "
            f"latency={result['latency']:.3f}s peak={result['peak_gpu_memory_bytes'] / 2**30:.3f}GiB",
            flush=True,
        )
    document = {
        "instruction": instruction,
        "temperature": 0.5,
        "do_sample": True,
        "max_new_tokens": 1024,
        "records": records,
    }
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
