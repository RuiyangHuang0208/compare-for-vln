#!/usr/bin/env python3
"""Convert the upstream DynaNav benchmark into the workspace episode schema."""

from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path

import yaml


def scene_name(scene_asset: str) -> str:
    value = scene_asset.lower()
    for name in ("hospital", "office", "outdoor", "warehouse"):
        if name in value:
            return name
    raise ValueError(f"Unsupported DynaNav scene asset: {scene_asset}")


def convert(source: dict) -> dict:
    seed = int(source.get("seed", 666))
    episodes = {}
    source_episodes = sorted(
        ((int(key.split("_", 1)[1]), value) for key, value in source.items() if key.startswith("episode_")),
        key=lambda item: item[0],
    )
    if [index for index, _ in source_episodes] != list(range(1, 86)):
        raise ValueError("Expected the official contiguous DynaNav episode_1..episode_85 set")
    for index, item in source_episodes:
        start = item["start"]
        goal = item["goal"]
        scene = scene_name(str(item["scene"]))
        episodes[f"dynanav_{index:03d}"] = {
            "suite": "dynanav_full",
            "source_episode": f"episode_{index}",
            "scene": scene,
            "instruction": str(item["instruction"]),
            "spawn": [float(start[0]), float(start[1]), math.radians(float(item.get("start_yaw", 0.0)))],
            "floor_z": float(start[2]) if len(start) > 2 else 0.0,
            "goal": [float(goal[0]), float(goal[1])],
            # The official benchmark overrides the global threshold on several
            # Outdoor episodes (for example 1, 2, or 4 m).  Preserve that
            # per-episode value instead of silently applying 1.5 m everywhere.
            "success_threshold": float(
                item.get("success_threshold", source.get("success_threshold", 1.5))
            ),
            "max_duration": float(item["timeout"]),
            "pedestrian_count": int(item.get("num_people", 0)),
            "seed": seed,
        }
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.source.open(encoding="utf-8") as stream:
        source = yaml.safe_load(stream)
    generated = convert(source)

    existing = {}
    if args.output.exists():
        with args.output.open(encoding="utf-8") as stream:
            existing = yaml.safe_load(stream) or {}
    custom = {
        key: value
        for key, value in existing.get("episodes", {}).items()
        if value.get("suite") != "dynanav_full" and not key.startswith("dynanav_")
    }
    payload = {
        "metadata": {
            "source": "third_party/TIC-VLA/DynaNav/configs/benchmark_full.yaml",
            "official_episode_count": 85,
            "yaw_unit": "radians",
        },
        "episodes": {**custom, **generated},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=False, width=120)
    counts = Counter(item["scene"] for item in generated.values())
    print(f"Wrote {len(generated)} official episodes to {args.output}: {dict(counts)}")


if __name__ == "__main__":
    main()
