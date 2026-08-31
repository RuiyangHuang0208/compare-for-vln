from __future__ import annotations

import importlib
from pathlib import Path
import sys
import time
import types

import numpy as np
from PIL import Image

from .contracts import ACTION_SHAPE, CHECKPOINT_VARIANT, LANGUAGE_ONLY_MODALITY_ID, RESUME_STEP


class OfficialOmniVLA:
    """Thin wrapper around NHirose/OmniVLA's public full-model inference path."""

    def __init__(self, repository, checkpoint, device="cuda:0"):
        self.repository = Path(repository).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if str(self.repository) not in sys.path:
            sys.path.insert(0, str(self.repository))

        # These package initializers eagerly import training configuration,
        # TensorFlow/RLDS, and metrics. The public inference file only needs
        # concrete modules below these paths, so expose the official directories
        # as packages without executing their training-only initializers.
        package_paths = {
            "prismatic": self.repository / "prismatic",
            "prismatic.models": self.repository / "prismatic" / "models",
            "prismatic.training": self.repository / "prismatic" / "training",
            "prismatic.vla": self.repository / "prismatic" / "vla",
        }
        for package_name, package_path in package_paths.items():
            if package_name in sys.modules:
                continue
            package = types.ModuleType(package_name)
            package.__path__ = [str(package_path)]
            package.__package__ = package_name
            sys.modules[package_name] = package

        import torch

        if not torch.cuda.is_available() or not str(device).startswith("cuda"):
            raise RuntimeError("omnivla-original requires a CUDA device; CPU fallback is disabled")
        self.torch = torch
        self.module = importlib.import_module("inference.run_omnivla")
        self.module.pose_goal = False
        self.module.satellite = False
        self.module.image_goal = False
        self.module.lan_prompt = True

        cfg = self.module.InferenceConfig()
        cfg.resume = True
        cfg.vla_path = str(self.checkpoint)
        cfg.resume_step = RESUME_STEP
        cfg.use_l1_regression = True
        cfg.use_diffusion = False
        cfg.use_film = False
        cfg.num_images_in_input = 2
        # The public run_omnivla.py loads merged model shards directly and does not
        # load lora_adapter separately. Keep the field for parity without double-loading.
        cfg.use_lora = True
        (
            self.vla,
            self.action_head,
            self.pose_projector,
            self.device,
            self.num_patches,
            self.action_tokenizer,
            self.processor,
        ) = self.module.define_model(cfg)
        self.vla.eval()
        self.action_head.eval()
        self.pose_projector.eval()
        self.helper = self.module.Inference(
            save_dir=str(self.repository / "inference"),
            lan_inst_prompt="",
            goal_utm=(0.0, 0.0),
            goal_compass=0.0,
            goal_image_PIL=Image.new("RGB", (1, 1)),
            action_tokenizer=self.action_tokenizer,
            processor=self.processor,
        )

    @property
    def metadata(self):
        return {
            "implementation": "NHirose/OmniVLA inference/run_omnivla.py",
            "variant": CHECKPOINT_VARIANT,
            "resume_step": RESUME_STEP,
            "action_shape": list(ACTION_SHAPE),
            "modality_id": LANGUAGE_ONLY_MODALITY_ID,
            "lora_loading": "merged model shards; no separate adapter load",
            "dist_head_used": False,
        }

    def _batch(self, current_rgb, instruction, goal_image, goal_pose):
        numpy_state = np.random.get_state()
        try:
            # Official preprocessing inserts dummy action tokens. Fix their random
            # values so repeated language-only leakage checks are reproducible.
            np.random.seed(0)
            return self.helper.data_transformer_omnivla(
                Image.fromarray(np.asarray(current_rgb, dtype=np.uint8), mode="RGB"),
                str(instruction),
                goal_image,
                np.asarray(goal_pose, dtype=np.float32),
                prompt_builder=self.module.PurePromptBuilder,
                action_tokenizer=self.action_tokenizer,
                processor=self.processor,
            )
        finally:
            np.random.set_state(numpy_state)

    def infer_with_placeholders(self, current_rgb, instruction, goal_image, goal_pose):
        batch = self._batch(current_rgb, instruction, goal_image, goal_pose)
        torch = self.torch
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        started = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            actions, modality = self.helper.run_forward_pass(
                vla=self.vla,
                action_head=self.action_head,
                noisy_action_projector=None,
                pose_projector=self.pose_projector,
                batch=batch,
                action_tokenizer=self.action_tokenizer,
                device_id=self.device,
                use_l1_regression=True,
                use_diffusion=False,
                use_film=False,
                num_patches=self.num_patches,
                compute_diffusion_l1=False,
                num_diffusion_steps_train=None,
                mode="inference",
            )
        raw = actions.float().detach().cpu().numpy()
        if raw.shape != (1, *ACTION_SHAPE):
            raise ValueError(f"official OmniVLA returned {raw.shape}, expected {(1, *ACTION_SHAPE)}")
        if int(modality.detach().cpu().numpy()[0]) != LANGUAGE_ONLY_MODALITY_ID:
            raise ValueError("official modality selection did not produce language-only id 7")
        return raw[0], time.perf_counter() - started

    def infer(self, current_rgb, instruction):
        image = np.asarray(current_rgb, dtype=np.uint8)
        black_goal = Image.new("RGB", (image.shape[1], image.shape[0]), color=(0, 0, 0))
        return self.infer_with_placeholders(image, instruction, black_goal, np.zeros(4, dtype=np.float32))

    def check_language_only_leakage(self, current_rgb, instruction, atol=1.0e-4, rtol=1.0e-4):
        image = np.asarray(current_rgb, dtype=np.uint8)
        size = (image.shape[1], image.shape[0])
        started = time.perf_counter()
        first, _ = self.infer_with_placeholders(
            image, instruction, Image.new("RGB", size, color=(0, 0, 0)), np.zeros(4, dtype=np.float32)
        )
        second, _ = self.infer_with_placeholders(
            image,
            instruction,
            Image.new("RGB", size, color=(255, 127, 31)),
            np.asarray((91.0, -73.0, 0.3, -0.7), dtype=np.float32),
        )
        difference = float(np.max(np.abs(first - second)))
        return (
            bool(np.allclose(first, second, atol=atol, rtol=rtol)),
            difference,
            first,
            time.perf_counter() - started,
        )

    def close(self):
        del self.helper, self.action_head, self.pose_projector, self.vla
        self.torch.cuda.empty_cache()


class StubOmniVLA:
    def __init__(self):
        self.metadata = {
            "implementation": "deterministic stub (not OmniVLA inference)",
            "variant": "stub",
            "resume_step": None,
            "action_shape": list(ACTION_SHAPE),
            "modality_id": LANGUAGE_ONLY_MODALITY_ID,
        }

    def infer(self, _current_rgb, _instruction):
        raw = np.zeros(ACTION_SHAPE, dtype=np.float32)
        raw[:, 0] = np.arange(1, ACTION_SHAPE[0] + 1, dtype=np.float32)
        raw[:, 2] = 1.0
        return raw, 0.001

    def close(self):
        return None
