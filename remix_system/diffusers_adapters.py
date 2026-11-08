"""Optional production adapters for the patched SD3.5 / SD1.5 pipelines.

The runtime itself has no torch or diffusers dependency. This module requires
the project's generate_with_profile and diffedit_rewrite_soft pipeline methods.
"""
from __future__ import annotations

import builtins
from contextlib import nullcontext
from typing import Any

from .types import EditPlan, Generated, Trace


class SD35Backend:
    def __init__(self, pipeline: Any, device: str, steps: int = 40,
                 height: int = 1024, width: int = 1024, guidance: float = 5.0,
                 seed: int = 42, negative_prompt: str = "",
                 capture_profile: bool = True):
        if not hasattr(pipeline, "generate_with_profile"):
            raise TypeError("SD3.5 pipeline must implement generate_with_profile")
        self.pipeline = pipeline
        self.device = device
        self.steps = steps
        self.height = height
        self.width = width
        self.guidance = guidance
        self.seed = seed
        self.negative_prompt = negative_prompt
        self.capture_profile = capture_profile
        # The patched Diffusers method prints one timing line per SD3.5 miss.
        # Filter that line in its module without muting other process output.
        method = getattr(pipeline.generate_with_profile, "__func__",
                         pipeline.generate_with_profile)
        namespace = method.__globals__
        current_print = namespace.get("print", builtins.print)
        if not getattr(current_print, "_remix_quiet_hds", False):
            def quiet_print(*args, **kwargs):
                if args and isinstance(args[0], str) and "[HDS inference]" in args[0]:
                    return None
                return current_print(*args, **kwargs)
            quiet_print._remix_quiet_hds = True
            namespace["print"] = quiet_print

    def generate(self, prompt: str) -> Generated:
        import torch
        if not self.capture_profile:
            result = self.pipeline(
                prompt=prompt, height=self.height, width=self.width,
                num_inference_steps=self.steps, guidance_scale=self.guidance,
                negative_prompt=self.negative_prompt,
                generator=torch.Generator(self.device).manual_seed(self.seed))
            return Generated(result.images[0])
        image, records, latents = self.pipeline.generate_with_profile(
            prompt=prompt, height=self.height, width=self.width,
            num_inference_steps=self.steps, guidance_scale=self.guidance,
            negative_prompt=self.negative_prompt,
            generator=torch.Generator(self.device).manual_seed(self.seed))
        deltas = tuple(float(row["noise_delta"]) for row in records)
        return Generated(image, Trace(tuple(latents), deltas))


class SD15Backend:
    def __init__(self, pipeline: Any, device: str, steps: int = 40,
                 guidance: float = 7.5, seed: int = 42,
                 negative_prompt: str = "", device_lock: Any = None):
        if not hasattr(pipeline, "diffedit_rewrite_soft"):
            raise TypeError("SD1.5 pipeline must implement diffedit_rewrite_soft")
        self.pipeline = pipeline
        self.device = device
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.negative_prompt = negative_prompt
        self._device_lock = device_lock or nullcontext()

    def edit(self, plan: EditPlan) -> Any:
        with self._device_lock:
            return self._edit(plan)

    def _edit(self, plan: EditPlan) -> Any:
        import torch
        generator = torch.Generator(self.device).manual_seed(self.seed)
        if plan.repaint_mode == "graduated" and plan.source_prompt != plan.target_prompt:
            result = self.pipeline.diffedit_rewrite_soft(
                image=plan.source_image, original_prompt=plan.source_prompt,
                new_prompt=plan.target_prompt, strength=plan.strength,
                num_inference_steps=self.steps, guidance_scale=self.guidance,
                generator=generator, output_type="pil",
                mask_expand_end=plan.mask_expand_end)
        elif plan.repaint_mode == "full" or plan.source_prompt == plan.target_prompt:
            result = self.pipeline(
                prompt=plan.target_prompt, image=plan.source_image,
                strength=plan.strength, num_inference_steps=self.steps,
                guidance_scale=self.guidance, negative_prompt=self.negative_prompt,
                generator=generator)
        else:
            raise ValueError(f"unsupported repaint mode: {plan.repaint_mode}")
        return result.images[0]


class SD35VAEDecoder:
    def __init__(self, vae: Any, device: str, device_lock: Any = None):
        self.vae = vae
        self.device = device
        self._device_lock = device_lock or nullcontext()

    def decode(self, latent: Any) -> Any:
        with self._device_lock:
            return self._decode(latent)

    def _decode(self, latent: Any) -> Any:
        import numpy as np
        import torch
        from PIL import Image
        with torch.inference_mode():
            data = latent.to(self.device, non_blocking=True) / self.vae.config.scaling_factor
            decoded = self.vae.decode(data, return_dict=False)[0]
            pixels = ((decoded / 2 + 0.5).clamp(0, 1)
                      .float().cpu().squeeze(0).permute(1, 2, 0).numpy())
        return Image.fromarray((pixels * 255).astype(np.uint8))


class CLIPScorer:
    """Resident scorer shared by cache lookup and profile scoring."""

    def __init__(self, model_path: str, device: str = "cpu", device_lock: Any = None):
        import threading
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(
            model_path, local_files_only=True).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        self._lock = device_lock or threading.Lock()

    def __call__(self, image: Any, prompt: str) -> float:
        return float(sum(a * b for a, b in zip(self.encode_image(image),
                                               self.encode_text(prompt))))

    def encode_text(self, prompt: str) -> tuple[float, ...]:
        import torch
        with self._lock, torch.inference_mode():
            data = self.processor(text=prompt, return_tensors="pt", padding=True,
                                  truncation=True, max_length=77)
            data = {key: value.to(self.device) for key, value in data.items()}
            value = self.model.get_text_features(**data)
            if not isinstance(value, torch.Tensor):
                value = value.pooler_output if hasattr(value, "pooler_output") else value[0]
            value = value / value.norm(dim=-1, keepdim=True)
            return tuple(value[0].float().cpu().tolist())

    def encode_image(self, image: Any) -> tuple[float, ...]:
        import torch
        with self._lock, torch.inference_mode():
            data = self.processor(images=image, return_tensors="pt")
            data = {key: value.to(self.device) for key, value in data.items()}
            value = self.model.get_image_features(**data)
            if not isinstance(value, torch.Tensor):
                value = value.pooler_output if hasattr(value, "pooler_output") else value[0]
            value = value / value.norm(dim=-1, keepdim=True)
            return tuple(value[0].float().cpu().tolist())

