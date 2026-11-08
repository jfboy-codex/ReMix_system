"""Mask implementation required by the experiment's patched SD1.5 pipeline.

Kept compatible with ``from src.mask_algo import ExpandingMask`` in
``diffedit_rewrite_soft``. The noise difference and mask schedule match the
original experiment implementation.
"""

import torch
import torch.nn.functional as F


class ExpandingMask:
    def __init__(self, unet, scheduler, timesteps, img_latents,
                 noise_shared, text_orig, text_new, uncond,
                 guidance_scale=7.5, do_cfg=True, expand_until: int | None = None):
        self._dtype = text_orig.dtype
        self.timesteps = timesteps
        self.img_latents = img_latents
        self.noise_shared = noise_shared
        self.k = len(timesteps)
        self.diff_map = self._compute(unet, scheduler, text_orig, text_new,
                                      uncond, do_cfg)
        d_f = self.diff_map.float()
        self._start = float(d_f.mean() + d_f.std())
        self._expand_until = expand_until

    def _compute(self, unet, scheduler, text_orig, text_new, uncond, do_cfg):
        t0_batch = self.timesteps[0].unsqueeze(0)
        xt = scheduler.add_noise(self.img_latents, self.noise_shared, t0_batch)
        latent_in = scheduler.scale_model_input(
            torch.cat([xt] * 2).to(dtype=self._dtype) if do_cfg else xt,
            t0_batch)
        if do_cfg:
            n_orig = unet(latent_in, t0_batch,
                encoder_hidden_states=torch.cat([uncond, text_orig]).to(dtype=self._dtype),
                return_dict=False)[0].chunk(2)[1].float()
            n_new = unet(latent_in, t0_batch,
                encoder_hidden_states=torch.cat([uncond, text_new]).to(dtype=self._dtype),
                return_dict=False)[0].chunk(2)[1].float()
        else:
            n_orig = unet(latent_in, t0_batch,
                encoder_hidden_states=text_orig, return_dict=False)[0].float()
            n_new = unet(latent_in, t0_batch,
                encoder_hidden_states=text_new, return_dict=False)[0].float()
        diff_map = torch.norm(n_new - n_orig, dim=1, keepdim=True)
        diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
        diff_map = F.max_pool2d(diff_map.float(), kernel_size=5, stride=1, padding=2)
        return diff_map.to(dtype=self._dtype)

    def get_mask(self, step_idx: int, total_steps: int) -> torch.Tensor:
        denom = self._expand_until if self._expand_until is not None else total_steps - 1
        thresh = self._start * max(0.0, 1.0 - step_idx / max(denom, 1))
        mask = (self.diff_map > thresh).float()
        mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        return mask.to(dtype=self._dtype)

    def classification(self) -> dict:
        return {"type": "expanding", "threshold_start": round(self._start, 6),
                "expand_until": self._expand_until}
