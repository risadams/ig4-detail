"""ComfyUI node definitions for the Ideogram 4 Detail Enhancer.

Two nodes are exposed:

* ``IG4DetailEnhancer`` — the full node with every knob, plus skin/eye mask
  outputs for dialing things in.
* ``IG4DetailEnhancerSimple`` — a one-slider version for quick drop-in use.

Both take IMAGE and return IMAGE, so they chain directly after any Ideogram
node (or any other image source) and in front of SaveImage.
"""

from __future__ import annotations

import numpy as np
import torch

from .enhancer import EnhanceSettings, enhance


def _image_to_numpy(image: torch.Tensor) -> np.ndarray:
    """ComfyUI IMAGE (B, H, W, C) float 0..1 -> numpy on CPU."""
    return image.detach().cpu().float().numpy()


def _run_batch(image: torch.Tensor,
               settings: EnhanceSettings,
               mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = _image_to_numpy(image)
    if batch.ndim != 4 or batch.shape[-1] < 3:
        raise ValueError(f"Expected IMAGE tensor of shape (B, H, W, 3+), got {tuple(image.shape)}")

    mask_np = None
    if mask is not None:
        mask_np = mask.detach().cpu().float().numpy()
        if mask_np.ndim == 2:
            mask_np = mask_np[None, ...]

    out_images, out_skin, out_eyes = [], [], []
    for i in range(batch.shape[0]):
        frame_settings = settings
        if mask_np is not None:
            frame_settings = EnhanceSettings(**{**settings.__dict__})
            frame_settings.external_mask = mask_np[min(i, mask_np.shape[0] - 1)]
        result = enhance(batch[i][..., :3], frame_settings)
        frame = batch[i].copy()
        frame[..., :3] = result.image
        out_images.append(frame)
        out_skin.append(result.skin_mask)
        out_eyes.append(result.eye_mask)

    device = image.device
    return (
        torch.from_numpy(np.stack(out_images)).to(device),
        torch.from_numpy(np.stack(out_skin)).to(device),
        torch.from_numpy(np.stack(out_eyes)).to(device),
    )


class IG4DetailEnhancer:
    """Drop-in detail enhancer for Ideogram 4 outputs.

    Adds photographic micro-detail — skin texture, crisp eyes, hair strands,
    clarity, sharpening and film grain — to any generated image without
    running a second generative model.
    """

    CATEGORY = "image/ideogram4"
    FUNCTION = "enhance"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "skin_mask", "eye_mask")
    DESCRIPTION = ("Adds realistic micro-detail (skin pores, hair strands, crisp eyes, "
                   "clarity, sharpening, film grain) to Ideogram 4 outputs. "
                   "Pure image processing — fast, deterministic, no extra models.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "overall_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Master wet/dry blend. 0 = bypass, 1 = full effect, "
                               ">1 exaggerates the current settings."}),
                "skin_smooth": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Removes waxy AI sheen/banding on skin before texture is added."}),
                "skin_texture": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Pore-level micro texture added inside detected skin areas."}),
                "eye_enhance": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Iris sharpening, local contrast and saturation on detected eyes."}),
                "hair_detail": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Strand-level high-frequency boost in detailed non-skin areas "
                               "(hair, fabric, foliage)."}),
                "clarity": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Midtone local contrast — adds punch without halos."}),
                "fine_sharpen": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Global unsharp-mask amount applied after detail passes."}),
                "sharpen_radius": ("FLOAT", {
                    "default": 1.2, "min": 0.4, "max": 3.0, "step": 0.1,
                    "tooltip": "Unsharp-mask radius in pixels."}),
                "grain_amount": ("FLOAT", {
                    "default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Photographic luminance grain. Small amounts hide residual "
                               "smoothness and sell realism."}),
                "grain_size": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1,
                    "tooltip": "Grain scale in pixels (bigger = coarser, more 'film')."}),
                "grain_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF,
                    "tooltip": "Seed for grain and skin-texture noise (deterministic)."}),
                "saturation": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 1.5, "step": 0.01,
                    "tooltip": "Final saturation trim (1.0 = unchanged)."}),
                "detect_faces": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable face/eye detection for targeted eye enhancement. "
                               "Skin masking stays active either way."}),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional mask limiting where the enhancement is applied "
                               "(white = enhance)."}),
            },
        }

    def enhance(self, image, overall_strength, skin_smooth, skin_texture,
                eye_enhance, hair_detail, clarity, fine_sharpen, sharpen_radius,
                grain_amount, grain_size, grain_seed, saturation, detect_faces,
                mask=None):
        settings = EnhanceSettings(
            overall_strength=overall_strength,
            skin_smooth=skin_smooth,
            skin_texture=skin_texture,
            eye_enhance=eye_enhance,
            hair_detail=hair_detail,
            clarity=clarity,
            fine_sharpen=fine_sharpen,
            sharpen_radius=sharpen_radius,
            grain_amount=grain_amount,
            grain_size=grain_size,
            grain_seed=grain_seed,
            saturation=saturation,
            detect_faces=detect_faces,
        )
        return _run_batch(image, settings, mask)


class IG4DetailEnhancerSimple:
    """One-knob version: sensible portrait defaults scaled by a single strength."""

    CATEGORY = "image/ideogram4"
    FUNCTION = "enhance"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    DESCRIPTION = ("Single-slider Ideogram 4 detail enhancer using tuned portrait "
                   "defaults. Use the full IG4 Detail Enhancer node for fine control.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Scales the whole enhancement (skin, eyes, hair, "
                               "clarity, sharpen, grain)."}),
                "grain_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF,
                    "tooltip": "Seed for grain and skin-texture noise."}),
            },
        }

    def enhance(self, image, strength, grain_seed):
        settings = EnhanceSettings(
            overall_strength=strength,
            skin_smooth=0.25,
            skin_texture=1.0,
            eye_enhance=1.0,
            hair_detail=1.0,
            clarity=0.5,
            fine_sharpen=0.6,
            sharpen_radius=1.2,
            grain_amount=0.12,
            grain_size=1.0,
            grain_seed=grain_seed,
            saturation=1.0,
            detect_faces=True,
        )
        out, _, _ = _run_batch(image, settings, None)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "IG4DetailEnhancer": IG4DetailEnhancer,
    "IG4DetailEnhancerSimple": IG4DetailEnhancerSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IG4DetailEnhancer": "IG4 Detail Enhancer (Ideogram 4)",
    "IG4DetailEnhancerSimple": "IG4 Detail Enhancer — Simple",
}
