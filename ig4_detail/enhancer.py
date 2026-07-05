"""Core image-processing engine for the Ideogram 4 Detail Enhancer.

Everything here operates on a single float32 RGB numpy array in [0, 1],
shape (H, W, 3). The ComfyUI node layer (nodes.py) handles torch tensor
conversion and batching. Keeping this module torch-free makes it easy to
test standalone.

The pipeline is classic "retouching in reverse": Ideogram 4 (like most
diffusion/AR image models) tends to output slightly waxy skin, mushy hair
strands and soft eyes. We rebuild plausible micro-detail with frequency
separation, targeted masks (skin / eyes / hair) and film grain, without
running another generative model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .detection import detect_faces_and_eyes


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

@dataclass
class EnhanceSettings:
    overall_strength: float = 1.0   # master wet/dry blend (0 = bypass, >1 = extrapolate)
    skin_smooth: float = 0.25       # remove waxy banding before re-texturing (0..1)
    skin_texture: float = 1.0       # synthetic pore-level micro texture in skin areas
    eye_enhance: float = 1.0        # iris sharpening + local contrast + saturation
    hair_detail: float = 1.0        # strand-level high-frequency boost outside skin
    clarity: float = 0.5            # midtone local contrast ("punch")
    fine_sharpen: float = 0.6       # global unsharp mask
    sharpen_radius: float = 1.2     # unsharp mask sigma in pixels
    grain_amount: float = 0.12      # photographic luminance grain
    grain_size: float = 1.0         # grain scale in pixels (bigger = coarser)
    grain_seed: int = 0
    saturation: float = 1.0         # 1.0 = unchanged
    detect_faces: bool = True       # enables eye targeting + face-weighted skin mask
    external_mask: np.ndarray | None = field(default=None, repr=False)  # (H, W) 0..1


@dataclass
class EnhanceResult:
    image: np.ndarray
    skin_mask: np.ndarray  # (H, W) float32 0..1
    eye_mask: np.ndarray   # (H, W) float32 0..1


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img.copy()
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _luma(img: np.ndarray) -> np.ndarray:
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def _midtone_weight(luma: np.ndarray) -> np.ndarray:
    """1.0 in the midtones, falling off toward pure black/white.

    Keeps texture/grain/clarity from crushing shadows or clipping highlights.
    """
    return np.clip(1.0 - (2.0 * luma - 1.0) ** 2, 0.0, 1.0)


def _feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    return np.clip(_gaussian(mask.astype(np.float32), sigma), 0.0, 1.0)


# --------------------------------------------------------------------------
# Masks: skin, eyes
# --------------------------------------------------------------------------

def skin_color_mask_raw(img: np.ndarray) -> np.ndarray:
    """Binary-ish color-based skin mask (YCrCb range) with morphological
    cleanup. Works for faces, hands, arms, etc."""
    h, w = img.shape[:2]
    u8 = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    ycrcb = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
    mask = ((cr >= 133) & (cr <= 180) &
            (cb >= 80) & (cb <= 135) &
            (y >= 30)).astype(np.float32)
    k = max(3, (min(h, w) // 200) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def build_eye_mask(shape: tuple[int, int],
                   eyes: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), np.float32)
    for (cx, cy, r) in eyes:
        cv2.circle(mask, (int(cx), int(cy)), int(max(2, r)), 1.0, -1)
    if eyes:
        sigma = max(1.5, 0.35 * float(np.mean([r for _, _, r in eyes])))
        mask = _feather(mask, sigma)
    return mask


def finalize_skin_mask(raw: np.ndarray, eye_mask: np.ndarray) -> np.ndarray:
    """Feather the raw skin mask and subtract eye regions so skin smoothing
    never softens the irises we just sharpened."""
    h, w = raw.shape[:2]
    mask = _feather(raw, max(2.0, min(h, w) / 150.0))
    return np.clip(mask * (1.0 - eye_mask), 0.0, 1.0)


# --------------------------------------------------------------------------
# Enhancement stages
# --------------------------------------------------------------------------

def _smooth_skin(img: np.ndarray, skin_mask: np.ndarray, amount: float) -> np.ndarray:
    """Gentle bilateral pass inside skin only — removes the 'plastic' AI sheen
    and color banding so the synthetic texture we add next reads as pores,
    not as noise on top of noise."""
    if amount <= 0 or skin_mask.max() <= 0.01:
        return img
    h, w = img.shape[:2]
    d = max(5, int(min(h, w) * 0.01) | 1)
    smoothed = cv2.bilateralFilter(img, d=d, sigmaColor=0.10, sigmaSpace=d)
    weight = (amount * skin_mask)[..., None]
    return img * (1.0 - weight) + smoothed * weight


def _add_skin_texture(img: np.ndarray, skin_mask: np.ndarray,
                      amount: float, rng: np.random.Generator) -> np.ndarray:
    """Band-passed noise at pore scale, luminance-weighted, skin-masked."""
    if amount <= 0 or skin_mask.max() <= 0.01:
        return img
    h, w = img.shape[:2]
    noise = rng.standard_normal((h, w)).astype(np.float32)
    # Band-pass: keep only the ~1-2 px "pore" frequency band.
    band = _gaussian(noise, 0.6) - _gaussian(noise, 1.8)
    std = float(band.std())
    if std > 1e-6:
        band /= std
    weight = amount * 0.045 * skin_mask * _midtone_weight(_luma(img))
    return img + band[..., None] * weight[..., None]


def _enhance_eyes(img: np.ndarray, eye_mask: np.ndarray, amount: float) -> np.ndarray:
    """Sharpen, add local contrast and a touch of saturation inside eyes —
    makes irises crisp and catchlights pop."""
    if amount <= 0 or eye_mask.max() <= 0.01:
        return img
    sharp = img + 1.2 * amount * (img - _gaussian(img, 1.0))
    contrasted = (sharp - 0.5) * (1.0 + 0.15 * amount) + 0.5
    gray = _luma(contrasted)[..., None]
    saturated = gray + (contrasted - gray) * (1.0 + 0.18 * amount)
    weight = np.clip(eye_mask * min(1.0, amount), 0.0, 1.0)[..., None]
    return img * (1.0 - weight) + np.clip(saturated, 0.0, 1.0) * weight


def _boost_hair_detail(img: np.ndarray, skin_mask: np.ndarray, amount: float) -> np.ndarray:
    """High-frequency boost weighted toward strand-dense regions outside skin.

    Hair (and fabric, foliage, etc.) is where Ideogram outputs go mushy; we
    find high-frequency-energy areas that are not skin and amplify the fine
    structure that is already there.
    """
    if amount <= 0:
        return img
    hp = img - _gaussian(img, 1.5)
    energy = _gaussian(np.abs(hp).mean(axis=2), 5.0)
    p95 = float(np.percentile(energy, 95))
    if p95 > 1e-6:
        energy = np.clip(energy / p95, 0.0, 1.0)
    weight = amount * 0.9 * energy * (1.0 - skin_mask)
    return img + hp * weight[..., None]


def _apply_clarity(img: np.ndarray, amount: float) -> np.ndarray:
    """Large-radius, midtone-weighted local contrast (a la Lightroom clarity)."""
    if amount <= 0:
        return img
    h, w = img.shape[:2]
    sigma = float(np.clip(min(h, w) * 0.02, 6.0, 40.0))
    luma = _luma(img)
    detail = luma - _gaussian(luma, sigma)
    boosted = luma + amount * 0.4 * detail * _midtone_weight(luma)
    ratio = (np.clip(boosted, 0.0, 1.0) + 1e-5) / (luma + 1e-5)
    return img * ratio[..., None]


def _fine_sharpen(img: np.ndarray, amount: float, radius: float) -> np.ndarray:
    if amount <= 0:
        return img
    return img + amount * 0.6 * (img - _gaussian(img, max(0.4, radius)))


def _apply_saturation(img: np.ndarray, saturation: float) -> np.ndarray:
    if abs(saturation - 1.0) < 1e-4:
        return img
    gray = _luma(img)[..., None]
    return gray + (img - gray) * saturation


def _add_grain(img: np.ndarray, amount: float, size: float,
               rng: np.random.Generator) -> np.ndarray:
    """Photographic luminance grain. Grain is the cheapest realism win there
    is: it masks residual smoothness and ties added texture together."""
    if amount <= 0:
        return img
    h, w = img.shape[:2]
    size = max(0.5, size)
    nh, nw = max(2, int(h / size)), max(2, int(w / size))
    noise = rng.standard_normal((nh, nw)).astype(np.float32)
    if (nh, nw) != (h, w):
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    weight = amount * 0.05 * (0.25 + 0.75 * _midtone_weight(_luma(img)))
    return img + noise[..., None] * weight[..., None]


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def enhance(img: np.ndarray, settings: EnhanceSettings) -> EnhanceResult:
    """Run the full detail-enhancement pipeline on one float32 RGB image."""
    img = np.clip(img.astype(np.float32), 0.0, 1.0)
    original = img.copy()
    h, w = img.shape[:2]

    skin_raw = skin_color_mask_raw(img)
    if settings.detect_faces:
        faces = detect_faces_and_eyes(img, skin_raw)
        eyes = [eye for face in faces for eye in face.eyes]
    else:
        eyes = []
    eye_mask = build_eye_mask((h, w), eyes)
    skin_mask = finalize_skin_mask(skin_raw, eye_mask)

    if settings.external_mask is not None:
        ext = settings.external_mask.astype(np.float32)
        if ext.shape != (h, w):
            ext = cv2.resize(ext, (w, h), interpolation=cv2.INTER_LINEAR)
        ext = np.clip(ext, 0.0, 1.0)
    else:
        ext = None

    rng = np.random.default_rng(settings.grain_seed & 0xFFFFFFFF)

    out = img
    out = _smooth_skin(out, skin_mask, settings.skin_smooth)
    out = np.clip(_apply_clarity(out, settings.clarity), 0.0, 1.0)
    out = np.clip(_boost_hair_detail(out, skin_mask, settings.hair_detail), 0.0, 1.0)
    out = np.clip(_add_skin_texture(out, skin_mask, settings.skin_texture, rng), 0.0, 1.0)
    out = np.clip(_enhance_eyes(out, eye_mask, settings.eye_enhance), 0.0, 1.0)
    out = np.clip(_fine_sharpen(out, settings.fine_sharpen, settings.sharpen_radius), 0.0, 1.0)
    out = np.clip(_apply_saturation(out, settings.saturation), 0.0, 1.0)
    out = np.clip(_add_grain(out, settings.grain_amount, settings.grain_size, rng), 0.0, 1.0)

    # Master blend. Values > 1 extrapolate past the processed image, which is
    # a quick way to audition an exaggerated version of the current knobs.
    s = settings.overall_strength
    out = np.clip(original + (out - original) * s, 0.0, 1.0)

    if ext is not None:
        out = original * (1.0 - ext[..., None]) + out * ext[..., None]
        skin_mask = skin_mask * ext
        eye_mask = eye_mask * ext

    return EnhanceResult(image=out.astype(np.float32),
                         skin_mask=skin_mask.astype(np.float32),
                         eye_mask=eye_mask.astype(np.float32))
