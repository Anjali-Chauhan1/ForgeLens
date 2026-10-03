"""Social-media laundering simulation.

Applied identically to authentic AND forged images. If it were only
applied to forged images the model would learn "detect JPEG artifacts"
rather than "detect tampering" — the most common silent failure mode
in this field, called out explicitly in the blueprint. This is also
the single augmentation with the largest measured effect on
cross-dataset / real-world generalization.
"""
from __future__ import annotations
import io
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def _jpeg_recompress(img: np.ndarray, quality: int) -> np.ndarray:
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def _resize_random_interp(img: np.ndarray, scale: float) -> np.ndarray:
    pil = Image.fromarray(img)
    w, h = pil.size
    new_w, new_h = max(8, int(w * scale)), max(8, int(h * scale))
    interp = random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
    resized = pil.resize((new_w, new_h), interp)
    return np.asarray(resized.resize((w, h), interp))  # round-trip back


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    pil = Image.fromarray(img)
    return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=sigma)))


def _add_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _color_jitter(img: np.ndarray, brightness: float, contrast: float,
                   saturation: float) -> np.ndarray:
    pil = Image.fromarray(img)
    pil = ImageEnhance.Brightness(pil).enhance(brightness)
    pil = ImageEnhance.Contrast(pil).enhance(contrast)
    pil = ImageEnhance.Color(pil).enhance(saturation)
    return np.asarray(pil)


def _simulate_screenshot(img: np.ndarray) -> np.ndarray:
    """Crude screenshot proxy: downscale-upscale (introduces a resampling
    grid) plus a very light interference pattern to emulate moire."""
    pil = Image.fromarray(img)
    w, h = pil.size
    small = pil.resize((max(8, int(w * 0.6)), max(8, int(h * 0.6))), Image.BILINEAR)
    back = np.asarray(small.resize((w, h), Image.NEAREST)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    interference = 6.0 * np.sin(xx / 3.0) * np.sin(yy / 3.0)
    back += interference[..., None]
    return np.clip(back, 0, 255).astype(np.uint8)


class LaunderingAugment:
    """Callable: (img_uint8 HWC, mask_float01 HW) -> (img, mask).
    The mask is never touched by pixel-level ops (JPEG, blur, noise,
    color) since those don't move pixels; it IS touched by the resize
    round-trip and the screenshot path, whose nearest-neighbor final
    step keeps the mask binary-safe.
    """

    def __init__(self, p_apply: float = 0.8):
        self.p_apply = p_apply

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        if random.random() >= self.p_apply:
            return img, mask

        if random.random() < 0.6:
            scale = random.uniform(0.5, 1.0)
            img = _resize_random_interp(img, scale)
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            w, h = mask_pil.size
            new_w, new_h = max(8, int(w * scale)), max(8, int(h * scale))
            mask_pil = mask_pil.resize((new_w, new_h), Image.NEAREST).resize((w, h), Image.NEAREST)
            mask = (np.asarray(mask_pil).astype(np.float32) / 255.0 > 0.5).astype(np.float32)

        if random.random() < 0.9:
            img = _jpeg_recompress(img, random.randint(55, 95))

        if random.random() < 0.4:
            img = _jpeg_recompress(img, random.randint(40, 80))

        if random.random() < 0.2:
            img = _gaussian_blur(img, random.uniform(0.3, 1.2))

        if random.random() < 0.2:
            img = _add_noise(img, random.uniform(1, 5))

        if random.random() < 0.3:
            img = _color_jitter(
                img,
                brightness=random.uniform(0.85, 1.15),
                contrast=random.uniform(0.85, 1.15),
                saturation=random.uniform(0.85, 1.15),
            )

        if random.random() < 0.05:
            img = _simulate_screenshot(img)

        return img, mask


def degradation_grid_variants(img: np.ndarray) -> dict[str, np.ndarray]:
    """Deterministic variants for the robustness evaluation grid
    (JPEG QF x scale) described in the blueprint's evaluation section."""
    variants = {}
    for qf in (90, 70, 50, 30):
        variants[f"jpeg_q{qf}"] = _jpeg_recompress(img, qf)
    for scale in (0.75, 0.5):
        variants[f"scale_{scale}"] = _resize_random_interp(img, scale)
    return variants
