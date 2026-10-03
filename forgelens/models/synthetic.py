"""Branch C: AI-generated content detection.

Key finding from the literature review: a CNN trained from scratch on
real-vs-fake gets excellent in-distribution accuracy and collapses to
near-chance on unseen generators, because it overfits to one generator's
specific upsampling/spectral fingerprint. What generalizes is a FROZEN
large pretrained encoder (CLIP) with a light head on top -- freezing
prevents the encoder from re-specializing to a single generator family.

Two heads share the frozen encoder:
  - global head: "is this whole image generated?"
  - patch head: run over a grid of crops, catches PARTIAL generative
    fill (e.g. Photoshop generative fill on one region of a real photo)
    that a single global embedding would average away.

A frequency auxiliary (azimuthally-averaged FFT power spectrum) is
concatenated to the head input -- cheap, and it catches the periodic
upsampling artifacts that CLIP's semantic features tend to miss.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import open_clip
    _BACKEND = "open_clip"
except Exception:  # pragma: no cover
    try:
        from transformers import CLIPModel, CLIPImageProcessor
        _BACKEND = "transformers"
    except Exception:
        _BACKEND = None


FREQ_BINS = 32


def azimuthal_fft_features(x: torch.Tensor, bins: int = FREQ_BINS) -> torch.Tensor:
    """x: (B, C, H, W) in [0,1]. Returns (B, bins) log-magnitude spectrum
    radially averaged -- a compact, resolution-invariant frequency
    fingerprint sensitive to GAN/diffusion upsampling artifacts.
    """
    gray = x.mean(dim=1)  # (B, H, W)
    spec = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
    mag = torch.log1p(spec.abs())  # (B, H, W)

    b, h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device) - cy,
        torch.arange(w, device=x.device) - cx,
        indexing="ij",
    )
    r = torch.sqrt((yy.float() ** 2 + xx.float() ** 2))
    r_max = r.max().clamp_min(1e-6)
    bin_idx = (r / r_max * (bins - 1)).long().clamp(0, bins - 1)  # (H, W)

    out = torch.zeros(b, bins, device=x.device, dtype=mag.dtype)
    counts = torch.zeros(bins, device=x.device, dtype=mag.dtype)
    flat_idx = bin_idx.view(-1)
    for i in range(bins):
        counts[i] = (flat_idx == i).float().sum().clamp_min(1.0)
    for i in range(bins):
        mask = (bin_idx == i)
        if mask.any():
            out[:, i] = mag[:, mask].mean(dim=-1)
    return out


class FrozenCLIPEncoder(nn.Module):
    """Wraps whichever CLIP backend is installed behind one interface.
    Always frozen (requires_grad=False) -- only the head trains.
    """

    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai"):
        super().__init__()
        if _BACKEND == "open_clip":
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self.model = model.visual
            self.embed_dim = model.visual.output_dim
            self._forward_fn = self._forward_open_clip
        elif _BACKEND == "transformers":
            self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").vision_model
            self.embed_dim = self.model.config.hidden_size
            self._forward_fn = self._forward_hf
        else:
            raise ImportError(
                "Neither open_clip nor transformers is installed. "
                "Install one: `pip install open_clip_torch` (recommended) "
                "or `pip install transformers`."
            )
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def _forward_open_clip(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _forward_hf(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        return out.pooler_output

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_fn(x)


class SyntheticHead(nn.Module):
    def __init__(self, in_dim: int, freq_bins: int = FREQ_BINS, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + freq_bins, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, embed: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([embed, freq], dim=-1))


class SyntheticDetector(nn.Module):
    """Global classification + patch-grid classification over a frozen
    CLIP encoder. Train ONLY `global_head` and `patch_head` -- the
    encoder stays frozen throughout (see train/trainer_synthetic.py).
    """

    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai",
                 patch_size: int = 224, patch_stride: int = 168):
        super().__init__()
        self.encoder = FrozenCLIPEncoder(model_name, pretrained)
        self.global_head = SyntheticHead(self.encoder.embed_dim)
        self.patch_head = SyntheticHead(self.encoder.embed_dim)
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def _clip_input_size(self) -> int:
        return 224  # standard CLIP ViT input resolution

    def forward_global(self, image_224: torch.Tensor) -> torch.Tensor:
        embed = self.encoder(image_224)
        freq = azimuthal_fft_features(image_224)
        return self.global_head(embed, freq).squeeze(-1)

    def forward_patches(self, image_full: torch.Tensor) -> torch.Tensor:
        """image_full: (B, 3, H, W) at native resolution, values in [0,1].
        Returns a coarse (B, gh, gw) logit map by tiling non-overlapping
        (with stride) crops through the encoder+patch_head.
        """
        b, c, h, w = image_full.shape
        ps, st = self.patch_size, self.patch_stride
        ys = list(range(0, max(h - ps, 0) + 1, st)) or [0]
        xs = list(range(0, max(w - ps, 0) + 1, st)) or [0]

        patches = []
        for y in ys:
            for x in xs:
                crop = image_full[:, :, y:y + ps, x:x + ps]
                if crop.shape[-2:] != (ps, ps):
                    crop = F.pad(crop, (0, ps - crop.shape[-1], 0, ps - crop.shape[-2]))
                patches.append(crop)
        grid = torch.stack(patches, dim=1)  # (B, N, 3, ps, ps)
        bn, n = grid.shape[0], grid.shape[1]
        flat = grid.view(bn * n, c, ps, ps)
        clip_in = F.interpolate(flat, size=(224, 224), mode="bilinear", align_corners=False)

        embed = self.encoder(clip_in)
        freq = azimuthal_fft_features(clip_in)
        logits = self.patch_head(embed, freq).view(bn, n)
        return logits.view(bn, len(ys), len(xs))

    def forward(self, image_224: torch.Tensor, image_full: torch.Tensor | None = None) -> dict:
        out = {"global_logit": self.forward_global(image_224)}
        if image_full is not None:
            patch_logits = self.forward_patches(image_full)
            out["patch_logits"] = patch_logits
            out["patch_max"] = patch_logits.amax(dim=(-2, -1))
            out["patch_area_ratio"] = (torch.sigmoid(patch_logits) > 0.5).float().mean(dim=(-2, -1))
        return out
