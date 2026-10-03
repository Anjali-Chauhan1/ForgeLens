"""Shared encoder-decoder for Branch A (manipulation localization).

Uses a timm ConvNeXt-Tiny encoder fed by the ForensicStem's early-fused
RGB+Bayar+SRM tensor, and a lightweight FPN-style decoder producing two
heads: a per-pixel tamper-region mask and a per-pixel edge/boundary map.
Dual region+edge supervision is the second highest-leverage design
choice in the blueprint after the forensic stem itself — boundary
supervision measurably sharpens masks and improves generalization.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from forgelens.models.stems import ForensicStem


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsample + concat skip + two conv blocks (standard U-Net decoder)."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNAct(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


class LocalizationNet(nn.Module):
    """RGB+residual stem -> ConvNeXt-Tiny encoder -> U-Net decoder ->
    {region mask, edge map, image-level tamper score}.
    """

    def __init__(self, backbone_name: str = "convnext_tiny", pretrained: bool = True,
                 decoder_channels=(256, 128, 64, 32)):
        super().__init__()
        self.stem = ForensicStem(bayar_out=8)

        self.encoder = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True,
            in_chans=self.stem.out_channels,
        )
        enc_channels = self.encoder.feature_info.channels()  # shallow -> deep

        # Build decoder from deepest to shallowest.
        chans = list(reversed(enc_channels))  # e.g. [768, 384, 192, 96]
        dec_ch = list(decoder_channels)
        self.decoders = nn.ModuleList()
        in_ch = chans[0]
        for skip_ch, out_ch in zip(chans[1:], dec_ch[: len(chans) - 1]):
            self.decoders.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        final_ch = dec_ch[len(chans) - 2] if len(chans) >= 2 else in_ch

        self.region_head = nn.Sequential(
            ConvBNAct(final_ch, final_ch // 2),
            nn.Conv2d(final_ch // 2, 1, kernel_size=1),
        )
        self.edge_head = nn.Sequential(
            ConvBNAct(final_ch, final_ch // 2),
            nn.Conv2d(final_ch // 2, 1, kernel_size=1),
        )
        # Image-level head: global pool of the deepest features.
        self.image_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(chans[0], 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        """rgb: (B, 3, H, W) in [0, 1], NOT resized — pad to a multiple
        of the backbone stride upstream (see data/datasets.py)."""
        x = self.stem(rgb)
        feats = self.encoder(x)  # shallow -> deep
        feats = list(reversed(feats))  # deep -> shallow

        image_logit = self.image_head(feats[0])

        y = feats[0]
        for dec, skip in zip(self.decoders, feats[1:]):
            y = dec(y, skip)

        region_logit = self.region_head(y)
        edge_logit = self.edge_head(y)

        region_logit = F.interpolate(region_logit, size=rgb.shape[-2:],
                                      mode="bilinear", align_corners=False)
        edge_logit = F.interpolate(edge_logit, size=rgb.shape[-2:],
                                    mode="bilinear", align_corners=False)

        return {
            "region_logit": region_logit,   # (B,1,H,W)
            "edge_logit": edge_logit,       # (B,1,H,W)
            "image_logit": image_logit,     # (B,1)
        }
