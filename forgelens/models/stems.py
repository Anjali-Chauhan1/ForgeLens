"""Forensic residual extractors: the learned and fixed high-pass front
ends that force the localization network to model noise/compression
statistics rather than image semantics. This is the single design
choice with the largest effect on cross-dataset generalization —
a network that only ever sees RGB content learns to recognize *objects*,
not *tampering*.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class BayarConv2d(nn.Module):
    """Constrained convolution: center weight fixed at -1, all other
    weights in the kernel constrained to sum to +1. This forces every
    output to be a *residual* — a weighted local prediction error —
    rather than a content filter. Introduced by Bayar & Stamm (2018)
    for camera-model forensics and reused across most modern
    manipulation-localization architectures.

    The non-center taps are normalized with softmax rather than plain
    division by their sum: a raw learnable sum is a mean-zero random
    variable early in training and dividing by it explodes whenever it
    passes near zero (verified empirically — the naive `w / w.sum()`
    version produces kernel magnitudes in the tens of thousands from a
    small-variance random init). Softmax guarantees a positive sum
    bounded away from zero at every step, at the cost of restricting
    non-center taps to positive weights — an acceptable trade for a
    stable, always-well-defined high-pass filter.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        assert k % 2 == 1, "kernel size must be odd so there is a center tap"
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = k
        self.center = (k * k) // 2
        # One learnable weight per non-center tap, per (out, in) pair.
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k * k - 1) * 0.05)

    def _constrained_kernel(self) -> torch.Tensor:
        # Softmax normalizes the non-center taps to sum to exactly 1
        # along the kernel dimension -> the full kernel (with center =
        # -1) sums to 0 by construction, for any value of self.weight.
        w = F.softmax(self.weight, dim=-1)
        oc, ic, _ = w.shape
        full = w.new_zeros(oc, ic, self.k * self.k)
        full[:, :, : self.center] = w[:, :, : self.center]
        full[:, :, self.center] = -1.0
        full[:, :, self.center + 1:] = w[:, :, self.center:]
        return full.view(oc, ic, self.k, self.k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self._constrained_kernel()
        return F.conv2d(x, kernel, padding=self.k // 2)


def _srm_kernels() -> torch.Tensor:
    """A compact bank of classic Spatial Rich Model high-pass filters
    from steganalysis (Fridrich & Kodovsky, 2012). Fixed, not learned —
    the point is to hand the network a content-suppressed, noise-amplified
    view it cannot "unlearn" into a content detector.
    """
    k1 = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32) / 4.0

    k2 = torch.tensor([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ], dtype=torch.float32) / 12.0

    k3 = torch.zeros(5, 5, dtype=torch.float32)
    k3[1:4, 1:4] = torch.tensor([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],
                                 dtype=torch.float32)
    k3 = k3 / 4.0

    edge = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, -1, 4, -1, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32)

    return torch.stack([k1, k2, k3, edge])  # (4, 5, 5)


class SRMConv2d(nn.Module):
    """Fixed (non-trainable) SRM filter bank applied per input channel."""

    def __init__(self, in_ch: int = 3):
        super().__init__()
        base = _srm_kernels()  # (4, 5, 5)
        n_filters = base.shape[0]
        # Apply each filter independently per input channel (depthwise)
        # by expanding to (in_ch*n_filters, 1, k, k) and using groups=in_ch.
        weight = base.unsqueeze(1).repeat(in_ch, 1, 1, 1)  # (in_ch*n_filters, 1, 5, 5)
        self.register_buffer("weight", weight)
        self.in_ch = in_ch
        self.n_filters = n_filters
        self.out_ch = in_ch * n_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # groups=in_ch applies the same n_filters-filter bank to each
        # input channel independently, output channels = in_ch * n_filters.
        return F.conv2d(x, self.weight, padding=2, groups=self.in_ch)


class ForensicStem(nn.Module):
    """Combines RGB, a learned Bayar residual, and a fixed SRM residual
    into one early-fusion tensor. This is the "multi-stream early fusion"
    input to the shared encoder — concatenation at the stem outperforms
    late fusion of separately-trained streams (per the blueprint).
    """

    def __init__(self, bayar_out: int = 8):
        super().__init__()
        self.bayar = BayarConv2d(in_ch=3, out_ch=bayar_out, k=5)
        self.srm = SRMConv2d(in_ch=3)  # out_ch = 3 * 4 = 12
        self.out_channels = 3 + bayar_out + self.srm.out_ch

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        bayar_feat = torch.tanh(self.bayar(rgb))  # bounded residual
        srm_feat = torch.clamp(self.srm(rgb), -3.0, 3.0)
        return torch.cat([rgb, bayar_feat, srm_feat], dim=1)
