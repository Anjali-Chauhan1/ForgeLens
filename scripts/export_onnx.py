"""Exports the trained localization model to ONNX for the ~2-4x latency
win from ONNX Runtime / TensorRT called out in the blueprint's stack
section. Fixed input size (matches the training canonical crop) since
the model is trained on pad-not-resize crops, not arbitrary resolutions.
"""
from __future__ import annotations
import argparse
import torch

from forgelens.models.backbone import LocalizationNet


def main(args):
    model = LocalizationNet(backbone_name=args.backbone, pretrained=False)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("ema", state["model"]))
    model.eval()

    dummy = torch.randn(1, 3, args.size, args.size)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["image"],
        output_names=["region_logit", "edge_logit", "image_logit"],
        opset_version=17,
        dynamic_axes=None,  # fixed size by design -- see module docstring
    )
    print(f"Exported to {args.out} (fixed input {args.size}x{args.size})")


def build_argparser():
    p = argparse.ArgumentParser(description="Export Branch A to ONNX")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="runs/localize/model.onnx")
    p.add_argument("--backbone", default="convnext_tiny")
    p.add_argument("--size", type=int, default=512)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
