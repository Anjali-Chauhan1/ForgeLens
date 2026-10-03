"""Training loop for Branch A (manipulation localization).

Implements every accuracy-maximizing detail called out in the blueprint's
training-recipe section: bf16 AMP, discriminative learning rates
(backbone vs head), cosine schedule with warmup, EMA of weights, and the
authentic-image batch composition enforced via AuthenticMixBatchSampler.
Two-stage curriculum (large-scale synthetic pretrain, then benchmark
finetune) is expressed as two separate invocations sharing --resume.
"""
from __future__ import annotations
import argparse
import copy
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from forgelens.models.backbone import LocalizationNet
from forgelens.train.losses import LocalizationLoss
from forgelens.data.datasets import ManipulationDataset, AuthenticMixBatchSampler
from forgelens.data.laundering import LaunderingAugment
from forgelens.eval.metrics import pixel_f1, pixel_iou


class EMA:
    """Exponential moving average of model weights. Reliably worth
    ~1 F1 point per the blueprint on a full-length run -- but a fixed
    decay=0.999 has a ~1000-step time constant, so on a short run (a
    demo, an ablation, early experimentation) the shadow weights stay
    dominated by the near-random initialization for a long time
    (0.999^312 ~= 0.73 after a full 312-step training run, i.e. the
    "EMA" checkpoint is still ~73% initial noise -- verified empirically:
    an evaluate_test_set.py run using such a checkpoint predicted nothing
    on every image despite the raw model scoring val pixel_f1=0.55).
    The standard fix (used in e.g. MoCo, torchvision's EMA reference) is
    a warmup schedule that ramps the effective decay up from 0 instead of
    applying the target decay from step 1.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.num_updates = 0
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.num_updates += 1
        # Ramps from 0 toward `decay` with a ~10-step time constant early
        # on, reaching the target decay only after enough updates that a
        # slow-moving average is actually meaningful.
        effective_decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(effective_decay).add_(v, alpha=1 - effective_decay)
            else:
                self.shadow[k] = v.clone()

    def apply_to(self, model: torch.nn.Module):
        model.load_state_dict(self.shadow)


def cosine_warmup_schedule(optimizer, total_steps: int, warmup_ratio: float = 0.05):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())

    return LambdaLR(optimizer, lr_lambda)


def build_optimizer(model: LocalizationNet, backbone_lr: float, head_lr: float, wd: float):
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            backbone_params.append(p)
        else:
            head_params.append(p)
    return AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ], weight_decay=wd)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    f1s, ious = [], []
    for batch in loader:
        img = batch["image"].to(device)
        mask = batch["mask"].to(device)
        out = model(img)
        prob = torch.sigmoid(out["region_logit"]).cpu().numpy()
        gt = mask.cpu().numpy()
        for p, g in zip(prob, gt):
            f1s.append(pixel_f1(p[0], g[0], mode="fixed"))
            ious.append(pixel_iou(p[0], g[0]))
    model.train()
    return {"pixel_f1": sum(f1s) / max(len(f1s), 1),
            "pixel_iou": sum(ious) / max(len(ious), 1)}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LocalizationNet(backbone_name=args.backbone, pretrained=True).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        print(f"Resumed weights from {args.resume}")

    loss_fn = LocalizationLoss()
    ema = EMA(model, decay=args.ema_decay)

    # NOTE: build_manifest() is dataset-specific and lives in
    # scripts/build_manifest.py -- it returns list[Sample].
    from scripts.build_manifest import load_manifest
    train_samples = load_manifest(args.train_manifest)
    val_samples = load_manifest(args.val_manifest)

    augment = LaunderingAugment(p_apply=args.laundering_p)
    train_ds = ManipulationDataset(train_samples, canonical=args.crop_size,
                                    augment=augment, train=True)
    val_ds = ManipulationDataset(val_samples, canonical=args.crop_size, train=False)

    sampler = AuthenticMixBatchSampler(train_samples, batch_size=args.batch_size,
                                        min_authentic_ratio=args.min_authentic_ratio)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=args.workers,
                               pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers)

    optimizer = build_optimizer(model, args.backbone_lr, args.head_lr, args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = cosine_warmup_schedule(optimizer, total_steps, args.warmup_ratio)
    # GradScaler is a CUDA-only mechanism (loss scaling for fp16); it is a
    # documented no-op on CPU/bf16, which is what makes CPU training work
    # without a separate code path here.
    scaler = torch.amp.GradScaler(device.type, enabled=(args.amp and device.type == "cuda"))
    # bf16 autocast is a CUDA/AMX-class-hardware optimization. On CPUs
    # without fast bf16 kernels it falls back to slow reference paths --
    # measured 35x slower than plain fp32 on this machine (56s/step vs
    # 1.6s/step for identical work) rather than the speedup the blueprint's
    # "bf16 AMP" recommendation assumes for GPU training. Gate it to CUDA
    # only so CPU runs (development, small demos, environments without a
    # GPU) aren't silently crippled by a flag that helps only on GPU.
    use_autocast = args.amp and device.type == "cuda"

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    step = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        running = {}
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            image_label = batch["image_label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
                out = model(img)
                losses = loss_fn(out, mask, image_label)

            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach())
            step += 1

            if step % args.log_every == 0:
                msg = " ".join(f"{k}={v/args.log_every:.4f}" for k, v in running.items())
                print(f"epoch {epoch} step {step}/{total_steps} {msg}")
                running = {}

        metrics = evaluate(model, val_loader, device)
        print(f"[epoch {epoch}] val pixel_f1={metrics['pixel_f1']:.4f} "
              f"pixel_iou={metrics['pixel_iou']:.4f} "
              f"({time.time() - t0:.1f}s)")

        ckpt = {"model": model.state_dict(), "ema": ema.shadow, "epoch": epoch,
                "metrics": metrics}
        torch.save(ckpt, Path(args.out_dir) / "last.pt")
        if metrics["pixel_f1"] > best_f1:
            best_f1 = metrics["pixel_f1"]
            torch.save(ckpt, Path(args.out_dir) / "best.pt")
            print(f"  -> new best (pixel_f1={best_f1:.4f}), saved.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Branch A: manipulation localization")
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--out-dir", default="runs/localize")
    p.add_argument("--backbone", default="convnext_tiny")
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--min-authentic-ratio", type=float, default=0.3)
    p.add_argument("--laundering-p", type=float, default=0.8)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--resume", default=None)
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
