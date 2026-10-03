"""Training loop for Branch C (synthetic/AI-generated content detection).

Only `global_head` and `patch_head` train -- the CLIP encoder stays
frozen throughout, which is the specific design choice that gives this
branch its cross-generator generalization (see models/synthetic.py for
the reasoning). Evaluation MUST use leave-one-generator-out per the
blueprint; same-generator numbers are not reported as a headline.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from PIL import Image
import numpy as np

from forgelens.models.synthetic import SyntheticDetector
from forgelens.eval.metrics import image_auc, expected_calibration_error


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


class SyntheticImageDataset(Dataset):
    """manifest: list of {"path": str, "label": 0|1, "generator": str}.
    label=1 means AI-generated/synthetic, 0 means camera-captured real.
    """

    def __init__(self, manifest: list[dict], crop: int = 224):
        self.manifest = manifest
        self.crop = crop

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        rec = self.manifest[idx]
        img = Image.open(rec["path"]).convert("RGB")
        img = img.resize((self.crop, self.crop), Image.BICUBIC)
        arr = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
        arr = (arr - CLIP_MEAN) / CLIP_STD
        return {"image": arr, "label": torch.tensor(float(rec["label"])),
                "generator": rec.get("generator", "unknown")}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SyntheticDetector(model_name=args.clip_model).to(device)

    from scripts.build_manifest import load_synthetic_manifest
    train_manifest = load_synthetic_manifest(args.train_manifest)
    val_manifest = load_synthetic_manifest(args.val_manifest)

    train_ds = SyntheticImageDataset(train_manifest)
    val_ds = SyntheticImageDataset(val_manifest)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers)

    # Only optimize the two heads -- the encoder is frozen inside the model.
    trainable = list(model.global_head.parameters()) + list(model.patch_head.parameters())
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.01)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    best_auc = -1.0

    for epoch in range(args.epochs):
        model.train()
        model.encoder.eval()  # keep frozen encoder in eval mode (BN/dropout)
        running_loss = 0.0
        for batch in train_loader:
            img = batch["image"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_global(img)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            loss.backward()
            optimizer.step()
            running_loss += float(loss)

        # Validation with per-generator breakdown -- the honest reporting
        # format the blueprint asks for instead of one pooled number.
        model.eval()
        probs, labels, gens = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                logits = model.forward_global(img)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                labels.append(batch["label"].numpy())
                gens.extend(batch["generator"])
        probs = np.concatenate(probs)
        labels = np.concatenate(labels)

        auc = image_auc(probs, labels)
        ece = expected_calibration_error(probs, labels)
        print(f"[epoch {epoch}] loss={running_loss/len(train_loader):.4f} "
              f"val_auc={auc:.4f} val_ece={ece:.4f}")

        for gen in sorted(set(gens)):
            idx = [i for i, g in enumerate(gens) if g == gen]
            if len(idx) < 5:
                continue
            gen_auc = image_auc(probs[idx], labels[idx])
            print(f"    generator={gen:20s} n={len(idx):5d} auc={gen_auc:.4f}")

        ckpt = {"global_head": model.global_head.state_dict(),
                "patch_head": model.patch_head.state_dict(), "epoch": epoch, "auc": auc}
        torch.save(ckpt, Path(args.out_dir) / "last.pt")
        if auc > best_auc:
            best_auc = auc
            torch.save(ckpt, Path(args.out_dir) / "best.pt")
            print(f"  -> new best (auc={best_auc:.4f}), saved.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Branch C: synthetic content detection")
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--out-dir", default="runs/synthetic")
    p.add_argument("--clip-model", default="ViT-L-14")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
