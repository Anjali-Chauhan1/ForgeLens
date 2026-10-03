"""Content hashing for chain-of-custody and perceptual near-duplicate lookup."""
from __future__ import annotations
import hashlib
from pathlib import Path
from PIL import Image
import imagehash


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hashes(img: Image.Image) -> dict:
    """Multiple hash families: robust to different perturbation types."""
    return {
        "phash": str(imagehash.phash(img, hash_size=16)),
        "dhash": str(imagehash.dhash(img, hash_size=16)),
        "whash": str(imagehash.whash(img)),
        "ahash": str(imagehash.average_hash(img)),
    }


def hamming_distance(hash_a: str, hash_b: str) -> int:
    a = imagehash.hex_to_hash(hash_a)
    b = imagehash.hex_to_hash(hash_b)
    return a - b
