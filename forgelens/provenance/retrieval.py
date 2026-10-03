"""Near-duplicate retrieval: perceptual-hash + CLIP-embedding index over
a corpus of known/original images. Answers three high-value questions
per the blueprint: has this image been seen before in a different form,
has it been submitted before (insurance fraud rings), and is it a known
stock photo. When a near-duplicate is found, diffing against it gives a
GROUND-TRUTH tamper mask -- the highest-precision result the whole
system can produce, and it gets its own verdict band in the pipeline.

Uses FAISS for the embedding index when installed; degrades to a
perceptual-hash-only index (dependency already required by
utils/hashing.py) otherwise, which is slower to query at scale but
requires no extra install for local development.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import cv2

from forgelens.utils.hashing import perceptual_hashes, hamming_distance, sha256_file

try:
    import faiss
    _HAS_FAISS = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_FAISS = False


@dataclass
class IndexEntry:
    sha256: str
    path: str
    phash: str
    label: str = ""  # e.g. "stock", "prior_submission", "known_original"


@dataclass
class RetrievalMatch:
    entry: IndexEntry
    hamming_distance: int
    embedding_similarity: float | None = None


class PerceptualHashIndex:
    """Dependency-light fallback: linear scan over stored phash values.
    Fine up to tens of thousands of entries; swap for the FAISS-backed
    embedding index below at real production scale.
    """

    def __init__(self, store_path: str | None = None):
        self.entries: list[IndexEntry] = []
        self.store_path = store_path
        if store_path and Path(store_path).exists():
            self._load()

    def _load(self):
        with open(self.store_path) as f:
            for line in f:
                rec = json.loads(line)
                self.entries.append(IndexEntry(**rec))

    def _persist(self, entry: IndexEntry):
        if self.store_path:
            Path(self.store_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.store_path, "a") as f:
                f.write(json.dumps(entry.__dict__) + "\n")

    def add(self, image_path: str, label: str = "") -> IndexEntry:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        hashes = perceptual_hashes(img)
        entry = IndexEntry(sha256=sha256_file(image_path), path=image_path,
                            phash=hashes["phash"], label=label)
        self.entries.append(entry)
        self._persist(entry)
        return entry

    def query(self, image_path: str, max_hamming: int = 10, top_k: int = 5) -> list[RetrievalMatch]:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        query_hash = perceptual_hashes(img)["phash"]

        matches = []
        for entry in self.entries:
            dist = hamming_distance(query_hash, entry.phash)
            if dist <= max_hamming:
                matches.append(RetrievalMatch(entry=entry, hamming_distance=dist))
        matches.sort(key=lambda m: m.hamming_distance)
        return matches[:top_k]


class EmbeddingIndex:
    """FAISS-backed CLIP-embedding index for approximate near-duplicate
    search at scale, per the blueprint's stack recommendation. Embeddings
    must be produced by the same encoder used elsewhere (e.g. the frozen
    CLIP encoder in models/synthetic.py) for the similarity scores to be
    meaningful across the two branches.
    """

    def __init__(self, dim: int = 768):
        if not _HAS_FAISS:
            raise ImportError("faiss is not installed; `pip install faiss-cpu` "
                               "to use EmbeddingIndex, or use PerceptualHashIndex.")
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors
        self.entries: list[IndexEntry] = []

    def add(self, embedding: np.ndarray, entry: IndexEntry):
        vec = embedding.astype(np.float32).reshape(1, -1)
        vec /= (np.linalg.norm(vec) + 1e-8)
        self.index.add(vec)
        self.entries.append(entry)

    def query(self, embedding: np.ndarray, top_k: int = 5,
              min_similarity: float = 0.9) -> list[RetrievalMatch]:
        vec = embedding.astype(np.float32).reshape(1, -1)
        vec /= (np.linalg.norm(vec) + 1e-8)
        sims, idx = self.index.search(vec, min(top_k, len(self.entries) or 1))
        matches = []
        for sim, i in zip(sims[0], idx[0]):
            if i < 0 or sim < min_similarity:
                continue
            matches.append(RetrievalMatch(entry=self.entries[i], hamming_distance=-1,
                                           embedding_similarity=float(sim)))
        return matches


def diff_against_match(query_path: str, matched_path: str) -> dict:
    """Aligns (via ORB feature matching + homography) and differences the
    query image against a retrieved near-duplicate original. This is the
    ground-truth-diff path called out in the blueprint as the system's
    highest-precision result when a match is found -- much stronger than
    any of the four inferential branches alone.
    """
    from PIL import Image
    q = np.asarray(Image.open(query_path).convert("RGB"))
    m = np.asarray(Image.open(matched_path).convert("RGB"))

    q_gray = cv2.cvtColor(q, cv2.COLOR_RGB2GRAY)
    m_gray = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(q_gray, None)
    kp2, des2 = orb.detectAndCompute(m_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return {"aligned": False, "reason": "Insufficient keypoints to align images."}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:200]
    if len(matches) < 4:
        return {"aligned": False, "reason": "Insufficient reliable matches to align images."}

    src_pts = np.float32([kp1[m_.queryIdx].pt for m_ in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m_.trainIdx].pt for m_ in matches]).reshape(-1, 1, 2)
    H, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return {"aligned": False, "reason": "Homography estimation failed."}

    h, w = m_gray.shape
    warped = cv2.warpPerspective(q, H, (w, h))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)

    valid = (warped_gray > 0)
    diff = cv2.absdiff(warped_gray, m_gray).astype(np.float32)
    diff[~valid] = 0

    thresh = np.percentile(diff[valid], 90) if valid.any() else 255
    diff_mask = ((diff > max(thresh, 25)) & valid).astype(np.uint8)

    return {
        "aligned": True,
        "diff_mask": diff_mask,
        "diff_area_ratio": float(diff_mask.mean()),
        "inlier_matches": int(mask_h.sum()) if mask_h is not None else len(matches),
        "homography": H.tolist(),
    }
